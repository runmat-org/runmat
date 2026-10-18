//! Async execution of a selected private host preparation. The existing
//! dispatcher retains the outer FullCall clock and final publication fence.
use super::*;
use runmat_runtime::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
};

pub(in crate::native_auto::builtin_call) async fn prepare_async(
    auto: &NativeAutoOffload,
    context: &RuntimeContext,
    request: AdmittedBuiltinPreparation,
) -> PreparationResult<PreparedBuiltinCall> {
    let root = runmat_runtime::execution::RootedValueLease::for_values(request.operands().iter())
        .map_err(|error| Box::new(error.into_runtime_error()))?;
    let form = request.form();
    let compound = request
        .plan_compound(Some(context), auto.provider)
        .map_err(Box::new)?;
    let complete_host = request.has_prepared_completion_right()
        && runmat_runtime::context::preparation::PreparedHostExecution::supports_form(form);
    let admission = request
        .has_prepared_completion_right()
        .then(|| super::super::host_readback::admit(auto, context, &request))
        .flatten();
    // A host-only selected call has no resident-source Admission, but its
    // canonical resource envelope is still authenticated by this request.
    let host_only_envelope = if admission.is_none() && complete_host {
        request.host_resource_bound(context)
    } else {
        None
    };
    // A newly eligible promoting resident call has no provider-domain proof
    // yet. Without an executable private-host envelope this optional route is
    // not applicable: leave the original domain-owned implementation intact.
    // This is before planning/resource admission; no rejection is caught here.
    if admission.is_none()
        && compound.is_none()
        && request.numerical_domain() == runmat_accelerate_api::NumericalDomainEvidence::Unknown
        && matches!(
            form.operation,
            Operation::RealUnary(_) | Operation::RealPower
        )
        && request
            .operands()
            .iter()
            .any(runmat_runtime::dispatcher::value_contains_gpu)
    {
        return Ok(request
            .into_operands()
            .with_unbounded_values(PreparedBuiltinCall::unobserved)
            .map_err(Box::new)?
            .with_request_input_root(root));
    }
    let mut prepared = prepare_with_inner(
        auto,
        context,
        request,
        admission
            .as_ref()
            .map(|admission| (&admission.resources, admission.identity)),
        compound.as_ref(),
    )?
    .with_request_input_root(root);
    if prepared
        .observation
        .as_ref()
        .is_some_and(|observation| observation.provider_selected())
    {
        prepared.compound = compound;
    }
    let Some(observation) = prepared
        .observation
        .as_mut()
        .filter(|observation| complete_host && observation.host_selected())
    else {
        return Ok(prepared);
    };
    let Some(proof) = observation.select_private_host(context, form) else {
        observation.preparation_failed();
        return Err(failed(
            "selected prepared host authority is no longer current",
        ));
    };
    if !proof.is_current() {
        observation.preparation_failed();
        return Err(failed("prepared host authority expired before readback"));
    }
    let managed = HostAllocationReceipt::policy(Some(context)).map_err(|cause| {
        Box::new(host_allocation_runtime_error(
            Some(context),
            "RunMat:Placement:HostCastOutput",
            "selected host allocation policy changed",
            cause,
        ))
    })? == HostAllocationPolicy::Managed;
    if managed {
        let Some(envelope) = admission
            .as_ref()
            .map(|admission| &admission.envelope)
            .or(host_only_envelope.as_ref())
        else {
            observation.preparation_failed();
            return Err(failed(
                "selected host call has no exact numerical output envelope",
            ));
        };
        prepared.host_numerical_output = Some(
            runmat_runtime::context::preparation::PreparedHostNumericalOutput::prepare_admitted(
                context, envelope,
            )
            .map_err(|cause| {
                Box::new(host_allocation_runtime_error(
                    Some(context),
                    "RunMat:Placement:HostNumericalOutput",
                    "selected host numerical output admission failed",
                    cause,
                ))
            })?
            .ok_or_else(|| failed("selected host call lost its finite output owner"))?,
        );
    }
    let Some(admission) = admission else {
        return Ok(prepared);
    };
    let readback = if managed {
        admission
            .plan
            .prepare_inputs_managed(&prepared.args, context)
            .await
    } else {
        admission.plan.prepare_inputs(&prepared.args).await
    };
    let mut inputs = match readback {
        Ok(inputs) => inputs,
        Err(cause) => {
            observation.preparation_failed();
            let mut error = match cause {
                crate::fusion_exec::host_readback::CpuReadbackFailure::BeforeEffect(error)
                | crate::fusion_exec::host_readback::CpuReadbackFailure::AfterEffect(error) => {
                    error
                }
            };
            error.gpu_gather_retry = runmat_runtime::GpuGatherRetry::Never;
            return Err(Box::new(error));
        }
    };
    if !proof.is_current() {
        observation.preparation_failed();
        return Err(failed("prepared host authority expired during readback"));
    }
    // All source checks and downloads completed before publishing even one
    // operand. These values belong to the invocation, not the workspace.
    for (offset, value) in std::mem::take(&mut inputs.entry) {
        prepared.args[offset] = value;
    }
    prepared.readbacks = Some(inputs);
    Ok(prepared)
}

fn failed(message: &'static str) -> Box<RuntimeError> {
    let mut error = runmat_runtime::build_runtime_error(message)
        .with_identifier("RunMat:Placement:StaleHostReadback")
        .build();
    error.gpu_gather_retry = runmat_runtime::GpuGatherRetry::Never;
    Box::new(error)
}
