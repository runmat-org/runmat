//! Async execution of a selected private host preparation. The existing
//! dispatcher retains the outer FullCall clock and final publication fence.
use super::*;
use runmat_runtime::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
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
    // Host policy is needed to price a host candidate, but a missing or stale
    // host ledger cannot reject a provider-only or unobserved preparation.
    let host_policy = complete_host.then(|| HostAllocationReceipt::policy(Some(context)));
    let managed = matches!(host_policy, Some(Ok(HostAllocationPolicy::Managed)));
    let initial_host_epoch = host_policy
        .as_ref()
        .filter(|policy| policy.is_ok())
        .and_then(|_| {
            context
                .service_ports()
                .parallel()
                .map(|parallel| parallel.placement_resources().epoch)
        });
    let initial_host_owner = if managed {
        context
            .service_ports()
            .parallel()
            .and_then(|parallel| parallel.managed_host_effect_ledger())
    } else {
        None
    };
    let admission = host_policy
        .as_ref()
        .filter(|policy| policy.is_ok())
        .and_then(|_| super::super::host_readback::admit(auto, context, &request, managed));
    // A host-only selected call has no resident-source Admission, but its
    // canonical resource envelope is still authenticated by this request.
    let host_only = if admission.is_none()
        && host_policy.as_ref().is_some_and(Result::is_ok)
        && !request
            .operands()
            .iter()
            .any(runmat_runtime::dispatcher::value_contains_gpu)
    {
        request.host_resource_bound(context).and_then(|envelope| {
            let (resources, identity) =
                super::super::host_readback::resources_for_envelope(&[], &envelope, managed)?;
            Some((envelope, resources, identity))
        })
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
            .map(|admission| (&admission.resources, admission.identity))
            .or_else(|| {
                host_only
                    .as_ref()
                    .map(|(_, resources, identity)| (resources, *identity))
            }),
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
    let Some(initial_policy) = host_policy else {
        observation.preparation_failed();
        return Err(failed("selected host call has no prepared host policy"));
    };
    let initial_policy = initial_policy.map_err(|cause| {
        observation.preparation_failed();
        policy_error(context, cause)
    })?;
    let current_policy = HostAllocationReceipt::policy(Some(context)).map_err(|cause| {
        observation.preparation_failed();
        policy_error(context, cause)
    })?;
    let current_host_epoch = context
        .service_ports()
        .parallel()
        .map(|parallel| parallel.placement_resources().epoch);
    let current_host_owner = if managed {
        context
            .service_ports()
            .parallel()
            .and_then(|parallel| parallel.managed_host_effect_ledger())
    } else {
        None
    };
    let same_host_owner = !managed
        || initial_host_owner
            .as_ref()
            .zip(current_host_owner.as_ref())
            .is_some_and(|(initial, current)| initial.same_owner(current));
    if current_policy != initial_policy
        || current_host_epoch != initial_host_epoch
        || !same_host_owner
    {
        observation.preparation_failed();
        return Err(policy_error(context, HostAllocationError::StaleAuthority));
    }
    if managed {
        let Some(envelope) = admission
            .as_ref()
            .map(|admission| &admission.envelope)
            .or_else(|| host_only.as_ref().map(|(envelope, _, _)| envelope))
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

fn policy_error(context: &RuntimeContext, cause: HostAllocationError) -> Box<RuntimeError> {
    Box::new(host_allocation_runtime_error(
        Some(context),
        "RunMat:Placement:HostNumericalOutput",
        "selected host allocation policy changed",
        cause,
    ))
}
