//! Async execution of a selected private host preparation. The existing
//! dispatcher retains the outer FullCall clock and final publication fence.
use super::*;

pub(in crate::native_auto::builtin_call) async fn prepare_async(
    auto: &NativeAutoOffload,
    context: &RuntimeContext,
    request: AdmittedBuiltinPreparation,
) -> PreparationResult<PreparedBuiltinCall> {
    let root = runmat_runtime::execution::RootedValueLease::for_values(request.operands().iter())
        .map_err(|error| Box::new(error.into_runtime_error()))?;
    let form = request.form();
    let complete_host = request.has_prepared_completion_right()
        && runmat_runtime::context::preparation::PreparedHostExecution::supports_form(form);
    let admission = request
        .has_prepared_completion_right()
        .then(|| super::super::host_readback::admit(auto, context, &request))
        .flatten();
    let mut prepared = prepare_with_inner(
        auto,
        context,
        request,
        admission
            .as_ref()
            .map(|admission| (&admission.resources, admission.identity)),
    )?
    .with_request_input_root(root);
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
    let Some(admission) = admission else {
        return Ok(prepared);
    };
    let mut inputs = match admission.plan.prepare_inputs(&prepared.args).await {
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
