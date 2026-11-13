//! Actual paired POWER body for its delivered compound. No second placement
//! decision, provider-hook replay or inferred-domain permission is used.
use crate::context::{preparation::AdmittedRealPowerCompound, RuntimeContext};
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_value::Value;

pub(super) async fn execute(
    runtime: Option<&RuntimeContext>,
    plan: AdmittedRealPowerCompound,
    left: Value,
    right: Value,
) -> BuiltinResult<RootedValue> {
    let result = execute_inner(runtime, &plan, &left, &right).await;
    result.map_err(|mut error| {
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}

async fn execute_inner(
    runtime: Option<&RuntimeContext>,
    plan: &AdmittedRealPowerCompound,
    left: &Value,
    right: &Value,
) -> BuiltinResult<RootedValue> {
    let recorder = runmat_accelerate_api::observation::current_provider_observation();
    if recorder
        .as_ref()
        .is_some_and(|recorder| !recorder.expects_power_pair(plan.physical()))
    {
        return Err(invalid());
    }
    plan.preflight_execution(runtime, &[left, right], recorder.is_some())?;
    let [base, exponent] = plan.physical().probe().sources();
    let owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(base)
        .ok_or_else(invalid)?;
    let cancellation = runtime.map(RuntimeContext::cancellation);
    let executed = plan
        .physical()
        .execute(cancellation.as_ref())
        .await
        .map_err(|cause| {
            use std::error::Error;
            let cancelled = runtime.is_some_and(RuntimeContext::is_cancelled)
                || cause.source().is_some_and(|source| {
                    source.is::<runmat_accelerate_api::ProviderCompoundCancelled>()
                });
            if cancelled {
                crate::runtime_error::with_provider_execution_source(
                    crate::build_runtime_error("power: compound execution cancelled")
                        .with_identifier("RunMat:Cancelled")
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build(),
                    cause,
                )
            } else {
                super::provider::typed_provider_error("compound execution", cause)
            }
        })?;
    let (output, evidence) = executed.into_parts();
    if let Err(cause) = plan.validate_current(runtime, &[left, right]) {
        return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
            output, cause,
        ));
    }
    let value = super::provider::publish_pair(
        owner,
        base,
        exponent,
        &plan.physical().probe().physical().output_shape,
        evidence.domain(),
        output,
    )?;
    let output = RootedValue::new(value).map_err(|cause| cause.into_runtime_error())?;
    plan.validate_current(runtime, &[left, right])?;
    if let Some(recorder) = recorder {
        recorder.confirm_power_pair_execution(&evidence);
    }
    Ok(output)
}

fn invalid() -> crate::RuntimeError {
    crate::build_runtime_error(
        "power: selected compound authority does not match its admitted body",
    )
    .with_identifier("RunMat:Placement:InvalidDecision")
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}
