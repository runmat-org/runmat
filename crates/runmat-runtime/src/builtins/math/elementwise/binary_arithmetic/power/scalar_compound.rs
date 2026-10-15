//! Selected scalar POWER body. A pending projection becomes a completed
//! receipt only after its actual upload, paired probe/evaluation and original
//! scalar publication have succeeded.
use crate::context::{preparation::AdmittedRealPowerScalarCompound, RuntimeContext};
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_accelerate_api::power_service::execute_resident_scalar_power_projected;
use runmat_value::Value;

pub(super) async fn execute(
    runtime: Option<&RuntimeContext>,
    mut plan: AdmittedRealPowerScalarCompound,
    left: Value,
    right: Value,
) -> BuiltinResult<RootedValue> {
    let result = execute_inner(runtime, &mut plan, &left, &right).await;
    result.map_err(|mut error| {
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}

async fn execute_inner(
    runtime: Option<&RuntimeContext>,
    plan: &mut AdmittedRealPowerScalarCompound,
    left: &Value,
    right: &Value,
) -> BuiltinResult<RootedValue> {
    let recorder = runmat_accelerate_api::observation::current_provider_observation();
    if recorder
        .as_ref()
        .is_some_and(|recorder| !recorder.expects_power_scalar(plan.projection(), plan.source()))
    {
        return Err(invalid());
    }
    plan.preflight_execution(runtime, &[left, right], recorder.is_some())?;
    let prepared = plan.take_prepared().ok_or_else(invalid)?;
    let cancellation = runtime.map(RuntimeContext::cancellation);
    let executed =
        execute_resident_scalar_power_projected(prepared, plan.projection(), cancellation.as_ref())
            .await
            .map_err(|cause| {
                use std::error::Error;
                let cause = cause.into_execution_error();
                let cancelled = runtime.is_some_and(RuntimeContext::is_cancelled)
                    || cause.source().is_some_and(|source| {
                        source.is::<runmat_accelerate_api::ProviderCompoundCancelled>()
                    });
                crate::runtime_error::with_provider_execution_source(
                    crate::build_runtime_error(if cancelled {
                        "power: selected scalar compound cancelled"
                    } else {
                        "power: selected scalar compound failed"
                    })
                    .with_identifier(if cancelled {
                        "RunMat:Cancelled"
                    } else {
                        "RunMat:Placement:InvalidDecision"
                    })
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build(),
                    cause,
                )
            })?;
    if let Err(cause) = plan.validate_current(runtime, &[left, right]) {
        return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
            executed.handle,
            cause,
        ));
    }
    let Some(evidence) = executed.scalar_evidence else {
        return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
            executed.handle,
            invalid(),
        ));
    };
    let value = match executed.domain {
        runmat_accelerate_api::PowerResultDomain::Real => {
            crate::builtins::common::gpu_helpers::resident_gpu_value(executed.handle)?
        }
        runmat_accelerate_api::PowerResultDomain::Complex => {
            crate::builtins::common::gpu_helpers::complex_gpu_value(executed.handle)?
        }
    };
    let output = RootedValue::new(value).map_err(|cause| cause.into_runtime_error())?;
    plan.validate_current(runtime, &[left, right])?;
    if let Some(recorder) = recorder {
        recorder.confirm_power_scalar_execution(&evidence);
    }
    Ok(output)
}

fn invalid() -> crate::RuntimeError {
    crate::build_runtime_error("power: selected scalar compound authority is no longer current")
        .with_identifier("RunMat:Placement:InvalidDecision")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
