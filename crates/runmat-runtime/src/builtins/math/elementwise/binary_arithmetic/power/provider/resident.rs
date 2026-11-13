use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::{NumericScalar, Value};

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::BuiltinResult;

use super::super::super::provider_support::host_real_scalar;
use super::super::super::provider_support::HostFallbackAdmission;
use super::super::{builtin_error, power_host, BUILTIN_NAME};
use super::conversion::{
    gather_value, is_complex, is_integer, tensor_matches_handle, value_to_real_tensor,
};

#[derive(Clone, Copy)]
pub(super) enum ResidentOperand {
    Base,
    Exponent,
}

pub(in super::super) async fn power_gpu_host_left(
    runtime: Option<&crate::context::RuntimeContext>,
    base: GpuTensorHandle,
    exponent: Value,
) -> BuiltinResult<Value> {
    power_gpu_host(runtime, base, exponent, ResidentOperand::Base).await
}

pub(in super::super) async fn power_gpu_host_right(
    runtime: Option<&crate::context::RuntimeContext>,
    base: Value,
    exponent: GpuTensorHandle,
) -> BuiltinResult<Value> {
    power_gpu_host(runtime, exponent, base, ResidentOperand::Exponent).await
}

async fn power_gpu_host(
    runtime: Option<&crate::context::RuntimeContext>,
    resident: GpuTensorHandle,
    host: Value,
    resident_operand: ResidentOperand,
) -> BuiltinResult<Value> {
    if is_complex(&host)
        || is_integer(&host)
        || runmat_accelerate_api::handle_integer_type(&resident).is_some()
    {
        return gather_and_evaluate(runtime, resident, host, resident_operand).await;
    }

    if let Some(provider) = gpu_helpers::exact_provider_for_handle(&resident) {
        if host_real_scalar(&host).is_some() {
            if let Some(output) =
                try_resident_scalar(provider, &resident, &host, resident_operand).await?
            {
                return Ok(output);
            }
        } else if let Some(host_tensor) = value_to_real_tensor(&host).await? {
            if host_tensor.shape() == resident.shape()
                && tensor_matches_handle(&host_tensor, &resident)
            {
                let uploaded = {
                    let _metadata = gpu_helpers::ResidencyRestoreGuard::new(&resident);
                    gpu_helpers::upload_tensor_protecting(provider, &host_tensor, &[&resident])?
                };
                let uploaded_owner = uploaded;
                let uploaded = &uploaded_owner;
                let result = resident_operand
                    .typed_power(provider, &resident, uploaded)
                    .await;
                let result = result.map_err(|error| {
                    crate::runtime_error::with_provider_execution_source(
                        builtin_error("power: uploaded pair failed after operand materialization"),
                        runmat_accelerate_api::ProviderExecutionError::from_source(error)
                            .after_effect(),
                    )
                })?;
                if let Some(output) = result {
                    return Ok(output);
                }
                return Err(super::pair::typed_provider_error(
                    "uploaded pair admission",
                    runmat_accelerate_api::ProviderExecutionError::from(
                        runmat_accelerate_api::unsupported_provider_operation(
                            "POWER pair declined after host operand upload",
                        ),
                    ),
                ));
            }
        }
    }

    gather_and_evaluate(runtime, resident, host, resident_operand).await
}

async fn try_resident_scalar(
    provider: &'static dyn AccelProvider,
    resident: &GpuTensorHandle,
    host: &Value,
    resident_operand: ResidentOperand,
) -> BuiltinResult<Option<Value>> {
    use runmat_accelerate_api::power_service::{
        execute_resident_scalar_power, prepare_resident_scalar_power,
    };
    use runmat_accelerate_api::{
        PowerResultDomain, PowerScalarControl, ProviderOperationIdentity, RealPowerOperand,
    };
    let (control, host_shape) = match host {
        Value::Num(value) => (PowerScalarControl::DoubleBits(value.to_bits()), &[1, 1][..]),
        Value::Tensor(tensor) if tensor.integer_storage().is_none() => {
            let control = match tensor.numeric_value_at(0) {
                Some(NumericScalar::F32(value)) => PowerScalarControl::SingleBits(value.to_bits()),
                Some(NumericScalar::F64(value)) => PowerScalarControl::DoubleBits(value.to_bits()),
                _ => return Ok(None),
            };
            (control, tensor.shape())
        }
        _ => return Ok(None),
    };
    let role = match resident_operand {
        ResidentOperand::Base => RealPowerOperand::Base,
        ResidentOperand::Exponent => RealPowerOperand::Exponent,
    };
    let prepared = prepare_resident_scalar_power(
        provider,
        resident,
        control,
        role,
        host_shape,
        ProviderOperationIdentity::new(runmat_builtins::POWER_CATALOG_ENTRY.identity.name),
    )
    .map_err(|error| service_error("scalar preparation", error))?;
    if let Some(prepared) = prepared {
        let executed = execute_resident_scalar_power(prepared)
            .await
            .map_err(|error| service_error("scalar execution", error))?;
        return if executed.domain == PowerResultDomain::Complex {
            gpu_helpers::complex_gpu_value(executed.handle).map(Some)
        } else {
            gpu_helpers::resident_gpu_value(executed.handle).map(Some)
        };
    }
    Ok(None)
}

fn service_error(
    stage: &str,
    error: runmat_accelerate_api::power_service::PowerServiceError,
) -> crate::RuntimeError {
    let diagnostic = builtin_error(format!("power: {stage} failed: {error}"));
    crate::runtime_error::with_provider_execution_source(diagnostic, error.into_execution_error())
}

async fn gather_and_evaluate(
    runtime: Option<&crate::context::RuntimeContext>,
    resident: GpuTensorHandle,
    host: Value,
    resident_operand: ResidentOperand,
) -> BuiltinResult<Value> {
    let fallback = match resident_operand {
        ResidentOperand::Base => HostFallbackAdmission::new(
            runtime,
            &runmat_builtins::POWER_CATALOG_ENTRY,
            &Value::GpuTensor(resident.clone()),
            &host,
        )?,
        ResidentOperand::Exponent => HostFallbackAdmission::new(
            runtime,
            &runmat_builtins::POWER_CATALOG_ENTRY,
            &host,
            &Value::GpuTensor(resident.clone()),
        )?,
    };
    let resident = gpu_helpers::gather_tensor_async(&resident)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let host = gather_value(host).await?;
    let output = match resident_operand {
        ResidentOperand::Base => power_host(Value::Tensor(resident), host),
        ResidentOperand::Exponent => power_host(host, Value::Tensor(resident)),
    }?;
    fallback.finish(output)
}

impl ResidentOperand {
    async fn typed_power(
        self,
        provider: &'static dyn AccelProvider,
        resident: &GpuTensorHandle,
        host: &GpuTensorHandle,
    ) -> BuiltinResult<Option<Value>> {
        match self {
            Self::Base => super::pair::try_real_pair(provider, resident, host).await,
            Self::Exponent => super::pair::try_real_pair(provider, host, resident).await,
        }
    }
}

#[cfg(test)]
mod error_tests;
