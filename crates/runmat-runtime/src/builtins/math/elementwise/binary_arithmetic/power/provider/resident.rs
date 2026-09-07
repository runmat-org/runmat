use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::BuiltinResult;

use super::super::super::provider_support::{host_real_scalar, valid_real_binary_output};
use super::super::{builtin_error, power_host, BUILTIN_NAME};
use super::conversion::{
    gather_value, is_complex, is_integer, tensor_matches_handle, value_to_real_tensor,
};

#[derive(Clone, Copy)]
enum ResidentOperand {
    Base,
    Exponent,
}

pub(in super::super) async fn power_gpu_host_left(
    base: GpuTensorHandle,
    exponent: Value,
) -> BuiltinResult<Value> {
    power_gpu_host(base, exponent, ResidentOperand::Base).await
}

pub(in super::super) async fn power_gpu_host_right(
    base: Value,
    exponent: GpuTensorHandle,
) -> BuiltinResult<Value> {
    power_gpu_host(exponent, base, ResidentOperand::Exponent).await
}

async fn power_gpu_host(
    resident: GpuTensorHandle,
    host: Value,
    resident_operand: ResidentOperand,
) -> BuiltinResult<Value> {
    if is_complex(&host)
        || is_integer(&host)
        || runmat_accelerate_api::handle_integer_type(&resident).is_some()
    {
        return gather_and_evaluate(resident, host, resident_operand).await;
    }

    if let Some(provider) = gpu_helpers::exact_provider_for_handle(&resident) {
        if let Some(scalar) = host_real_scalar(&host) {
            if let Some(output) =
                try_resident_scalar(provider, &resident, scalar, resident_operand).await?
            {
                return Ok(Value::GpuTensor(output));
            }
        } else if let Some(host_tensor) = value_to_real_tensor(&host).await? {
            if host_tensor.shape == resident.shape && tensor_matches_handle(&host_tensor, &resident)
            {
                let uploaded = gpu_helpers::upload_tensor(provider, &host_tensor)?;
                let result = accept_result(
                    resident_operand
                        .elem_pow(provider, &resident, &uploaded)
                        .await,
                    provider,
                    &resident,
                    Some(&uploaded),
                );
                gpu_helpers::free_unprotected_exact_owner(&uploaded, &[&resident]);
                if let Some(output) = result? {
                    return Ok(Value::GpuTensor(output));
                }
            }
        }
    }

    gather_and_evaluate(resident, host, resident_operand).await
}

async fn try_resident_scalar(
    provider: &dyn AccelProvider,
    resident: &GpuTensorHandle,
    scalar: f64,
    resident_operand: ResidentOperand,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    if let Some(uploaded) =
        gpu_helpers::upload_exact_integer_scalar_like(provider, resident, scalar)
            .map_err(|error| builtin_error(format!("power: {error}")))?
    {
        let result = accept_result(
            resident_operand
                .elem_pow(provider, resident, &uploaded)
                .await,
            provider,
            resident,
            Some(&uploaded),
        );
        gpu_helpers::free_unprotected_exact_owner(&uploaded, &[resident]);
        if let Some(output) = result? {
            return Ok(Some(output));
        }
    }

    let filled = match provider.fill_like(resident, scalar) {
        Ok(handle) => handle,
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => return Ok(None),
        Err(error) => return Err(builtin_error(format!("power: {error}"))),
    };
    if !valid_real_binary_output(&filled, resident, None, provider, &resident.shape) {
        gpu_helpers::free_rejected_provider_output(&filled, &[resident], provider);
        return Err(builtin_error(
            "power: provider returned invalid scalar-fill metadata",
        ));
    }
    let result = accept_result(
        resident_operand.elem_pow(provider, resident, &filled).await,
        provider,
        resident,
        Some(&filled),
    );
    gpu_helpers::free_unprotected_exact_owner(&filled, &[resident]);
    result
}

fn accept_result(
    result: anyhow::Result<GpuTensorHandle>,
    provider: &dyn AccelProvider,
    class_source: &GpuTensorHandle,
    other_source: Option<&GpuTensorHandle>,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    match result {
        Ok(output)
            if valid_real_binary_output(
                &output,
                class_source,
                other_source,
                provider,
                &class_source.shape,
            ) =>
        {
            Ok(Some(output))
        }
        Ok(output) => {
            let protected =
                other_source.map_or_else(|| vec![class_source], |other| vec![class_source, other]);
            gpu_helpers::free_rejected_provider_output(&output, &protected, provider);
            Err(builtin_error(
                "power: provider returned an invalid element-wise power result",
            ))
        }
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => Ok(None),
        Err(error) => Err(builtin_error(format!("power: {error}"))),
    }
}

async fn gather_and_evaluate(
    resident: GpuTensorHandle,
    host: Value,
    resident_operand: ResidentOperand,
) -> BuiltinResult<Value> {
    let resident = gpu_helpers::gather_tensor_async(&resident)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let host = gather_value(host).await?;
    match resident_operand {
        ResidentOperand::Base => power_host(Value::Tensor(resident), host),
        ResidentOperand::Exponent => power_host(host, Value::Tensor(resident)),
    }
}

impl ResidentOperand {
    async fn elem_pow(
        self,
        provider: &dyn AccelProvider,
        resident: &GpuTensorHandle,
        host: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        match self {
            Self::Base => provider.elem_pow(resident, host).await,
            Self::Exponent => provider.elem_pow(host, resident).await,
        }
    }
}
