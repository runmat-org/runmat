use super::super::{builtin_error, power_host, BUILTIN_NAME};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::BuiltinResult;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, PowerResultDomain};
use runmat_value::Value;

pub(in super::super) async fn power_gpu_pair(
    lhs: GpuTensorHandle,
    rhs: GpuTensorHandle,
) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&lhs).is_none()
        && runmat_accelerate_api::handle_integer_type(&rhs).is_none()
    {
        if let Some(provider) = common_owner(&lhs, &rhs) {
            if let Some(output) = try_real_pair(provider, &lhs, &rhs).await? {
                return Ok(output);
            }
        }
    }
    gather_pair_and_evaluate(lhs, rhs).await
}

pub(super) async fn try_real_pair(
    provider: &'static dyn AccelProvider,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
) -> BuiltinResult<Option<Value>> {
    if !compatible_real_inputs(lhs, rhs) {
        return Ok(None);
    }
    let shape =
        crate::builtins::common::broadcast::broadcast_shapes(BUILTIN_NAME, &lhs.shape, &rhs.shape)
            .map_err(builtin_error)?;
    let source =
        gpu_helpers::select_resident_output_source([lhs.clone(), rhs.clone()], BUILTIN_NAME)?;
    if !super::admission::may_execute(provider, lhs, rhs)
        .map_err(|error| builtin_error(format!("power: {error}")))?
    {
        return Ok(None);
    }
    let domain = {
        let _left = gpu_helpers::ResidencyRestoreGuard::new(lhs);
        let _right = gpu_helpers::ResidencyRestoreGuard::new(rhs);
        match provider.classify_real_power(lhs, rhs).await {
            Ok(domain) => domain,
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => return Ok(None),
            Err(error) => {
                return Err(builtin_error(format!(
                    "power: domain classification failed: {error}"
                )))
            }
        }
    };
    let result = {
        let _left = gpu_helpers::ResidencyRestoreGuard::new(lhs);
        let _right = gpu_helpers::ResidencyRestoreGuard::new(rhs);
        provider.real_power_in_domain(lhs, rhs, domain).await
    };
    match result {
        Ok(mut output) if valid_output(&output, lhs, rhs, provider, &shape, domain) => {
            if let Some(source) = source {
                let provenance = runmat_accelerate_api::handle_provenance(&source)
                    .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
                runmat_accelerate_api::set_handle_provenance(&mut output, provenance);
            }
            Ok(Some(if domain == PowerResultDomain::Complex {
                gpu_helpers::complex_gpu_value(output)
            } else {
                gpu_helpers::resident_gpu_value(output)
            }))
        }
        Ok(output) => {
            gpu_helpers::free_rejected_provider_output(&output, &[lhs, rhs], provider);
            Err(builtin_error(
                "power: provider returned malformed domain-preserving output",
            ))
        }
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => Ok(None),
        Err(error) => Err(builtin_error(format!("power: {error}"))),
    }
}

fn valid_output(
    output: &GpuTensorHandle,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    shape: &[usize],
    domain: PowerResultDomain,
) -> bool {
    use gpu_helpers::{BinaryGpuOutputContract, GpuOutputAliasPolicy};
    let storage = match domain {
        PowerResultDomain::Real => GpuTensorStorage::Real,
        PowerResultDomain::Complex => GpuTensorStorage::ComplexInterleaved,
    };
    output.descriptor.storage == Some(storage)
        && gpu_helpers::binary_gpu_output_matches(
            output,
            lhs,
            rhs,
            provider,
            &BinaryGpuOutputContract {
                shape: shape.to_vec(),
                storage,
                precision: runmat_accelerate_api::handle_precision(lhs),
                integer: None,
                logical: false,
                alias: GpuOutputAliasPolicy::RequireDistinct,
            },
        )
}

fn common_owner(
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
) -> Option<&'static dyn AccelProvider> {
    let left = gpu_helpers::exact_provider_for_handle(lhs)?;
    let right = gpu_helpers::exact_provider_for_handle(rhs)?;
    (std::ptr::eq(left, right) && lhs.device_id == rhs.device_id).then_some(left)
}

fn compatible_real_inputs(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    [lhs, rhs].into_iter().all(|handle| {
        handle.descriptor.storage == Some(GpuTensorStorage::Real)
            && !runmat_accelerate_api::handle_is_logical(handle)
            && runmat_accelerate_api::handle_integer_type(handle).is_none()
            && gpu_helpers::expected_handle_numeric_element_type(handle).is_ok()
    }) && runmat_accelerate_api::handle_precision(lhs).is_some()
        && runmat_accelerate_api::handle_precision(lhs)
            == runmat_accelerate_api::handle_precision(rhs)
}

async fn gather_pair_and_evaluate(
    lhs: GpuTensorHandle,
    rhs: GpuTensorHandle,
) -> BuiltinResult<Value> {
    let left = gpu_helpers::gather_tensor_async(&lhs)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let right = gpu_helpers::gather_tensor_async(&rhs)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    power_host(Value::Tensor(left), Value::Tensor(right))
}
