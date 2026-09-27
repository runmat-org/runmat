use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::BuiltinResult;

use super::super::super::provider_support::{
    broadcast_repetitions, valid_real_binary_output, ExpandedPair,
};
use super::super::{builtin_error, power_host, BUILTIN_NAME};

pub(in super::super) async fn power_gpu_pair(
    lhs: GpuTensorHandle,
    rhs: GpuTensorHandle,
) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&lhs).is_some()
        || runmat_accelerate_api::handle_integer_type(&rhs).is_some()
    {
        return gather_pair_and_evaluate(lhs, rhs).await;
    }

    if let Some(provider) = common_owner(&lhs, &rhs) {
        if !compatible_real_inputs(&lhs, &rhs) {
            return gather_pair_and_evaluate(lhs, rhs).await;
        }
        if lhs.shape == rhs.shape {
            match power(provider, &lhs, &rhs).await {
                Ok(handle)
                    if valid_real_binary_output(
                        &handle,
                        &lhs,
                        Some(&rhs),
                        provider,
                        &lhs.shape,
                    ) =>
                {
                    return Ok(Value::GpuTensor(handle));
                }
                Ok(handle) => {
                    gpu_helpers::free_rejected_provider_output(&handle, &[&lhs, &rhs], provider);
                    return Err(builtin_error(
                        "power: provider returned an invalid element-wise power result",
                    ));
                }
                Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
                    return gather_pair_and_evaluate(lhs, rhs).await
                }
                Err(error) => return Err(builtin_error(format!("power: {error}"))),
            }
        }
        if let Some((out_shape, lhs_repetitions, rhs_repetitions)) =
            broadcast_repetitions(&lhs.shape, &rhs.shape)
        {
            let expanded_lhs = lhs_repetitions.iter().any(|&count| count != 1);
            let expanded_rhs = rhs_repetitions.iter().any(|&count| count != 1);
            let mut expanded = ExpandedPair {
                provider,
                original_left: &lhs,
                original_right: &rhs,
                left: lhs.clone(),
                right: rhs.clone(),
                owns_left: false,
                owns_right: false,
            };
            if expanded_lhs {
                expanded.left = match replicate(provider, &lhs, &lhs_repetitions, &[&lhs, &rhs]) {
                    Ok(handle) => handle,
                    Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
                        return gather_pair_and_evaluate(lhs.clone(), rhs.clone()).await;
                    }
                    Err(error) => return Err(builtin_error(format!("power: {error}"))),
                };
                expanded.owns_left = true;
            }
            if expanded_rhs {
                expanded.right = match replicate(provider, &rhs, &rhs_repetitions, &[&lhs, &rhs]) {
                    Ok(handle) => handle,
                    Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
                        drop(expanded);
                        return gather_pair_and_evaluate(lhs, rhs).await;
                    }
                    Err(error) => return Err(builtin_error(format!("power: {error}"))),
                };
                expanded.owns_right = true;
            }
            if (expanded_lhs
                && !valid_real_binary_output(
                    &expanded.left,
                    &lhs,
                    Some(&rhs),
                    provider,
                    &out_shape,
                ))
                || (expanded_rhs
                    && !valid_real_binary_output(
                        &expanded.right,
                        &rhs,
                        Some(&lhs),
                        provider,
                        &out_shape,
                    ))
                || (expanded_lhs
                    && expanded_rhs
                    && gpu_helpers::same_gpu_handle(&expanded.left, &expanded.right))
            {
                expanded.release(&[]);
                return Err(builtin_error(
                    "power: provider returned invalid singleton-expanded input metadata",
                ));
            }
            let result = power(provider, &expanded.left, &expanded.right).await;
            match result {
                Ok(handle)
                    if !gpu_helpers::same_gpu_handle(&handle, &lhs)
                        && !gpu_helpers::same_gpu_handle(&handle, &rhs)
                        && valid_real_binary_output(
                            &handle,
                            &expanded.left,
                            Some(&expanded.right),
                            provider,
                            &out_shape,
                        ) =>
                {
                    expanded.release(&[&handle]);
                    return Ok(Value::GpuTensor(handle));
                }
                Ok(handle) => {
                    gpu_helpers::free_rejected_provider_output(
                        &handle,
                        &[&expanded.left, &expanded.right, &lhs, &rhs],
                        provider,
                    );
                    expanded.release(&[]);
                    return Err(builtin_error(
                        "power: provider returned an invalid element-wise power result",
                    ));
                }
                Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
                    expanded.release(&[]);
                }
                Err(error) => {
                    expanded.release(&[]);
                    return Err(builtin_error(format!("power: {error}")));
                }
            }
        }
    }

    gather_pair_and_evaluate(lhs, rhs).await
}

fn common_owner(
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
) -> Option<&'static dyn runmat_accelerate_api::AccelProvider> {
    let lhs_owner = gpu_helpers::exact_provider_for_handle(lhs)?;
    let rhs_owner = gpu_helpers::exact_provider_for_handle(rhs)?;
    (std::ptr::eq(lhs_owner, rhs_owner) && lhs.device_id == rhs.device_id).then_some(lhs_owner)
}

async fn power(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> anyhow::Result<GpuTensorHandle> {
    let _left = gpu_helpers::HandleMetadataRestoreGuard::new(left);
    let _right = gpu_helpers::HandleMetadataRestoreGuard::new(right);
    provider.elem_pow(left, right).await
}

fn replicate(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    repetitions: &[usize],
    protected: &[&GpuTensorHandle],
) -> anyhow::Result<GpuTensorHandle> {
    let _metadata: Vec<_> = protected
        .iter()
        .map(|handle| gpu_helpers::HandleMetadataRestoreGuard::new(handle))
        .collect();
    provider.repmat(input, repetitions)
}

fn compatible_real_inputs(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    runmat_accelerate_api::handle_storage(lhs) == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_storage(rhs)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && !runmat_accelerate_api::handle_is_logical(lhs)
        && !runmat_accelerate_api::handle_is_logical(rhs)
        && runmat_accelerate_api::handle_precision(lhs)
            == runmat_accelerate_api::handle_precision(rhs)
        && runmat_accelerate_api::handle_precision(lhs).is_some()
}

async fn gather_pair_and_evaluate(
    lhs: GpuTensorHandle,
    rhs: GpuTensorHandle,
) -> BuiltinResult<Value> {
    let lhs = gpu_helpers::gather_tensor_async(&lhs)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let rhs = gpu_helpers::gather_tensor_async(&rhs)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    power_host(Value::Tensor(lhs), Value::Tensor(rhs))
}
