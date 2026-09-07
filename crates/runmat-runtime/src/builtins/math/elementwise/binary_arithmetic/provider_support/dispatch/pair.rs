use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::BuiltinCatalogIdentity;
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

use super::super::{
    broadcast_repetitions, device_real_scalar, ArithmeticProviderOperation, ExpandedPair,
};
use super::common::{
    accept_output, accepted_output, common_owner, compatible_inputs, float_handle_supported,
    provider_contract_error, provider_error,
};

pub(in crate::builtins::math::elementwise::binary_arithmetic) async fn try_pair(
    operation: ArithmeticProviderOperation,
    identity: BuiltinCatalogIdentity,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> BuiltinResult<Option<Value>> {
    let Some(provider) = common_owner(left, right) else {
        return Ok(None);
    };
    if !compatible_inputs(left, right) {
        return Ok(None);
    }
    if left.shape == right.shape {
        match operation.apply(provider, left, right).await {
            Ok(output) => {
                return accept_output(operation, output, left, Some(right), provider, &left.shape)
            }
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
            Err(error) => return Err(provider_error(operation, error)),
        }
    }
    if left.shape != right.shape {
        if let Some(output) = try_broadcast(operation, provider, left, right).await? {
            return Ok(Some(output));
        }
    }
    try_device_scalar(operation, identity, left, right).await
}

async fn try_broadcast(
    operation: ArithmeticProviderOperation,
    provider: &dyn AccelProvider,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> BuiltinResult<Option<Value>> {
    let Some((output_shape, left_repetitions, right_repetitions)) =
        broadcast_repetitions(&left.shape, &right.shape)
    else {
        return Ok(None);
    };
    let Some(expanded) = expand_pair(
        operation,
        provider,
        left,
        right,
        &left_repetitions,
        &right_repetitions,
    )?
    else {
        return Ok(None);
    };
    if !expanded_inputs_are_valid(&expanded, &output_shape) {
        expanded.release(&[]);
        return Err(provider_contract_error(
            operation,
            "provider returned invalid singleton-expanded input metadata",
        ));
    }
    let result = match operation
        .apply(provider, &expanded.left, &expanded.right)
        .await
    {
        Ok(output) => accept_output(
            operation,
            output,
            &expanded.left,
            Some(&expanded.right),
            provider,
            &output_shape,
        ),
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => Ok(None),
        Err(error) => Err(provider_error(operation, error)),
    };
    let protected = accepted_output(&result);
    expanded.release(&protected.into_iter().collect::<Vec<_>>());
    result
}

async fn try_device_scalar(
    operation: ArithmeticProviderOperation,
    identity: BuiltinCatalogIdentity,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> BuiltinResult<Option<Value>> {
    let Some(provider) = common_owner(left, right) else {
        return Ok(None);
    };
    if !float_handle_supported(left) || !float_handle_supported(right) {
        return Ok(None);
    }
    let (result, class_source) = if left.shape.iter().product::<usize>() <= 1 {
        let Some(scalar) = device_real_scalar(identity, left).await? else {
            return Ok(None);
        };
        (operation.scalar_left(provider, right, scalar), right)
    } else if right.shape.iter().product::<usize>() <= 1 {
        let Some(scalar) = device_real_scalar(identity, right).await? else {
            return Ok(None);
        };
        (operation.scalar_right(provider, left, scalar), left)
    } else {
        return Ok(None);
    };
    match result {
        Ok(output) => accept_output(
            operation,
            output,
            class_source,
            None,
            provider,
            &class_source.shape,
        ),
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => Ok(None),
        Err(error) => Err(provider_error(operation, error)),
    }
}

fn expand_pair<'a>(
    operation: ArithmeticProviderOperation,
    provider: &'a dyn AccelProvider,
    left: &'a GpuTensorHandle,
    right: &'a GpuTensorHandle,
    left_repetitions: &[usize],
    right_repetitions: &[usize],
) -> BuiltinResult<Option<ExpandedPair<'a>>> {
    let owns_left = left_repetitions.iter().any(|&count| count != 1);
    let owns_right = right_repetitions.iter().any(|&count| count != 1);
    let expanded_left = if owns_left {
        match provider.repmat(left, left_repetitions) {
            Ok(output) => output,
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => return Ok(None),
            Err(error) => return Err(provider_error(operation, error)),
        }
    } else {
        left.clone()
    };
    let expanded_right = if owns_right {
        match provider.repmat(right, right_repetitions) {
            Ok(output) => output,
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
                if owns_left {
                    gpu_helpers::free_rejected_provider_output(&expanded_left, &[left], provider);
                }
                return Ok(None);
            }
            Err(error) => {
                if owns_left {
                    gpu_helpers::free_rejected_provider_output(&expanded_left, &[left], provider);
                }
                return Err(provider_error(operation, error));
            }
        }
    } else {
        right.clone()
    };
    Ok(Some(ExpandedPair {
        provider,
        original_left: left,
        original_right: right,
        left: expanded_left,
        right: expanded_right,
        owns_left,
        owns_right,
    }))
}

fn expanded_inputs_are_valid(expanded: &ExpandedPair<'_>, expected_shape: &[usize]) -> bool {
    use super::super::valid_real_binary_output;

    (!expanded.owns_left
        || valid_real_binary_output(
            &expanded.left,
            expanded.original_left,
            None,
            expanded.provider,
            expected_shape,
        ))
        && (!expanded.owns_right
            || valid_real_binary_output(
                &expanded.right,
                expanded.original_right,
                None,
                expanded.provider,
                expected_shape,
            ))
}
