//! Singleton expansion and validation of temporary provider-owned inputs.

use super::super::super::{valid_real_binary_output, ExpandedPair};
use super::*;
use crate::builtins::common::gpu_helpers;

pub(super) fn expand_pair<'a>(
    operation: ArithmeticProviderOperation,
    provider: &'a dyn AccelProvider,
    left: &'a GpuTensorHandle,
    right: &'a GpuTensorHandle,
    left_repetitions: &[usize],
    right_repetitions: &[usize],
) -> BuiltinResult<Option<ExpandedPair<'a>>> {
    let _left_metadata = gpu_helpers::ResidencyRestoreGuard::new(left);
    let _right_metadata = gpu_helpers::ResidencyRestoreGuard::new(right);
    let mut expanded = ExpandedPair {
        provider,
        original_left: left,
        original_right: right,
        left: left.clone(),
        right: right.clone(),
        expanded_left: false,
        expanded_right: false,
    };
    if left_repetitions.iter().any(|&count| count != 1) {
        expanded.left = match provider.repmat(left, left_repetitions) {
            Ok(Some(output)) => output,
            Ok(None) => return Ok(None),
            Err(error) => return Err(provider_error(operation, error)),
        };
        expanded.expanded_left = true;
        let shape = gpu_helpers::replication_shape(left.shape(), left_repetitions);
        if !shape.as_ref().is_some_and(|shape| {
            gpu_helpers::replication_output_matches(left, &expanded.left, provider, shape)
        }) || expanded.left.shares_allocation_with(right)
        {
            return Err(provider_contract_error(
                operation,
                "provider returned invalid left expansion",
            ));
        }
    }
    if right_repetitions.iter().any(|&count| count != 1) {
        expanded.right = match provider.repmat(right, right_repetitions) {
            Ok(Some(output)) => output,
            Ok(None) => {
                return if expanded.expanded_left {
                    Err(provider_error_after_effect(
                        operation,
                        anyhow::anyhow!(
                            "required right expansion unavailable after left expansion"
                        ),
                    ))
                } else {
                    Ok(None)
                };
            }
            Err(error) => {
                return Err(if expanded.expanded_left {
                    provider_error_after_effect(operation, error)
                } else {
                    provider_error(operation, error)
                });
            }
        };
        expanded.expanded_right = true;
        let shape = gpu_helpers::replication_shape(right.shape(), right_repetitions);
        if !shape.as_ref().is_some_and(|shape| {
            gpu_helpers::replication_output_matches(right, &expanded.right, provider, shape)
        }) || expanded.right.shares_allocation_with(left)
            || (expanded.expanded_left && expanded.right.shares_allocation_with(&expanded.left))
        {
            return Err(provider_contract_error(
                operation,
                "provider returned invalid right expansion",
            ));
        }
    }
    Ok(Some(expanded))
}

pub(super) fn expanded_inputs_are_valid(
    expanded: &ExpandedPair<'_>,
    expected_shape: &[usize],
) -> bool {
    !(expanded.expanded_left
        && expanded.expanded_right
        && expanded.left.shares_allocation_with(&expanded.right))
        && (!expanded.expanded_left
            || valid_real_binary_output(
                &expanded.left,
                expanded.original_left,
                Some(expanded.original_right),
                expanded.provider,
                expected_shape,
            ))
        && (!expanded.expanded_right
            || valid_real_binary_output(
                &expanded.right,
                expanded.original_right,
                Some(expanded.original_left),
                expanded.provider,
                expected_shape,
            ))
}
