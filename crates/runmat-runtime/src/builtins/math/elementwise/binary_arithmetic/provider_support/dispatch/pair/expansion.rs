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
            Ok(output) => output,
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => return Ok(None),
            Err(error) => return Err(provider_error(operation, error)),
        };
        expanded.expanded_left = true;
    }
    if right_repetitions.iter().any(|&count| count != 1) {
        expanded.right = match provider.repmat(right, right_repetitions) {
            Ok(output) => output,
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
                return if expanded.expanded_left {
                    Err(provider_error_after_effect(operation, error))
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
