//! Shared execution mechanics for resident real-unary functions across math families.
//!
//! Builtin identities own their public errors, numerical kernels, and provider
//! hooks. This module owns the family invariant for resident real floating-point
//! inputs so ownership and output validation do not drift across families.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_value::Tensor;

use crate::builtins::common::gpu_helpers;

mod scalar_shape;
pub(super) use scalar_shape::preserve_elementwise_scalar_shape;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum InvalidResidentInput {
    Integer,
    Logical,
    Complex,
}

impl InvalidResidentInput {
    pub(super) const fn detail(self) -> &'static str {
        match self {
            Self::Integer => "integer-class gpuArray inputs are not supported",
            Self::Logical => "logical gpuArray inputs are not supported",
            Self::Complex => "complex gpuArray inputs are not supported",
        }
    }
}

pub(super) fn validate_input(handle: &GpuTensorHandle) -> Result<(), InvalidResidentInput> {
    if runmat_accelerate_api::handle_integer_type(handle).is_some() {
        return Err(InvalidResidentInput::Integer);
    }
    if runmat_accelerate_api::handle_is_logical(handle) {
        return Err(InvalidResidentInput::Logical);
    }
    if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved {
        return Err(InvalidResidentInput::Complex);
    }
    Ok(())
}

pub(super) fn exact_owner(
    handle: &GpuTensorHandle,
    error: impl FnOnce() -> crate::RuntimeError,
) -> crate::BuiltinResult<&'static dyn AccelProvider> {
    gpu_helpers::require_resident_compute_owner(handle, error)
}

pub(super) fn output_matches(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> bool {
    let precision = runmat_accelerate_api::handle_precision(input);
    matches!(
        precision,
        Some(ProviderPrecision::F32 | ProviderPrecision::F64)
    ) && gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision,
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    )
}

pub(super) fn reject_output(
    output: GpuTensorHandle,
    error: crate::RuntimeError,
) -> crate::RuntimeError {
    gpu_helpers::reject_gpu_output(output, error)
}

pub(super) fn preserve_residency_intent(
    output: &mut GpuTensorHandle,
    input: &GpuTensorHandle,
) -> crate::BuiltinResult<()> {
    gpu_helpers::propagate_output_provenance(output, [input])
}

pub(super) fn restore_fallback(
    result: &Tensor,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> crate::BuiltinResult<GpuTensorHandle> {
    let mut output = gpu_helpers::upload_tensor_protecting(provider, result, &[input])?;
    if output_matches(&output, input, provider) {
        preserve_residency_intent(&mut output, input)?;
        Ok(output)
    } else {
        Err(reject_output(
            output,
            crate::build_runtime_error("provider upload returned malformed fallback output")
                .build(),
        ))
    }
}

pub(super) fn hook_allows_host_fallback(error: &anyhow::Error) -> bool {
    gpu_helpers::provider_hook_allows_host_fallback(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_view_does_not_retire_independently_retained_input() {
        use runmat_accelerate_api::{test_support::TestAllocationOwner, NumericElementType};
        let owner = TestAllocationOwner::new();
        let allocation = owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let rejected = allocation
            .handle
            .with_view(allocation.handle.view().clone().with_shape(vec![1, 2]))
            .unwrap();
        let error = reject_output(
            rejected,
            crate::build_runtime_error("invalid output").build(),
        );
        drop(error);
        assert_eq!(allocation.probe.retirements(), 0);
        allocation
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        drop(allocation.handle);
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn resident_input_reason_is_stable() {
        assert_eq!(
            InvalidResidentInput::Integer.detail(),
            "integer-class gpuArray inputs are not supported"
        );
        assert_eq!(
            InvalidResidentInput::Logical.detail(),
            "logical gpuArray inputs are not supported"
        );
        assert_eq!(
            InvalidResidentInput::Complex.detail(),
            "complex gpuArray inputs are not supported"
        );
    }
}
