//! Shared execution mechanics for resident real-unary functions across math families.
//!
//! Builtin identities own their public errors, numerical kernels, and provider
//! hooks. This module owns the family invariant for resident real floating-point
//! inputs so ownership and output validation do not drift across families.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_value::Tensor;

use crate::builtins::common::gpu_helpers;

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

pub(super) fn exact_owner(handle: &GpuTensorHandle) -> Option<&'static dyn AccelProvider> {
    gpu_helpers::exact_provider_for_handle(handle)
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
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        },
    )
}

pub(super) fn reject_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) {
    gpu_helpers::free_rejected_provider_output(output, &[input], provider);
}

pub(super) fn preserve_residency_intent(output: &mut GpuTensorHandle, input: &GpuTensorHandle) {
    let provenance = runmat_accelerate_api::handle_provenance(input)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    runmat_accelerate_api::set_handle_provenance(output, provenance);
}

pub(super) fn restore_fallback(
    result: &Tensor,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> Result<GpuTensorHandle, String> {
    let mut output = gpu_helpers::upload_tensor(provider, result)?;
    if output_matches(&output, input, provider) {
        preserve_residency_intent(&mut output, input);
        Ok(output)
    } else {
        reject_output(&output, input, provider);
        Err("provider upload returned malformed fallback output".into())
    }
}

pub(super) fn hook_is_unsupported(error: &anyhow::Error) -> bool {
    gpu_helpers::provider_hook_is_unsupported(error)
}

#[cfg(test)]
mod tests {
    use super::*;

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
