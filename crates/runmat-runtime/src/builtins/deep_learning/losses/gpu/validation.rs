use super::*;
use runmat_accelerate_api::{ProviderExecutionError, ProviderPrecision};

pub(super) fn provider_error(error: impl Into<ProviderExecutionError>) -> crate::RuntimeError {
    let source = error.into();
    crate::runtime_error::with_provider_execution_source(
        deep_learning_error("crossentropy", format!("crossentropy: {source}")),
        source,
    )
}

pub(super) fn checked_len(shape: &[usize]) -> BuiltinResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| {
            deep_learning_error(
                "crossentropy",
                "crossentropy: tensor element count exceeds addressable storage",
            )
        })
}

pub(super) fn validate_live(handle: &GpuTensorHandle) -> BuiltinResult<()> {
    handle.allocation_lease().validate_live().map_err(|source| {
        crate::build_runtime_error("crossentropy: provider input allocation is retired")
            .with_builtin("crossentropy")
            .with_identifier("RunMat:deepLearning:InvalidInput")
            .with_source(source)
            .build()
    })
}

pub(super) fn validate_result_handle(
    result: &GpuTensorHandle,
    shape: &[usize],
    precision: Option<ProviderPrecision>,
    provider: &dyn AccelProvider,
    borrowed: &[&GpuTensorHandle],
) -> BuiltinResult<()> {
    if result.shape() != shape
        || runmat_accelerate_api::handle_storage(result) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_is_logical(result)
        || runmat_accelerate_api::handle_precision(result) != precision
        || crate::builtins::common::gpu_helpers::exact_provider_for_handle(result)
            .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
        || borrowed
            .iter()
            .any(|input| result.allocation_identity() == input.allocation_identity())
    {
        return Err(deep_learning_error("crossentropy", "crossentropy: provider returned a loss tensor with the wrong owner, shape, class, or allocation"));
    }
    validate_live(result)
}
