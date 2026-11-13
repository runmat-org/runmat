//! Imfilter's provider-error presentation around the shared pure diagnostic.

use super::super::kernel::ImfilterKernelError;

pub(super) fn with_provider_source(
    mut fallback: crate::RuntimeError,
    source: runmat_accelerate_api::ProviderExecutionError,
) -> crate::RuntimeError {
    let mut cursor: &(dyn std::error::Error + 'static) = &source;
    loop {
        if let Some(kernel) = cursor.downcast_ref::<ImfilterKernelError>() {
            // Match the host adapter's descriptor while retaining the entire
            // provider cause below, including the original pure diagnostic.
            fallback = super::imfilter_error_with_detail(
                &super::IMFILTER_ERROR_INVALID_INPUT,
                &kernel.message,
            );
            break;
        }
        let Some(next) = cursor.source() else { break };
        cursor = next;
    }
    crate::runtime_error::with_provider_execution_source(fallback, source)
}
