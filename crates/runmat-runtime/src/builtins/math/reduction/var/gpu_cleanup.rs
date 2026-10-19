//! Complete the owned std intermediate after the variance square attempt.
use super::*;

pub(super) async fn complete_std_square(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    std_handle: GpuTensorHandle,
    expected_shape: &[usize],
) -> BuiltinResult<GpuTensorHandle> {
    let result = match provider.elem_mul(&std_handle, &std_handle).await {
        Ok(square) => {
            if square.shares_allocation_with(&std_handle)
                || !super::super::std::std_output_matches(provider, input, &square, expected_shape)
            {
                let mut error =
                    var_internal_error("var: provider square returned an invalid resident result");
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                Err(gpu_helpers::reject_gpu_output(square, error))
            } else {
                Ok(square)
            }
        }
        Err(source) => {
            let source = source.after_effect();
            let mut error =
                var_internal_error(format!("var: resident variance square failed: {source}"));
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            Err(crate::runtime_error::with_provider_execution_source(
                error, source,
            ))
        }
    };

    // The provider result is an owned intermediate even when the later square
    // fails. A failed retirement must not disappear behind Drop.
    let cleanup = std_handle.release().map(|_| ()).map_err(|source| {
        let mut error = var_internal_error(format!(
            "var: resident std intermediate retirement failed: {source}"
        ));
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error.source = Some(Box::new(source));
        error
    });

    match (result, cleanup) {
        (Ok(square), Err(cleanup)) => Err(gpu_helpers::reject_gpu_output(square, cleanup)),
        (result, cleanup) => crate::runtime_error::complete_with_cleanup(result, cleanup),
    }
}
