//! Typed uploads preserve numeric storage and admit logical producer output.
use super::*;

pub(super) fn provider_failure(
    error: runmat_accelerate_api::ProviderExecutionError,
) -> RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        gpu_array_error_with_message(format!("gpuArray: {error}"), &GPUARRAY_ERROR_PROVIDER_IO),
        error,
    )
}

pub(super) fn upload_real_tensor(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    tensor: &Tensor,
    logical: bool,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<GpuTensorHandle> {
    if logical {
        // cast_tensor has already checked the logical conversion and produced
        // canonical zero/one values. Classification belongs to the producer.
        let bits: Vec<u8> = tensor
            .materialize_f64()
            .iter()
            .map(|value| u8::from(*value != 0.0))
            .collect();
        runmat_accelerate_api::upload_logical_protecting(
            provider,
            &runmat_accelerate_api::HostLogicalTensorView {
                data: &bits,
                shape: &tensor.shape,
            },
            protected,
        )
        .map_err(|error| {
            provider_failure(runmat_accelerate_api::ProviderExecutionError::from_source(
                error,
            ))
        })
    } else {
        gpu_helpers::upload_tensor_protecting_typed(provider, tensor, protected).map_err(|error| {
            provider_failure(runmat_accelerate_api::ProviderExecutionError::from_source(
                error,
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    fn check_logical_producer(provider: &'static dyn runmat_accelerate_api::AccelProvider) {
        for (data, shape) in [(vec![0.0, 1.0], vec![2, 1]), (vec![], vec![0, 3])] {
            let tensor = Tensor::new(data.clone(), shape.clone()).unwrap();
            let handle = upload_real_tensor(provider, &tensor, true, &[]).unwrap();
            assert!(runmat_accelerate_api::handle_is_logical(&handle));
            assert_eq!(handle.shape(), shape);
            validate_prepared_handle(
                &handle,
                provider,
                &shape,
                runmat_accelerate_api::GpuTensorStorage::Real,
                DataClass::Logical,
                true,
            )
            .unwrap();
            let weak = handle.downgrade();
            let downloaded = futures::executor::block_on(provider.download(&handle)).unwrap();
            assert_eq!(downloaded.data, data);
            drop(handle);
            assert!(weak.acquire().is_err());
        }
    }

    #[test]
    fn logical_upload_has_producer_class_on_both_provider_precisions() {
        test_support::with_test_provider(|provider| check_logical_producer(provider));
        test_support::with_f32_test_provider(|provider| check_logical_producer(provider));
    }
}
