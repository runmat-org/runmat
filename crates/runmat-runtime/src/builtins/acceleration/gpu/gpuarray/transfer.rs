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
                shape: tensor.shape(),
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
        for (data, shape) in [
            (vec![0.0, 1.0], vec![2, 1]),
            (vec![], vec![0, 3]),
            (vec![1.0], Vec::new()),
            (vec![0.0, 1.0], vec![2]),
        ] {
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

    #[test]
    fn integer_conversion_retains_borrowed_source_and_native_result() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new_integer(IntegerStorage::U16(vec![1, 257]), vec![2, 1]).unwrap();
            let source = upload_real_tensor(provider, &tensor, false, &[]).unwrap();
            let weak = source.downgrade();
            let converted = futures::executor::block_on(convert_device_value(
                source.clone(),
                DataClass::UInt32,
            ))
            .unwrap();
            assert!(source.allocation_lease().validate_live().is_ok());
            let gathered =
                futures::executor::block_on(gpu_helpers::gather_tensor_async(&converted.handle))
                    .unwrap();
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U32(vec![1, 257]))
            );
            assert_eq!(gathered.shape(), &[2, 1]);
            drop(converted);
            assert!(weak.acquire().is_ok());
            drop(source);
            assert!(weak.acquire().is_err());
        });
    }

    #[test]
    fn provider_error_adapter_retains_nested_runtime_identity() {
        let error = crate::build_runtime_error("download failed")
            .with_identifier("RunMat:TestDownload")
            .build();
        let converted = provider_failure(
            runmat_accelerate_api::ProviderExecutionError::from_source(error),
        );
        assert_eq!(converted.identifier(), Some("RunMat:TestDownload"));
        assert_eq!(converted.message(), "download failed");
    }

    #[test]
    fn conversion_does_not_route_an_unregistered_owner_to_the_active_provider() {
        test_support::with_test_provider(|_| {
            let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
            let allocation = owner
                .allocate_zeroed(
                    vec![2, 1],
                    runmat_accelerate_api::NumericElementType::U16,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap();
            let result = futures::executor::block_on(convert_device_value(
                allocation.handle.clone(),
                DataClass::UInt32,
            ));
            let error = match result {
                Ok(_) => panic!("unregistered owner must not use ambient provider"),
                Err(error) => error,
            };
            assert_eq!(
                error.identifier(),
                gpu_array_error(&GPUARRAY_ERROR_NO_PROVIDER).identifier()
            );
            allocation
                .handle
                .allocation_lease()
                .validate_live()
                .unwrap();
            assert_eq!(allocation.probe.retirements(), 0);
            drop(allocation.handle);
            assert_eq!(allocation.probe.retirements(), 1);
        });
    }
}
