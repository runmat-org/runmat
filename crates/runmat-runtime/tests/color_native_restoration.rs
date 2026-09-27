use futures::executor::block_on;
use runmat_accelerate_api::*;
use runmat_runtime::call_builtin_async_with_outputs;
use runmat_value::{NumericDType, Tensor, Value};
#[path = "support/owning_restore_provider.rs"]
mod fixture;
use fixture::{Mode, Provider};

#[test]
fn color_native_single_uses_actual_owner_capability_not_preferred_precision() {
    for explicit in [false, true] {
        for mode in [
            Mode::Supported,
            Mode::Unsupported,
            Mode::Failure,
            Mode::Alias,
        ] {
            let inner = Box::leak(Box::new(
                runmat_accelerate::simple_provider::InProcessProvider::new(),
            ));
            let provider = Provider::new(inner, mode);
            let _active = ThreadProviderGuard::set(Some(provider));
            assert_eq!(provider.precision(), ProviderPrecision::F64);
            let host = Tensor::from_f32(vec![70.0, 5.0, 10.0], vec![1, 1, 3]).unwrap();
            let input = inner
                .upload_numeric_raw(&HostNumericTensorView {
                    data: HostNumericDataView::F32(host.as_f32_slice().unwrap()),
                    shape: &host.shape,
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
                .with_provenance(if explicit {
                    GpuHandleProvenance::Explicit
                } else {
                    GpuHandleProvenance::Automatic
                })
                .unwrap();
            let borrowed = input.allocation_lease().downgrade();
            if matches!(mode, Mode::Alias) {
                *provider.alias.lock().unwrap() = Some(input.clone());
            }
            let expected = block_on(call_builtin_async_with_outputs(
                "lab2rgb",
                &[Value::Tensor(host.clone())],
                1,
            ))
            .unwrap();
            let Value::Tensor(expected) = expected else {
                panic!("host RGB")
            };
            let result = block_on(call_builtin_async_with_outputs(
                "lab2rgb",
                &[Value::GpuTensor(input.clone())],
                1,
            ));
            assert_eq!(
                *provider.trace.lock().unwrap(),
                ["download", "numeric-upload"]
            );
            match mode {
                Mode::Supported => {
                    let Value::GpuTensor(output) = result.unwrap() else {
                        panic!("supported native Single must stay resident")
                    };
                    assert_eq!(output.physical().element_type(), NumericElementType::F32);
                    assert_eq!(output.shape(), &[1, 1, 3]);
                    assert_eq!(handle_is_explicit(&output), explicit);
                    assert!(!output.shares_allocation_with(&input));
                    let retired = output.allocation_lease().downgrade();
                    let payload = block_on(inner.download_numeric(&output)).unwrap();
                    let HostNumericDataOwned::F32(data) = payload.data else {
                        panic!("native Single output")
                    };
                    assert_eq!(data, expected.as_f32_slice().unwrap());
                    drop(output);
                    assert!(retired.acquire().is_none());
                }
                Mode::Unsupported if !explicit => {
                    let Value::Tensor(output) = result.unwrap() else {
                        panic!("typed unsupported automatic result")
                    };
                    assert_eq!(output.numeric_dtype(), NumericDType::F32);
                    assert_eq!(output, expected);
                }
                Mode::Unsupported => {
                    let error = result.unwrap_err();
                    assert_eq!(
                        error.gpu_gather_retry,
                        runmat_runtime::GpuGatherRetry::Never
                    );
                    let upload = error
                        .source
                        .as_ref()
                        .unwrap()
                        .downcast_ref::<NumericUploadError>()
                        .unwrap();
                    assert!(
                        matches!(upload, NumericUploadError::Provider(cause) if is_unsupported_provider_operation(cause))
                    );
                }
                Mode::Failure | Mode::Alias => fixture::assert_upload_failure(
                    &result.unwrap_err(),
                    matches!(mode, Mode::Alias),
                ),
            }
            provider.alias.lock().unwrap().take();
            assert!(borrowed.acquire().is_some());
            assert!(
                matches!(block_on(inner.download_numeric(&input)).unwrap().data, HostNumericDataOwned::F32(data) if data == host.as_f32_slice().unwrap())
            );
            drop(input);
            assert!(borrowed.acquire().is_none());
        }
    }
}
