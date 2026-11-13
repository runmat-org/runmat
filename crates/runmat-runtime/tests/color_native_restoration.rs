use futures::executor::block_on;
use runmat_accelerate_api::*;
use runmat_runtime::call_builtin_async_with_outputs;
use runmat_value::{NumericDType, Tensor, Value};
#[path = "support/owning_restore_provider.rs"]
mod fixture;
use fixture::{Mode, Provider};

#[test]
fn color_native_single_uses_actual_owner_capability_not_preferred_precision() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
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
                    shape: host.shape(),
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
            let Value::Tensor(expected) = expected.value() else {
                panic!("host RGB")
            };
            let result = block_on(call_builtin_async_with_outputs(
                "lab2rgb",
                &[Value::GpuTensor(input.clone())],
                1,
            ));
            assert_eq!(
                *provider.trace.lock().unwrap(),
                ["download", "numeric-upload"],
                "explicit={explicit}, result={:?}",
                result.as_ref().map(|_| "success")
            );
            match mode {
                Mode::Supported => {
                    let result = result.unwrap();
                    let Value::GpuTensor(output) = result.value() else {
                        panic!("supported native Single must stay resident")
                    };
                    assert_eq!(output.physical().element_type(), NumericElementType::F32);
                    assert_eq!(output.shape(), &[1, 1, 3]);
                    assert_eq!(handle_is_explicit(output), explicit);
                    assert!(!output.shares_allocation_with(&input));
                    let retired = output.allocation_lease().downgrade();
                    let payload = block_on(inner.download_numeric(output)).unwrap();
                    let HostNumericDataOwned::F32(data) = payload.data else {
                        panic!("native Single output")
                    };
                    assert_eq!(data, expected.as_f32_slice().unwrap());
                    drop(result);
                    assert!(retired.acquire().is_none());
                }
                Mode::Unsupported if !explicit => {
                    let result = result.unwrap();
                    let Value::Tensor(output) = result.value() else {
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
                        matches!(upload, NumericUploadError::Provider(cause) if cause.is_unsupported())
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

#[test]
fn explicit_color_extension_rejection_precedes_all_provider_effects() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let inner = Box::leak(Box::new(
        runmat_accelerate::simple_provider::InProcessProvider::new(),
    ));
    let provider = Provider::new(inner, Mode::Supported);
    let _active = ThreadProviderGuard::set(Some(provider));
    let input = inner
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[70., 5., 10.]),
            shape: &[1, 1, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let borrowed = input.allocation_lease().downgrade();
    let error = block_on(call_builtin_async_with_outputs(
        "lab2rgb",
        &[Value::GpuTensor(input.clone())],
        1,
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:Lab2rgbExplicitGpuExtension")
    );
    assert!(provider.trace.lock().unwrap().is_empty());
    assert!(input.allocation_lease().validate_live().is_ok());
    drop(input);
    assert!(borrowed.acquire().is_none());
}

#[test]
fn scalar_color_fallback_preserves_the_original_host_carrier() {
    for explicit in [false, true] {
        let inner = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        let provider = Provider::new(inner, Mode::Unsupported);
        let _active = ThreadProviderGuard::set(Some(provider));
        let input = inner
            .upload(&HostTensorView {
                data: &[0.25],
                shape: &[1, 1],
            })
            .unwrap()
            .with_provenance(if explicit {
                GpuHandleProvenance::Explicit
            } else {
                GpuHandleProvenance::Automatic
            })
            .unwrap();
        let borrowed = input.allocation_lease().downgrade();
        let expected = block_on(call_builtin_async_with_outputs(
            "im2double",
            &[Value::Num(0.25)],
            1,
        ))
        .unwrap();
        assert_eq!(expected, Value::Num(0.25));
        let result = block_on(call_builtin_async_with_outputs(
            "im2double",
            &[Value::GpuTensor(input.clone())],
            1,
        ));
        assert_eq!(
            *provider.trace.lock().unwrap(),
            ["download", "numeric-upload"],
            "explicit={explicit}, result={:?}",
            result.as_ref().map(|_| "success")
        );
        if explicit {
            let error = result.unwrap_err();
            assert_eq!(
                error.gpu_gather_retry,
                runmat_runtime::GpuGatherRetry::Never
            );
            assert!(
                matches!(error.source.as_ref().unwrap().downcast_ref::<NumericUploadError>().unwrap(), NumericUploadError::Provider(cause) if cause.is_unsupported())
            );
        } else {
            assert_eq!(result.unwrap(), expected);
        }
        assert!(borrowed.acquire().is_some());
        drop(input);
        assert!(borrowed.acquire().is_none());
    }
}
