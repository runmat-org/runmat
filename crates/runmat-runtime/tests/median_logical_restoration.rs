use futures::executor::block_on;
use runmat_accelerate_api::*;
use runmat_runtime::call_builtin_async_with_outputs;
use runmat_value::{LogicalArray, Value};
#[path = "support/owning_restore_provider.rs"]
mod fixture;
#[path = "../src/builtins/common/test_support/native_logical.rs"]
mod native_logical;
use fixture::{Mode, Provider};

#[test]
fn logical_weighted_median_uses_true_native_producer_and_retains_operational_errors() {
    for native_u8 in [false, true] {
        for explicit in [false, true] {
            for mode in [
                Mode::Supported,
                Mode::Unsupported,
                Mode::Failure,
                Mode::Alias,
            ] {
                let inner: &'static dyn AccelProvider = if native_u8 {
                    Box::leak(Box::new(native_logical::NativeLogicalProvider::default()))
                } else {
                    Box::leak(Box::new(
                        runmat_accelerate::simple_provider::InProcessProvider::new(),
                    ))
                };
                let provider = Provider::new(inner, mode);
                let _active = ThreadProviderGuard::set(Some(provider));
                let input = inner
                    .upload_logical_raw(&HostLogicalTensorView {
                        data: &[0, 1, 0],
                        shape: &[3, 1, 1],
                    })
                    .unwrap()
                    .with_provenance(if explicit {
                        GpuHandleProvenance::Explicit
                    } else {
                        GpuHandleProvenance::Automatic
                    })
                    .unwrap();
                let weights = inner
                    .upload_numeric_raw(&HostNumericTensorView {
                        data: HostNumericDataView::F32(&[1., 5., 1.]),
                        shape: &[3, 1],
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap()
                    .with_provenance(GpuHandleProvenance::Automatic)
                    .unwrap();
                let borrowed = input.allocation_lease().downgrade();
                let borrowed_weights = weights.allocation_lease().downgrade();
                if matches!(mode, Mode::Alias) {
                    *provider.alias.lock().unwrap() = Some(input.clone());
                }
                let result = block_on(call_builtin_async_with_outputs(
                    "median",
                    &[
                        Value::GpuTensor(input.clone()),
                        Value::String("Weights".into()),
                        Value::GpuTensor(weights.clone()),
                    ],
                    1,
                ));
                assert_eq!(
                    *provider.trace.lock().unwrap(),
                    ["download", "download", "logical-upload"]
                );
                match mode {
                    Mode::Supported => {
                        let result = result.unwrap();
                        let Value::GpuTensor(output) = result.value() else {
                            panic!("resident logical output")
                        };
                        assert_eq!(output.shape(), &[1, 1, 1]);
                        assert_eq!(
                            output.physical().element_type(),
                            if native_u8 {
                                NumericElementType::U8
                            } else {
                                NumericElementType::F64
                            }
                        );
                        assert!(handle_is_logical(output));
                        assert_eq!(handle_is_explicit(output), explicit);
                        assert!(!output.shares_allocation_with(&input));
                        assert!(!output.shares_allocation_with(&weights));
                        let retired = output.allocation_lease().downgrade();
                        let gathered = block_on(runmat_runtime::gather_if_needed_async(
                            &Value::GpuTensor(output.clone()),
                        ))
                        .unwrap();
                        assert_eq!(
                            gathered,
                            Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1, 1]).unwrap())
                        );
                        drop(result);
                        assert!(retired.acquire().is_none());
                    }
                    Mode::Unsupported => {
                        if explicit {
                            let error = result.unwrap_err();
                            assert_eq!(
                                error.gpu_gather_retry,
                                runmat_runtime::GpuGatherRetry::Never
                            );
                            let source = error
                                .source
                                .as_ref()
                                .unwrap()
                                .downcast_ref::<NumericUploadError>()
                                .unwrap();
                            assert!(
                                matches!(source, NumericUploadError::Provider(cause) if cause.is_unsupported())
                            );
                        } else {
                            assert_eq!(
                                result.unwrap().value(),
                                &Value::LogicalArray(
                                    LogicalArray::new(vec![1], vec![1, 1, 1]).unwrap()
                                )
                            );
                        }
                    }
                    Mode::Failure | Mode::Alias => fixture::assert_upload_failure(
                        &result.unwrap_err(),
                        matches!(mode, Mode::Alias),
                    ),
                }
                provider.alias.lock().unwrap().take();
                assert!(borrowed.acquire().is_some());
                assert!(borrowed_weights.acquire().is_some());
                assert!(handle_is_logical(&input));
                assert!(!handle_is_logical(&weights));
                drop((input, weights));
                assert!(borrowed.acquire().is_none());
                assert!(borrowed_weights.acquire().is_none());
            }
        }
    }
}

#[test]
fn weighted_numeric_median_rejects_upload_alias_to_either_original_operand() {
    for alias_weights in [false, true] {
        let inner = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        let provider = Provider::new(inner, Mode::Alias);
        let _active = ThreadProviderGuard::set(Some(provider));
        let input = inner
            .upload(&HostTensorView {
                data: &[7.],
                shape: &[1, 1],
            })
            .unwrap();
        let weights = inner
            .upload(&HostTensorView {
                data: &[1.],
                shape: &[1, 1],
            })
            .unwrap();
        *provider.alias.lock().unwrap() = Some(if alias_weights {
            weights.clone()
        } else {
            input.clone()
        });
        let borrowed = input.allocation_lease().downgrade();
        let borrowed_weights = weights.allocation_lease().downgrade();
        let error = block_on(call_builtin_async_with_outputs(
            "median",
            &[
                Value::GpuTensor(input.clone()),
                Value::String("Weights".into()),
                Value::GpuTensor(weights.clone()),
            ],
            1,
        ))
        .unwrap_err();
        fixture::assert_upload_failure(&error, true);
        assert_eq!(
            *provider.trace.lock().unwrap(),
            ["download", "download", "numeric-upload"]
        );
        provider.alias.lock().unwrap().take();
        assert_eq!(block_on(inner.download(&input)).unwrap().data, [7.]);
        assert_eq!(block_on(inner.download(&weights)).unwrap().data, [1.]);
        assert!(borrowed.acquire().is_some());
        assert!(borrowed_weights.acquire().is_some());
        drop((input, weights));
        assert!(borrowed.acquire().is_none());
        assert!(borrowed_weights.acquire().is_none());
    }
}
