use futures::{task::noop_waker_ref, FutureExt};
use runmat_accelerate_api::*;
use runmat_value::Value;
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::task::{Context, Poll};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[derive(Debug)]
struct Operational;
impl std::fmt::Display for Operational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("dot not supported by provider")
    }
}
impl Error for Operational {}
struct Provider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    unsupported: bool,
    hooks: AtomicUsize,
    downloads: AtomicUsize,
    alias: Mutex<Option<GpuTensorHandle>>,
    pending: bool,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "dot typed fallback fixture".into()
    }
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(view)
    }
    fn upload_numeric_raw(
        &self,
        view: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if let Some(alias) = self.alias.lock().unwrap().as_ref() {
            return Ok(alias.with_view(alias.view().clone()).unwrap());
        }
        self.inner.upload_numeric_raw(view)
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(input)
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(input)
    }
    fn dot<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
        _: Option<usize>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.hooks.fetch_add(1, Ordering::SeqCst);
            if self.pending {
                return futures::future::pending().await;
            }
            if self.unsupported {
                Err(unsupported_provider_operation(
                    "dot not supported by provider",
                ))
            } else {
                Err(anyhow::Error::new(Operational))
            }
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn only_typed_unsupported_can_gather_dot_inputs() {
    for unsupported in [false, true] {
        let provider = Box::leak(Box::new(Provider {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            unsupported,
            hooks: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            alias: Mutex::new(None),
            pending: false,
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let left = provider
            .inner
            .upload(&HostTensorView {
                data: &[1., 2.],
                shape: &[2, 1],
            })
            .unwrap();
        let right = provider
            .inner
            .upload(&HostTensorView {
                data: &[3., 4.],
                shape: &[2, 1],
            })
            .unwrap();
        let result = futures::executor::block_on(runmat_runtime::call_builtin_async(
            "dot",
            &[
                Value::GpuTensor(left.clone()),
                Value::GpuTensor(right.clone()),
            ],
        ));
        assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
        if unsupported {
            let output = result.unwrap();
            let value = match output.value() {
                Value::Num(value) => *value,
                Value::Tensor(tensor) => tensor.materialize_f64()[0],
                Value::GpuTensor(handle) => {
                    futures::executor::block_on(provider.inner.download(handle))
                        .unwrap()
                        .data[0]
                }
                other => panic!("unexpected dot result {other:?}"),
            };
            assert_eq!(value, 11.);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 2);
        } else {
            let error = result.unwrap_err();
            assert_eq!(
                error.gpu_gather_retry,
                runmat_runtime::GpuGatherRetry::Never
            );
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            let mut cursor: &(dyn Error + 'static) = &error;
            while !cursor.is::<Operational>() {
                cursor = cursor.source().expect("original typed error retained");
            }
        }
        left.allocation_lease().validate_live().unwrap();
        right.allocation_lease().validate_live().unwrap();
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn native_and_fallback_dot_preserve_input_driven_class_under_f64_preference() {
    for fallback in [false, true] {
        let provider: &'static dyn AccelProvider = if fallback {
            Box::leak(Box::new(Provider {
                inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
                unsupported: true,
                hooks: AtomicUsize::new(0),
                downloads: AtomicUsize::new(0),
                alias: Mutex::new(None),
                pending: false,
            }))
        } else {
            Box::leak(Box::new(
                runmat_accelerate::simple_provider::InProcessProvider::new(),
            ))
        };
        let _selected = ThreadProviderGuard::set(Some(provider));
        assert_eq!(provider.precision(), ProviderPrecision::F64);
        for (single_left, single_right) in
            [(true, true), (true, false), (false, true), (false, false)]
        {
            for explicit in [false, true] {
                let provenance = if explicit {
                    GpuHandleProvenance::Explicit
                } else {
                    GpuHandleProvenance::Automatic
                };
                let left = provider
                    .upload_numeric(&HostNumericTensorView {
                        data: if single_left {
                            HostNumericDataView::F32(&[16777216., 1., -16777216.])
                        } else {
                            HostNumericDataView::F64(&[16777216., 1., -16777216.])
                        },
                        shape: &[3, 1],
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap()
                    .with_provenance(provenance)
                    .unwrap();
                let right = provider
                    .upload_numeric(&HostNumericTensorView {
                        data: if single_right {
                            HostNumericDataView::F32(&[1.; 3])
                        } else {
                            HostNumericDataView::F64(&[1.; 3])
                        },
                        shape: &[3, 1],
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap()
                    .with_provenance(provenance)
                    .unwrap();
                let result = futures::executor::block_on(runmat_runtime::call_builtin_async(
                    "dot",
                    &[
                        Value::GpuTensor(left.clone()),
                        Value::GpuTensor(right.clone()),
                    ],
                ))
                .unwrap();
                let Value::GpuTensor(output) = result.value() else {
                    panic!("actual native class is supported, including nonpreferred Single")
                };
                assert_eq!(handle_provenance(output), Some(provenance));
                assert_eq!(
                    handle_precision(output),
                    Some(if single_left && single_right {
                        ProviderPrecision::F32
                    } else {
                        ProviderPrecision::F64
                    })
                );
                assert_eq!(output.shape(), &[1, 1]);
                assert!(!output.shares_allocation_with(&left));
                assert!(!output.shares_allocation_with(&right));
                let downloaded =
                    futures::executor::block_on(provider.download_numeric(output)).unwrap();
                match downloaded.data {
                    HostNumericDataOwned::F32(values) => assert_eq!(values, [1.]),
                    HostNumericDataOwned::F64(values) => assert_eq!(values, [1.]),
                    other => panic!("wrong native storage {other:?}"),
                }
                left.allocation_lease().validate_live().unwrap();
                right.allocation_lease().validate_live().unwrap();
            }
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn restoration_rejects_alias_to_either_original_and_pending_call_retains_both() {
    for target_right in [false, true] {
        let provider = Box::leak(Box::new(Provider {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            unsupported: true,
            hooks: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            alias: Mutex::new(None),
            pending: false,
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let left = provider
            .inner
            .upload(&HostTensorView {
                data: &[2.],
                shape: &[1, 1],
            })
            .unwrap();
        let right = provider
            .inner
            .upload(&HostTensorView {
                data: &[3.],
                shape: &[1, 1],
            })
            .unwrap();
        *provider.alias.lock().unwrap() = Some(if target_right {
            right.clone()
        } else {
            left.clone()
        });
        let error = futures::executor::block_on(runmat_runtime::call_builtin_async(
            "dot",
            &[
                Value::GpuTensor(left.clone()),
                Value::GpuTensor(right.clone()),
            ],
        ))
        .unwrap_err();
        assert_eq!(
            error.gpu_gather_retry,
            runmat_runtime::GpuGatherRetry::Never
        );
        assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
        left.allocation_lease().validate_live().unwrap();
        right.allocation_lease().validate_live().unwrap();
        provider.alias.lock().unwrap().take();
    }
    let provider = Box::leak(Box::new(Provider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        unsupported: false,
        hooks: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        alias: Mutex::new(None),
        pending: true,
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let left = provider
        .inner
        .upload(&HostTensorView {
            data: &[2.],
            shape: &[1, 1],
        })
        .unwrap();
    let right = provider
        .inner
        .upload(&HostTensorView {
            data: &[3.],
            shape: &[1, 1],
        })
        .unwrap();
    let left_weak = left.downgrade();
    let right_weak = right.downgrade();
    let mut pending = async move {
        runmat_runtime::call_builtin_async(
            "dot",
            &[Value::GpuTensor(left), Value::GpuTensor(right)],
        )
        .await
    }
    .boxed_local();
    assert!(matches!(
        pending.poll_unpin(&mut Context::from_waker(noop_waker_ref())),
        Poll::Pending
    ));
    assert!(left_weak.acquire().is_ok());
    assert!(right_weak.acquire().is_ok());
    drop(pending);
    assert!(left_weak.acquire().is_err());
    assert!(right_weak.acquire().is_err());
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn complex_single_host_route_restores_authentic_complex_single() {
    let provider = Box::leak(Box::new(
        runmat_accelerate::simple_provider::InProcessProvider::new(),
    ));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let left = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1., 1., 2., -1.]),
            shape: &[2, 1],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap()
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let right = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1., 0., 0., 1.]),
            shape: &[2, 1],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    let result = futures::executor::block_on(runmat_runtime::call_builtin_async(
        "dot",
        &[
            Value::GpuTensor(left.clone()),
            Value::GpuTensor(right.clone()),
        ],
    ))
    .unwrap();
    let Value::GpuTensor(output) = result.value() else {
        panic!("explicit native complex result")
    };
    assert_eq!(handle_precision(output), Some(ProviderPrecision::F32));
    assert_eq!(
        output.physical().storage(),
        GpuTensorStorage::ComplexInterleaved
    );
    assert_eq!(output.shape(), &[1, 1]);
    let result = futures::executor::block_on(provider.download_numeric(output)).unwrap();
    assert!(matches!(result.data, HostNumericDataOwned::F32(values) if values == [0.,1.]));
    assert!(!output.shares_allocation_with(&left));
    assert!(!output.shares_allocation_with(&right));
}
