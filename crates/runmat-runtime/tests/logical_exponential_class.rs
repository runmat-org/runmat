use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, AccelProviderFuture,
    GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorOwned, HostNumericTensorView, HostTensorView,
    NumericElementType, ProviderPrecision, ThreadProviderGuard,
};
use runmat_value::{NumericDType, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct Provider {
    owner: TestAllocationOwner,
    supports_double: bool,
    fail_upload: bool,
    records: Mutex<Vec<(u64, RetirementProbe)>>,
    downloads: AtomicUsize,
    unary_calls: AtomicUsize,
}

impl Provider {
    fn remember(
        &self,
        allocation: runmat_accelerate_api::test_support::TestAllocation,
    ) -> GpuTensorHandle {
        self.records
            .lock()
            .unwrap()
            .push((allocation.handle.buffer_id(), allocation.probe));
        allocation.handle
    }
    fn read(&self, handle: &GpuTensorHandle) -> anyhow::Result<HostNumericTensorOwned> {
        let probe = self
            .records
            .lock()
            .unwrap()
            .iter()
            .find(|(id, _)| *id == handle.buffer_id())
            .ok_or_else(|| anyhow::anyhow!("unknown fixture allocation"))?
            .1
            .clone();
        let data = probe.with_data(handle, |data| match data {
            HostNumericDataView::F32(data) => HostNumericDataOwned::F32(data.to_vec()),
            HostNumericDataView::F64(data) => HostNumericDataOwned::F64(data.to_vec()),
            HostNumericDataView::I16(data) => HostNumericDataOwned::I16(data.to_vec()),
            _ => panic!("fixture only allocates floating and I16 inputs"),
        })?;
        Ok(HostNumericTensorOwned {
            data,
            shape: handle.shape().to_vec(),
            storage: GpuTensorStorage::Real,
        })
    }
}

impl AccelProvider for Provider {
    fn download_integer<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelIntegerDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::Relaxed);
            let host = self.read(input)?;
            let HostNumericDataOwned::I16(data) = host.data else {
                anyhow::bail!("expected integer input");
            };
            Ok(runmat_accelerate_api::HostIntegerTensorOwned {
                data: runmat_accelerate_api::HostIntegerDataOwned::I16(data),
                shape: host.shape,
            })
        })
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "preferred F32, typed Double capability fixture".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F32
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("resolved Double must use typed upload")
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        host.validate()?;
        let HostNumericDataView::F64(data) = host.data else {
            panic!("expected resolved Double output");
        };
        if self.fail_upload {
            return Err(anyhow::anyhow!("injected typed Double upload failure").into());
        }
        if !self.supports_double {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "fixture cannot encode Double",
            )
            .into());
        }
        Ok(
            self.remember(self.owner.allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(data.to_vec()),
                shape: host.shape.to_vec(),
                storage: host.storage,
            })?),
        )
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::Relaxed);
            self.read(input)
        })
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("native class download must not use legacy projection") })
    }
    fn unary_exp<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.unary_calls.fetch_add(1, Ordering::Relaxed);
        Box::pin(async { panic!("logical/integer input must use admitted host semantics") })
    }
    fn unary_expm1<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.unary_exp(input)
    }
}

#[test]
fn exponential_language_class_uses_typed_capability_not_preferred_precision() {
    for supports_double in [false, true] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            supports_double,
            fail_upload: false,
            records: Mutex::new(Vec::new()),
            downloads: AtomicUsize::new(0),
            unary_calls: AtomicUsize::new(0),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        for (name, mapped) in [("exp", false), ("expm1", false), ("exp", true)] {
            for class in [
                NumericElementType::F32,
                NumericElementType::F64,
                NumericElementType::I16,
            ] {
                for explicit in [false, true] {
                    let allocation = if class == NumericElementType::I16 {
                        provider
                            .owner
                            .allocate_zeroed(vec![1, 1, 1], class, GpuTensorStorage::Real)
                    } else {
                        provider.owner.allocate_logical_zeroed(
                            vec![1, 1, 1],
                            if class == NumericElementType::F32 {
                                ProviderPrecision::F32
                            } else {
                                ProviderPrecision::F64
                            },
                        )
                    }
                    .unwrap();
                    let input = provider
                        .remember(allocation)
                        .with_provenance(if explicit {
                            GpuHandleProvenance::Explicit
                        } else {
                            GpuHandleProvenance::Automatic
                        })
                        .unwrap();
                    let mut arguments = vec![Value::GpuTensor(input.clone())];
                    let invoked_name = if mapped {
                        arguments.insert(0, Value::FunctionHandle(name.into()));
                        "arrayfun"
                    } else {
                        name
                    };
                    {
                        let _strict =
                            runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
                        let before = provider.downloads.load(Ordering::Relaxed);
                        block_on(runmat_runtime::call_builtin_async_with_outputs(
                            invoked_name,
                            &arguments,
                            1,
                        ))
                        .unwrap_err();
                        assert_eq!(provider.downloads.load(Ordering::Relaxed), before);
                    }
                    let _extensions =
                        runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
                    let result = block_on(runmat_runtime::call_builtin_async_with_outputs(
                        invoked_name,
                        &arguments,
                        1,
                    ));
                    if explicit && !supports_double {
                        let error = result.unwrap_err();
                        if mapped {
                            assert!(error.message().starts_with("arrayfun:"));
                            let cause = error
                                .source
                                .as_ref()
                                .and_then(|source| {
                                    source.downcast_ref::<runmat_runtime::RuntimeError>()
                                })
                                .expect("arrayfun retains canonical callback failure");
                            assert_eq!(
                                cause.gpu_gather_retry,
                                runmat_runtime::GpuGatherRetry::Never
                            );
                        } else {
                            assert_eq!(
                                error.gpu_gather_retry,
                                runmat_runtime::GpuGatherRetry::Never
                            );
                        }
                        assert!(error.source.is_some());
                    } else {
                        let expected = if name == "exp" { 1.0 } else { 0.0 };
                        let output = result.unwrap();
                        match output.value() {
                            Value::GpuTensor(output) => {
                                assert!(supports_double);
                                assert_eq!(output.device_id(), input.device_id());
                                assert_eq!(output.shape(), &[1, 1, 1]);
                                assert_eq!(
                                    output.physical().element_type(),
                                    NumericElementType::F64
                                );
                                assert!(!runmat_accelerate_api::handle_is_logical(output));
                                let HostNumericDataOwned::F64(data) =
                                    provider.read(output).unwrap().data
                                else {
                                    panic!("expected Double payload");
                                };
                                assert_eq!(data, vec![expected]);
                            }
                            Value::Tensor(output) => {
                                assert!(!supports_double && !explicit);
                                assert_eq!(output.numeric_dtype(), NumericDType::F64);
                                assert_eq!(output.shape(), &[1, 1, 1]);
                                assert_eq!(output.materialize_f64(), vec![expected]);
                            }
                            other => panic!("unexpected output {other:?}"),
                        }
                    }
                    assert_eq!(provider.unary_calls.load(Ordering::Relaxed), 0);
                    input.allocation_lease().validate_live().unwrap();
                }
            }
        }
    }
}

#[test]
fn mapped_automatic_exponential_never_hides_operational_upload_failure() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        supports_double: true,
        fail_upload: true,
        records: Mutex::new(Vec::new()),
        downloads: AtomicUsize::new(0),
        unary_calls: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let allocation = provider
        .owner
        .allocate_logical_zeroed(vec![1, 1, 1], ProviderPrecision::F32)
        .unwrap();
    let input = provider
        .remember(allocation)
        .with_provenance(GpuHandleProvenance::Automatic)
        .unwrap();
    for mapped in [false, true] {
        let mut arguments = vec![Value::GpuTensor(input.clone())];
        let name = if mapped {
            arguments.insert(0, Value::FunctionHandle("exp".into()));
            "arrayfun"
        } else {
            "exp"
        };
        let before = provider.downloads.load(Ordering::Relaxed);
        let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
            name, &arguments, 1,
        ))
        .unwrap_err();
        assert!(
            error
                .message()
                .contains("injected typed Double upload failure"),
            "{error:?}"
        );
        assert!(error.source.is_some());
        assert_eq!(
            provider.downloads.load(Ordering::Relaxed),
            before + 1,
            "no scalar retry after operational failure"
        );
        assert_eq!(
            provider.records.lock().unwrap().len(),
            1,
            "only original allocation remains"
        );
        input.allocation_lease().validate_live().unwrap();
    }
    assert_eq!(provider.unary_calls.load(Ordering::Relaxed), 0);
}
