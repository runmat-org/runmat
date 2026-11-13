use futures::{executor::block_on, FutureExt};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::allocation::{WeakAllocationLease, WeakGpuTensorHandle};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorHandle,
    HostNumericTensorView, HostTensorView, ProviderPrecision, ThreadProviderGuard,
};
use runmat_value::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

struct Provider {
    inner: InProcessProvider,
    mode: AtomicUsize,
    repmats: AtomicUsize,
    right: Mutex<Option<WeakGpuTensorHandle>>,
    first: Mutex<Option<WeakGpuTensorHandle>>,
    outputs: Mutex<Vec<WeakAllocationLease>>,
}
impl Provider {
    fn observe(&self, output: GpuTensorHandle) -> GpuTensorHandle {
        self.outputs
            .lock()
            .unwrap()
            .push(output.allocation_lease().downgrade());
        output
    }
    fn assert_retired(&self, count: usize) {
        let outputs = self.outputs.lock().unwrap();
        assert_eq!(outputs.len(), count);
        assert!(outputs.iter().all(|lease| lease.acquire().is_none()));
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
    fn query_feasibility(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        // This facade owns the deliberately pending native-integer division
        // stage; the inner provider supplies storage, not that execution hook.
        use runmat_accelerate_api::{ProviderElementType, ProviderLayout, ProviderStorage};
        if query.operation
            == runmat_accelerate_api::ProviderOperationIdentity::new(
                runmat_builtins::RDIVIDE_CATALOG_ENTRY.identity.name,
            )
            && query.inputs.len() == 2
            && query.outputs.len() == 1
            && query.inputs.iter().chain(&query.outputs).all(|value| {
                value.element_type == ProviderElementType::U64
                    && value.storage == ProviderStorage::DenseReal
                    && value.layout == ProviderLayout::ColumnMajorContiguous
            })
            && ((query.inputs[0].shape == [1, 2] && query.inputs[1].shape == [1, 1])
                || (query.inputs[0].shape == [1, 1] && query.inputs[1].shape == [1, 2]))
            && query.outputs[0].shape == [1, 2]
        {
            return runmat_accelerate_api::ProviderFeasibility::supported(Default::default());
        }
        self.inner.query_feasibility(query)
    }
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(view).map(|output| self.observe(output))
    }
    fn upload_numeric_raw(
        &self,
        view: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner
            .upload_numeric_raw(view)
            .map(|output| self.observe(output))
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn fill_like(
        &self,
        handle: &GpuTensorHandle,
        value: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.inner
            .fill_like(handle, value)
            .map(|output| output.map(|output| self.observe(output)))
    }
    fn repmat(
        &self,
        handle: &GpuTensorHandle,
        repetitions: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let index = self.repmats.fetch_add(1, Ordering::SeqCst);
        match (self.mode.load(Ordering::SeqCst), index) {
            (1, 0) => Ok(Some(
                self.right.lock().unwrap().as_ref().unwrap().acquire()?,
            )),
            (1, _) => Err(
                runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(
                    "second expansion failed"
                ))
                .with_unproven_effects(),
            ),
            (2, 1) => Ok(Some(
                self.first.lock().unwrap().as_ref().unwrap().acquire()?,
            )),
            _ => {
                let Some(output) = self.inner.repmat(handle, repetitions)? else {
                    return Ok(None);
                };
                let expanded = self.observe(output);
                if index == 0 {
                    *self.first.lock().unwrap() = Some(expanded.downgrade());
                }
                Ok(Some(expanded))
            }
        }
    }
    fn elem_add<'a>(
        &'a self,
        _left: &'a GpuTensorHandle,
        _right: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(futures::future::pending())
    }
    fn elem_div<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.elem_add(left, right)
    }
    fn elem_pow<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.elem_add(left, right)
    }
    fn classify_real_power<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, runmat_accelerate_api::PowerResultDomain> {
        if self.mode.load(Ordering::SeqCst) == 3 {
            Box::pin(futures::future::pending())
        } else {
            self.inner.classify_real_power(left, right)
        }
    }
    fn real_power_in_domain<'a>(
        &'a self,
        _left: &'a GpuTensorHandle,
        _right: &'a GpuTensorHandle,
        _: runmat_accelerate_api::PowerResultDomain,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(futures::future::pending())
    }
}

#[test]
fn cancelled_broadcast_and_failed_alias_expansions_release_only_owned_temporaries() {
    // The runtime retains provider facades for the worker lifetime. A leaked
    // test-local facade supplies that lifetime without requiring browser-local
    // allocation leases to implement Send or Sync for a process-wide static.
    let provider: &'static Provider = Box::leak(Box::new(Provider {
        inner: InProcessProvider::new(),
        mode: AtomicUsize::new(0),
        repmats: AtomicUsize::new(0),
        right: Mutex::new(None),
        first: Mutex::new(None),
        outputs: Mutex::new(vec![]),
    }));
    let _provider = ThreadProviderGuard::set(Some(provider));
    for (name, mode) in ["plus"]
        .into_iter()
        .flat_map(|name| (0..3).map(move |mode| (name, mode)))
    {
        provider.mode.store(mode, Ordering::SeqCst);
        provider.repmats.store(0, Ordering::SeqCst);
        provider.outputs.lock().unwrap().clear();
        let left = provider
            .inner
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[2, 1],
            })
            .unwrap();
        let right = provider
            .inner
            .upload(&HostTensorView {
                data: &[3.0, 4.0, 5.0],
                shape: &[1, 3],
            })
            .unwrap();
        *provider.right.lock().unwrap() = Some(right.downgrade());
        let physical = [
            left.allocation_lease().downgrade(),
            right.allocation_lease().downgrade(),
        ];
        let descriptors = [left.wire_descriptor(), right.wire_descriptor()];
        let args = [
            Value::GpuTensor(left.clone()),
            Value::GpuTensor(right.clone()),
        ];
        let mut execution = Box::pin(runmat_runtime::call_builtin_async_with_outputs(
            name, &args, 1,
        ));
        if mode == 0 {
            assert!(execution.as_mut().now_or_never().is_none());
            drop(execution);
        } else {
            assert!(block_on(execution).is_err());
        }
        provider.assert_retired(match mode {
            0 => 2,
            1 => 0,
            _ => 1,
        });
        assert_eq!(
            [left.wire_descriptor(), right.wire_descriptor()],
            descriptors
        );
        for handle in [&left, &right] {
            assert!(!runmat_accelerate_api::handle_is_logical(handle));
            assert!(block_on(provider.inner.download(handle)).is_ok());
        }
        drop(args);
        assert!(physical.iter().all(|lease| lease.acquire().is_some()));
        drop(left);
        drop(right);
        assert!(physical.iter().all(|lease| lease.acquire().is_none()));
    }
    // Power's typed hook supports broadcast directly. Cancellation during
    // either the domain probe or the actual kernel must restore both inputs.
    for mode in [0, 3] {
        provider.mode.store(mode, Ordering::SeqCst);
        provider.outputs.lock().unwrap().clear();
        let left = provider
            .inner
            .upload(&HostTensorView {
                data: &[-4.0, 4.0],
                shape: &[1, 2],
            })
            .unwrap();
        let right = provider
            .inner
            .upload(&HostTensorView {
                data: &[2.0, 0.5],
                shape: &[2, 1],
            })
            .unwrap();
        let physical = [
            left.allocation_lease().downgrade(),
            right.allocation_lease().downgrade(),
        ];
        let descriptors = [left.wire_descriptor(), right.wire_descriptor()];
        let args = [
            Value::GpuTensor(left.clone()),
            Value::GpuTensor(right.clone()),
        ];
        let mut execution = Box::pin(runmat_runtime::call_builtin_async_with_outputs(
            "power", &args, 1,
        ));
        assert!(execution.as_mut().now_or_never().is_none());
        drop(execution);
        provider.assert_retired(0);
        assert_eq!(
            [left.wire_descriptor(), right.wire_descriptor()],
            descriptors
        );
        for handle in [&left, &right] {
            assert!(!runmat_accelerate_api::handle_is_logical(handle));
            assert!(block_on(provider.inner.download(handle)).is_ok());
        }
        drop(args);
        drop(left);
        drop(right);
        assert!(physical.iter().all(|lease| lease.acquire().is_none()));
    }
    provider.mode.store(0, Ordering::SeqCst);
    // A mixed resident power call owns either a scalar fill or uploaded host
    // tensor until its asynchronous operation finishes, in both operand orders.
    for resident_first in [false, true] {
        for host in [
            Value::Num(2.0),
            Value::Tensor(runmat_value::Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
        ] {
            provider.outputs.lock().unwrap().clear();
            let resident = provider
                .inner
                .upload(&HostTensorView {
                    data: &[4.0, 9.0],
                    shape: &[1, 2],
                })
                .unwrap();
            let physical = resident.allocation_lease().downgrade();
            let descriptor = resident.wire_descriptor();
            let args = if resident_first {
                vec![Value::GpuTensor(resident.clone()), host]
            } else {
                vec![host, Value::GpuTensor(resident.clone())]
            };
            let mut execution = Box::pin(runmat_runtime::call_builtin_async_with_outputs(
                "power", &args, 1,
            ));
            assert!(execution.as_mut().now_or_never().is_none());
            drop(execution);
            provider.assert_retired(1);
            assert_eq!(resident.wire_descriptor(), descriptor);
            assert!(!runmat_accelerate_api::handle_is_logical(&resident));
            assert_eq!(
                block_on(provider.inner.download(&resident)).unwrap().data,
                vec![4.0, 9.0]
            );
            drop(args);
            drop(resident);
            assert!(physical.acquire().is_none());
        }
    }
    // Exact integer-scalar temporaries cross the same asynchronous provider
    // boundary, both for arithmetic and for either direction of division.
    for (name, resident_first) in [("plus", true), ("rdivide", true), ("rdivide", false)] {
        use runmat_accelerate_api::{GpuTensorStorage, HostNumericDataView};
        provider.outputs.lock().unwrap().clear();
        let resident = provider
            .inner
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::U64(&[7, 9]),
                shape: &[1, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let physical = resident.allocation_lease().downgrade();
        let descriptor = resident.wire_descriptor();
        let args = if resident_first {
            vec![Value::GpuTensor(resident.clone()), Value::Num(2.0)]
        } else {
            vec![Value::Num(2.0), Value::GpuTensor(resident.clone())]
        };
        let mut execution = Box::pin(runmat_runtime::call_builtin_async_with_outputs(
            name, &args, 1,
        ));
        let polled = execution.as_mut().now_or_never();
        assert!(polled.is_none(), "{name}: {polled:?}");
        drop(execution);
        provider.assert_retired(1);
        assert_eq!(resident.wire_descriptor(), descriptor);
        assert!(!runmat_accelerate_api::handle_is_logical(&resident));
        let downloaded = block_on(provider.inner.download_numeric(&resident)).unwrap();
        assert_eq!(downloaded.shape, vec![1, 2]);
        assert!(
            matches!(downloaded.data, runmat_accelerate_api::HostNumericDataOwned::U64(ref values) if values == &[7, 9])
        );
        drop(args);
        drop(resident);
        assert!(physical.acquire().is_none());
    }
}
