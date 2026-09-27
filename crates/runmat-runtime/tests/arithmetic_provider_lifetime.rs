use futures::{executor::block_on, FutureExt};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorHandle,
    HostNumericTensorView, HostTensorView, ProviderPrecision, ThreadProviderGuard,
};
use runmat_value::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

struct Provider {
    inner: InProcessProvider,
    mode: AtomicUsize,
    repmats: AtomicUsize,
    right: Mutex<Option<GpuTensorHandle>>,
    first: Mutex<Option<GpuTensorHandle>>,
    freed: Mutex<Vec<u64>>,
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
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(view)
    }
    fn upload_numeric(&self, view: &HostNumericTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload_numeric(view)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.freed.lock().unwrap().push(handle.buffer_id);
        self.inner.free(handle)
    }
    fn repmat(
        &self,
        handle: &GpuTensorHandle,
        repetitions: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        let index = self.repmats.fetch_add(1, Ordering::SeqCst);
        match (self.mode.load(Ordering::SeqCst), index) {
            (1, 0) => Ok(self.right.lock().unwrap().as_ref().unwrap().clone()),
            (1, _) => Err(anyhow::anyhow!("second expansion failed")),
            (2, 1) => Ok(self.first.lock().unwrap().as_ref().unwrap().clone()),
            _ => {
                let expanded = self.inner.repmat(handle, repetitions)?;
                if index == 0 {
                    *self.first.lock().unwrap() = Some(expanded.clone());
                }
                Ok(expanded)
            }
        }
    }
    fn elem_add<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        runmat_accelerate_api::set_handle_logical(left, true);
        runmat_accelerate_api::set_handle_logical(right, true);
        Box::pin(futures::future::pending())
    }
    fn elem_div<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.elem_add(left, right)
    }
    fn elem_pow<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.elem_add(left, right)
    }
}

#[test]
fn cancelled_broadcast_and_failed_alias_expansions_release_only_owned_temporaries() {
    static PROVIDER: OnceLock<Provider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(|| Provider {
        inner: InProcessProvider::new(),
        mode: AtomicUsize::new(0),
        repmats: AtomicUsize::new(0),
        right: Mutex::new(None),
        first: Mutex::new(None),
        freed: Mutex::new(vec![]),
    });
    let _provider = ThreadProviderGuard::set(Some(provider));
    for (name, mode) in ["plus", "power"]
        .into_iter()
        .flat_map(|name| (0..3).map(move |mode| (name, mode)))
    {
        provider.mode.store(mode, Ordering::SeqCst);
        provider.repmats.store(0, Ordering::SeqCst);
        provider.freed.lock().unwrap().clear();
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
        *provider.right.lock().unwrap() = Some(right.clone());
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
        let freed = provider.freed.lock().unwrap().clone();
        assert_eq!(
            freed.len(),
            match mode {
                0 => 2,
                1 => 0,
                _ => 1,
            }
        );
        assert!(!freed.contains(&left.buffer_id) && !freed.contains(&right.buffer_id));
        for handle in [&left, &right] {
            assert!(!runmat_accelerate_api::handle_is_logical(handle));
            assert!(block_on(provider.inner.download(handle)).is_ok());
            provider.inner.free(handle).unwrap();
        }
    }
    // Exact integer-scalar temporaries cross the same asynchronous provider
    // boundary, both for arithmetic and for either direction of division.
    for (name, resident_first) in [("plus", true), ("rdivide", true), ("rdivide", false)] {
        use runmat_accelerate_api::{GpuTensorStorage, HostNumericDataView};
        provider.freed.lock().unwrap().clear();
        let resident = provider
            .inner
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::U64(&[7, 9]),
                shape: &[1, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let args = if resident_first {
            vec![Value::GpuTensor(resident.clone()), Value::Num(2.0)]
        } else {
            vec![Value::Num(2.0), Value::GpuTensor(resident.clone())]
        };
        let mut execution = Box::pin(runmat_runtime::call_builtin_async_with_outputs(
            name, &args, 1,
        ));
        assert!(execution.as_mut().now_or_never().is_none(), "{name}");
        drop(execution);
        let freed = provider.freed.lock().unwrap().clone();
        assert_eq!(freed.len(), 1, "{name}");
        assert_ne!(freed[0], resident.buffer_id);
        assert!(!runmat_accelerate_api::handle_is_logical(&resident));
        provider.inner.free(&resident).unwrap();
    }
}
