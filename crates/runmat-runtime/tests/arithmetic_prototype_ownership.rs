use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorHandle,
    HostNumericTensorView, HostTensorView, ProviderPrecision, ThreadProviderGuard,
};
use runmat_value::{Tensor, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

struct Provider {
    inner: InProcessProvider,
    borrowed: Mutex<Option<GpuTensorHandle>>,
    frees: AtomicUsize,
    uploads: AtomicUsize,
    hooks: AtomicUsize,
    mode: AtomicUsize,
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
    fn upload_numeric_raw(
        &self,
        view: &HostNumericTensorView<'_>,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        if self.mode.load(Ordering::SeqCst) == 2 {
            let mut fresh = self.inner.upload_numeric(view)?;
            fresh.shape = vec![99, 1];
            return Ok(fresh);
        }
        let mut alias = self.borrowed.lock().unwrap().as_ref().unwrap().clone();
        runmat_accelerate_api::set_handle_logical(&alias, true);
        if self.mode.load(Ordering::SeqCst) == 1 {
            alias.shape = vec![99, 1];
        }
        Ok(alias)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.frees.fetch_add(1, Ordering::SeqCst);
        self.inner.free(handle)
    }
    fn elem_add<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.hooks.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Err(anyhow::anyhow!("unexpected arithmetic hook")) })
    }
}

#[test]
fn prototype_preflight_precedes_hooks_and_upload_cleanup_protects_borrowed_handles() {
    static PROVIDER: OnceLock<Provider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(|| Provider {
        inner: InProcessProvider::new(),
        borrowed: Mutex::new(None),
        frees: AtomicUsize::new(0),
        uploads: AtomicUsize::new(0),
        hooks: AtomicUsize::new(0),
        mode: AtomicUsize::new(0),
    });
    let _provider = ThreadProviderGuard::set(Some(provider));
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let prototype = provider
        .inner
        .upload(&HostTensorView {
            data: &[7.0, 8.0],
            shape: &[1, 2],
        })
        .unwrap();
    *provider.borrowed.lock().unwrap() = Some(prototype.clone());
    let args = [
        Value::GpuTensor(prototype.clone()),
        Value::GpuTensor(prototype.clone()),
        Value::String("like".into()),
        Value::String("invalid prototype".into()),
    ];
    assert!(block_on(runmat_runtime::call_builtin_async_with_outputs(
        "plus", &args, 1
    ))
    .is_err());
    assert_eq!(provider.hooks.load(Ordering::SeqCst), 0);
    for mode in 0..3 {
        provider.mode.store(mode, Ordering::SeqCst);
        let before = provider.frees.load(Ordering::SeqCst);
        let args = [
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            Value::String("like".into()),
            Value::GpuTensor(prototype.clone()),
        ];
        assert!(block_on(runmat_runtime::call_builtin_async_with_outputs(
            "plus", &args, 1
        ))
        .is_err());
        assert_eq!(
            provider.frees.load(Ordering::SeqCst) - before,
            usize::from(mode == 2)
        );
        assert!(!runmat_accelerate_api::handle_is_logical(&prototype));
        assert_eq!(
            block_on(provider.inner.download(&prototype)).unwrap().data,
            vec![7.0, 8.0]
        );
    }
    // Construction uploads a host component alongside a borrowed resident
    // component. Either argument order must protect the resident allocation.
    for resident_first in [false, true] {
        for host in [
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
            Value::String("not a real numeric component".into()),
            Value::Complex(1.0, 2.0),
        ] {
            let before = provider.uploads.load(Ordering::SeqCst);
            let resident = Value::GpuTensor(prototype.clone());
            let args = if resident_first {
                vec![resident, host]
            } else {
                vec![host, resident]
            };
            assert!(block_on(runmat_runtime::call_builtin_async_with_outputs(
                "complex", &args, 1
            ))
            .is_err());
            assert_eq!(provider.uploads.load(Ordering::SeqCst), before);
        }
    }
    for mode in 0..3 {
        provider.mode.store(mode, Ordering::SeqCst);
        for resident_first in [false, true] {
            let before = provider.frees.load(Ordering::SeqCst);
            let host = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
            let resident = Value::GpuTensor(prototype.clone());
            let args = if resident_first {
                vec![resident, host]
            } else {
                vec![host, resident]
            };
            assert!(block_on(runmat_runtime::call_builtin_async_with_outputs(
                "complex", &args, 1
            ))
            .is_err());
            assert_eq!(
                provider.frees.load(Ordering::SeqCst) - before,
                usize::from(mode == 2)
            );
            assert!(!runmat_accelerate_api::handle_is_logical(&prototype));
            assert_eq!(
                block_on(provider.inner.download(&prototype)).unwrap().data,
                vec![7.0, 8.0]
            );
        }
    }
    provider.inner.free(&prototype).unwrap();
    *provider.borrowed.lock().unwrap() = None;
}
