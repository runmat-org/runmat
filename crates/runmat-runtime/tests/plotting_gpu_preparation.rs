#![cfg(all(feature = "plot-core", feature = "wgpu"))]

use futures::{executor::block_on, task::noop_waker, FutureExt};
use runmat_accelerate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
use runmat_accelerate_api::{
    AccelContextHandle, AccelContextKind, AccelDownloadFuture, AccelProvider, AccelProviderFuture,
    GpuTensorHandle, HostTensorView, ProviderPrecision, ThreadProviderGuard, WgpuBufferRef,
};
use runmat_runtime::builtins::plotting::{clone_figure, lock_plot_test_context, new_figure_handle};
use runmat_value::{Tensor, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

struct FailingBounds {
    inner: WgpuProvider,
    mode: AtomicUsize,
    downloads: AtomicUsize,
    bounds: AtomicUsize,
    frees: AtomicUsize,
}
impl AccelProvider for FailingBounds {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
    fn upload(&self, input: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(input)
    }
    fn export_context(&self, kind: AccelContextKind) -> Option<AccelContextHandle> {
        self.inner.export_context(kind)
    }
    fn export_wgpu_buffer(&self, input: &GpuTensorHandle) -> Option<WgpuBufferRef> {
        self.inner.export_wgpu_buffer(input)
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(input)
    }
    fn reduce_min<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.bounds.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if self.mode.load(Ordering::SeqCst) == 0 {
                futures::future::pending::<()>().await;
            }
            anyhow::bail!("injected plotting bounds failure")
        })
    }
    fn free(&self, input: &GpuTensorHandle) -> anyhow::Result<()> {
        self.frees.fetch_add(1, Ordering::SeqCst);
        self.inner.free(input)
    }
}

#[test]
#[ignore = "requires a compatible WGPU device; run explicitly for hardware qualification"]
fn exported_gpu_geometry_does_not_retry_device_errors_or_publish_pending_plots() {
    let inner = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("required WGPU plotting qualification device is unavailable");
    let provider = Box::leak(Box::new(FailingBounds {
        inner,
        mode: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        bounds: AtomicUsize::new(0),
        frees: AtomicUsize::new(0),
    }));
    // This fixture owns the newly allocated device namespace for the process
    // lifetime. Register that owner independently of ambient provider selection.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    let _plot = lock_plot_test_context();
    let _provider = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[2, 1],
        })
        .unwrap();
    assert!(provider.export_wgpu_buffer(&input).is_some());
    assert_eq!(
        input.device_id,
        provider.device_id(),
        "uploaded device identity"
    );
    assert!(
        runmat_accelerate_api::provider_for_device(input.device_id).is_some(),
        "registered device owner"
    );
    assert!(
        runmat_accelerate_api::export_wgpu_buffer(&input).is_some(),
        "registered buffer export"
    );
    for name in ["stairs", "area", "quiver", "errorbar", "scatter3"] {
        let figure = new_figure_handle();
        let resident = Value::GpuTensor(input.clone());
        let mut args = match name {
            "area" => vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
                resident,
            ],
            "quiver" => vec![
                resident.clone(),
                resident.clone(),
                resident.clone(),
                resident,
            ],
            "errorbar" | "scatter3" => vec![resident.clone(), resident.clone(), resident],
            _ => vec![resident.clone(), resident],
        };
        args.extend([Value::from("Visible"), Value::from("off")]);
        provider.mode.store(0, Ordering::SeqCst);
        provider.bounds.store(0, Ordering::SeqCst);
        let mut pending = Box::pin(runmat_runtime::call_builtin_async(name, &args));
        if let Poll::Ready(result) = pending.poll_unpin(&mut Context::from_waker(&noop_waker())) {
            panic!("{name}: bounds probe did not suspend: {result:?}");
        }
        assert_eq!(provider.bounds.load(Ordering::SeqCst), 1, "{name}");
        assert_eq!(clone_figure(figure).unwrap().plots().count(), 0, "{name}");
        drop(pending);
        provider.mode.store(1, Ordering::SeqCst);
        let error = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("injected plotting bounds failure"),
            "{name}: {error}"
        );
        assert_eq!(clone_figure(figure).unwrap().plots().count(), 0, "{name}");
        assert_eq!(
            provider.downloads.load(Ordering::SeqCst),
            0,
            "{name} retried on host"
        );
        assert_eq!(
            provider.frees.load(Ordering::SeqCst),
            0,
            "{name} freed borrowed input"
        );
        assert!(block_on(provider.inner.download(&input)).is_ok());
    }
    provider.free(&input).unwrap();
}
