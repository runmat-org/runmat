#![cfg(all(feature = "plot-core", feature = "wgpu"))]

use futures::{executor::block_on, task::noop_waker, FutureExt};
use runmat_accelerate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
use runmat_accelerate_api::{
    AccelContextHandle, AccelContextKind, AccelDownloadFuture, AccelProvider, GpuTensorHandle,
    HostTensorView, ProviderPrecision, ThreadProviderGuard, WgpuBufferRef,
};
use runmat_runtime::builtins::plotting::{clone_figure, lock_plot_test_context, new_figure_handle};
use runmat_value::{Tensor, Value};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::task::{Context, Poll};

struct FailingBounds {
    inner: WgpuProvider,
    mode: AtomicUsize,
    downloads: AtomicUsize,
    bounds: AtomicUsize,
    refused_export: AtomicU64,
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
    fn export_wgpu_buffer(
        &self,
        input: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<WgpuBufferRef>> {
        match self.mode.load(Ordering::SeqCst) {
            2 => Ok(None),
            5 if input.buffer_id() == self.refused_export.load(Ordering::SeqCst) => Ok(None),
            3 | 4 => {
                let error = runmat_accelerate_api::ProviderExecutionError::from(
                    runmat_accelerate_api::unsupported_provider_operation(
                        "injected export refusal",
                    ),
                );
                Err(if self.mode.load(Ordering::SeqCst) == 4 {
                    error.after_effect()
                } else {
                    error
                })
            }
            _ => self.inner.export_wgpu_buffer(input),
        }
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(input)
    }
    fn reduce_min<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.bounds.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if self.mode.load(Ordering::SeqCst) == 0 {
                futures::future::pending::<()>().await;
            }
            Err(anyhow::anyhow!("injected plotting bounds failure").into())
        })
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
        refused_export: AtomicU64::new(0),
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
    let physical = input.allocation_lease().downgrade();
    let descriptor = input.wire_descriptor();
    assert!(provider.export_wgpu_buffer(&input).unwrap().is_some());
    assert_eq!(
        input.device_id(),
        provider.device_id(),
        "uploaded device identity"
    );
    assert!(
        runmat_accelerate_api::registered_providers()
            .iter()
            .any(|owner| owner.device_id() == input.device_id()),
        "registered device owner"
    );
    assert!(
        runmat_accelerate_api::export_wgpu_buffer(&input)
            .unwrap()
            .is_some(),
        "registered buffer export"
    );
    for name in [
        "stairs", "area", "quiver", "errorbar", "scatter3", "plot", "line",
    ] {
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
        provider.downloads.store(0, Ordering::SeqCst);
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
        assert!(
            physical.acquire().is_some(),
            "{name} lost borrowed allocation"
        );
        assert_eq!(input.wire_descriptor(), descriptor);
        assert_eq!(
            block_on(provider.inner.download(&input)).unwrap().data,
            vec![1.0, 2.0]
        );
        // A typed error (even a retained Unsupported cause) is not None.
        // Both pre-effect and post-effect failures remain terminal.
        for mode in [3, 4] {
            provider.mode.store(mode, Ordering::SeqCst);
            provider.downloads.store(0, Ordering::SeqCst);
            provider.bounds.store(0, Ordering::SeqCst);
            let error = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap_err();
            assert_eq!(
                error.gpu_gather_retry,
                runmat_runtime::GpuGatherRetry::Never,
                "{name}"
            );
            assert!(
                error.to_string().contains("injected export refusal"),
                "{name}: {error}"
            );
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0, "{name}");
            assert_eq!(provider.bounds.load(Ordering::SeqCst), 0, "{name}");
            assert_eq!(clone_figure(figure).unwrap().plots().count(), 0, "{name}");
            assert_eq!(input.wire_descriptor(), descriptor);
        }
        // Pure absence preserves supported host rendering, with no reduction
        // attempt or hidden export materialization preceding the gathers.
        provider.mode.store(2, Ordering::SeqCst);
        provider.downloads.store(0, Ordering::SeqCst);
        block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap();
        assert!(provider.downloads.load(Ordering::SeqCst) > 0, "{name}");
        assert_eq!(provider.bounds.load(Ordering::SeqCst), 0, "{name}");
        assert!(clone_figure(figure).unwrap().plots().count() > 0, "{name}");
        assert!(physical.acquire().is_some(), "{name}");
    }
    // A nested attribute refusal must retain the coordinate export stage.
    // The transposed vector actually materializes before scatter asks for S.
    let transposed = provider.inner.transpose(&input).unwrap();
    assert!(runmat_accelerate_api::handle_transpose_info(&transposed).is_some());
    let attribute = provider
        .inner
        .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
            data: runmat_accelerate_api::HostNumericDataView::F32(&[2.0, 3.0]),
            shape: &[2, 1],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
        .unwrap();
    let figure = new_figure_handle();
    provider
        .refused_export
        .store(attribute.buffer_id(), Ordering::SeqCst);
    provider.mode.store(5, Ordering::SeqCst);
    provider.downloads.store(0, Ordering::SeqCst);
    provider.bounds.store(0, Ordering::SeqCst);
    let error = block_on(runmat_runtime::call_builtin_async(
        "scatter",
        &[
            Value::GpuTensor(transposed.clone()),
            Value::GpuTensor(transposed.clone()),
            Value::GpuTensor(attribute.clone()),
            Value::String("Visible".into()),
            Value::String("off".into()),
        ],
    ))
    .unwrap_err();
    assert_eq!(
        error.gpu_gather_retry,
        runmat_runtime::GpuGatherRetry::Never
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.bounds.load(Ordering::SeqCst), 0);
    assert_eq!(clone_figure(figure).unwrap().plots().count(), 0);
    transposed.allocation_lease().validate_live().unwrap();
    attribute.allocation_lease().validate_live().unwrap();
    drop((transposed, attribute));
    drop(input);
    assert!(physical.acquire().is_none());
}
