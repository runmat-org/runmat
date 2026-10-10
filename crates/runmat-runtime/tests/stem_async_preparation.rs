#![cfg(feature = "plot-core")]
//! Real dispatcher coverage with one linked runtime state/registry owner.
use futures::{executor::block_on, task::noop_waker, FutureExt};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, HostTensorView, ProviderPrecision,
    ThreadProviderGuard,
};
use runmat_runtime::builtins::plotting::{clone_figure, lock_plot_test_context, new_figure_handle};
use runmat_value::{Tensor, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

#[path = "support/retained_provider.rs"]
mod retained_provider;

struct Provider {
    inner: InProcessProvider,
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
    fn upload(&self, input: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(input)
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            futures::future::poll_fn(|_| match self.mode.load(Ordering::SeqCst) {
                0 => Poll::Pending,
                mode => Poll::Ready(mode),
            })
            .await;
            if self.mode.load(Ordering::SeqCst) == 1 {
                anyhow::bail!("injected download failure");
            }
            self.inner.download(input).await
        })
    }
}

#[test]
fn dispatched_stem_and_stairs_publish_only_after_preparation_completes() {
    let provider = retained_provider::retained_test_provider!(PROVIDER: Provider = Provider {
        inner: InProcessProvider::new(),
        mode: AtomicUsize::new(0),
    });
    let _plot = lock_plot_test_context();
    let _current = ThreadProviderGuard::set(Some(provider));
    for name in [
        "stem", "stairs", "quiver3", "area", "quiver", "errorbar", "scatter3",
    ] {
        provider.mode.store(0, Ordering::SeqCst);
        assert_preparation_lifecycle(name, provider);
    }
}

fn assert_preparation_lifecycle(name: &str, provider: &Provider) {
    let figure = new_figure_handle();
    let count = || clone_figure(figure).unwrap().plots().count();
    let before = count();
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    let physical = input.allocation_lease().downgrade();
    let descriptor = input.wire_descriptor();
    let host = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let mut args = if name == "quiver3" {
        vec![
            host.clone(),
            host.clone(),
            host.clone(),
            host.clone(),
            host,
            Value::GpuTensor(input.clone()),
        ]
    } else if name == "errorbar" || name == "scatter3" {
        vec![host.clone(), host, Value::GpuTensor(input.clone())]
    } else if name == "quiver" {
        vec![
            host.clone(),
            host.clone(),
            host,
            Value::GpuTensor(input.clone()),
        ]
    } else {
        vec![host, Value::GpuTensor(input.clone())]
    };
    args.extend([Value::from("Visible"), Value::from("off")]);
    let mut invalid = args.to_vec();
    *invalid.last_mut().unwrap() = Value::from("invalid");
    let mut rejected = Box::pin(runmat_runtime::call_builtin_async(name, &invalid));
    assert!(matches!(
        rejected.poll_unpin(&mut Context::from_waker(&noop_waker())),
        Poll::Ready(Err(_))
    ));
    assert_eq!(count(), before);
    drop(rejected);
    let mut future = Box::pin(runmat_runtime::call_builtin_async(name, &args));
    assert!(matches!(
        future.poll_unpin(&mut Context::from_waker(&noop_waker())),
        Poll::Pending
    ));
    assert_eq!(count(), before);
    drop(future);
    assert_eq!(count(), before);
    assert!(physical.acquire().is_some());
    assert_eq!(input.wire_descriptor(), descriptor);
    provider.mode.store(1, Ordering::SeqCst);
    assert!(block_on(runmat_runtime::call_builtin_async(name, &args)).is_err());
    assert_eq!(count(), before);
    provider.mode.store(0, Ordering::SeqCst);
    let mut future = Box::pin(runmat_runtime::call_builtin_async(name, &args));
    assert!(matches!(
        future.poll_unpin(&mut Context::from_waker(&noop_waker())),
        Poll::Pending
    ));
    assert_eq!(count(), before);
    provider.mode.store(2, Ordering::SeqCst);
    block_on(future).unwrap();
    assert_eq!(count(), before + 1);
    assert!(clone_figure(figure)
        .unwrap()
        .plots()
        .all(|plot| !plot.is_visible()));
    assert!(physical.acquire().is_some());
    assert_eq!(input.wire_descriptor(), descriptor);
    assert_eq!(
        block_on(provider.inner.download(&input)).unwrap().data,
        vec![1.0, 2.0]
    );
    drop(invalid);
    drop(args);
    drop(input);
    assert!(physical.acquire().is_none());
}
