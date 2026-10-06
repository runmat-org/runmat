use super::*;
use crate::builtins::common::test_support;
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};
use futures::task::noop_waker;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, HostTensorView, ProviderPrecision,
    ThreadProviderGuard,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::error::Error;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::task::{Context, Poll};

#[derive(Clone, Copy)]
enum DownloadMode {
    Pending,
    Ready,
}

struct PendingDownloadProvider {
    inner: InProcessProvider,
    mode: Mutex<DownloadMode>,
    attempts: AtomicUsize,
    finished_futures: AtomicUsize,
}

struct DownloadFutureEnd<'a>(&'a AtomicUsize);
impl Drop for DownloadFutureEnd<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl AccelProvider for PendingDownloadProvider {
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
        self.attempts.fetch_add(1, Ordering::SeqCst);
        let mode = *self.mode.lock().unwrap();
        Box::pin(async move {
            let _end = DownloadFutureEnd(&self.finished_futures);
            match mode {
                DownloadMode::Pending => futures::future::pending().await,
                DownloadMode::Ready => self.inner.download(input).await,
            }
        })
    }
}

#[test]
fn pending_coordinate_download_cancel_and_drop_do_not_publish_or_consume_input() {
    let _global = test_support::accel_test_lock();
    let _plots = lock_plot_registry();
    ensure_plot_test_env();
    let _ = clear_figure(None);
    let provider: &'static PendingDownloadProvider = Box::leak(Box::new(PendingDownloadProvider {
        inner: InProcessProvider::new(),
        mode: Mutex::new(DownloadMode::Pending),
        attempts: AtomicUsize::new(0),
        finished_futures: AtomicUsize::new(0),
    }));
    let _current = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload(&HostTensorView {
            data: &[2.0],
            shape: &[1, 1],
        })
        .unwrap();
    let figure = current_figure_handle();
    let count = || clone_figure(figure).unwrap().axes_text_annotations(0).len();
    let original = count();
    let args = || {
        vec![
            Value::GpuTensor(input.clone()),
            Value::Num(3.0),
            Value::String("pending".into()),
        ]
    };

    let mut dropped = Box::pin(text_builtin(None, args()));
    assert!(matches!(
        dropped
            .as_mut()
            .poll(&mut Context::from_waker(&noop_waker())),
        Poll::Pending
    ));
    assert_eq!(count(), original);
    drop(dropped);
    assert_eq!(count(), original);
    assert_eq!(provider.attempts.load(Ordering::SeqCst), 1);
    assert_eq!(provider.finished_futures.load(Ordering::SeqCst), 1);

    let cancellation = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::default()),
        cancellation.token(),
    );
    let mut cancelled = Box::pin(text_builtin(Some(runtime), args()));
    assert!(matches!(
        cancelled
            .as_mut()
            .poll(&mut Context::from_waker(&noop_waker())),
        Poll::Pending
    ));
    cancellation.cancel(CancellationReason::User);
    let error = match cancelled
        .as_mut()
        .poll(&mut Context::from_waker(&noop_waker()))
    {
        Poll::Ready(Err(error)) => error,
        _ => panic!("cancelled text download must terminate"),
    };
    assert!(matches!(
        error
            .source()
            .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>()),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
    drop(cancelled);
    assert_eq!(count(), original);
    assert_eq!(provider.attempts.load(Ordering::SeqCst), 2);
    assert_eq!(provider.finished_futures.load(Ordering::SeqCst), 2);

    *provider.mode.lock().unwrap() = DownloadMode::Ready;
    futures::executor::block_on(text_builtin(None, args())).unwrap();
    assert_eq!(count(), original + 1);
    assert!(futures::executor::block_on(provider.inner.download(&input)).is_ok());
}
