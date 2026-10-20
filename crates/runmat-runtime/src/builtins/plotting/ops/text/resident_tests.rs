use super::*;
use crate::builtins::common::test_support;
use crate::builtins::plotting::state::{axes_handle_exists, reset_plot_state, FigureHandle};
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::builtins::plotting::{clone_figure, current_figure_handle};
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
    reset_plot_state();
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
    let figure = FigureHandle::default();
    let absent = || !axes_handle_exists(figure, 0);
    assert!(absent());
    let args = || {
        vec![
            Value::GpuTensor(input.clone()),
            Value::Num(3.0),
            Value::String("pending".into()),
        ]
    };

    let invalid = futures::executor::block_on(text_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::GpuTensor(input.clone()),
            Value::Num(3.0),
            Value::Num(4.0),
        ]),
    ))
    .expect_err("missing label must reject before download or scene creation");
    assert_eq!(invalid.identifier(), TEXT_ERROR_INVALID_ARGUMENT.identifier);
    assert!(absent());
    assert_eq!(provider.attempts.load(Ordering::SeqCst), 0);

    let nonscalar = provider
        .upload(&HostTensorView {
            data: &[2.0, 3.0],
            shape: &[1, 2],
        })
        .unwrap();
    let invalid = futures::executor::block_on(text_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::GpuTensor(nonscalar),
            Value::Num(3.0),
            Value::String("invalid shape".into()),
        ]),
    ))
    .expect_err("nonscalar resident coordinate must reject before download");
    assert_eq!(invalid.identifier(), TEXT_ERROR_INVALID_ARGUMENT.identifier);
    assert!(absent());
    assert_eq!(provider.attempts.load(Ordering::SeqCst), 0);

    let pre_cancel = CancellationSource::new();
    pre_cancel.cancel(CancellationReason::User);
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::default()),
        pre_cancel.token(),
    );
    let error = futures::executor::block_on(text_builtin(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(args()),
    ))
    .expect_err("pre-cancelled call must not poll provider or create scene");
    assert!(matches!(
        error
            .source()
            .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>()),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
    assert!(absent());
    assert_eq!(provider.attempts.load(Ordering::SeqCst), 0);

    let mut dropped = Box::pin(text_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(args()),
    ));
    assert!(matches!(
        dropped
            .as_mut()
            .poll(&mut Context::from_waker(&noop_waker())),
        Poll::Pending
    ));
    assert!(absent());
    drop(dropped);
    assert!(absent());
    assert_eq!(provider.attempts.load(Ordering::SeqCst), 1);
    assert_eq!(provider.finished_futures.load(Ordering::SeqCst), 1);

    let cancellation = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::default()),
        cancellation.token(),
    );
    let mut cancelled = Box::pin(text_builtin(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(args()),
    ));
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
    assert!(absent());
    assert_eq!(provider.attempts.load(Ordering::SeqCst), 2);
    assert_eq!(provider.finished_futures.load(Ordering::SeqCst), 2);

    *provider.mode.lock().unwrap() = DownloadMode::Ready;
    futures::executor::block_on(text_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(args()),
    ))
    .unwrap();
    assert_eq!(current_figure_handle(), figure);
    assert_eq!(
        clone_figure(figure).unwrap().axes_text_annotations(0).len(),
        1
    );
    assert!(futures::executor::block_on(provider.inner.download(&input)).is_ok());
}
