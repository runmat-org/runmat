use super::*;
use futures::{executor::block_on, task::noop_waker, FutureExt};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProviderFuture, HostTensorView, ProviderPrecision,
    ThreadProviderGuard,
};
use std::sync::{Mutex, OnceLock};
use std::task::{Context, Poll};

#[derive(Clone)]
enum Mode {
    Ready,
    PendingMax,
    ErrorMax,
    Alias(GpuTensorHandle),
    PendingDownload,
    ErrorDownload,
}
struct Provider {
    inner: InProcessProvider,
    mode: Mutex<Mode>,
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
    fn upload(&self, input: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(input)
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        let mode = self.mode.lock().unwrap().clone();
        Box::pin(async move {
            match mode {
                Mode::PendingDownload => futures::future::pending().await,
                Mode::ErrorDownload => anyhow::bail!("injected download failure"),
                _ => self.inner.download(input).await,
            }
        })
    }
    fn reduce_min<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        let mode = self.mode.lock().unwrap().clone();
        Box::pin(async move {
            match mode {
                Mode::Alias(handle) => Ok(handle),
                _ => self.inner.reduce_min(input).await,
            }
        })
    }
    fn reduce_max<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        let mode = self.mode.lock().unwrap().clone();
        Box::pin(async move {
            match mode {
                Mode::PendingMax => futures::future::pending().await,
                Mode::ErrorMax => anyhow::bail!("injected max failure"),
                _ => self.inner.reduce_max(input).await,
            }
        })
    }
    fn free(&self, input: &GpuTensorHandle) -> anyhow::Result<()> {
        self.freed.lock().unwrap().push(input.buffer_id);
        self.inner.free(input)
    }
}
fn provider(mode: Mode) -> &'static Provider {
    static PROVIDER: OnceLock<Provider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(|| Provider {
        inner: InProcessProvider::new(),
        mode: Mutex::new(Mode::Ready),
        freed: Mutex::new(Vec::new()),
    });
    *provider.mode.lock().unwrap() = mode;
    provider.freed.lock().unwrap().clear();
    provider
}
fn upload(provider: &Provider) -> GpuTensorHandle {
    provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap()
}

#[test]
fn bounds_drop_error_and_cross_coordinate_alias_preserve_borrowed_inputs() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider(Mode::Ready);
    let _current = ThreadProviderGuard::set(Some(provider));
    let x = upload(provider);
    let y = upload(provider);
    for mode in [Mode::PendingMax, Mode::ErrorMax, Mode::Alias(y.clone())] {
        *provider.mode.lock().unwrap() = mode.clone();
        provider.freed.lock().unwrap().clear();
        let protected = [&x, &y];
        let mut future = Box::pin(axis_bounds_protecting(&x, &protected, "stem"));
        if matches!(mode, Mode::PendingMax) {
            assert!(matches!(
                future.poll_unpin(&mut Context::from_waker(&noop_waker())),
                Poll::Pending
            ));
            drop(future);
        } else {
            assert!(block_on(future).is_err());
        }
        let freed = provider.freed.lock().unwrap().clone();
        assert_eq!(freed.len(), usize::from(!matches!(mode, Mode::Alias(_))));
        assert!(!freed.contains(&x.buffer_id) && !freed.contains(&y.buffer_id));
        assert!(block_on(provider.inner.download(&x)).is_ok());
        assert!(block_on(provider.inner.download(&y)).is_ok());
    }
    provider.free(&x).unwrap();
    provider.free(&y).unwrap();
}

#[test]
fn dispatched_stem_preparation_can_suspend_fail_and_drop_before_plot_mutation() {
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let _plots = lock_plot_registry();
    ensure_plot_test_env();
    let _ = clear_figure(None);
    let figure = current_figure_handle();
    let count = || clone_figure(figure).unwrap().plots().count();
    let initial = count();
    let provider = provider(Mode::PendingDownload);
    let _current = ThreadProviderGuard::set(Some(provider));
    let input = upload(provider);
    // Mixed host/resident coordinates choose the owner-gather route without
    // depending on whether the test machine has a WGPU rendering context.
    let args = [
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        Value::GpuTensor(input.clone()),
    ];
    let mut future = Box::pin(crate::dispatcher::call_builtin_async("stem", &args));
    assert!(matches!(
        future.poll_unpin(&mut Context::from_waker(&noop_waker())),
        Poll::Pending
    ));
    assert_eq!(count(), initial);
    drop(future);
    assert_eq!(count(), initial);
    assert!(provider.freed.lock().unwrap().is_empty());
    *provider.mode.lock().unwrap() = Mode::ErrorDownload;
    assert!(block_on(crate::dispatcher::call_builtin_async("stem", &args)).is_err());
    assert_eq!(count(), initial);
    *provider.mode.lock().unwrap() = Mode::Ready;
    block_on(crate::dispatcher::call_builtin_async("stem", &args)).unwrap();
    assert_eq!(count(), initial + 1);
    assert!(block_on(provider.inner.download(&input)).is_ok());
    assert!(provider.freed.lock().unwrap().is_empty());
    provider.free(&input).unwrap();
}
