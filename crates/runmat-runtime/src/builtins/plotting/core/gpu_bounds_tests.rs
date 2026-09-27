use super::*;
use futures::{executor::block_on, task::noop_waker, FutureExt};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProviderFuture, HostTensorView, ProviderPrecision,
    ThreadProviderGuard,
};
use std::sync::Mutex;
use std::task::{Context, Poll};

#[derive(Clone)]
enum Mode {
    Ready,
    PendingMax,
    ErrorMax,
    Alias(GpuTensorHandle),
    PendingDownload,
    ErrorDownload,
    RejectLogicalView,
}
struct Provider {
    inner: InProcessProvider,
    mode: Mutex<Mode>,
    produced: Mutex<Vec<runmat_accelerate_api::GpuTensorWireDescriptor>>,
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
                Mode::RejectLogicalView => {
                    let output = self.inner.reduce_min(input).await?;
                    assert!(input
                        .with_view(input.view().clone().with_logical(true))
                        .is_err());

                    self.produced.lock().unwrap().push(output.wire_descriptor());
                    Ok(output)
                }
                _ => {
                    let output = self.inner.reduce_min(input).await?;
                    self.produced.lock().unwrap().push(output.wire_descriptor());
                    Ok(output)
                }
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
                Mode::PendingMax => {
                    assert!(input
                        .with_view(input.view().clone().with_logical(true))
                        .is_err());
                    futures::future::pending().await
                }
                Mode::ErrorMax => anyhow::bail!("injected max failure"),
                Mode::RejectLogicalView => {
                    assert!(!runmat_accelerate_api::handle_is_logical(input));
                    assert!(runmat_accelerate_api::handle_transpose_info(input).is_none());
                    {
                        let output = self.inner.reduce_max(input).await?;
                        self.produced.lock().unwrap().push(output.wire_descriptor());
                        Ok(output)
                    }
                }
                _ => {
                    let output = self.inner.reduce_max(input).await?;
                    self.produced.lock().unwrap().push(output.wire_descriptor());
                    Ok(output)
                }
            }
        })
    }
}
fn provider(mode: Mode) -> &'static Provider {
    Box::leak(Box::new(Provider {
        inner: InProcessProvider::new(),
        mode: Mutex::new(mode),
        produced: Mutex::new(Vec::new()),
    }))
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
fn bounds_reject_logical_reinterpretation_and_keep_admitted_output_type() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider(Mode::RejectLogicalView);
    let _current = ThreadProviderGuard::set(Some(provider));
    let input = upload(provider);
    assert_eq!(
        block_on(axis_bounds_protecting(&input, &[&input], "stem")).unwrap(),
        (1.0, 2.0)
    );
    assert!(!runmat_accelerate_api::handle_is_logical(&input));
    let produced = provider.produced.lock().unwrap().clone();
    assert_eq!(produced.len(), 2);
    assert!(produced
        .iter()
        .all(|wire| provider.inner.acquire_live(wire).is_err()));
    assert!(block_on(provider.inner.download(&input)).is_ok());
    input.release().unwrap();
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
        provider.produced.lock().unwrap().clear();
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
        let retired = provider.produced.lock().unwrap().clone();
        assert_eq!(retired.len(), usize::from(!matches!(mode, Mode::Alias(_))));
        assert!(retired
            .iter()
            .all(|wire| provider.inner.acquire_live(wire).is_err()));
        assert!(!runmat_accelerate_api::handle_is_logical(&x));
        assert!(block_on(provider.inner.download(&x)).is_ok());
        assert!(block_on(provider.inner.download(&y)).is_ok());
    }
    x.release().unwrap();
    y.release().unwrap();
}

#[test]
fn stem_body_preparation_can_suspend_fail_and_drop_before_plot_mutation() {
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
    let mut future = Box::pin(crate::builtins::plotting::stem::stem_builtin(args.to_vec()));
    assert!(matches!(
        future.poll_unpin(&mut Context::from_waker(&noop_waker())),
        Poll::Pending
    ));
    assert_eq!(count(), initial);
    drop(future);
    assert_eq!(count(), initial);
    assert!(provider.produced.lock().unwrap().is_empty());
    *provider.mode.lock().unwrap() = Mode::ErrorDownload;
    assert!(block_on(crate::builtins::plotting::stem::stem_builtin(args.to_vec())).is_err());
    assert_eq!(count(), initial);
    *provider.mode.lock().unwrap() = Mode::Ready;
    block_on(crate::builtins::plotting::stem::stem_builtin(args.to_vec())).unwrap();
    assert_eq!(count(), initial + 1);
    assert!(block_on(provider.inner.download(&input)).is_ok());
    assert!(provider.produced.lock().unwrap().is_empty());
    input.release().unwrap();
}
