use super::*;
use crate::builtins::common::test_support;
use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProviderFuture, HostTensorView, ThreadProviderGuard,
};
use std::{
    future::Future,
    sync::{Mutex, OnceLock},
    task::{Context, Poll},
};

#[derive(Clone, Copy, Default)]
enum Fault {
    #[default]
    None,
    Fail,
    Unsupported,
    Pending,
    Alias,
    BadStorage,
}
#[derive(Default)]
struct State {
    calls: usize,
    at: usize,
    fault: Fault,
    created: Vec<GpuTensorHandle>,
    freed: Vec<u64>,
    downloads: usize,
}
struct Provider {
    inner: InProcessProvider,
    state: Mutex<State>,
}
impl Provider {
    fn step(&self) -> Fault {
        let mut state = self.state.lock().unwrap();
        state.calls += 1;
        if state.calls == state.at {
            state.fault
        } else {
            Fault::None
        }
    }
    fn track(&self, mut output: GpuTensorHandle, fault: Fault) -> GpuTensorHandle {
        self.state.lock().unwrap().created.push(output.clone());
        if matches!(fault, Fault::BadStorage) {
            output.descriptor.storage = None;
        }
        output
    }
    async fn before(&self, fault: Fault) -> anyhow::Result<()> {
        match fault {
            Fault::Fail => anyhow::bail!("injected provider failure"),
            Fault::Unsupported => Err(runmat_accelerate_api::unsupported_provider_operation(
                "injected unavailable hook",
            )),
            Fault::Pending => std::future::pending().await,
            _ => Ok(()),
        }
    }
    fn assert_reclaimed(&self, published: usize) {
        let state = self.state.lock().unwrap();
        assert_eq!(state.created.len(), state.freed.len() + published);
        for handle in &state.created {
            assert!(
                state
                    .freed
                    .iter()
                    .filter(|id| **id == handle.buffer_id)
                    .count()
                    <= 1
            );
        }
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        self.inner.precision()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        let fault = self.step();
        if matches!(fault, Fault::Fail) {
            anyhow::bail!("injected upload failure");
        }
        Ok(self.track(self.inner.upload(host)?, fault))
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.state.lock().unwrap().downloads += 1;
        self.inner.download(handle)
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.state.lock().unwrap().freed.push(handle.buffer_id);
        self.inner.free(handle)
    }
    fn unary_cos<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let fault = self.step();
            self.before(fault).await?;
            if matches!(fault, Fault::Alias) {
                return Ok(input.clone());
            }
            Ok(self.track(self.inner.unary_cos(input).await?, fault))
        })
    }
    fn unary_sin<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let fault = self.step();
            self.before(fault).await?;
            if matches!(fault, Fault::Alias) {
                return Ok(input.clone());
            }
            Ok(self.track(self.inner.unary_sin(input).await?, fault))
        })
    }
    fn elem_mul<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let fault = self.step();
            self.before(fault).await?;
            if matches!(fault, Fault::Alias) {
                return Ok(right.clone());
            }
            Ok(self.track(self.inner.elem_mul(left, right).await?, fault))
        })
    }
}
fn provider(at: usize, fault: Fault) -> &'static Provider {
    static PROVIDER: OnceLock<Provider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(|| Provider {
        inner: InProcessProvider::new(),
        state: Mutex::default(),
    });
    *provider.state.lock().unwrap() = State {
        at,
        fault,
        ..State::default()
    };
    provider
}
fn inputs(provider: &Provider) -> (GpuTensorHandle, Vec<Value>) {
    let borrowed = provider
        .inner
        .upload(&HostTensorView {
            data: &[0.0, 0.0],
            shape: &[2, 1],
        })
        .unwrap();
    let args = vec![
        Value::GpuTensor(borrowed.clone()),
        Value::Num(2.0),
        Value::Tensor(Tensor::new(vec![7.0, 8.0, 9.0], vec![1, 3]).unwrap()),
    ];
    (borrowed, args)
}
fn run(args: &[Value], outputs: usize) -> BuiltinResult<Value> {
    block_on(crate::dispatcher::call_builtin_async_with_outputs(
        NAME, args, outputs,
    ))
}
fn assert_borrowed_live(provider: &Provider, borrowed: &GpuTensorHandle) {
    assert!(block_on(provider.inner.download(borrowed)).is_ok());
    assert!(!provider
        .state
        .lock()
        .unwrap()
        .freed
        .contains(&borrowed.buffer_id));
    provider.inner.free(borrowed).unwrap();
}

#[test]
fn every_provider_failure_reclaims_prior_temporaries_without_host_retry() {
    let _lock = test_support::accel_test_lock();
    for at in 1..=10 {
        let provider = provider(at, Fault::Fail);
        let _provider = ThreadProviderGuard::set(Some(provider));
        let (borrowed, args) = inputs(provider);
        assert!(run(&args, 3).is_err(), "failure point {at}");
        assert_eq!(provider.state.lock().unwrap().calls, at);
        assert_eq!(provider.state.lock().unwrap().downloads, 0);
        provider.assert_reclaimed(0);
        assert_borrowed_live(provider, &borrowed);
    }
}

#[test]
fn provider_zero_through_three_outputs_expand_xy_with_z_and_release_discarded_handles() {
    let _lock = test_support::accel_test_lock();
    for count in 0..=3 {
        let provider = provider(0, Fault::None);
        let _provider = ThreadProviderGuard::set(Some(provider));
        let (borrowed, args) = inputs(provider);
        let Value::OutputList(values) = run(&args, count).unwrap() else {
            panic!("outputs")
        };
        assert_eq!(values.len(), count);
        provider.assert_reclaimed(count);
        for (index, value) in values.into_iter().enumerate() {
            let Value::GpuTensor(handle) = value else {
                panic!("resident output")
            };
            assert_eq!(handle.shape, vec![2, 3]);
            let data = block_on(provider.inner.download(&handle)).unwrap().data;
            let expected = match index {
                0 => vec![2.0; 6],
                1 => vec![0.0; 6],
                _ => vec![7.0, 7.0, 8.0, 8.0, 9.0, 9.0],
            };
            assert_eq!(data, expected);
            provider.free(&handle).unwrap();
        }
        provider.assert_reclaimed(0);
        assert_borrowed_live(provider, &borrowed);
    }
}

#[test]
fn malformed_and_aliased_provider_outputs_are_rejected_without_freeing_inputs() {
    let _lock = test_support::accel_test_lock();
    for (at, fault) in [(3, Fault::Alias), (5, Fault::Alias), (3, Fault::BadStorage)] {
        let provider = provider(at, fault);
        let _provider = ThreadProviderGuard::set(Some(provider));
        let (borrowed, args) = inputs(provider);
        let error = run(&args, 3).unwrap_err();
        assert_eq!(error.identifier(), ERROR_INTERNAL.identifier);
        provider.assert_reclaimed(0);
        assert_borrowed_live(provider, &borrowed);
    }
}

#[test]
fn only_unsupported_hooks_fall_back_and_pending_drop_reclaims_owned_uploads() {
    let _lock = test_support::accel_test_lock();
    let provider = provider(4, Fault::Unsupported);
    let _provider = ThreadProviderGuard::set(Some(provider));
    let (borrowed, args) = inputs(provider);
    let Value::OutputList(values) = run(&args, 3).unwrap() else {
        panic!("outputs")
    };
    assert!(values.iter().all(|v| matches!(v, Value::Tensor(_))));
    provider.assert_reclaimed(0);
    assert_borrowed_live(provider, &borrowed);
    *provider.state.lock().unwrap() = State {
        at: 4,
        fault: Fault::Pending,
        ..State::default()
    };
    let (borrowed, args) = inputs(provider);
    let mut future = Box::pin(crate::dispatcher::call_builtin_async_with_outputs(
        NAME, &args, 3,
    ));
    let waker = noop_waker();
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(provider.state.lock().unwrap().calls, 4);
    drop(future);
    provider.assert_reclaimed(0);
    assert_borrowed_live(provider, &borrowed);
}

#[test]
fn invalid_form_and_unpolled_drop_never_touch_provider() {
    let _lock = test_support::accel_test_lock();
    let provider = provider(0, Fault::None);
    let _provider = ThreadProviderGuard::set(Some(provider));
    let (borrowed, mut args) = inputs(provider);
    drop(crate::dispatcher::call_builtin_async_with_outputs(
        NAME, &args, 3,
    ));
    assert!(run(&args, 4).is_err());
    args.push(Value::Num(0.0));
    assert!(run(&args, 3).is_err());
    args.pop();
    args[2] = Value::Tensor(Tensor::new(vec![1.0; 3], vec![3, 1]).unwrap());
    assert!(run(&args, 3).is_err());
    args[2] = Value::String("invalid".into());
    assert!(run(&args, 3).is_err());
    args[2] = Value::Num(1.0);
    let mut malformed = borrowed.clone();
    malformed.descriptor.storage = None;
    args[0] = Value::GpuTensor(malformed);
    assert!(run(&args, 3).is_err());
    let state = provider.state.lock().unwrap();
    assert_eq!(state.calls, 0);
    assert_eq!(state.downloads, 0);
    drop(state);
    assert_borrowed_live(provider, &borrowed);
}
