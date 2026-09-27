use super::*;
use crate::builtins::common::test_support;
use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProviderFuture, HostTensorView, ThreadProviderGuard,
};
use runmat_value::Value;
use std::{
    future::Future,
    sync::{Mutex, OnceLock},
    task::{Context, Poll},
};

#[derive(Clone, Copy, Default)]
enum Fault {
    #[default]
    None,
    Failure,
    Unsupported,
    Pending,
    BorrowedAlias,
    TemporaryAlias,
    MissingStorage,
}

#[derive(Default)]
struct State {
    at: usize,
    calls: usize,
    fault: Fault,
    owned: Vec<GpuTensorHandle>,
    freed: Vec<u64>,
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

    async fn before(&self, fault: Fault) -> anyhow::Result<()> {
        match fault {
            Fault::Failure => anyhow::bail!("injected range failure"),
            Fault::Unsupported => Err(runmat_accelerate_api::unsupported_provider_operation(
                "range test",
            )),
            Fault::Pending => std::future::pending().await,
            _ => Ok(()),
        }
    }

    async fn reduce(
        &self,
        input: &GpuTensorHandle,
        minimum: bool,
    ) -> anyhow::Result<GpuTensorHandle> {
        let fault = self.step();
        self.before(fault).await?;
        if matches!(fault, Fault::BorrowedAlias) {
            return Ok(input.clone());
        }
        if matches!(fault, Fault::TemporaryAlias) {
            return Ok(self.state.lock().unwrap().owned[0].clone());
        }
        let mut output = if minimum {
            self.inner.reduce_min(input).await?
        } else {
            self.inner.reduce_max(input).await?
        };
        self.state.lock().unwrap().owned.push(output.clone());
        if matches!(fault, Fault::MissingStorage) {
            output.descriptor.storage = None;
        }
        Ok(output)
    }

    fn assert_reclaimed(&self, input: &GpuTensorHandle) {
        let state = self.state.lock().unwrap();
        assert_eq!(state.owned.len(), state.freed.len());
        for output in &state.owned {
            assert_eq!(
                state
                    .freed
                    .iter()
                    .filter(|id| **id == output.buffer_id)
                    .count(),
                1
            );
        }
        assert!(!state.freed.contains(&input.buffer_id));
        drop(state);
        assert!(block_on(self.inner.download(input)).is_ok());
        self.inner.free(input).unwrap();
    }
}

impl AccelProvider for Provider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        self.inner.precision()
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let fault = self.step();
            self.before(fault).await?;
            self.inner.download(input).await
        })
    }
    fn free(&self, input: &GpuTensorHandle) -> anyhow::Result<()> {
        self.state.lock().unwrap().freed.push(input.buffer_id);
        self.inner.free(input)
    }
    fn reduce_min<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(self.reduce(input, true))
    }
    fn reduce_max<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(self.reduce(input, false))
    }
}

fn fixture(at: usize, fault: Fault) -> (&'static Provider, GpuTensorHandle) {
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
    let input = provider
        .inner
        .upload(&HostTensorView {
            data: &[0.0, 0.5],
            shape: &[1, 2],
        })
        .unwrap();
    (provider, input)
}

#[test]
fn inverse_range_actual_call_failure_is_not_host_fallback_and_reclaims_temporaries() {
    let _serial = test_support::accel_test_lock();
    for (name, steps) in [("asin", 4), ("acos", 4), ("atanh", 4), ("acosh", 2)] {
        for step in 1..=steps {
            let (provider, input) = fixture(step, Fault::Failure);
            let _provider = ThreadProviderGuard::set(Some(provider));
            let result = block_on(crate::dispatcher::call_builtin_async_with_outputs(
                name,
                &[Value::GpuTensor(input.clone())],
                1,
            ));
            assert!(result.is_err(), "{name} step {step}");
            assert_eq!(
                provider.state.lock().unwrap().calls,
                step,
                "must not retry on host"
            );
            provider.assert_reclaimed(&input);
        }
    }
}

#[test]
fn inverse_range_actual_call_drop_reclaims_each_suspended_stage() {
    let _serial = test_support::accel_test_lock();
    for (name, steps) in [("asin", 4), ("acos", 4), ("atanh", 4), ("acosh", 2)] {
        for step in 0..=steps {
            let (provider, input) = fixture(step, Fault::Pending);
            let _provider = ThreadProviderGuard::set(Some(provider));
            let args = [Value::GpuTensor(input.clone())];
            let mut call = Box::pin(crate::dispatcher::call_builtin_async_with_outputs(
                name, &args, 1,
            ));
            if step != 0 {
                assert!(matches!(
                    call.as_mut().poll(&mut Context::from_waker(&noop_waker())),
                    Poll::Pending
                ));
            }
            drop(call);
            assert_eq!(provider.state.lock().unwrap().calls, step);
            provider.assert_reclaimed(&input);
        }
    }
}

#[test]
fn inverse_range_rejects_aliases_and_missing_storage_without_freeing_borrowed_input() {
    let _serial = test_support::accel_test_lock();
    for (step, fault) in [
        (1, Fault::BorrowedAlias),
        (2, Fault::TemporaryAlias),
        (1, Fault::MissingStorage),
        (2, Fault::MissingStorage),
    ] {
        let (provider, input) = fixture(step, fault);
        let _provider = ThreadProviderGuard::set(Some(provider));
        assert!(block_on(requires_complex(
            provider,
            &input,
            RealDomain::ClosedUnitInterval,
            "asin"
        ))
        .is_err());
        assert_eq!(provider.state.lock().unwrap().calls, step);
        provider.assert_reclaimed(&input);
    }
}

#[test]
fn inverse_range_only_unsupported_hook_is_inconclusive() {
    let _serial = test_support::accel_test_lock();
    for step in 1..=2 {
        let (provider, input) = fixture(step, Fault::Unsupported);
        let _provider = ThreadProviderGuard::set(Some(provider));
        assert_eq!(
            block_on(requires_complex(
                provider,
                &input,
                RealDomain::ClosedUnitInterval,
                "asin"
            ))
            .unwrap(),
            None
        );
        provider.assert_reclaimed(&input);
    }
    let (provider, input) = fixture(0, Fault::None);
    let _provider = ThreadProviderGuard::set(Some(provider));
    assert_eq!(
        block_on(requires_complex(
            provider,
            &input,
            RealDomain::ClosedUnitInterval,
            "asin"
        ))
        .unwrap(),
        Some(false)
    );
    provider.assert_reclaimed(&input);
}

#[test]
fn inverse_range_abandonment_keeps_its_admitted_provider_after_selection_is_removed() {
    let _serial = test_support::accel_test_lock();
    let (provider, input) = fixture(2, Fault::Pending);
    let selected = ThreadProviderGuard::set(Some(provider));
    let mut call = Box::pin(requires_complex(
        provider,
        &input,
        RealDomain::ClosedUnitInterval,
        "asin",
    ));
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&noop_waker())),
        Poll::Pending
    ));
    drop(selected);
    for output in &provider.state.lock().unwrap().owned {
        runmat_accelerate_api::clear_handle_metadata(output);
    }
    drop(call);
    provider.assert_reclaimed(&input);
}
