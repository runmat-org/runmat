use super::*;
use crate::builtins::common::test_support;
use crate::builtins::math::trigonometry::test_owner;
use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate_api::test_support::RetirementProbe;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProviderFuture, HostTensorView, ThreadProviderGuard,
};
use std::{
    future::Future,
    sync::Mutex,
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
    InvalidLogicalViewAlias,
}
#[derive(Default)]
struct State {
    calls: usize,
    at: usize,
    fault: Fault,
    created: Vec<RetirementProbe>,
    downloads: usize,
}
struct Provider {
    inner: test_owner::Owner,
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
    fn track(&self, output: GpuTensorHandle, fault: Fault) -> GpuTensorHandle {
        let output = if matches!(fault, Fault::BadStorage) {
            let host = self.inner.download(&output).unwrap();
            let data: Vec<_> = host
                .data
                .into_iter()
                .flat_map(|value| [value, 0.0])
                .collect();
            let wrong = self
                .inner
                .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                    data: runmat_accelerate_api::HostNumericDataView::F64(&data),
                    shape: &host.shape,
                    storage: runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
                })
                .unwrap();
            output.release().unwrap();
            wrong
        } else {
            output
        };
        self.state
            .lock()
            .unwrap()
            .created
            .push(self.inner.probe(&output));
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
        let retired: usize = state.created.iter().map(|probe| probe.retirements()).sum();
        assert_eq!(state.created.len(), retired + published);
        for probe in &state.created {
            assert!(probe.retirements() <= 1);
        }
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "pol2cart owning fixture".into()
    }
    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        runmat_accelerate_api::ProviderPrecision::F64
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
        Box::pin(async move { self.inner.download(handle) })
    }
    fn unary_cos<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let fault = self.step();
            self.before(fault).await?;
            if matches!(fault, Fault::InvalidLogicalViewAlias) {
                assert!(input
                    .with_view(input.view().clone().with_logical(true))
                    .is_err());
                return Ok(input.clone());
            }
            if matches!(fault, Fault::Alias) {
                return Ok(input.clone());
            }
            let host = self.inner.download(input)?;
            let values: Vec<_> = host.data.into_iter().map(f64::cos).collect();
            Ok(self.track(
                self.inner.upload(&HostTensorView {
                    data: &values,
                    shape: &host.shape,
                })?,
                fault,
            ))
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
            let host = self.inner.download(input)?;
            let values: Vec<_> = host.data.into_iter().map(f64::sin).collect();
            Ok(self.track(
                self.inner.upload(&HostTensorView {
                    data: &values,
                    shape: &host.shape,
                })?,
                fault,
            ))
        })
    }
    fn elem_mul<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let fault = self.step();
            self.before(fault).await?;
            if matches!(fault, Fault::Alias) {
                return Ok(right.clone());
            }
            // Keep the existing native arithmetic/broadcast kernel as this fixture's
            // numerical implementation; only its publication owner is replaced.
            let reference = runmat_accelerate::simple_provider::InProcessProvider::new();
            let lhs = self.inner.download(left)?;
            let rhs = self.inner.download(right)?;
            let lhs = reference.upload(&HostTensorView {
                data: &lhs.data,
                shape: &lhs.shape,
            })?;
            let rhs = reference.upload(&HostTensorView {
                data: &rhs.data,
                shape: &rhs.shape,
            })?;
            let result = reference.elem_mul(&lhs, &rhs).await?;
            let host = reference.download(&result).await?;
            Ok(self.track(
                self.inner.upload(&HostTensorView {
                    data: &host.data,
                    shape: &host.shape,
                })?,
                fault,
            ))
        })
    }
}
fn provider(at: usize, fault: Fault) -> &'static Provider {
    let provider = Box::leak(Box::new(Provider {
        inner: test_owner::Owner::new(),
        state: Mutex::default(),
    }));
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
fn resident_inputs(provider: &Provider) -> (GpuTensorHandle, Vec<Value>) {
    let theta = provider
        .inner
        .upload(&HostTensorView {
            data: &[0.0, 0.0],
            shape: &[2, 1],
        })
        .unwrap();
    let rho = provider
        .inner
        .upload(&HostTensorView {
            data: &[2.0],
            shape: &[1, 1],
        })
        .unwrap();
    let z = provider
        .inner
        .upload(&HostTensorView {
            data: &[7.0, 8.0, 9.0],
            shape: &[1, 3],
        })
        .unwrap();
    let args = vec![
        Value::GpuTensor(theta.clone()),
        Value::GpuTensor(rho),
        Value::GpuTensor(z),
    ];
    (theta, args)
}
fn run(args: &[Value], outputs: usize) -> BuiltinResult<crate::execution::RootedValue> {
    block_on(crate::dispatcher::call_builtin_async_with_outputs(
        NAME, args, outputs,
    ))
}
fn assert_borrowed_live(provider: &Provider, borrowed: &GpuTensorHandle) {
    assert!(provider.inner.download(borrowed).is_ok());
    assert_eq!(provider.inner.probe(borrowed).retirements(), 0);
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
        run(&args, count).unwrap().with_owned(|value| {
            let Value::OutputList(values) = value else {
                panic!("outputs")
            };
            assert_eq!(values.len(), count);
            provider.assert_reclaimed(count);
            for (index, value) in values
                .try_into_owned()
                .expect("unique output list")
                .into_iter()
                .enumerate()
            {
                let Value::GpuTensor(handle) = value else {
                    panic!("resident output")
                };
                assert_eq!(handle.shape(), vec![2, 3]);
                let data = provider.inner.download(&handle).unwrap().data;
                let expected = match index {
                    0 => vec![2.0; 6],
                    1 => vec![0.0; 6],
                    _ => vec![7.0, 7.0, 8.0, 8.0, 9.0, 9.0],
                };
                assert_eq!(data, expected);
                handle.release().unwrap();
            }
        });
        provider.assert_reclaimed(0);
        assert_borrowed_live(provider, &borrowed);
    }
}

#[test]
fn malformed_and_aliased_provider_outputs_are_rejected_without_freeing_inputs() {
    let _lock = test_support::accel_test_lock();
    for (at, fault) in [
        (3, Fault::Alias),
        (5, Fault::Alias),
        (3, Fault::BadStorage),
        (3, Fault::InvalidLogicalViewAlias),
    ] {
        let provider = provider(at, fault);
        let _provider = ThreadProviderGuard::set(Some(provider));
        let (borrowed, args) = inputs(provider);
        let metadata = borrowed.wire_descriptor();
        let error = run(&args, 3).unwrap_err();
        assert_eq!(error.identifier(), ERROR_INTERNAL.identifier);
        assert_eq!(borrowed.wire_descriptor(), metadata);
        provider.assert_reclaimed(0);
        assert_borrowed_live(provider, &borrowed);
    }
}

#[test]
fn pol2cart_outputs_preserve_automatic_and_explicit_input_provenance() {
    let _lock = test_support::accel_test_lock();
    for provenance in [
        runmat_accelerate_api::GpuHandleProvenance::Automatic,
        runmat_accelerate_api::GpuHandleProvenance::Explicit,
    ] {
        let provider = provider(0, Fault::None);
        let _provider = ThreadProviderGuard::set(Some(provider));
        let (borrowed, mut args) = inputs(provider);
        let borrowed = borrowed.with_provenance(provenance).unwrap();
        args[0] = Value::GpuTensor(borrowed.clone());
        run(&args, 3).unwrap().with_owned(|value| {
            let Value::OutputList(outputs) = value else {
                panic!("outputs")
            };
            for output in outputs.try_into_owned().expect("unique output list") {
                let Value::GpuTensor(handle) = output else {
                    panic!("resident output")
                };
                assert_eq!(
                    runmat_accelerate_api::handle_provenance(&handle),
                    Some(provenance)
                );
                handle.release().unwrap();
            }
        });
        provider.assert_reclaimed(0);
        assert_borrowed_live(provider, &borrowed);
    }
}

#[test]
fn pre_effect_unsupported_hook_falls_back_late_hook_is_terminal_and_pending_drop_reclaims() {
    let _lock = test_support::accel_test_lock();
    let provider = provider(1, Fault::Unsupported);
    let _provider = ThreadProviderGuard::set(Some(provider));
    let (borrowed, args) = resident_inputs(provider);
    let output = run(&args, 3).unwrap();
    let Value::OutputList(values) = output.value() else {
        panic!("outputs")
    };
    assert!(values.iter().all(|v| matches!(v, Value::Tensor(_))));
    provider.assert_reclaimed(0);
    assert_borrowed_live(provider, &borrowed);
    *provider.state.lock().unwrap() = State {
        at: 4,
        fault: Fault::Unsupported,
        ..State::default()
    };
    let (borrowed, args) = inputs(provider);
    let error = run(&args, 3).expect_err("late unsupported cannot replay staged effects");
    assert!(error.message().contains("injected unavailable hook"));
    assert_eq!(provider.state.lock().unwrap().calls, 4);
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
    assert!(borrowed
        .with_view(borrowed.view().clone().with_shape(vec![9, 9]))
        .is_err());
    let state = provider.state.lock().unwrap();
    assert_eq!(state.calls, 0);
    assert_eq!(state.downloads, 0);
    drop(state);
    assert_borrowed_live(provider, &borrowed);
}

#[test]
fn polled_drop_retains_admitted_owner_after_provider_selection_ends() {
    let _lock = test_support::accel_test_lock();
    let provider = provider(4, Fault::Pending);
    let selected = ThreadProviderGuard::set(Some(provider));
    let (borrowed, args) = inputs(provider);
    let mut call = Box::pin(crate::dispatcher::call_builtin_async_with_outputs(
        NAME, &args, 3,
    ));
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&noop_waker())),
        Poll::Pending
    ));
    drop(selected);
    drop(call);
    provider.assert_reclaimed(0);
    assert_borrowed_live(provider, &borrowed);
}
