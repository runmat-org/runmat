use super::*;
use crate::builtins::common::test_support;
use crate::builtins::math::trigonometry::test_owner;
use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate_api::{
    allocation::WeakGpuTensorHandle, test_support::RetirementProbe, GpuHandleProvenance,
    GpuTensorStorage,
};
use runmat_accelerate_api::{AccelDownloadFuture, HostTensorView, ThreadProviderGuard};
use runmat_value::Value;
use std::{
    future::Future,
    sync::Mutex,
    task::{Context, Poll},
};

fn asin_diagnostic() -> gpu_helpers::RealProbeDiagnostic {
    gpu_helpers::RealProbeDiagnostic {
        builtin: runmat_builtins::ASIN_CATALOG_ENTRY.identity,
        internal_error: &runmat_builtins::ASIN_ERROR_INTERNAL,
    }
}

#[derive(Clone, Copy, Default)]
enum Fault {
    #[default]
    None,
    Failure,
    Unsupported,
    Pending,
    BorrowedAlias,
    TemporaryAlias,
    WrongStorage,
}

#[derive(Default)]
struct State {
    at: usize,
    calls: usize,
    fault: Fault,
    owned: Vec<(WeakGpuTensorHandle, RetirementProbe)>,
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
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let fault = self.step();
        // The injected Unsupported result is certified before any provider
        // work. Later download/upload errors cannot make that claim.
        self.before(fault)
            .await
            .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
        if matches!(fault, Fault::BorrowedAlias) {
            return Ok(input.clone());
        }
        if matches!(fault, Fault::TemporaryAlias) {
            return Ok(self.state.lock().unwrap().owned[0].0.acquire()?);
        }
        let host = self.inner.download(input).map_err(|error| {
            runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
        })?;
        let value = if minimum {
            host.data.into_iter().fold(f64::INFINITY, f64::min)
        } else {
            host.data.into_iter().fold(f64::NEG_INFINITY, f64::max)
        };
        // Missing metadata cannot be minted. A genuine complex allocation is
        // instead an invalid real-range result and must be retired on rejection.
        let output = if matches!(fault, Fault::WrongStorage) {
            self.inner
                .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                    data: runmat_accelerate_api::HostNumericDataView::F64(&[value, 0.0]),
                    shape: &[1, 1],
                    storage: GpuTensorStorage::ComplexInterleaved,
                })
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error)
                        .with_unproven_effects()
                })?
        } else {
            self.inner
                .upload(&HostTensorView {
                    data: &[value],
                    shape: &[1, 1],
                })
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error)
                        .with_unproven_effects()
                })?
        };
        self.state
            .lock()
            .unwrap()
            .owned
            .push((output.downgrade(), self.inner.probe(&output)));
        Ok(output)
    }

    fn assert_reclaimed(&self, input: &GpuTensorHandle) {
        let state = self.state.lock().unwrap();
        for (_, probe) in &state.owned {
            assert_eq!(probe.retirements(), 1);
        }
        assert_eq!(self.inner.probe(input).retirements(), 0);
        drop(state);
        assert!(self.inner.download(input).is_ok());
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
        "range owning fixture".into()
    }
    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        runmat_accelerate_api::ProviderPrecision::F64
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let fault = self.step();
            self.before(fault).await?;
            self.inner.download(input)
        })
    }

    fn reduce_min<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce(input, true).await })
    }
    fn reduce_max<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce(input, false).await })
    }
}

fn fixture(at: usize, fault: Fault) -> (&'static Provider, GpuTensorHandle) {
    fixture_values(at, fault, &[0.0, 0.5])
}

fn fixture_values(at: usize, fault: Fault, values: &[f64]) -> (&'static Provider, GpuTensorHandle) {
    let provider = Box::leak(Box::new(Provider {
        inner: test_owner::Owner::new(),
        state: Mutex::default(),
    }));
    *provider.state.lock().unwrap() = State {
        at,
        fault,
        ..State::default()
    };
    let input = provider
        .inner
        .upload(&HostTensorView {
            data: values,
            shape: &[1, values.len()],
        })
        .unwrap();
    (provider, input)
}

#[test]
fn inverse_range_all_nan_is_inconclusive_not_complex_evidence() {
    let _serial = test_support::accel_test_lock();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    for name in ["asin", "acos", "atanh", "acosh"] {
        let (provider, input) = fixture_values(0, Fault::None, &[f64::NAN, f64::NAN]);
        let input = input
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let _provider = ThreadProviderGuard::set(Some(provider));
        let result = block_on(crate::dispatcher::call_builtin_async_with_outputs(
            name,
            &[Value::GpuTensor(input.clone())],
            1,
        ))
        .unwrap();
        let Value::GpuTensor(output) = result.value() else {
            panic!("{name}: explicit input must retain residency");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(output));
        assert_eq!(output.descriptor().storage, Some(GpuTensorStorage::Real));
        let host = provider.inner.download(output).unwrap();
        assert!(host.data.iter().all(|value| value.is_nan()));
        drop(result);
        provider.assert_reclaimed(&input);
    }
}

#[test]
fn inverse_range_complex_policy_is_independent_of_probe_support_and_respects_provenance() {
    let _serial = test_support::accel_test_lock();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    for name in ["asin", "acos", "atanh", "acosh"] {
        let outside = if name == "acosh" { 0.5 } else { 2.0 };
        for fault in [Fault::None, Fault::Unsupported] {
            for automatic in [false, true] {
                let (provider, input) = fixture_values(1, fault, &[outside]);
                let input = input
                    .with_provenance(if automatic {
                        GpuHandleProvenance::Automatic
                    } else {
                        GpuHandleProvenance::Explicit
                    })
                    .unwrap();
                let _provider = ThreadProviderGuard::set(Some(provider));
                let result = block_on(crate::dispatcher::call_builtin_async_with_outputs(
                    name,
                    &[Value::GpuTensor(input.clone())],
                    1,
                ));
                if automatic {
                    let output = result.unwrap();
                    assert!(matches!(
                        output.value(),
                        Value::Complex(_, _) | Value::ComplexTensor(_)
                    ));
                    let expected = block_on(crate::dispatcher::call_builtin_async_with_outputs(
                        name,
                        &[Value::Num(outside)],
                        1,
                    ))
                    .unwrap();
                    assert_eq!(
                        output, expected,
                        "{name}: automatic residency changed semantics"
                    );
                } else {
                    assert!(
                        result.is_err(),
                        "{name}: explicit complex extension must be gated"
                    );
                }
                provider.assert_reclaimed(&input);
            }
        }
    }
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
fn inverse_range_rejects_aliases_and_wrong_storage_without_freeing_borrowed_input() {
    let _serial = test_support::accel_test_lock();
    for (step, fault) in [
        (1, Fault::BorrowedAlias),
        (2, Fault::TemporaryAlias),
        (1, Fault::WrongStorage),
        (2, Fault::WrongStorage),
    ] {
        let (provider, input) = fixture(step, fault);
        let _provider = ThreadProviderGuard::set(Some(provider));
        assert!(block_on(requires_complex(
            provider,
            &input,
            RealDomain::ClosedUnitInterval,
            asin_diagnostic()
        ))
        .is_err());
        assert_eq!(provider.state.lock().unwrap().calls, step);
        provider.assert_reclaimed(&input);
    }
}

#[test]
fn inverse_range_only_first_unsupported_hook_is_inconclusive() {
    let _serial = test_support::accel_test_lock();
    let (provider, input) = fixture(1, Fault::Unsupported);
    let _provider = ThreadProviderGuard::set(Some(provider));
    assert_eq!(
        block_on(requires_complex(
            provider,
            &input,
            RealDomain::ClosedUnitInterval,
            asin_diagnostic()
        ))
        .unwrap(),
        None
    );
    provider.assert_reclaimed(&input);

    // Once reduce_min has completed, an unsupported reduce_max cannot
    // authorize replay: its failure follows a provider effect.
    let (provider, input) = fixture(2, Fault::Unsupported);
    let _provider = ThreadProviderGuard::set(Some(provider));
    let error = block_on(requires_complex(
        provider,
        &input,
        RealDomain::ClosedUnitInterval,
        asin_diagnostic(),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:asin:Internal"));
    let source = error
        .source
        .as_deref()
        .and_then(|source| source.downcast_ref::<runmat_accelerate_api::ProviderExecutionError>())
        .expect("post-effect provider cause remains attached");
    assert!(source.is_after_effect());
    assert!(!source.is_unsupported());
    provider.assert_reclaimed(&input);
    let (provider, input) = fixture(0, Fault::None);
    let _provider = ThreadProviderGuard::set(Some(provider));
    assert_eq!(
        block_on(requires_complex(
            provider,
            &input,
            RealDomain::ClosedUnitInterval,
            asin_diagnostic()
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
        asin_diagnostic(),
    ));
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&noop_waker())),
        Poll::Pending
    ));
    drop(selected);
    drop(call);
    provider.assert_reclaimed(&input);
}
