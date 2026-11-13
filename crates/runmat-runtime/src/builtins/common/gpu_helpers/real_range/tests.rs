use super::*;
use crate::builtins::common::test_support;
use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate_api::{
    allocation::WeakGpuTensorHandle,
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProviderFuture, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorOwned, HostTensorOwned, HostTensorView,
    ThreadProviderGuard,
};
use std::{
    future::Future,
    pin::Pin,
    sync::Mutex,
    task::{Context, Poll},
};

fn acosh_diagnostic() -> RealProbeDiagnostic {
    RealProbeDiagnostic {
        builtin: runmat_builtins::ACOSH_CATALOG_ENTRY.identity,
        internal_error: &runmat_builtins::ACOSH_ERROR_INTERNAL,
    }
}

fn atanh_diagnostic() -> RealProbeDiagnostic {
    RealProbeDiagnostic {
        builtin: runmat_builtins::ATANH_CATALOG_ENTRY.identity,
        internal_error: &runmat_builtins::ATANH_ERROR_INTERNAL,
    }
}

#[derive(Clone, Copy)]
enum Fault {
    None,
    UnsupportedMask,
    UnsupportedReduction,
    FailedReduction,
    PendingReduction,
    PendingMinimum,
    PendingMaximum,
    BorrowedMask,
    AliasedReduction,
    WrongOwnerMask,
    FloatingLogical,
    NonlogicalMask,
    NonlogicalReduction,
}

struct Provider {
    owner: TestAllocationOwner,
    precision: ProviderPrecision,
    fault: Fault,
    owned: Mutex<Vec<(WeakGpuTensorHandle, RetirementProbe)>>,
}

impl Provider {
    fn track(&self, handle: &GpuTensorHandle, probe: RetirementProbe) {
        self.owned.lock().unwrap().push((handle.downgrade(), probe));
    }

    fn probe(&self, handle: &GpuTensorHandle) -> anyhow::Result<RetirementProbe> {
        self.owned
            .lock()
            .unwrap()
            .iter()
            .find(|(_, probe)| probe.with_data(handle, |_| ()).is_ok())
            .map(|(_, probe)| probe.clone())
            .ok_or_else(|| anyhow::anyhow!("unowned fixture handle"))
    }

    fn real(&self, data: Vec<f64>) -> GpuTensorHandle {
        let data = match self.precision {
            ProviderPrecision::F32 => {
                HostNumericDataOwned::F32(data.into_iter().map(|value| value as f32).collect())
            }
            ProviderPrecision::F64 => HostNumericDataOwned::F64(data),
        };
        let allocation = self
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                shape: vec![1, data.len()],
                data,
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        self.track(&allocation.handle, allocation.probe);
        allocation.handle
    }

    fn logical(&self, data: Vec<u8>, shape: Vec<usize>) -> GpuTensorHandle {
        if matches!(self.fault, Fault::FloatingLogical) {
            assert!(data.iter().all(|value| *value == 0));
            let allocation = self
                .owner
                .allocate_logical_zeroed(shape, self.precision)
                .unwrap();
            self.track(&allocation.handle, allocation.probe);
            return allocation.handle;
        }
        let allocation = self.owner.allocate_logical(data, shape).unwrap();
        self.track(&allocation.handle, allocation.probe);
        allocation.handle
    }

    fn assert_temporaries_reclaimed(&self, input: &GpuTensorHandle) {
        let owned = self.owned.lock().unwrap();
        for (weak, probe) in owned.iter() {
            if weak
                .acquire()
                .is_ok_and(|handle| handle.shares_allocation_with(input))
            {
                assert_eq!(probe.retirements(), 0, "input must stay live");
            } else {
                assert_eq!(probe.retirements(), 1, "probe temporary must retire");
            }
        }
        drop(owned);
        assert!(self.probe(input).is_ok());
    }
}

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "owned NaN range fixture".into()
    }

    fn precision(&self) -> ProviderPrecision {
        self.precision
    }

    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        Ok(self.real(host.data.to_vec()))
    }

    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let data = self.probe(input)?.with_data(input, |data| match data {
                HostNumericDataView::F64(values) => values.to_vec(),
                HostNumericDataView::F32(values) => values.iter().copied().map(f64::from).collect(),
                HostNumericDataView::U8(values) => values.iter().copied().map(f64::from).collect(),
                _ => panic!("fixture used unexpected physical class"),
            })?;
            Ok(HostTensorOwned {
                data,
                shape: input.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }

    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            let data = self.probe(input)?.with_data(input, |data| match data {
                HostNumericDataView::F64(values) => HostNumericDataOwned::F64(values.to_vec()),
                HostNumericDataView::F32(values) => HostNumericDataOwned::F32(values.to_vec()),
                HostNumericDataView::U8(values) => HostNumericDataOwned::U8(values.to_vec()),
                _ => panic!("fixture used unexpected physical class"),
            })?;
            Ok(HostNumericTensorOwned {
                data,
                shape: input.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }

    fn logical_isnan(&self, input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        match self.fault {
            Fault::UnsupportedMask => {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "isnan",
                ));
            }
            Fault::BorrowedMask => return Ok(input.clone()),
            Fault::NonlogicalMask => {
                return Ok(self.real(vec![0.0; input.shape().iter().product()]));
            }
            Fault::WrongOwnerMask => {
                let other = TestAllocationOwner::new();
                let count = input.shape().iter().product();
                let allocation = other.allocate_logical(vec![0; count], input.shape().to_vec())?;
                self.track(&allocation.handle, allocation.probe);
                return Ok(allocation.handle);
            }
            _ => {}
        }
        let data = self.probe(input)?.with_data(input, |data| match data {
            HostNumericDataView::F64(values) => values
                .iter()
                .map(|value| u8::from(value.is_nan()))
                .collect(),
            HostNumericDataView::F32(values) => values
                .iter()
                .map(|value| u8::from(value.is_nan()))
                .collect(),
            _ => panic!("NaN probe input must be floating"),
        })?;
        Ok(self.logical(data, input.shape().to_vec()))
    }

    fn reduce_any<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            match self.fault {
                Fault::UnsupportedReduction => {
                    return Err(runmat_accelerate_api::unsupported_provider_operation("any").into());
                }
                Fault::FailedReduction => anyhow::bail!("injected reduction failure"),
                Fault::PendingReduction => std::future::pending().await,
                Fault::AliasedReduction => return Ok(input.clone()),
                Fault::NonlogicalReduction => return Ok(self.real(vec![0.0])),
                _ => {}
            }
            let any = self.probe(input)?.with_data(input, |data| match data {
                HostNumericDataView::U8(values) => values.iter().any(|value| *value != 0),
                HostNumericDataView::F32(values) => values.iter().any(|value| *value != 0.0),
                HostNumericDataView::F64(values) => values.iter().any(|value| *value != 0.0),
                _ => panic!("reduction input must be logical"),
            })?;
            Ok(self.logical(vec![u8::from(any)], vec![1, 1]))
        })
    }

    fn reduce_min<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if matches!(self.fault, Fault::PendingMinimum) {
                std::future::pending::<()>().await;
            }
            let data = self.probe(input)?.with_data(input, |data| match data {
                HostNumericDataView::F64(values) => values.to_vec(),
                HostNumericDataView::F32(values) => values.iter().copied().map(f64::from).collect(),
                _ => panic!("extrema input must be floating"),
            })?;
            let minimum = data
                .into_iter()
                .filter(|value| !value.is_nan())
                .reduce(f64::min)
                .ok_or_else(|| anyhow::anyhow!("empty extrema input"))?;
            Ok(self.real(vec![minimum]))
        })
    }

    fn reduce_max<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if matches!(self.fault, Fault::PendingMaximum) {
                std::future::pending::<()>().await;
            }
            let data = self.probe(input)?.with_data(input, |data| match data {
                HostNumericDataView::F64(values) => values.to_vec(),
                HostNumericDataView::F32(values) => values.iter().copied().map(f64::from).collect(),
                _ => panic!("extrema input must be floating"),
            })?;
            let maximum = data
                .into_iter()
                .filter(|value| !value.is_nan())
                .reduce(f64::max)
                .ok_or_else(|| anyhow::anyhow!("empty extrema input"))?;
            Ok(self.real(vec![maximum]))
        })
    }
}

fn fixture(fault: Fault) -> (&'static Provider, GpuTensorHandle) {
    fixture_values(fault, ProviderPrecision::F64, vec![0.25, f64::NAN, 0.75])
}

fn fixture_values(
    fault: Fault,
    precision: ProviderPrecision,
    values: Vec<f64>,
) -> (&'static Provider, GpuTensorHandle) {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        precision,
        fault,
        owned: Mutex::new(Vec::new()),
    }));
    let input = provider.real(values);
    (provider, input)
}

#[test]
fn nan_absence_requires_logical_owner_proof_and_reclaims_every_temporary() {
    let _lock = test_support::accel_test_lock();
    let (provider, clean) =
        fixture_values(Fault::None, ProviderPrecision::F64, vec![0.25, 0.5, 0.75]);
    let _selected = ThreadProviderGuard::set(Some(provider));
    assert_eq!(
        block_on(probe_real_nan_absence(provider, &clean, acosh_diagnostic())).unwrap(),
        Some(true)
    );
    provider.assert_temporaries_reclaimed(&clean);
    drop(_selected);
    for (fault, expected) in [
        (Fault::None, Some(false)),
        (Fault::UnsupportedMask, None),
        (Fault::UnsupportedReduction, None),
    ] {
        let (provider, input) = fixture(fault);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let mut stages = Vec::new();
        assert_eq!(
            block_on(probe_real_nan_absence_observed(
                provider,
                &input,
                acosh_diagnostic(),
                &mut |stage| match stage {
                    RealProbeStage::Provider { operation, .. } => stages.push(operation.0),
                    RealProbeStage::ScalarReadback => stages.push("scalar_readback".into()),
                },
            ))
            .unwrap(),
            expected
        );
        let expected_stages: &[&str] = match fault {
            Fault::None => &["logical_isnan", "reduce_any", "scalar_readback"],
            Fault::UnsupportedMask => &["logical_isnan"],
            Fault::UnsupportedReduction => &["logical_isnan", "reduce_any"],
            _ => unreachable!(),
        };
        assert_eq!(
            stages
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .as_slice(),
            expected_stages
        );
        provider.assert_temporaries_reclaimed(&input);
    }
}

#[test]
fn native_precision_extrema_and_nan_proof_use_typed_result_storage() {
    let _lock = test_support::accel_test_lock();
    for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
        let (provider, input) = fixture_values(Fault::None, precision, vec![0.25, 0.5, 0.75]);
        let _selected = ThreadProviderGuard::set(Some(provider));
        assert_eq!(
            block_on(probe_real_nan_absence(provider, &input, atanh_diagnostic())).unwrap(),
            Some(true)
        );
        let mut stages = Vec::new();
        let bounds = block_on(probe_real_extrema_observed(
            provider,
            &input,
            true,
            atanh_diagnostic(),
            &mut |stage| stages.push(stage),
        ))
        .unwrap()
        .unwrap();
        assert_eq!(bounds.precision, precision);
        assert_eq!(bounds.minimum, 0.25);
        assert_eq!(bounds.maximum, Some(0.75));
        assert_eq!(
            stages.len(),
            4,
            "two reductions and two native scalar reads"
        );
        provider.assert_temporaries_reclaimed(&input);
    }
}

#[test]
fn trusted_floating_logical_probe_outputs_preserve_encoding_and_retire() {
    let _lock = test_support::accel_test_lock();
    for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
        let (provider, input) =
            fixture_values(Fault::FloatingLogical, precision, vec![0.25, 0.5, 0.75]);
        let _selected = ThreadProviderGuard::set(Some(provider));
        assert_eq!(
            block_on(probe_real_nan_absence(provider, &input, acosh_diagnostic())).unwrap(),
            Some(true)
        );
        provider.assert_temporaries_reclaimed(&input);
    }
}

#[test]
fn nan_probe_failures_and_bad_results_do_not_become_unknown() {
    let _lock = test_support::accel_test_lock();
    for fault in [
        Fault::FailedReduction,
        Fault::BorrowedMask,
        Fault::AliasedReduction,
        Fault::WrongOwnerMask,
        Fault::NonlogicalMask,
        Fault::NonlogicalReduction,
    ] {
        let (provider, input) = fixture(fault);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let error = block_on(probe_real_nan_absence(provider, &input, acosh_diagnostic()))
            .expect_err("malformed or failed proof must be a typed error");
        assert_eq!(
            error.identifier(),
            runmat_builtins::ACOSH_ERROR_INTERNAL.identifier
        );
        if matches!(fault, Fault::FailedReduction) {
            assert!(error
                .source
                .as_ref()
                .is_some_and(|source| source.to_string().contains("injected reduction failure")));
        }
        provider.assert_temporaries_reclaimed(&input);
    }
    let (provider, input) = fixture(Fault::FailedReduction);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let error = block_on(probe_real_nan_absence(provider, &input, atanh_diagnostic()))
        .expect_err("atanh owner must retain its catalog diagnostic");
    assert_eq!(
        error.identifier(),
        runmat_builtins::ATANH_ERROR_INTERNAL.identifier
    );
    assert!(error
        .source
        .as_ref()
        .is_some_and(|source| source.to_string().contains("injected reduction failure")));
    provider.assert_temporaries_reclaimed(&input);
}

#[test]
fn abandoning_nan_reduction_retires_mask_but_not_input() {
    let _lock = test_support::accel_test_lock();
    let (provider, input) = fixture(Fault::PendingReduction);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let mut probe = Box::pin(probe_real_nan_absence(provider, &input, acosh_diagnostic()));
    assert!(matches!(
        Pin::as_mut(&mut probe).poll(&mut Context::from_waker(&noop_waker())),
        Poll::Pending
    ));
    drop(probe);
    provider.assert_temporaries_reclaimed(&input);
}

#[test]
fn abandoning_either_extrema_stage_retires_completed_temporaries() {
    let _lock = test_support::accel_test_lock();
    for fault in [Fault::PendingMinimum, Fault::PendingMaximum] {
        let (provider, input) =
            fixture_values(fault, ProviderPrecision::F32, vec![0.25, 0.5, 0.75]);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let mut probe = Box::pin(probe_real_extrema(
            provider,
            &input,
            true,
            atanh_diagnostic(),
        ));
        assert!(matches!(
            Pin::as_mut(&mut probe).poll(&mut Context::from_waker(&noop_waker())),
            Poll::Pending
        ));
        drop(probe);
        provider.assert_temporaries_reclaimed(&input);
    }
}
