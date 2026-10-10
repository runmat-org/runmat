use super::*;
use crate::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, AccelProviderFuture,
    GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorOwned, HostNumericTensorView, HostTensorView, NumericElementType,
    NumericalDomainEvidence, PowerResultDomain, PowerScalarControl, ProviderCapabilityOperation,
    ProviderCapabilitySnapshot, ProviderExecutionFuture, ProviderFeasibility,
    ProviderFeasibilityQuery, ProviderOperationFamily, ProviderOperationIdentity,
    ProviderPrecision, ProviderRejectionCode, ProviderResourceEstimate, RealPowerArithmeticMode,
    RealPowerOperand,
};
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

struct Recorder {
    owner: TestAllocationOwner,
    uploads: AtomicUsize,
    probes: AtomicUsize,
    last_bits: AtomicU32,
    last_double_bits: AtomicU64,
    widenings: AtomicUsize,
    fills: AtomicUsize,
    publications: AtomicUsize,
    uploaded_probe: Mutex<Option<crate::test_support::RetirementProbe>>,
    widened: Mutex<Option<crate::allocation::WeakGpuTensorHandle>>,
    widened_probe: Mutex<Option<crate::test_support::RetirementProbe>>,
    alias_widened: AtomicBool,
    wrong_power_shape: AtomicBool,
    real_only: AtomicBool,
    complex_domain: AtomicBool,
    power_calls: AtomicUsize,
    fail_probe: bool,
    suspend_probe: bool,
    admit: AtomicBool,
    advertise_power: AtomicBool,
    expected_mode: RealPowerArithmeticMode,
}

impl AccelProvider for Recorder {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "power-service-recorder".into()
    }
    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        let mut snapshot = ProviderCapabilitySnapshot::conservative(self);
        if self.advertise_power.load(Ordering::SeqCst) {
            snapshot.operations.push(ProviderCapabilityOperation {
                identity: ProviderOperationIdentity::new("test.power"),
                family: ProviderOperationFamily::Elementwise,
            });
        }
        snapshot
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("untyped upload forbidden")
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> crate::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        let data = match host.data {
            HostNumericDataView::F32(values) => {
                self.last_bits.store(values[0].to_bits(), Ordering::SeqCst);
                HostNumericDataOwned::F32(values.to_vec())
            }
            HostNumericDataView::F64(values) => {
                self.last_double_bits
                    .store(values[0].to_bits(), Ordering::SeqCst);
                if self.alias_widened.load(Ordering::SeqCst) {
                    return Ok(self
                        .widened
                        .lock()
                        .unwrap()
                        .as_ref()
                        .expect("widened scalar source")
                        .acquire()?);
                }
                HostNumericDataOwned::F64(values.to_vec())
            }
            _ => return Err(anyhow::anyhow!("expected floating POWER scalar").into()),
        };
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data,
            shape: host.shape.to_vec(),
            storage: GpuTensorStorage::Real,
        })?;
        *self.uploaded_probe.lock().unwrap() = Some(allocation.probe);
        Ok(allocation.handle)
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("host replay forbidden") })
    }
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        assert_eq!(
            query.operation,
            ProviderOperationIdentity::new("test.power")
        );
        assert_eq!(query.workload.real_power_mode, Some(self.expected_mode));
        if self.admit.load(Ordering::SeqCst)
            && !(self.real_only.load(Ordering::SeqCst)
                && query.workload.numerical_domain
                    == NumericalDomainEvidence::RealPower(PowerResultDomain::Complex))
        {
            ProviderFeasibility::supported(ProviderResourceEstimate::default())
        } else {
            ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedElementType,
                "test.power.unsupported",
            )
        }
    }
    fn classify_real_power<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, PowerResultDomain> {
        self.probes.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if self.suspend_probe {
                std::future::pending::<()>().await;
            }
            if self.fail_probe {
                anyhow::bail!("post-upload probe failure")
            }
            Ok(if self.complex_domain.load(Ordering::SeqCst) {
                PowerResultDomain::Complex
            } else {
                PowerResultDomain::Real
            })
        })
    }
    fn real_power_in_domain<'a>(
        &'a self,
        base: &'a GpuTensorHandle,
        exponent: &'a GpuTensorHandle,
        _: PowerResultDomain,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.power_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let shape = if self.wrong_power_shape.load(Ordering::SeqCst) {
                base.shape().to_vec()
            } else {
                runmat_types::geometry::broadcast_concrete_shape(base.shape(), exponent.shape())?
            };
            let element = match crate::handle_precision(base) {
                Some(ProviderPrecision::F32) => NumericElementType::F32,
                Some(ProviderPrecision::F64) => NumericElementType::F64,
                None => anyhow::bail!("invalid base precision"),
            };
            Ok(self
                .owner
                .allocate_zeroed(shape, element, GpuTensorStorage::Real)?
                .handle)
        })
    }
    fn unary_double<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.widenings.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let allocation = self.owner.allocate_zeroed(
                input.shape().to_vec(),
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?;
            *self.widened.lock().unwrap() = Some(allocation.handle.downgrade());
            *self.widened_probe.lock().unwrap() = Some(allocation.probe);
            Ok(allocation.handle)
        })
    }
    fn fill_like(&self, input: &GpuTensorHandle, value: f64) -> anyhow::Result<GpuTensorHandle> {
        self.fills.fetch_add(1, Ordering::SeqCst);
        self.last_double_bits
            .store(value.to_bits(), Ordering::SeqCst);
        Ok(self
            .owner
            .allocate_zeroed(
                input.shape().to_vec(),
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?
            .handle)
    }
    fn unary_single<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.publications.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            Ok(self
                .owner
                .allocate_zeroed(
                    input.shape().to_vec(),
                    NumericElementType::F32,
                    GpuTensorStorage::Real,
                )
                .map_err(crate::ProviderExecutionError::from)?
                .handle)
        })
    }
}

fn provider(fail_probe: bool) -> &'static Recorder {
    provider_with_admission(fail_probe, true)
}

fn provider_with_admission(fail_probe: bool, admit: bool) -> &'static Recorder {
    provider_with_mode(fail_probe, admit, RealPowerArithmeticMode::NativeSingle)
}

fn provider_with_mode(
    fail_probe: bool,
    admit: bool,
    expected_mode: RealPowerArithmeticMode,
) -> &'static Recorder {
    provider_with_suspension(fail_probe, admit, expected_mode, false)
}

fn provider_with_suspension(
    fail_probe: bool,
    admit: bool,
    expected_mode: RealPowerArithmeticMode,
    suspend_probe: bool,
) -> &'static Recorder {
    let provider = Box::leak(Box::new(Recorder {
        owner: TestAllocationOwner::new(),
        uploads: AtomicUsize::new(0),
        probes: AtomicUsize::new(0),
        last_bits: AtomicU32::new(0),
        last_double_bits: AtomicU64::new(0),
        widenings: AtomicUsize::new(0),
        fills: AtomicUsize::new(0),
        publications: AtomicUsize::new(0),
        uploaded_probe: Mutex::new(None),
        widened: Mutex::new(None),
        widened_probe: Mutex::new(None),
        alias_widened: AtomicBool::new(false),
        wrong_power_shape: AtomicBool::new(false),
        real_only: AtomicBool::new(false),
        complex_domain: AtomicBool::new(false),
        power_calls: AtomicUsize::new(0),
        fail_probe,
        suspend_probe,
        admit: AtomicBool::new(admit),
        advertise_power: AtomicBool::new(true),
        expected_mode,
    }));
    crate::provider_registry::register(provider.device_id(), provider);
    provider
}

fn ready<T>(future: impl std::future::Future<Output = T>) -> T {
    struct Noop;
    impl std::task::Wake for Noop {
        fn wake(self: std::sync::Arc<Self>) {}
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(Noop));
    let mut future = Box::pin(future);
    match future
        .as_mut()
        .poll(&mut std::task::Context::from_waker(&waker))
    {
        std::task::Poll::Ready(value) => value,
        std::task::Poll::Pending => panic!("deterministic recorder unexpectedly suspended"),
    }
}

mod admission;
mod lifecycle;
mod mixed;
mod native;
mod symbolic;
