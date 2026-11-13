use super::super::*;
use runmat_accelerate_api::test_support::TestAllocationOwner;
use runmat_value::Value;
use std::sync::atomic::{AtomicUsize, Ordering};

struct LegacyRealOnly {
    owner: TestAllocationOwner,
    fills: AtomicUsize,
    powers: AtomicUsize,
    probes: AtomicUsize,
    downloads: AtomicUsize,
    power_fault: usize,
}

impl AccelProvider for LegacyRealOnly {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "legacy-real-only-power".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("upload not used")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("host fallback reached") })
    }
    fn fill_like(
        &self,
        prototype: &GpuTensorHandle,
        _: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<GpuTensorHandle> = (|| {
            self.fills.fetch_add(1, Ordering::SeqCst);
            Ok(self
                .owner
                .allocate_zeroed(
                    prototype.shape().to_vec(),
                    NumericElementType::F64,
                    GpuTensorStorage::Real,
                )?
                .handle)
        })();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
    fn elem_pow<'a>(
        &'a self,
        base: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.powers.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            match self.power_fault {
                1 => Err(
                    unsupported_provider_operation("legacy power unsupported after fill").into(),
                ),
                2 => Ok(base.clone()),
                _ => Ok(self
                    .owner
                    .allocate_zeroed(
                        base.shape().to_vec(),
                        NumericElementType::F64,
                        GpuTensorStorage::Real,
                    )?
                    .handle),
            }
        })
    }
    fn classify_real_power<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, PowerResultDomain> {
        self.probes.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("legacy provider cannot classify") })
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn unadvertised_legacy_integral_power_stays_resident_without_typed_probe() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(LegacyRealOnly {
        owner: TestAllocationOwner::new(),
        fills: AtomicUsize::new(0),
        powers: AtomicUsize::new(0),
        probes: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        power_fault: 0,
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let result = futures::executor::block_on(super::super::super::resident::power_gpu_host_left(
        None,
        resident.handle.clone(),
        Value::Num(2.0),
    ))
    .unwrap();
    assert!(matches!(result, Value::GpuTensor(_)));
    assert_eq!(provider.fills.load(Ordering::SeqCst), 1);
    assert_eq!(provider.powers.load(Ordering::SeqCst), 1);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    let unproven = runmat_accelerate_api::power_service::prepare_resident_scalar_power(
        provider,
        &resident.handle,
        runmat_accelerate_api::PowerScalarControl::DoubleBits(0.0f64.to_bits()),
        RealPowerOperand::Base,
        &[1, 1],
        ProviderOperationIdentity::new(runmat_builtins::POWER_CATALOG_ENTRY.identity.name),
    )
    .unwrap();
    assert!(unproven.is_none());
    assert_eq!(provider.fills.load(Ordering::SeqCst), 1);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn legacy_after_fill_unsupported_is_terminal_without_host_replay() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(LegacyRealOnly {
        owner: TestAllocationOwner::new(),
        fills: AtomicUsize::new(0),
        powers: AtomicUsize::new(0),
        probes: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        power_fault: 1,
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = futures::executor::block_on(super::super::super::resident::power_gpu_host_left(
        None,
        resident.handle.clone(),
        Value::Num(2.0),
    ))
    .err()
    .unwrap();
    assert!(error
        .message()
        .contains("legacy power unsupported after fill"));
    assert_eq!(provider.fills.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn legacy_malformed_alias_does_not_retire_borrowed_resident() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(LegacyRealOnly {
        owner: TestAllocationOwner::new(),
        fills: AtomicUsize::new(0),
        powers: AtomicUsize::new(0),
        probes: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        power_fault: 2,
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = futures::executor::block_on(super::super::super::resident::power_gpu_host_left(
        None,
        resident.handle.clone(),
        Value::Num(2.0),
    ))
    .err()
    .unwrap();
    assert!(error.message().contains("invalid legacy POWER result"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
    assert!(resident.handle.allocation_lease().validate_live().is_ok());
}
