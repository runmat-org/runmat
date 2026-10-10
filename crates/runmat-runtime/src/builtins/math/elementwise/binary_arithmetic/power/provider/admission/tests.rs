use super::*;
use runmat_accelerate_api::test_support::TestAllocationOwner;
use runmat_value::{Tensor, Value};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Recorder {
    owner: TestAllocationOwner,
    mode: AtomicUsize,
    queries: AtomicUsize,
    probes: AtomicUsize,
}

impl AccelProvider for Recorder {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "power-admission-recorder".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        unreachable!()
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        unreachable!()
    }
    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        let mut snapshot = ProviderCapabilitySnapshot::conservative(self);
        if self.mode.load(Ordering::SeqCst) != 3 {
            snapshot.operations.push(ProviderCapabilityOperation {
                identity: ProviderOperationIdentity::new(
                    runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
                ),
                family: ProviderOperationFamily::Elementwise,
            });
        }
        snapshot
    }
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        self.queries.fetch_add(1, Ordering::SeqCst);
        assert_eq!(query.inputs[0].element_type, ProviderElementType::F32);
        assert_eq!(query.inputs[0].layout, ProviderLayout::Strided);
        assert_eq!(query.inputs[0].shape, vec![3, 2]);
        let supports = match self.mode.load(Ordering::SeqCst) {
            1 => {
                query.workload.numerical_domain
                    == NumericalDomainEvidence::RealPower(PowerResultDomain::Real)
            }
            2 => {
                query.workload.numerical_domain
                    == NumericalDomainEvidence::RealPower(PowerResultDomain::Complex)
            }
            _ => false,
        };
        if supports {
            ProviderFeasibility::supported(ProviderResourceEstimate::default())
        } else {
            ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedElementType,
                "test.execution.unsupported",
            )
        }
    }
    fn classify_real_power<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, PowerResultDomain> {
        self.probes.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("probe was reached") })
    }
}

#[test]
fn standalone_admission_skips_only_explicitly_rejected_outcomes() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let owner = Box::leak(Box::new(Recorder {
        owner: TestAllocationOwner::new(),
        mode: AtomicUsize::new(0),
        queries: AtomicUsize::new(0),
        probes: AtomicUsize::new(0),
    }));
    let _provider = ThreadProviderGuard::set(Some(owner));
    for (mode, queries, probes) in [(0, 2, 0), (1, 1, 1), (2, 2, 1), (3, 0, 1)] {
        owner.mode.store(mode, Ordering::SeqCst);
        owner.queries.store(0, Ordering::SeqCst);
        owner.probes.store(0, Ordering::SeqCst);
        let left_allocation = owner
            .owner
            .allocate_zeroed(vec![2, 3], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let right_allocation = owner
            .owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let left = left_allocation
            .handle
            .with_view(GpuTensorView::contiguous(vec![3, 2]).with_layout(
                GpuTensorLayout::Transpose(TransposeInfo {
                    base_rows: 2,
                    base_cols: 3,
                }),
            ))
            .expect("admitted transpose view");
        let right = right_allocation.handle;
        let before = left.wire_descriptor();
        let result =
            futures::executor::block_on(super::super::pair::try_real_pair(owner, &left, &right));
        if mode == 0 {
            assert!(result.unwrap().is_none());
        } else {
            assert!(result.unwrap_err().message().contains("probe was reached"));
        }
        assert_eq!(owner.queries.load(Ordering::SeqCst), queries);
        assert_eq!(owner.probes.load(Ordering::SeqCst), probes);
        assert_eq!(left.wire_descriptor(), before);
        assert_eq!(left_allocation.probe.retirements(), 0);
        assert_eq!(right_allocation.probe.retirements(), 0);
        drop((left, left_allocation.handle, right));
        assert_eq!(left_allocation.probe.retirements(), 1);
        assert_eq!(right_allocation.probe.retirements(), 1);
    }
}

struct UnsupportedAfterScalarUpload {
    owner: TestAllocationOwner,
    uploads: AtomicUsize,
    downloads: AtomicUsize,
    probes: AtomicUsize,
    widenings: AtomicUsize,
    expected_mode: RealPowerArithmeticMode,
}

impl AccelProvider for UnsupportedAfterScalarUpload {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    fn device_info(&self) -> String {
        "native-single-unsupported-after-upload".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        unreachable!("typed scalar upload must use the numeric hook")
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let HostNumericDataView::F32(values) = host.data else {
            return Err(anyhow::anyhow!("Single scalar upload expected").into());
        };
        self.uploads.fetch_add(1, Ordering::SeqCst);
        Ok(self
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F32(values.to_vec()),
                shape: host.shape.to_vec(),
                storage: host.storage,
            })?
            .handle)
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("host replay is forbidden after scalar upload") })
    }

    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        let mut snapshot = ProviderCapabilitySnapshot::conservative(self);
        snapshot.operations.push(ProviderCapabilityOperation {
            identity: ProviderOperationIdentity::new(
                runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
            ),
            family: ProviderOperationFamily::Elementwise,
        });
        snapshot
    }

    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        assert_eq!(query.workload.real_power_mode, Some(self.expected_mode));
        assert_eq!(query.inputs[0].element_type, ProviderElementType::F32);
        assert_eq!(
            query.inputs[1].element_type,
            if self.expected_mode == RealPowerArithmeticMode::NativeSingle {
                ProviderElementType::F32
            } else {
                ProviderElementType::F64
            }
        );
        ProviderFeasibility::supported(ProviderResourceEstimate::default())
    }

    fn classify_real_power<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, PowerResultDomain> {
        self.probes.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Err(unsupported_provider_operation(
                "selected NativeSingle probe became unsupported",
            ))
        })
    }

    fn unary_double<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.widenings.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Err(anyhow::anyhow!("mixed Double widening was selected").into()) })
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn native_single_unsupported_after_typed_upload_never_replays_on_host() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(UnsupportedAfterScalarUpload {
        owner: TestAllocationOwner::new(),
        uploads: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        probes: AtomicUsize::new(0),
        widenings: AtomicUsize::new(0),
        expected_mode: RealPowerArithmeticMode::NativeSingle,
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let error = futures::executor::block_on(super::super::resident::power_gpu_host_left(
        resident.handle.clone(),
        Value::Tensor(Tensor::from_f32(vec![1.8], vec![1, 1]).unwrap()),
    ))
    .expect_err("admitted upload must not permit host replay on later unsupported probe");
    assert!(error
        .message()
        .contains("selected NativeSingle probe became unsupported"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 1);
    assert_eq!(provider.widenings.load(Ordering::SeqCst), 0);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn exact_double_two_literal_keeps_mixed_route_from_runtime_value_parser() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(UnsupportedAfterScalarUpload {
        owner: TestAllocationOwner::new(),
        uploads: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        probes: AtomicUsize::new(0),
        widenings: AtomicUsize::new(0),
        expected_mode: RealPowerArithmeticMode::MixedDoubleThenSingle {
            single_operand: RealPowerOperand::Base,
        },
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let error = futures::executor::block_on(super::super::resident::power_gpu_host_left(
        resident.handle.clone(),
        Value::Num(2.0),
    ))
    .err()
    .unwrap();
    assert!(error
        .message()
        .contains("mixed Double widening was selected"));
    assert_eq!(provider.widenings.load(Ordering::SeqCst), 1);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.probes.load(Ordering::SeqCst), 0);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

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
    fn fill_like(&self, prototype: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.fills.fetch_add(1, Ordering::SeqCst);
        Ok(self
            .owner
            .allocate_zeroed(
                prototype.shape().to_vec(),
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?
            .handle)
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
    let result = futures::executor::block_on(super::super::resident::power_gpu_host_left(
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
    let error = futures::executor::block_on(super::super::resident::power_gpu_host_left(
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
    let error = futures::executor::block_on(super::super::resident::power_gpu_host_left(
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
