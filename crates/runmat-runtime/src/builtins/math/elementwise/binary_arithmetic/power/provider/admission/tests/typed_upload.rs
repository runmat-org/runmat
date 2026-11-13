use super::super::*;
use runmat_accelerate_api::test_support::TestAllocationOwner;
use runmat_value::{Tensor, Value};
use std::sync::atomic::{AtomicUsize, Ordering};

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
    let error = futures::executor::block_on(super::super::super::resident::power_gpu_host_left(
        None,
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
    let error = futures::executor::block_on(super::super::super::resident::power_gpu_host_left(
        None,
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
