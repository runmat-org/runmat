use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, HostTensorView, ThreadProviderGuard,
};

struct Provider(TestAllocationOwner);
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
    fn device_info(&self) -> String {
        "extrema output admission fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("no upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("no download during descriptor admission") })
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn output_admission_checks_resolved_class_storage_owner_shape_and_live_lease() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let foreign = TestAllocationOwner::new();
    let input = provider
        .0
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let contract = pair_contract(&input.handle, &input.handle, vec![2, 1]).unwrap();
    for fault in ["class", "storage", "owner", "shape", "logical", "released"] {
        let output = match fault {
            "class" => provider
                .0
                .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap(),
            "storage" => provider
                .0
                .allocate_zeroed(
                    vec![2, 1],
                    NumericElementType::F32,
                    GpuTensorStorage::ComplexInterleaved,
                )
                .unwrap(),
            "owner" => foreign
                .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
                .unwrap(),
            "shape" => provider
                .0
                .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
                .unwrap(),
            "logical" => provider
                .0
                .allocate_logical_zeroed(vec![2, 1], runmat_accelerate_api::ProviderPrecision::F32)
                .unwrap(),
            _ => provider
                .0
                .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
                .unwrap(),
        };
        // A normal release of a clone cannot invalidate another owning alias;
        // it must still be admitted. The production lease, not a forged stale
        // descriptor, decides liveness.
        if fault == "released" {
            output.handle.clone().release().unwrap();
            let output_handle = ExtremaKind::Min
                .admit_output(
                    provider,
                    &input.handle,
                    &input.handle,
                    &contract,
                    output.handle,
                )
                .unwrap();
            assert_eq!(output.probe.retirements(), 0);
            drop(output_handle);
        } else {
            let error = ExtremaKind::Min
                .admit_output(
                    provider,
                    &input.handle,
                    &input.handle,
                    &contract,
                    output.handle,
                )
                .unwrap_err();
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        }
        assert_eq!(output.probe.retirements(), 1, "{fault}");
        assert_eq!(input.probe.retirements(), 0);
    }
    input.handle.allocation_lease().validate_live().unwrap();
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pair_class_resolution_matches_native_floating_and_integer_rules() {
    let owner = TestAllocationOwner::new();
    for (left, right, right_shape, expected) in [
        (
            NumericElementType::F32,
            NumericElementType::F32,
            vec![2, 1],
            Some(NumericElementType::F32),
        ),
        (
            NumericElementType::F32,
            NumericElementType::F64,
            vec![2, 1],
            Some(NumericElementType::F64),
        ),
        (
            NumericElementType::U64,
            NumericElementType::U64,
            vec![2, 1],
            Some(NumericElementType::U64),
        ),
        (
            NumericElementType::U64,
            NumericElementType::F64,
            vec![1, 1],
            Some(NumericElementType::U64),
        ),
        (
            NumericElementType::U64,
            NumericElementType::F64,
            vec![2, 1],
            None,
        ),
        (
            NumericElementType::U64,
            NumericElementType::F32,
            vec![1, 1],
            None,
        ),
    ] {
        let left = owner
            .allocate_zeroed(vec![2, 1], left, GpuTensorStorage::Real)
            .unwrap();
        let right = owner
            .allocate_zeroed(right_shape, right, GpuTensorStorage::Real)
            .unwrap();
        let contract = pair_contract(&left.handle, &right.handle, vec![2, 1]);
        match (contract, expected) {
            (Some(contract), Some(expected)) => {
                assert_eq!(contract.precision, expected.precision());
                assert_eq!(contract.integer, expected.integer_type());
                assert!(!contract.logical);
            }
            (None, None) => (),
            other => panic!("unexpected class admission {other:?}"),
        }
        drop((left.handle, right.handle));
        assert_eq!(left.probe.retirements(), 1);
        assert_eq!(right.probe.retirements(), 1);
    }
}
