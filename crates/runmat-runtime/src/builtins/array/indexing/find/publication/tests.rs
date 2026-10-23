use super::*;
use runmat_accelerate_api::{
    allocation::GpuHandleReleaseError,
    test_support::{RetirementProbe, TestAllocation, TestAllocationOwner},
    GpuTensorStorage, NumericElementType,
};

fn fixture() -> (ProviderFindResult, [RetirementProbe; 4]) {
    let owner = TestAllocationOwner::new();
    let [linear, rows, cols, values]: [TestAllocation; 4] = std::array::from_fn(|_| {
        owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap()
    });
    let probes = [linear.probe, rows.probe, cols.probe, values.probe];
    (
        ProviderFindResult {
            linear: linear.handle,
            rows: rows.handle,
            cols: cols.handle,
            values: Some(values.handle),
        },
        probes,
    )
}
fn kept(count: Option<usize>) -> [bool; 4] {
    match count {
        None | Some(1) => [true, false, false, false],
        Some(0) => [false; 4],
        Some(2) => [false, true, true, false],
        _ => [false, true, true, true],
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn publication_transfers_only_selected_originals_without_invalidating_outputs() {
    for count in [None, Some(0), Some(1), Some(2), Some(3)] {
        let (result, probes) = fixture();
        let output = FindEval::from_gpu(result).into_publication(count).unwrap();
        for (probe, keep) in probes.iter().zip(kept(count)) {
            assert_eq!(probe.retirements(), usize::from(!keep));
        }
        let check = |value: &Value| {
            let Value::GpuTensor(handle) = value else {
                panic!("selected provider handle")
            };
            handle.allocation_lease().validate_live().unwrap();
            assert_eq!(handle.physical().element_type(), NumericElementType::F64);
        };
        match &output {
            Value::OutputList(values) => values.as_slice().iter().for_each(check),
            value => check(value),
        }
        drop(output);
        assert!(probes.iter().all(|probe| probe.retirements() == 1));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn discarded_failure_is_terminal_and_settles_every_unpublished_role_once() {
    for count in [None, Some(0), Some(1), Some(2), Some(3)] {
        let (result, probes) = fixture();
        let discarded = if count.is_none() || count == Some(1) {
            1
        } else {
            0
        };
        probes[discarded].fail_retirement("original discarded find result");
        let secondary = if discarded == 0 { 1 } else { 0 };
        probes[secondary].fail_retirement("secondary find cleanup");
        let error = FindEval::from_gpu(result)
            .into_publication(count)
            .expect_err("retirement cannot publish success");
        assert_eq!(error.identifier(), FIND_ERROR_PROVIDER_OUTPUT.identifier);
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        let mut cause = std::error::Error::source(&error);
        let mut original = false;
        while let Some(source) = cause {
            original |= source
                .downcast_ref::<GpuHandleReleaseError>()
                .is_some_and(|source| {
                    source
                        .to_string()
                        .contains("original discarded find result")
                });
            cause = source.source();
        }
        assert!(original, "original typed cause survives secondary cleanup");
        assert!(probes.iter().all(|probe| probe.retirements() == 1));
    }
}
