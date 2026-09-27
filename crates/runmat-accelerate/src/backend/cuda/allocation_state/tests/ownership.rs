use super::*;
use runmat_accelerate_api::allocation::{
    AllocationReleaseOutcome, AllocationRetirementOutcome, GpuTensorLayout,
};

#[test]
fn cloned_and_reshaped_views_share_one_physical_retirement() {
    let (state, driver) = owner();
    let input = allocate(&state, 1501);
    let cloned = input.clone();
    let reshaped = input
        .with_view(input.view().clone().with_shape(vec![2, 1]))
        .unwrap();
    assert_eq!(input.view_identity(), cloned.view_identity());
    assert_ne!(input.view_identity(), reshaped.view_identity());
    assert!(input.shares_allocation_with(&reshaped));
    let wire = input.wire_descriptor();
    let acquired = state.acquire(&wire).unwrap();
    assert!(input.shares_allocation_with(&acquired));
    assert_eq!(
        input.release().unwrap(),
        AllocationReleaseOutcome::ReferenceReleased
    );
    drop(cloned);
    drop(reshaped);
    assert!(driver.frees().is_empty());
    assert_eq!(
        acquired.release().unwrap(),
        AllocationReleaseOutcome::Retired
    );
    assert_eq!(driver.frees(), [1501]);
    assert!(state.acquire(&wire).is_err());
}

#[test]
fn descriptor_and_layout_acquisition_cannot_forge_payload_authority() {
    let (state, driver) = owner();
    let input = allocate(&state, 1502);
    let mut wire = input.wire_descriptor();
    wire.physical =
        GpuPhysicalDescriptor::new(NumericElementType::F64, GpuTensorStorage::Real, 2, 16, 16)
            .unwrap();
    assert!(state.acquire(&wire).is_err());
    let mut wire = input.wire_descriptor();
    wire.view = wire
        .view
        .with_shape(vec![2, 1])
        .with_layout(GpuTensorLayout::Transpose(
            runmat_accelerate_api::TransposeInfo {
                base_rows: 1,
                base_cols: 2,
            },
        ));
    assert!(state.acquire(&wire).is_err());
    let mut wire = input.wire_descriptor();
    wire.view = wire.view.with_logical(true);
    assert!(state.acquire(&wire).is_err());
    assert!(driver.frees().is_empty());
    assert!(state.allocation(&input).is_ok());
}

#[test]
fn producer_logical_constraint_survives_numeric_read_view() {
    let (state, driver) = owner();
    let logical = state
        .allocate(
            vec![2],
            NativeDeviceClass::Logical,
            GpuTensorStorage::Real,
            2,
            || Ok(1503),
            |_| Ok(()),
        )
        .unwrap();
    // Native uninitialized logical allocation carries ABI class intent; it is
    // not a claim that every byte has already been initialized to zero or one.
    let numeric = logical
        .with_view(logical.view().clone().with_logical(false))
        .unwrap();
    assert!(state.validate_native_write(&logical).is_ok());
    assert!(state.allocation(&numeric).is_ok());
    assert!(state.validate_native_write(&numeric).is_err());
    drop(logical);
    assert!(driver.frees().is_empty());
    drop(numeric);
    assert_eq!(driver.frees(), [1503]);
}

#[test]
fn release_failure_retains_exact_owner_and_known_disposition_without_retry() {
    let (state, driver) = owner();
    let input = allocate(&state, 1504);
    driver.fail_free.store(true, Ordering::SeqCst);
    let failure = input.release().unwrap_err();
    let physical = failure.physical.as_ref().unwrap_err();
    assert!(matches!(
        physical.outcome(),
        AllocationRetirementOutcome::Uncertain { .. }
    ));
    assert!(physical
        .retained_owner()
        .validate_live(physical.identity())
        .is_err());
    let weak = Arc::downgrade(&state);
    drop(state);
    assert!(weak.upgrade().is_some());
    drop(failure);
    assert!(weak.upgrade().is_none());
    assert_eq!(driver.frees(), [1504]);
}

#[test]
fn final_release_and_wire_acquisition_are_race_safe() {
    let (state, driver) = owner();
    let input = allocate(&state, 1505);
    let wire = input.wire_descriptor();
    let gate = Arc::new(std::sync::Barrier::new(2));
    let worker = {
        let state = state.clone();
        let gate = gate.clone();
        std::thread::spawn(move || {
            gate.wait();
            state.acquire(&wire)
        })
    };
    gate.wait();
    let _outcome = input.release().unwrap();
    if let Ok(acquired) = worker.join().unwrap() {
        assert!(state.allocation(&acquired).is_ok());
        assert!(driver.frees().is_empty());
        drop(acquired);
    }
    assert_eq!(driver.frees(), [1505]);
    assert!(state.records().is_empty());
}
