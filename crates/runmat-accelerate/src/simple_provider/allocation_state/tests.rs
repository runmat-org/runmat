use super::*;
use runmat_accelerate_api::allocation::{AllocationReleaseOutcome, GpuTensorLayout, GpuTensorView};
use runmat_accelerate_api::{GpuTensorStorage, HostIntegerDataOwned, NumericElementType};

fn identity(id: u64) -> AllocationIdentity {
    AllocationIdentity::new(runmat_accelerate_api::next_device_id(), id, 1)
}

#[test]
fn publication_moves_payload_without_copy_and_last_view_retires_same_record() {
    let owner = NumericOwner::new(NumericAllocationState::default());
    let mut data = Vec::with_capacity(16);
    data.extend([-0.0f64, f64::MAX]);
    let pointer = data.as_ptr();
    let requested = (data.capacity() * std::mem::size_of::<f64>()) as u64;
    let pending = owner
        .stage(
            identity(1),
            InProcessNumericData::F64(data),
            vec![1, 2],
            GpuTensorStorage::Real,
            false,
        )
        .unwrap();
    let input = pending.publish().unwrap();
    assert_eq!(input.physical().payload_bytes(), 16);
    assert_eq!(input.physical().requested_bytes(), requested);
    {
        let guard = owner.lock().unwrap();
        let Some(InProcessNumericData::F64(data)) = guard.get(&input) else {
            panic!("typed F64 payload")
        };
        assert_eq!(data.as_ptr(), pointer);
        assert_eq!(data[0].to_bits(), (-0.0f64).to_bits());
    }
    let alias = input
        .with_view(GpuTensorView::contiguous(vec![2, 1]))
        .unwrap();
    let clone = input.clone();
    assert_eq!(input.view_identity(), clone.view_identity());
    assert_ne!(input.view_identity(), alias.view_identity());
    assert!(input.shares_allocation_with(&alias));
    let wire = input.wire_descriptor();
    let acquired = owner.acquire(&wire).unwrap();
    drop(input);
    drop(clone);
    drop(alias);
    assert_eq!(owner.allocations.lock().unwrap().len(), 1);
    assert_eq!(
        acquired.release().unwrap(),
        AllocationReleaseOutcome::Retired
    );
    assert!(owner.allocations.lock().unwrap().is_empty());
    assert!(owner.acquire(&wire).is_err());
}

#[test]
fn pending_drop_duplicate_and_invalid_shape_never_publish_or_replace_payloads() {
    let owner = NumericOwner::new(NumericAllocationState::default());
    let id = identity(2);
    let pending = owner
        .stage(
            id,
            InProcessNumericData::F32(vec![3.0]),
            vec![1],
            GpuTensorStorage::Real,
            false,
        )
        .unwrap();
    assert!(owner
        .stage(
            id,
            InProcessNumericData::F32(vec![8.0]),
            vec![1],
            GpuTensorStorage::Real,
            false
        )
        .is_err());
    assert!(owner
        .stage(
            identity(3),
            InProcessNumericData::F32(vec![1.0]),
            vec![2],
            GpuTensorStorage::Real,
            false
        )
        .is_err());
    assert_eq!(owner.allocations.lock().unwrap().len(), 1);
    drop(pending);
    assert!(owner.allocations.lock().unwrap().is_empty());
}

#[test]
fn exact_integer_and_empty_shape_admission_preserve_storage() {
    let owner = NumericOwner::new(NumericAllocationState::default());
    let handle = owner
        .stage(
            identity(4),
            InProcessNumericData::Integer(HostIntegerDataOwned::U64(vec![
                9_007_199_254_740_993,
                u64::MAX,
            ])),
            vec![1, 2],
            GpuTensorStorage::Real,
            false,
        )
        .unwrap()
        .publish()
        .unwrap();
    assert_eq!(handle.physical().element_type(), NumericElementType::U64);
    let guard = owner.lock().unwrap();
    let Some(InProcessNumericData::Integer(HostIntegerDataOwned::U64(data))) = guard.get(&handle)
    else {
        panic!("exact storage")
    };
    assert_eq!(data, &[9_007_199_254_740_993, u64::MAX]);
    drop(guard);
    let empty = owner
        .stage(
            identity(5),
            InProcessNumericData::F64(Vec::new()),
            vec![usize::MAX, usize::MAX, 0],
            GpuTensorStorage::Real,
            false,
        )
        .unwrap()
        .publish()
        .unwrap();
    assert_eq!(empty.physical().payload_bytes(), 0);
    assert_eq!(empty.shape(), &[usize::MAX, usize::MAX, 0]);
}

#[test]
fn wire_cannot_relabel_numeric_as_logical_or_invent_lazy_layout() {
    let owner = NumericOwner::new(NumericAllocationState::default());
    let handle = owner
        .stage(
            identity(6),
            InProcessNumericData::F32(vec![0.0, 1.0]),
            vec![1, 2],
            GpuTensorStorage::Real,
            false,
        )
        .unwrap()
        .publish()
        .unwrap();
    let mut wire = handle.wire_descriptor();
    wire.view = wire.view.with_logical(true);
    assert!(owner.acquire(&wire).is_err());
    let mut wire = handle.wire_descriptor();
    wire.view = wire.view.with_shape(vec![1]);
    assert!(
        owner.acquire(&wire).is_err(),
        "prefix views need a real payload-range kernel contract"
    );
    let mut wire = handle.wire_descriptor();
    wire.view = wire
        .view
        .with_shape(vec![2, 1])
        .with_layout(GpuTensorLayout::Transpose(
            runmat_accelerate_api::TransposeInfo {
                base_rows: 1,
                base_cols: 2,
            },
        ));
    assert!(owner.acquire(&wire).is_err());
}

#[cfg(panic = "unwind")]
#[test]
fn pending_unwind_cleans_up_and_poisoned_live_access_fails_closed() {
    let owner = NumericOwner::new(NumericAllocationState::default());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _pending = owner
            .stage(
                identity(7),
                InProcessNumericData::F64(vec![1.0]),
                vec![1],
                GpuTensorStorage::Real,
                false,
            )
            .unwrap();
        panic!("abandon pending publication");
    }));
    assert!(result.is_err());
    assert!(owner.allocations.lock().unwrap().is_empty());
    let handle = owner
        .stage(
            identity(8),
            InProcessNumericData::F64(vec![2.0]),
            vec![1],
            GpuTensorStorage::Real,
            false,
        )
        .unwrap()
        .publish()
        .unwrap();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = owner.allocations.lock().unwrap();
        panic!("poison owner");
    }));
    {
        let guard = owner.lock().unwrap_or_else(|error| error.into_inner());
        assert!(guard.get(&handle).is_none());
    }
    assert!(owner.acquire(&handle.wire_descriptor()).is_err());
    assert_eq!(handle.release().unwrap(), AllocationReleaseOutcome::Retired);
    assert!(owner
        .allocations
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .is_empty());
}
