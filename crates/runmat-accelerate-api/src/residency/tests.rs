use super::*;
use crate::*;
use std::collections::HashMap;
use std::sync::{atomic::Ordering, LazyLock, Mutex};

static RECORDS: LazyLock<Mutex<HashMap<GpuHandleIdentity, (usize, bool)>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
fn cleanup(identity: GpuHandleIdentity) -> Result<(), ResidencyCleanupError> {
    let mut records = RECORDS.lock().unwrap();
    let Some((count, fail)) = records.get_mut(&identity) else {
        return Ok(());
    };
    *count += 1;
    if *fail {
        Err(ResidencyCleanupError::Unavailable)
    } else {
        Ok(())
    }
}
fn track(handle: &GpuTensorHandle, fail: bool) {
    register_residency_clear(cleanup);
    RECORDS
        .lock()
        .unwrap()
        .insert(handle.view_identity(), (0, fail));
}
fn calls(identity: GpuHandleIdentity) -> usize {
    RECORDS.lock().unwrap()[&identity].0
}

#[test]
fn each_last_view_cleans_residency_before_last_allocation_retires() {
    let (first, owner) = crate::test_support::owned(
        17,
        vec![1, 2],
        NumericElementType::F64,
        GpuTensorStorage::Real,
        false,
    );
    let same = first.clone();
    let second = first
        .with_view(first.view().clone().with_shape(vec![2, 1]))
        .unwrap();
    track(&first, false);
    track(&second, false);
    let first_id = first.view_identity();
    let second_id = second.view_identity();
    drop(first);
    assert_eq!(calls(first_id), 0);
    drop(same);
    assert_eq!(calls(first_id), 1);
    assert_eq!(calls(second_id), 0);
    assert_eq!(owner.retired.load(Ordering::SeqCst), 0);
    drop(second);
    assert_eq!(calls(second_id), 1);
    assert_eq!(owner.retired.load(Ordering::SeqCst), 1);
}

#[test]
fn explicit_residency_failure_still_releases_physical_once_without_drop_retry() {
    let (handle, owner) = crate::test_support::owned(
        17,
        vec![1],
        NumericElementType::F64,
        GpuTensorStorage::Real,
        false,
    );
    track(&handle, true);
    let identity = handle.view_identity();
    let failure = handle.release().unwrap_err();
    assert_eq!(failure.residency, Some(ResidencyCleanupError::Unavailable));
    assert!(matches!(
        failure.physical,
        Ok(allocation::AllocationReleaseOutcome::Retired)
    ));
    assert_eq!(calls(identity), 1);
    assert_eq!(owner.retired.load(Ordering::SeqCst), 1);
    drop(failure);
    assert_eq!(calls(identity), 1);
}
