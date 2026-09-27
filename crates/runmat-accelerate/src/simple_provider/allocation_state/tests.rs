use super::*;
use crate::simple_provider::InProcessProvider;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorView, HostTensorView,
};

#[test]
fn state_retains_every_exact_numeric_payload() {
    let state = NumericAllocationState::default();
    let payloads = [
        InProcessNumericData::F64(vec![-0.0, f64::MAX]),
        InProcessNumericData::F32(vec![-0.0, f32::MAX]),
        InProcessNumericData::Integer(HostIntegerDataOwned::I8(vec![i8::MIN, i8::MAX])),
        InProcessNumericData::Integer(HostIntegerDataOwned::I16(vec![i16::MIN, i16::MAX])),
        InProcessNumericData::Integer(HostIntegerDataOwned::I32(vec![i32::MIN, i32::MAX])),
        InProcessNumericData::Integer(HostIntegerDataOwned::I64(vec![i64::MIN, i64::MAX])),
        InProcessNumericData::Integer(HostIntegerDataOwned::U8(vec![0, u8::MAX])),
        InProcessNumericData::Integer(HostIntegerDataOwned::U16(vec![0, u16::MAX])),
        InProcessNumericData::Integer(HostIntegerDataOwned::U32(vec![0, u32::MAX])),
        InProcessNumericData::Integer(HostIntegerDataOwned::U64(vec![0, u64::MAX])),
    ];
    for (index, payload) in payloads.iter().enumerate() {
        assert!(state
            .lock()
            .unwrap()
            .insert(index as u64, payload.clone())
            .is_none());
    }
    for (index, expected) in payloads.into_iter().enumerate() {
        assert_eq!(state.take(index as u64), Some(expected));
        assert!(state.take(index as u64).is_none());
    }
    assert!(state.lock().unwrap().is_empty());
}

#[test]
fn detached_payload_keeps_storage_and_does_not_hold_the_map_lock() {
    let state = Arc::new(NumericAllocationState::default());
    let values = vec![3.0_f64; 1024];
    let pointer = values.as_ptr();
    state
        .lock()
        .unwrap()
        .insert(1, InProcessNumericData::F64(values));
    let retained_state = Arc::clone(&state);
    drop(state);
    let removed = retained_state.take(1).expect("allocation");
    let InProcessNumericData::F64(values) = &removed else {
        panic!("F64 payload")
    };
    assert_eq!(
        values.as_ptr(),
        pointer,
        "retirement must not copy the payload"
    );
    // A teardown callback may reenter the same allocation owner. Holding the
    // detached payload cannot retain the map's mutation lock.
    let mut reentrant = retained_state
        .allocations
        .try_lock()
        .expect("lock released before teardown");
    reentrant.insert(2, InProcessNumericData::F32(Vec::new()));
    drop(reentrant);
    drop(removed);
    assert_eq!(
        retained_state.take(2),
        Some(InProcessNumericData::F32(Vec::new()))
    );
}

#[test]
fn retirement_recovers_the_existing_poisoned_map_without_losing_payloads() {
    let state = NumericAllocationState::default();
    state
        .lock()
        .unwrap()
        .insert(1, InProcessNumericData::F64(vec![7.0]));
    let result = std::panic::catch_unwind(|| {
        let _guard = state.lock().unwrap();
        panic!("deterministic lock poisoning");
    });
    assert!(result.is_err());
    assert_eq!(state.take(1), Some(InProcessNumericData::F64(vec![7.0])));
    assert!(state.take(1).is_none());
}

#[test]
fn explicit_free_preserves_provider_namespace_and_existing_idempotence() {
    let first = InProcessProvider::new();
    let second = InProcessProvider::new();
    let view = HostTensorView {
        data: &[2.0, 5.0],
        shape: &[1, 2],
    };
    let a = first.upload(&view).unwrap();
    let b = second.upload(&view).unwrap();
    assert_ne!(a.device_id, b.device_id);
    assert_ne!(a.buffer_id, b.buffer_id);
    assert!(second.free(&a).is_err());
    assert_eq!(block_on(first.download(&a)).unwrap().data, view.data);
    first.free(&a).unwrap();
    first
        .free(&a)
        .expect("existing InProcess free remains idempotent");
    assert!(block_on(first.download(&a)).is_err());
    assert_eq!(block_on(second.download(&b)).unwrap().data, view.data);
    second.free(&b).unwrap();
}

#[test]
fn provider_numeric_routes_share_the_same_state_without_downcasting() {
    let provider = InProcessProvider::new();
    let single = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[f32::MIN_POSITIVE, -0.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let integer = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::U64(&[9_007_199_254_740_993, u64::MAX]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let HostNumericDataOwned::F32(values) =
        block_on(provider.download_numeric(&single)).unwrap().data
    else {
        panic!("Single storage must stay Single")
    };
    assert_eq!(
        values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vec![f32::MIN_POSITIVE.to_bits(), (-0.0_f32).to_bits()]
    );
    assert_eq!(
        block_on(provider.download_numeric(&integer)).unwrap().data,
        HostNumericDataOwned::U64(vec![9_007_199_254_740_993, u64::MAX])
    );
    provider.free(&single).unwrap();
    provider.free(&integer).unwrap();
    let state = numeric_registry().lock().unwrap();
    assert!(!state.contains_key(&single.buffer_id));
    assert!(!state.contains_key(&integer.buffer_id));
}
