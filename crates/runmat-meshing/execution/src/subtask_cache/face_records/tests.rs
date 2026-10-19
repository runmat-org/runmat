//! Transport fixture only: current numerical face/boundary admission is separate.
use super::*;
use crate::subtask_cache::tests::Store;
fn inventory(bytes: &[u8]) -> Result<u64, MeshingSubtaskCacheError> {
    runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate(bytes)
        .map_err(|error| MeshingSubtaskCacheError::Storage(error.to_string()))
}
#[test]
fn oversized_record_preserves_identity_and_releases_storage_across_misses_and_corruption() {
    let store = Store::default();
    let control = crate::geometry_control::tests::bounded_control(64 * 1024 * 1024);
    let key = StableDigest::from_bytes([11; 32]);
    let mut bytes = runmat_canonical_codec::encode_contract(
        b"face-transport-test/v1\0",
        "face-transport-test/v1",
        &"x".repeat(2 * 1024 * 1024 + 17),
        runmat_canonical_codec::CanonicalLimits::new(4 * 1024 * 1024, 16, 4 * 1024 * 1024, 16),
    )
    .unwrap();
    let workspace = inventory(&bytes).unwrap();
    record::frame(&mut bytes, key);
    assert!(bytes.len() as u64 > store.maximum_entry_bytes());
    assert!(retain(&store, key, &bytes, workspace, &control).unwrap());
    assert_eq!(control.usage().live_allocation_bytes, 0);
    {
        let records = store.0.lock().unwrap();
        assert!(records
            .values()
            .all(|(bytes, _)| bytes.len() as u64 <= store.maximum_entry_bytes()));
        assert!(records[&key].0.len() < 4096);
    }
    let restored = read(&store, key, &control, inventory).unwrap().unwrap();
    assert_eq!(restored.bytes, bytes);
    assert_eq!(
        control.usage().live_allocation_bytes,
        bytes.len() as u64 + workspace
    );
    drop(restored);
    assert_eq!(control.usage().live_allocation_bytes, 0);
    let chunk = *store
        .0
        .lock()
        .unwrap()
        .iter()
        .find(|(_, (bytes, _))| bytes.len() == store.maximum_entry_bytes() as usize)
        .unwrap()
        .0;
    store.0.lock().unwrap().get_mut(&chunk).unwrap().0[0] ^= 1;
    let called = std::cell::Cell::new(false);
    assert!(read(&store, key, &control, |_| {
        called.set(true);
        Ok(0)
    })
    .is_err());
    assert!(
        !called.get(),
        "corrupt chunks cannot reach typed materialization"
    );
    assert_eq!(control.usage().live_allocation_bytes, 0);
    store.0.lock().unwrap().remove(&chunk);
    assert!(read(&store, key, &control, inventory).unwrap().is_none());
    assert_eq!(control.usage().live_allocation_bytes, 0);
    assert!(retain(&store, key, &bytes, workspace, &control).unwrap());
    let other = StableDigest::from_bytes([12; 32]);
    let pointer = store.0.lock().unwrap()[&key].clone();
    store.0.lock().unwrap().insert(other, pointer);
    assert!(read(&store, other, &control, inventory).is_err());
    assert_eq!(control.usage().live_allocation_bytes, 0);
    let limited = crate::geometry_control::tests::bounded_control(512 * 1024);
    assert!(read(&store, key, &limited, inventory).is_err());
    assert_eq!(limited.usage().live_allocation_bytes, 0);
}
