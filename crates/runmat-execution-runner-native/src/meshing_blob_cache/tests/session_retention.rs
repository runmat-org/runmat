//! Numerical caches have a host lifetime; result objects keep their session lifetime.
use super::*;

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A host-lifetime bounded numerical record survives native session cleanup while
//   temporary session objects are removed; reopened reads preserve bytes and release storage. This
//   fixture does not promise indefinite cache retention, prevent quota eviction, authorize an
//   accepted mesh or establish whole-study performance.
#[test]
fn meshing_subtask_cache_survives_session_cleanup_without_retaining_session_objects() {
    let fixture = Fixture::new();
    let base = fixture.0.join("sessions");
    let cache_path = base.join("meshing-subtask-cache/v1/cache.sqlite");
    let mut config = crate::NativeExecutionConfig::for_current_executable().unwrap();
    config.store_root = base;
    let session = crate::NativeProgramSession::new(config).unwrap();
    let store = session.object_store();
    let scope_root = store.root().parent().unwrap().to_owned();
    let control = Control {
        limit: Some(8 * 1024 * 1024),
        ..Control::default()
    };
    let key = [23; 32];
    {
        let cache = BoundedMeshingBlobCache::new(cache_path.clone());
        assert!(cache
            .write(&key, b"bounded diagnostic input", 4096, &control)
            .unwrap());
    }
    drop(store);
    drop(session);
    assert!(
        !scope_root.exists(),
        "temporary execution objects must still be reclaimed"
    );
    let reopened = BoundedMeshingBlobCache::new(cache_path);
    let entry = reopened.read(&key, &control).unwrap().unwrap();
    assert_eq!(entry.bytes, b"bounded diagnostic input");
    drop(entry);
    assert_eq!(control.live.load(Ordering::Relaxed), 0);
}
