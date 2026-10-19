//! The native worker must preserve actual-buffer admission through both adapters.
use super::*;

#[test]
fn meshing_reads_admit_actual_file_capacity_through_remote_store() {
    let mut store = RemoteObjectStore::new().unwrap();
    let object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        "admitted-native",
        "test/v1",
        vec![7; 128],
    )
    .unwrap();
    store.write_verified(&object).unwrap();
    let mut amounts = Vec::new();
    let value = store
        .read_verified_bounded_with_admission(
            object.descriptor.digest,
            512 << 20,
            &mut |capacity| {
                assert_eq!(
                    capacity, 128,
                    "artifact quota is not the returned buffer inventory"
                );
                amounts.push(capacity);
                Ok(())
            },
        )
        .unwrap()
        .unwrap();
    assert_eq!(value, object.bytes);
    assert_eq!(amounts, [128]);
    let denied = store
        .read_verified_bounded_with_admission(object.descriptor.digest, 512 << 20, &mut |_| {
            Err(ArtifactError::Limit(
                "native buffer admission denied".into(),
            ))
        })
        .unwrap_err();
    assert!(denied
        .to_string()
        .contains("native buffer admission denied"));
    assert!(store
        .read_verified_bounded_with_admission(object.descriptor.digest, 127, &mut |_| panic!(
            "oversized object must reject before buffer admission"
        ))
        .is_err());
}
