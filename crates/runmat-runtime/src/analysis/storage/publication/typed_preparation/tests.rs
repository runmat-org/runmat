use super::*;
#[test]
fn typed_original_and_copy_reservations_accumulate_without_reusing_the_ceiling() {
    let mut builder =
        AnalysisPublicationBuilder::new(Digest::sha256(b"typed-preparation"), 13).unwrap();
    let reservation = builder.admit_preparation_storage(7, 6).unwrap();
    assert_eq!(reservation.resident_bytes, 7);
    assert_eq!(builder.preparation_resident_bytes, 13);
    assert!(builder.admit_preparation_storage(1, 0).is_err());
    assert!(builder.admit_preparation_storage(u64::MAX, 1).is_err());
    assert_eq!(builder.preparation_resident_bytes, 13);
    assert!(builder.store.runs.read().unwrap().is_empty());
}
// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A retained numeric history separately reserves original/copy storage and JSON
//   projection; an insufficient ceiling rejects before the materializing serialization and leaves
//   staged output empty. No complete-family inventory, native study or measured peak claim.
#[test]
fn retained_manifest_projection_requires_original_reservation_and_rejects_before_materialization() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Counted<'a>(&'a Vec<f64>, &'a AtomicUsize);
    impl Serialize for Counted<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.1.fetch_add(1, Ordering::SeqCst);
            self.0.serialize(serializer)
        }
    }
    let original = vec![1.25_f64; 4096];
    let profile = profile_serialized(
        &original,
        CanonicalLimits::new(16 << 20, 1_000_000, 65_536, 64),
    )
    .unwrap();
    let workspace = profile.json_projection_resident_bytes_estimate().unwrap();
    let retained =
        (std::mem::size_of::<Vec<f64>>() + original.capacity() * std::mem::size_of::<f64>()) as u64;
    let definition = Digest::sha256(b"retained-manifest");
    let mut accepted =
        AnalysisPublicationBuilder::new(definition, 2 * retained + workspace).unwrap();
    assert!(accepted.serialize_retained_manifest(&original).is_err());
    accepted
        .admit_preparation_storage(retained, retained)
        .unwrap();
    let value = accepted.serialize_retained_manifest(&original).unwrap();
    assert_eq!(value.as_array().unwrap().len(), original.len());
    let mut rejected =
        AnalysisPublicationBuilder::new(definition, 2 * retained + workspace - 1).unwrap();
    rejected
        .admit_preparation_storage(retained, retained)
        .unwrap();
    let passes = AtomicUsize::new(0);
    assert!(rejected
        .serialize_retained_manifest(&Counted(&original, &passes))
        .is_err());
    assert_eq!(passes.load(Ordering::SeqCst), 1);
    assert!(!*rejected.admission.lock().unwrap());
    assert!(rejected.store.runs.read().unwrap().is_empty());
}

#[test]
fn mesh_storage_resource_rejection_preserves_its_error_category() {
    for field in ["mesh retained storage", "mesh validation resources"] {
        assert!(matches!(
            storage_error(runmat_meshing_core::MeshingContractError::invalid(
                field,
                "bounded inventory rejection",
            )),
            AnalysisPublicationError::ResourceLimit { .. }
        ));
    }
    assert!(!matches!(
        storage_error(runmat_meshing_core::MeshingContractError::invalid(
            "publication storage inventory",
            "cancelled",
        )),
        AnalysisPublicationError::ResourceLimit { .. }
    ));
}
