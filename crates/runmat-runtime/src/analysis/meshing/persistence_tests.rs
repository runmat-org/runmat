use super::*;
#[test]
fn immutable_mesh_collision_comparison_handles_matches_changes_short_and_oversized_files() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("original.cbor");
    let original = vec![7_u8; 8193];
    persist_canonical(&path, &original).unwrap();
    persist_canonical(&path, &original).unwrap();
    assert_eq!(runmat_filesystem::read(&path).unwrap(), original);
    for incoming in [vec![8_u8; 8193], vec![7_u8; 8192], vec![7_u8; 8194]] {
        assert!(persist_canonical(&path, &incoming)
            .unwrap_err()
            .contains("collision"));
        assert_eq!(runmat_filesystem::read(&path).unwrap(), original);
    }
}

#[test]
fn immutable_mesh_collision_read_cancels_without_changing_original_or_later_replay() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("original.cbor");
    let original = vec![7_u8; 16385];
    persist_canonical(&path, &original).unwrap();
    {
        let _interrupt = crate::interrupt::replace_interrupt(Some(
            runmat_execution::CancellationToken::from_shared(std::sync::Arc::new(
                std::sync::atomic::AtomicBool::new(true),
            )),
        ));
        assert!(persist_canonical(&path, &original)
            .unwrap_err()
            .contains("cancelled"));
    }
    assert_eq!(runmat_filesystem::read(&path).unwrap(), original);
    persist_canonical(&path, &original).unwrap();
}

#[test]
fn borrowed_mesh_encoding_observes_runtime_cancellation_and_preserves_original() {
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
    let _guard = crate::analysis::tests::analysis_test_guard();
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let original = mesh.canonical_encode().unwrap();
        {
            let _interrupt = crate::interrupt::replace_interrupt(Some(
                runmat_execution::CancellationToken::from_shared(std::sync::Arc::new(
                    std::sync::atomic::AtomicBool::new(true),
                )),
            ));
            assert!(mesh
                .canonical_encode_with_cancellation(&MeshPersistenceCancellation)
                .is_err());
        }
        assert_eq!(
            mesh.canonical_encode_with_cancellation(&MeshPersistenceCancellation)
                .unwrap(),
            original
        );
    }
}
