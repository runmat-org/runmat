use super::*;

#[test]
fn persisted_identity_rejects_same_length_changes_truncation_and_growth() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("output");
    let original = vec![73_u8; 131_093];
    let identity = FileIdentity::of(&original);
    std::fs::write(&path, &original).unwrap();
    assert!(identity.matches(&path));
    assert_eq!(identity.read_verified(&path).unwrap(), original);
    let mut changed = original.clone();
    changed[65_537] ^= 1;
    let grown = [&original[..], &[0]].concat();
    for bytes in [&changed[..], &original[..original.len() - 1], &grown[..]] {
        std::fs::write(&path, bytes).unwrap();
        assert!(!identity.matches(&path));
        assert!(identity.read_verified(&path).is_err());
    }
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Both element orders preserve the complete canonical mesh/evidence pair and scalar
//   summary through consuming persistence and independent cold decode. This checks handoff
//   semantics; source ownership and separate physical-footprint measurements are required to
//   establish peak-memory scaling.
#[test]
fn consuming_persistence_preserves_complete_canonical_pair_and_summary() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("mesh.cbor");
    let evidence_path = directory.path().join("evidence.cbor");
    for order in [
        runmat_meshing_core::ElementOrder::Tet4,
        runmat_meshing_core::ElementOrder::Tet10,
    ] {
        let artifact = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
        let evidence =
            runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&artifact);
        let expected_artifact = artifact.canonical_encode().unwrap();
        let expected_evidence = evidence.canonical_encode().unwrap();
        let expected_digest = artifact.canonical_digest;
        let nodes = artifact.topology.nodes.len();
        let summary = persist_completed_mesh(
            &output,
            &evidence_path,
            artifact,
            evidence,
            ExistingOutputPolicy::Replace,
        )
        .unwrap();
        assert_eq!(summary.canonical_digest, expected_digest);
        assert_eq!(summary.node_count, nodes);
        assert_eq!(std::fs::read(&output).unwrap(), expected_artifact);
        assert_eq!(std::fs::read(&evidence_path).unwrap(), expected_evidence);
    }
}
