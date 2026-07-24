use super::*;

#[test]
fn empty_queue_reports_complete_boundary_recovery() {
    let faces = tetrahedron_faces();

    let queue = build_boundary_recovery_queue(&faces, &faces)
        .expect("matching boundary faces should queue no work");

    assert!(queue.items.is_empty());
    assert_eq!(queue.missing_edge_count, 0);
    assert_eq!(queue.missing_face_count, 0);
    assert_eq!(queue.interface_mismatch_count, 0);
    assert_eq!(queue.provenance_mismatch_count, 0);
}

#[test]
fn missing_face_queues_face_recovery_when_edges_are_still_present() {
    let expected = tetrahedron_faces();
    let candidate = expected[1..].to_vec();

    let queue = build_boundary_recovery_queue(&expected, &candidate)
        .expect("missing face should produce recovery work");

    assert_eq!(queue.missing_edge_count, 0);
    assert_eq!(queue.missing_face_count, 1);
    assert_eq!(queue.interface_mismatch_count, 0);
    assert_eq!(queue.items[0].priority, BoundaryRecoveryPriority::Face);
    assert_eq!(queue.items[0].reason, BoundaryRecoveryReason::MissingFace);
    assert_eq!(queue.items[0].face_node_ids, Some([0, 1, 2]));
}

#[test]
fn missing_edges_are_prioritized_before_face_recovery() {
    let expected = vec![face([0, 1, 2], 10, [100, 101, 102], &["loaded"])];
    let candidate = Vec::new();

    let queue = build_boundary_recovery_queue(&expected, &candidate)
        .expect("missing face and edges should produce recovery work");

    assert_eq!(queue.missing_edge_count, 3);
    assert_eq!(queue.missing_face_count, 1);
    assert_eq!(queue.interface_mismatch_count, 0);
    assert_eq!(
        queue
            .items
            .iter()
            .map(|item| item.priority)
            .collect::<Vec<_>>(),
        vec![
            BoundaryRecoveryPriority::Edge,
            BoundaryRecoveryPriority::Edge,
            BoundaryRecoveryPriority::Edge,
            BoundaryRecoveryPriority::Face,
        ]
    );
}

#[test]
fn provenance_mismatch_queues_source_and_region_repairs() {
    let expected = tetrahedron_faces();
    let mut candidate = expected.clone();
    candidate[0].source_face_id = Some(99);
    candidate[0].source_edge_ids[1] = Some(88);
    candidate[0].region_ids = vec!["other".to_string()];

    let queue = build_boundary_recovery_queue(&expected, &candidate)
        .expect("provenance mismatch should queue repair work");

    assert_eq!(queue.missing_edge_count, 0);
    assert_eq!(queue.missing_face_count, 0);
    assert_eq!(queue.interface_mismatch_count, 0);
    assert_eq!(queue.provenance_mismatch_count, 3);
    assert_eq!(
        queue
            .items
            .iter()
            .map(|item| item.reason.clone())
            .collect::<Vec<_>>(),
        vec![
            BoundaryRecoveryReason::SourceFaceMismatch,
            BoundaryRecoveryReason::RegionMismatch,
            BoundaryRecoveryReason::SourceEdgeMismatch,
        ]
    );
}

#[test]
fn outside_neighbor_mismatch_queues_face_recovery() {
    let mut expected = tetrahedron_faces();
    expected[0].outside_tetrahedron_ids = vec![42, 24, 42];
    let mut candidate = expected.clone();
    candidate[0].outside_tetrahedron_ids = vec![42];

    let queue = build_boundary_recovery_queue(&expected, &candidate)
        .expect("outside-neighbor mismatch should queue face recovery work");

    assert_eq!(queue.missing_edge_count, 0);
    assert_eq!(queue.missing_face_count, 0);
    assert_eq!(queue.interface_mismatch_count, 1);
    assert_eq!(queue.provenance_mismatch_count, 0);
    assert_eq!(queue.items.len(), 1);
    assert_eq!(queue.items[0].priority, BoundaryRecoveryPriority::Face);
    assert_eq!(
        queue.items[0].reason,
        BoundaryRecoveryReason::OutsideTetrahedronMismatch
    );
    assert_eq!(queue.items[0].face_node_ids, Some([0, 1, 2]));
    assert_eq!(queue.items[0].outside_tetrahedron_ids, vec![24, 42]);
}

#[test]
fn duplicate_expected_faces_are_rejected() {
    let mut faces = tetrahedron_faces();
    faces[1].node_ids = faces[0].node_ids;

    let err = build_boundary_recovery_queue(&faces, &[])
        .expect_err("duplicate expected faces should fail");

    assert_eq!(
        err,
        BoundaryRecoveryQueueError::DuplicateBoundaryFace {
            node_ids: [0, 1, 2]
        }
    );
}

fn tetrahedron_faces() -> Vec<ConstrainedCavityBoundaryFace> {
    vec![
        face([0, 1, 2], 10, [100, 101, 102], &["loaded", "fixed"]),
        face([0, 3, 1], 11, [103, 104, 100], &["fixed"]),
        face([1, 3, 2], 12, [104, 105, 101], &["solid"]),
        face([2, 3, 0], 13, [105, 103, 102], &["solid"]),
    ]
}

fn face(
    node_ids: [u32; 3],
    source_face_id: u32,
    source_edge_ids: [u32; 3],
    region_ids: &[&str],
) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: Vec::new(),
        source_face_id: Some(source_face_id),
        source_edge_ids: source_edge_ids.map(Some),
        region_ids: region_ids.iter().map(|region| region.to_string()).collect(),
    }
}
