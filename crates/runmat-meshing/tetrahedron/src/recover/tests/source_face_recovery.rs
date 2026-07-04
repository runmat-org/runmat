use super::*;

#[test]
fn recovery_stage_result_repairs_boundary_source_face_provenance_before_audit() {
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[0].source_face_id = entity(MeshingStage::SurfaceMesh, "other");

    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh)
        .expect("matching boundary topology should repair source-face provenance");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.tetrahedron_mesh.boundary_faces[0].source_face_id,
        entity(MeshingStage::SurfaceMesh, "face_1")
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_source_face_provenance_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["boundary_face_source_face_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["volume_face_source_face_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["interior_face_source_face_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_items"],
        0
    );
}

#[test]
fn recovery_stage_result_recovers_missing_exterior_boundary_face_before_audit() {
    let mut mesh = tetrahedron_mesh();
    let missing_face = mesh.boundary_faces.remove(2);
    let initial_queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("volume-face source face should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_face_boundary_face_items"],
        0
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_face_volume_face_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_face_interior_face_items"],
        0
    );

    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh)
        .expect("missing exterior PLC facet should be recovered from Tetrahedron topology");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert!(result.tetrahedron_mesh.boundary_faces.iter().any(|face| {
        sorted_face_ids(face.node_ids.clone()) == sorted_face_ids(missing_face.node_ids.clone())
            && face.source_face_id == missing_face.source_face_id
    }));
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_missing_boundary_faces"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_protected_edge_boundary_faces"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["volume_face_source_face_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["attempted_source_face_diagonal_recovery_pairs"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_face_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_items"],
        0
    );
}

#[test]
fn recovery_stage_result_recovers_source_faces_by_boundary_diagonal_flip_without_protected_edge() {
    let plc = source_face_boundary_diagonal_flip_plc();
    let mesh = boundary_diagonal_flip_tetrahedron_mesh();

    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("missing source faces should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_face_items"],
        2
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_face_absent_face_items"],
        2
    );
    assert_eq!(initial_queue.evidence.entity_counts["source_edge_items"], 0);
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_items"],
        0
    );

    let result = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("boundary diagonal flip should recover missing source faces");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert!(result.tetrahedron_mesh.boundary_faces.iter().any(|face| {
        sorted_face_ids(face.node_ids.clone())
            == sorted_face_ids([
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
            ])
            && face.source_face_id == entity(MeshingStage::SurfaceMesh, "face_1")
            && face.source_edge_ids == [None, None, None]
    }));
    assert!(result.tetrahedron_mesh.boundary_faces.iter().any(|face| {
        sorted_face_ids(face.node_ids.clone())
            == sorted_face_ids([
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
            ])
            && face.source_face_id == entity(MeshingStage::SurfaceMesh, "face_2")
            && face.source_edge_ids == [None, None, None]
    }));
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["attempted_source_face_diagonal_recovery_pairs"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_face_diagonal_pairs"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["recovered_source_face_diagonal_boundary_faces"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_face_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_items"],
        0
    );
}

#[test]
fn source_face_boundary_diagonal_flip_records_rejection_without_mutating_mesh() {
    let plc = source_face_boundary_diagonal_flip_plc();
    let mut mesh = boundary_diagonal_flip_tetrahedron_mesh();
    mesh.elements[1].material_region_id = "other_body".to_string();
    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("missing source faces should be reported before recovery");
    let original_elements = mesh.elements.clone();
    let original_boundary_faces = mesh.boundary_faces.clone();

    let recovery = crate::recover::source_faces::recover_source_faces_by_boundary_diagonal_flip(
        &plc,
        &initial_queue,
        &mut mesh,
    );

    assert_eq!(recovery.attempted_source_face_pair_count, 1);
    assert_eq!(recovery.source_face_pair_count, 0);
    assert_eq!(recovery.source_face_count, 0);
    assert_eq!(recovery.boundary_face_count, 0);
    assert_eq!(recovery.rejected_source_face_pair_count, 1);
    assert_eq!(
        recovery.rejection_counts
            ["rejected_source_face_diagonal_recovery_material_region_mismatch"],
        1
    );
    assert_eq!(mesh.elements, original_elements);
    assert_eq!(mesh.boundary_faces, original_boundary_faces);
}

#[test]
fn source_face_boundary_diagonal_recovery_uses_absent_source_face_queue_items_only() {
    let plc = tetrahedron_plc();
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[0].source_face_id = entity(MeshingStage::SurfaceMesh, "other");
    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("source-face provenance miss should be reported before recovery");
    let original_elements = mesh.elements.clone();
    let original_boundary_faces = mesh.boundary_faces.clone();

    let recovery = crate::recover::source_faces::recover_source_faces_by_boundary_diagonal_flip(
        &plc,
        &initial_queue,
        &mut mesh,
    );

    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_face_boundary_face_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_face_absent_face_items"],
        0
    );
    assert_eq!(recovery.attempted_source_face_pair_count, 0);
    assert_eq!(recovery.source_face_pair_count, 0);
    assert_eq!(recovery.boundary_face_count, 0);
    assert_eq!(mesh.elements, original_elements);
    assert_eq!(mesh.boundary_faces, original_boundary_faces);
}

#[test]
fn source_face_boundary_face_recovery_uses_volume_face_queue_items_only() {
    let plc = tetrahedron_plc();
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[0].source_face_id = entity(MeshingStage::SurfaceMesh, "other");
    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("source-face provenance miss should be reported before recovery");
    let original_boundary_faces = mesh.boundary_faces.clone();

    let recovered_count =
        crate::recover::boundary_faces::recover_volume_face_source_face_boundary_faces(
            &plc,
            &initial_queue,
            &mut mesh,
        );

    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_face_boundary_face_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_face_volume_face_items"],
        0
    );
    assert_eq!(recovered_count, 0);
    assert_eq!(mesh.boundary_faces, original_boundary_faces);
}

#[test]
fn source_face_provenance_repair_uses_boundary_face_queue_items_only() {
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces.remove(0);
    let initial_queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("volume-face source face should be reported before recovery");
    mesh.boundary_faces[0].source_face_id = entity(MeshingStage::SurfaceMesh, "other");

    let repaired_count = crate::recover::boundary_faces::repair_boundary_source_face_provenance(
        &initial_queue,
        &mut mesh,
    );

    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_face_volume_face_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_face_boundary_face_items"],
        0
    );
    assert_eq!(repaired_count, 0);
    assert_eq!(
        mesh.boundary_faces[0].source_face_id,
        entity(MeshingStage::SurfaceMesh, "other")
    );
}
