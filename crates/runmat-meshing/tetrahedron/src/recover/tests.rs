use super::*;
use runmat_meshing_core::contracts::{
    PlcFacet, PlcNode, PlcProtectedEdge, PlcValidationSummary, Tetrahedron4Element,
    TetrahedronBoundaryFace, TetrahedronMeshNode, TopologyEntityId,
};

mod input_validation;

#[test]
fn builds_recovery_queue_for_recovered_plc_constraints() {
    let queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &tetrahedron_mesh())
        .expect("matching Tetrahedron mesh should recover PLC constraints");

    assert_eq!(queue.items.len(), 6);
    assert_eq!(queue.evidence.stage, MeshingStage::ConstraintRecovery);
    assert_eq!(queue.evidence.entity_counts["source_face_items"], 4);
    assert_eq!(queue.evidence.entity_counts["source_edge_items"], 1);
    assert_eq!(queue.evidence.entity_counts["material_interface_items"], 1);
    assert_eq!(queue.evidence.entity_counts["recovered_items"], 6);
    assert_eq!(queue.evidence.entity_counts["missing_items"], 0);
    assert_eq!(
        queue.evidence.entity_counts["missing_source_edge_topology_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_edge_provenance_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_topology_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_provenance_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_boundary_owned_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_interior_face_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_absent_partition_items"],
        0
    );
    assert!(queue
        .items
        .iter()
        .filter(|item| item.kind == TetrahedronRecoveryKind::SourceFace)
        .all(|item| {
            item.source_face_topology == Some(TetrahedronSourceFaceTopology::BoundaryFace)
                && item.source_face_node_ids.is_some()
        }));
    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::SourceEdge
            && item.protected_edge_node_ids
                == Some([
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                ])
            && item.protected_edge_topology == Some(TetrahedronProtectedEdgeTopology::BoundaryEdge)
    }));
    assert!(queue
        .items
        .iter()
        .all(|item| item.status == TetrahedronRecoveryStatus::Recovered));
}

#[test]
fn marks_tetrahedron_mesh_recovered_when_recovery_queue_has_no_missing_items() {
    let mut mesh = tetrahedron_mesh();
    let queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("matching Tetrahedron mesh should recover PLC constraints");

    mark_tetrahedron_mesh_recovery_state(&mut mesh, &queue);

    assert!(mesh.recovery_complete);
}

#[test]
fn recovery_stage_result_carries_audited_mesh_and_queue_evidence() {
    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), tetrahedron_mesh())
        .expect("matching Tetrahedron mesh should become a recovered stage artifact");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.recovery_queue.evidence.stage,
        MeshingStage::ConstraintRecovery
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_items"],
        0
    );
    assert_eq!(result.tetrahedron_mesh.elements.len(), 1);
}

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
        result.recovery_queue.evidence.entity_counts["missing_items"],
        0
    );
}

#[test]
fn recovery_stage_result_repairs_boundary_face_identity_before_audit() {
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[0].face_id = entity(MeshingStage::ProtectedBoundaryComplex, "stale_facet");

    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh)
        .expect("matching boundary topology should repair boundary-face identity");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.tetrahedron_mesh.boundary_faces[0].face_id,
        entity(MeshingStage::ProtectedBoundaryComplex, "facet_1")
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_boundary_face_identity_items"],
        1
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
        initial_queue.evidence.entity_counts["missing_source_face_volume_face_items"],
        1
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
fn recovery_stage_result_records_protected_source_edge_recovered_by_boundary_faces() {
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces.retain(|face| {
        !(face
            .node_ids
            .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "0"))
            && face
                .node_ids
                .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "1")))
    });

    let initial_queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("missing protected edge should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_topology_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_provenance_items"],
        0
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_volume_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_absent_edge_items"],
        0
    );

    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh)
        .expect("missing protected edge should recover with its exterior boundary faces");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_missing_boundary_faces"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_protected_edge_boundary_faces"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["volume_edge_source_edge_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["deferred_absent_source_edge_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_face_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_source_edge_items"],
        0
    );
}

#[test]
fn recovery_stage_result_reconnects_absent_source_edge_by_boundary_diagonal_flip() {
    let plc = boundary_diagonal_flip_plc();
    let mesh = boundary_diagonal_flip_tetrahedron_mesh();

    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("absent protected edge should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_absent_edge_items"],
        1
    );
    assert!(initial_queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::SourceEdge
            && item.status == TetrahedronRecoveryStatus::Missing
            && item.protected_edge_topology == Some(TetrahedronProtectedEdgeTopology::Absent)
            && item.protected_edge_node_ids
                == Some([
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                ])
    }));

    let result = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("boundary diagonal flip should recover the absent protected edge");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert!(result.tetrahedron_mesh.elements.iter().any(|element| {
        element
            .node_ids
            .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "0"))
            && element
                .node_ids
                .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "1"))
    }));
    assert!(result.tetrahedron_mesh.boundary_faces.iter().any(|face| {
        sorted_face_ids(face.node_ids.clone())
            == sorted_face_ids([
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
            ])
            && face.source_edge_ids.iter().any(|source_edge_id| {
                source_edge_id
                    .as_ref()
                    .is_some_and(|source_edge_id| source_edge_id.id == "edge_1")
            })
    }));
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["reconnected_absent_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_absent_source_edge_boundary_faces"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["deferred_absent_source_edge_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["attempted_source_face_diagonal_recovery_pairs"],
        0
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
fn recovery_stage_result_recovers_source_faces_by_boundary_diagonal_flip_without_protected_edge() {
    let plc = source_face_boundary_diagonal_flip_plc();
    let mesh = boundary_diagonal_flip_tetrahedron_mesh();

    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("missing source faces should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_face_items"],
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
fn absent_source_edge_boundary_diagonal_flip_records_rejection_without_mutating_mesh() {
    let plc = boundary_diagonal_flip_plc();
    let mut mesh = boundary_diagonal_flip_tetrahedron_mesh();
    mesh.elements[1].material_region_id = "other_body".to_string();
    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("absent protected edge should be reported before recovery");
    let original_elements = mesh.elements.clone();
    let original_boundary_faces = mesh.boundary_faces.clone();

    let recovery = super::absent_edges::recover_absent_protected_edges_by_boundary_diagonal_flip(
        &plc,
        &initial_queue,
        &mut mesh,
    );

    assert_eq!(recovery.attempted_source_edge_count, 1);
    assert_eq!(recovery.source_edge_count, 0);
    assert_eq!(recovery.boundary_face_count, 0);
    assert_eq!(recovery.rejected_source_edge_count, 1);
    assert_eq!(
        recovery.rejection_counts["rejected_absent_source_edge_recovery_material_region_mismatch"],
        1
    );
    assert_eq!(mesh.elements, original_elements);
    assert_eq!(mesh.boundary_faces, original_boundary_faces);
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

    let recovery = super::source_faces::recover_source_faces_by_boundary_diagonal_flip(
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

    let recovery = super::source_faces::recover_source_faces_by_boundary_diagonal_flip(
        &plc,
        &initial_queue,
        &mut mesh,
    );

    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_face_boundary_face_items"],
        1
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

    let recovered_count = super::boundary_faces::recover_volume_face_source_face_boundary_faces(
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
    let plc = tetrahedron_plc();
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces.remove(0);
    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("volume-face source face should be reported before recovery");
    mesh.boundary_faces[0].source_face_id = entity(MeshingStage::SurfaceMesh, "other");

    let repaired_count =
        super::boundary_faces::repair_boundary_source_face_provenance(&initial_queue, &mut mesh);

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

#[test]
fn recovery_stage_result_repairs_boundary_source_edge_provenance_before_audit() {
    let mut mesh = tetrahedron_mesh();
    for boundary_face in &mut mesh.boundary_faces {
        boundary_face.source_edge_ids = [None, None, None];
    }

    let initial_queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("missing protected source-edge provenance should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_topology_items"],
        0
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_provenance_items"],
        1
    );
    assert!(initial_queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::SourceEdge
            && item.protected_edge_topology == Some(TetrahedronProtectedEdgeTopology::BoundaryEdge)
    }));

    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh)
        .expect("matching boundary topology should repair source-edge provenance");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_source_edge_provenance_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["boundary_edge_source_edge_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_source_edge_items"],
        0
    );
}

#[test]
fn recovery_stage_result_repairs_partial_boundary_source_edge_provenance_before_audit() {
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[0].source_edge_ids[2] = None;

    let initial_queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("partial protected source-edge provenance should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_topology_items"],
        0
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_provenance_items"],
        1
    );

    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh)
        .expect("partial protected source-edge provenance should be repaired");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_source_edge_provenance_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_source_edge_items"],
        0
    );
}

#[test]
fn recovery_stage_result_replaces_stale_boundary_source_edge_provenance_before_audit() {
    let mut mesh = tetrahedron_mesh();
    for boundary_face in &mut mesh.boundary_faces {
        boundary_face.source_edge_ids = [
            Some(entity(MeshingStage::CurveMesh, "stale_edge")),
            Some(entity(MeshingStage::CurveMesh, "stale_edge")),
            Some(entity(MeshingStage::CurveMesh, "stale_edge")),
        ];
    }

    let initial_queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("stale protected source-edge provenance should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_provenance_items"],
        1
    );

    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh)
        .expect("matching boundary topology should replace stale source-edge provenance");

    let protected_edge = [
        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
    ];
    for boundary_face in &result.tetrahedron_mesh.boundary_faces {
        for (edge_index, face_edge) in
            crate::protected_edges::face_edges(boundary_face.node_ids.clone())
                .into_iter()
                .enumerate()
        {
            if face_edge == protected_edge {
                assert_eq!(
                    boundary_face.source_edge_ids[edge_index],
                    Some(entity(MeshingStage::CurveMesh, "edge_1"))
                );
            } else {
                assert_eq!(boundary_face.source_edge_ids[edge_index], None);
            }
        }
    }
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_source_edge_provenance_items"],
        12
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["boundary_edge_source_edge_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_source_edge_items"],
        0
    );
}

#[test]
fn source_edge_provenance_repair_uses_boundary_edge_queue_items_only() {
    let mut plc = tetrahedron_plc();
    plc.protected_edges[0].node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        entity(MeshingStage::ProtectedBoundaryComplex, "2"),
    ];
    plc.protected_edges[0].source_edge_id = entity(MeshingStage::CurveMesh, "edge_2");
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[0].node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        entity(MeshingStage::ProtectedBoundaryComplex, "3"),
    ];
    mesh.boundary_faces[3].node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "2"),
        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        entity(MeshingStage::ProtectedBoundaryComplex, "3"),
    ];
    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("volume-edge source edge should be reported before recovery");
    let original_boundary_faces = mesh.boundary_faces.clone();

    let repaired_count = super::boundary_faces::repair_boundary_source_edge_provenance(
        &plc,
        &initial_queue,
        &mut mesh,
    );

    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_volume_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_provenance_items"],
        0
    );
    assert_eq!(repaired_count, 0);
    assert_eq!(mesh.boundary_faces, original_boundary_faces);
}

#[test]
fn recovery_stage_result_repairs_single_material_interface_before_audit() {
    let mut mesh = tetrahedron_mesh();
    mesh.elements[0].material_region_id = "other_body".to_string();

    let initial_queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("missing material interface should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_material_interface_items"],
        1
    );

    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh)
        .expect("single material interface should repair element material ownership");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.tetrahedron_mesh.elements[0].material_region_id,
        "solid_body"
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_material_interface_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_material_interface_elements"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["attempted_material_interface_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["global_material_interface_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["boundary_owned_material_interface_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["interior_material_interface_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["rejected_material_interface_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_material_interface_items"],
        0
    );
}

#[test]
fn recovery_stage_result_does_not_guess_multi_material_interface_repair() {
    let mut plc = tetrahedron_plc();
    plc.facets[0].material_interface_ids = vec!["other_body".to_string()];
    let mut mesh = tetrahedron_mesh();
    mesh.elements[0].material_region_id = "other_body".to_string();
    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("ambiguous material interface should be reported before recovery");
    let mut direct_recovery_mesh = mesh.clone();
    let recovery = super::material_interfaces::recover_material_interface_regions(
        &plc,
        &initial_queue,
        &mut direct_recovery_mesh,
    );

    assert_eq!(recovery.attempted_material_interface_count, 1);
    assert_eq!(recovery.repaired_element_count, 0);
    assert_eq!(recovery.rejected_material_interface_count, 1);
    assert_eq!(recovery.global_material_interface_count, 0);
    assert_eq!(recovery.boundary_owned_material_interface_count, 1);
    assert_eq!(recovery.interior_material_interface_count, 0);
    assert_eq!(recovery.absent_partition_material_interface_count, 0);
    assert_eq!(recovery.ambiguous_boundary_ownership_count, 1);
    assert_eq!(recovery.missing_boundary_ownership_count, 0);
    assert_eq!(recovery.missing_interior_ownership_count, 0);
    assert_eq!(recovery.absent_partition_rejection_count, 0);

    let recovery_evidence = assert_incomplete_recovery(
        recover_tetrahedron_mesh_from_plc(&plc, mesh).expect_err("ambiguous repair should fail"),
        1,
        0,
        0,
        1,
    );
    assert_eq!(
        recovery_evidence.entity_counts["rejected_material_interface_recovery_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["boundary_owned_material_interface_recovery_input_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["absent_partition_material_interface_recovery_items"],
        0
    );
}

#[test]
fn recovery_stage_result_repairs_boundary_facet_owned_material_interface() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    mesh.elements[0].material_region_id = "region_b".to_string();

    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("missing material interface should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["material_interface_items"],
        2
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_material_interface_items"],
        1
    );

    let result = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("boundary-facet material ownership should repair the missing region");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.tetrahedron_mesh.elements[0].material_region_id,
        "region_a"
    );
    assert_eq!(
        result.tetrahedron_mesh.elements[1].material_region_id,
        "region_b"
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_material_interface_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_material_interface_elements"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["attempted_material_interface_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["rejected_material_interface_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["global_material_interface_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["boundary_owned_material_interface_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["boundary_owned_material_interface_recovery_input_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["interior_material_interface_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_material_interface_items"],
        0
    );
}

#[test]
fn recovery_queue_reports_incomplete_material_interface_ownership() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    add_unclassified_region_a_boundary_neighbor(&mut mesh);

    let queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("incomplete material ownership should be reported before recovery");

    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_items"],
        1
    );
    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::MaterialInterface
            && item.status == TetrahedronRecoveryStatus::Missing
            && item.material_interface_id.as_deref() == Some("region_a")
            && item.material_interface_topology
                == Some(TetrahedronMaterialInterfaceTopology::BoundaryOwned)
    }));
    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::MaterialInterface
            && item.status == TetrahedronRecoveryStatus::Recovered
            && item.material_interface_id.as_deref() == Some("region_b")
            && item.material_interface_topology.is_none()
    }));
    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_boundary_owned_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_interior_face_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_absent_partition_items"],
        0
    );
}

#[test]
fn recovery_queue_classifies_interior_face_material_interface_ownership() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    replace_region_b_with_unclassified_region_a_interior_neighbor(&mut mesh);

    let queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("interior material ownership should be classified before recovery");

    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::MaterialInterface
            && item.status == TetrahedronRecoveryStatus::Missing
            && item.material_interface_id.as_deref() == Some("region_a")
            && item.material_interface_topology
                == Some(TetrahedronMaterialInterfaceTopology::InteriorFace)
    }));
    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_interior_face_items"],
        1
    );
}

#[test]
fn recovery_queue_classifies_absent_partition_material_interface_work() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    mesh.elements.remove(0);
    mesh.boundary_faces
        .retain(|face| face.source_face_id.id.starts_with("face_b"));

    let queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("absent material partition should be classified before recovery");

    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::MaterialInterface
            && item.status == TetrahedronRecoveryStatus::Missing
            && item.material_interface_id.as_deref() == Some("region_a")
            && item.material_interface_topology
                == Some(TetrahedronMaterialInterfaceTopology::AbsentPartition)
    }));
    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_absent_partition_items"],
        1
    );
}

#[test]
fn recovery_stage_result_inserts_bounded_absent_material_interface_partition() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    mesh.elements.remove(0);
    mesh.boundary_faces
        .retain(|face| face.source_face_id.id.starts_with("face_b"));

    let result = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("bounded absent material partition should be inserted");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert!(result
        .tetrahedron_mesh
        .elements
        .iter()
        .any(|element| element.material_region_id == "region_a"));
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_material_interface_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_source_face_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["absent_partition_material_interface_recovery_input_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["attempted_absent_material_partition_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["inserted_absent_material_partition_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["inserted_absent_material_partition_elements"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["inserted_absent_material_partition_boundary_faces"],
        3
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_material_interface_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_face_items"],
        3
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["absent_material_partition_topology_candidate_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["absent_material_partition_usable_candidate_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["rejected_absent_material_partition_quality_candidate_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["rejected_absent_material_partition_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["rejected_material_interface_absent_partition"],
        0
    );
}

#[test]
fn recovery_stage_result_preserves_protected_source_edge_on_inserted_material_partition() {
    let plc = two_region_bipyramid_plc_with_region_a_protected_edge();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    mesh.elements.remove(0);
    mesh.boundary_faces
        .retain(|face| face.source_face_id.id.starts_with("face_b"));

    let result = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("bounded absent material partition should preserve protected edge provenance");

    let protected_source_edge_id = entity(MeshingStage::CurveMesh, "edge_region_a_0_2");
    assert!(result
        .tetrahedron_mesh
        .boundary_faces
        .iter()
        .filter(|face| face.source_face_id.id.starts_with("face_a"))
        .any(|face| face
            .source_edge_ids
            .iter()
            .any(|source_edge_id| source_edge_id.as_ref() == Some(&protected_source_edge_id))));
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["inserted_absent_material_partition_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["rolled_back_absent_material_partition_recovery_items"],
        0
    );
}

#[test]
fn recovery_stage_result_inserts_two_element_absent_material_interface_partition() {
    let plc = two_element_material_partition_plc();
    let mesh = two_element_material_partition_seed_mesh();

    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("two-element material partition should be queued before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_material_interface_absent_partition_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_face_absent_face_items"],
        6
    );

    let result = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("bounded two-element material partition should be inserted");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result
            .tetrahedron_mesh
            .elements
            .iter()
            .filter(|element| element.material_region_id == "region_a")
            .count(),
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["inserted_absent_material_partition_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["inserted_absent_material_partition_elements"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["inserted_absent_material_partition_boundary_faces"],
        6
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_face_items"],
        6
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_material_interface_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["absent_material_partition_topology_candidate_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["absent_material_partition_usable_candidate_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["rejected_absent_material_partition_interior_candidate_sets"],
        2
    );
}

#[test]
fn recovery_stage_result_reports_absent_material_partition_quality_rejection() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    mesh.elements.remove(0);
    mesh.boundary_faces
        .retain(|face| face.source_face_id.id.starts_with("face_b"));
    mesh.nodes
        .iter_mut()
        .find(|node| node.node_id.id == "4")
        .expect("fixture should carry apex node")
        .coordinates_m = [0.5, 0.5, 0.0];

    let recovery_evidence = assert_incomplete_recovery(
        recover_tetrahedron_mesh_from_plc(&plc, mesh)
            .expect_err("degenerate material partition should fail the quality gate"),
        4,
        3,
        0,
        1,
    );

    assert_eq!(
        recovery_evidence.entity_counts["attempted_absent_material_partition_recovery_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["inserted_absent_material_partition_recovery_items"],
        0
    );
    assert_eq!(
        recovery_evidence.entity_counts["absent_material_partition_topology_candidate_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["absent_material_partition_usable_candidate_items"],
        0
    );
    assert_eq!(
        recovery_evidence.entity_counts
            ["rejected_absent_material_partition_quality_candidate_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["rejected_absent_material_partition_quality_gate"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["rejected_absent_material_partition_recovery_items"],
        1
    );
}

#[test]
fn recovery_stage_result_rolls_back_absent_material_partition_on_post_insert_audit_failure() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    mesh.elements.remove(0);
    mesh.boundary_faces
        .retain(|face| face.source_face_id.id.starts_with("face_b"));
    mesh.boundary_faces.push(boundary_face(
        "stale_facet_a_1",
        ["0", "2", "3"],
        "stale_face_a_1",
    ));

    let TetrahedronRecoveryError::IncompleteRecovery {
        recovery_evidence, ..
    } = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect_err("stale partition boundary face should roll back insertion")
    else {
        panic!("expected incomplete recovery error");
    };

    assert_eq!(
        recovery_evidence.entity_counts["attempted_absent_material_partition_recovery_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["inserted_absent_material_partition_recovery_items"],
        0
    );
    assert_eq!(
        recovery_evidence.entity_counts["rolled_back_absent_material_partition_recovery_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["rolled_back_absent_material_partition_elements"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["rolled_back_absent_material_partition_boundary_faces"],
        2
    );
    assert_eq!(
        recovery_evidence.entity_counts["rejected_absent_material_partition_post_insertion_audit"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["missing_source_face_items"],
        2
    );
    assert_eq!(
        recovery_evidence.entity_counts["missing_material_interface_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["missing_material_interface_absent_partition_items"],
        1
    );
}

#[test]
fn recovery_stage_result_rolls_back_material_partition_with_stale_source_edge_provenance() {
    let plc = two_region_bipyramid_plc_with_region_a_protected_edge();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    mesh.elements.remove(0);
    mesh.boundary_faces
        .retain(|face| face.source_face_id.id.starts_with("face_b"));
    mesh.boundary_faces
        .push(boundary_face("facet_a_1", ["0", "2", "3"], "face_a_1"));

    let TetrahedronRecoveryError::IncompleteRecovery {
        recovery_evidence, ..
    } = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect_err("stale source-edge provenance should roll back partition insertion")
    else {
        panic!("expected incomplete recovery error");
    };

    assert_eq!(
        recovery_evidence.entity_counts["attempted_absent_material_partition_recovery_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["inserted_absent_material_partition_recovery_items"],
        0
    );
    assert_eq!(
        recovery_evidence.entity_counts["rolled_back_absent_material_partition_recovery_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["rolled_back_absent_material_partition_elements"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["rolled_back_absent_material_partition_boundary_faces"],
        2
    );
    assert_eq!(
        recovery_evidence.entity_counts["rejected_absent_material_partition_post_insertion_audit"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["missing_material_interface_absent_partition_items"],
        1
    );
}

#[test]
fn recovery_stage_result_repairs_incomplete_material_interface_ownership_from_queue() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    add_unclassified_region_a_boundary_neighbor(&mut mesh);

    let result = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("queued incomplete material ownership should be repaired");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.tetrahedron_mesh.elements[2].material_region_id,
        "region_a"
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_material_interface_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_material_interface_elements"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["attempted_material_interface_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["boundary_owned_material_interface_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["boundary_owned_material_interface_recovery_input_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_material_interface_items"],
        0
    );
}

#[test]
fn material_interface_recovery_propagates_through_interior_faces() {
    let plc = interior_material_interface_propagation_plc();
    let initial_queue = TetrahedronRecoveryQueue {
        items: vec![TetrahedronRecoveryQueueItem {
            item_id: "material_interface:region_a".to_string(),
            kind: TetrahedronRecoveryKind::MaterialInterface,
            status: TetrahedronRecoveryStatus::Missing,
            source_entity_id: None,
            source_face_node_ids: None,
            source_face_topology: None,
            protected_edge_node_ids: None,
            protected_edge_topology: None,
            material_interface_topology: Some(TetrahedronMaterialInterfaceTopology::BoundaryOwned),
            material_interface_id: Some("region_a".to_string()),
        }],
        evidence: StageEvidence::complete(MeshingStage::ConstraintRecovery),
    };
    let mut mesh = interior_material_interface_propagation_mesh();

    let recovery = super::material_interfaces::recover_material_interface_regions(
        &plc,
        &initial_queue,
        &mut mesh,
    );

    assert_eq!(mesh.elements[0].material_region_id, "region_a");
    assert_eq!(mesh.elements[1].material_region_id, "region_a");
    assert_eq!(recovery.attempted_material_interface_count, 1);
    assert_eq!(recovery.repaired_element_count, 2);
    assert_eq!(recovery.rejected_material_interface_count, 0);
    assert_eq!(recovery.global_material_interface_count, 0);
    assert_eq!(recovery.boundary_owned_material_interface_count, 1);
    assert_eq!(recovery.interior_material_interface_count, 1);
    assert_eq!(recovery.absent_partition_material_interface_count, 0);
    assert_eq!(recovery.ambiguous_boundary_ownership_count, 0);
    assert_eq!(recovery.missing_boundary_ownership_count, 0);
    assert_eq!(recovery.missing_interior_ownership_count, 0);
    assert_eq!(recovery.absent_partition_rejection_count, 0);
}

#[test]
fn keeps_tetrahedron_mesh_unrecovered_when_recovery_queue_has_missing_items() {
    let plc = tetrahedron_plc();
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[0].source_face_id = entity(MeshingStage::SurfaceMesh, "other");
    let queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("missing source faces should be reported as recovery evidence");

    mark_tetrahedron_mesh_recovery_state(&mut mesh, &queue);

    assert!(!mesh.recovery_complete);
}

#[test]
fn recovery_stage_result_rejects_missing_queue_items() {
    let plc = tetrahedron_plc();
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces.remove(0);
    mesh.nodes.push(tetrahedron_node(
        entity(MeshingStage::ProtectedBoundaryComplex, "4"),
        [1.0, 1.0, 1.0],
    ));
    mesh.elements[0].node_ids[0] = entity(MeshingStage::ProtectedBoundaryComplex, "4");

    let recovery_evidence = assert_incomplete_recovery(
        recover_tetrahedron_mesh_from_plc(&plc, mesh).expect_err("missing face should fail"),
        1,
        1,
        0,
        0,
    );
    assert_eq!(
        recovery_evidence.entity_counts["missing_source_face_absent_face_items"],
        1
    );
}

#[test]
fn incomplete_recovery_error_carries_source_face_diagonal_rejection_evidence() {
    let plc = source_face_boundary_diagonal_flip_plc();
    let mut mesh = boundary_diagonal_flip_tetrahedron_mesh();
    mesh.elements[1].material_region_id = "other_body".to_string();

    let recovery_evidence = assert_incomplete_recovery(
        recover_tetrahedron_mesh_from_plc(&plc, mesh)
            .expect_err("material-region mismatch should reject source-face diagonal recovery"),
        2,
        2,
        0,
        0,
    );

    assert_eq!(
        recovery_evidence.entity_counts["attempted_source_face_diagonal_recovery_pairs"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["rejected_source_face_diagonal_recovery_pairs"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts
            ["rejected_source_face_diagonal_recovery_material_region_mismatch"],
        1
    );
}

#[test]
fn recovery_queue_reports_missing_source_face() {
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[0].source_face_id = entity(MeshingStage::SurfaceMesh, "other");

    let queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("missing source faces should be reported as recovery evidence");

    assert_eq!(queue.evidence.status, StageEvidenceStatus::Failed);
    assert_eq!(queue.evidence.entity_counts["missing_items"], 1);
    assert_eq!(queue.evidence.entity_counts["missing_source_face_items"], 1);
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_provenance_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_topology_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_boundary_face_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_volume_face_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_absent_face_items"],
        0
    );
    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::SourceFace
            && item.status == TetrahedronRecoveryStatus::Missing
            && item.source_face_topology == Some(TetrahedronSourceFaceTopology::BoundaryFace)
            && item.source_face_node_ids
                == Some([
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                ])
            && item
                .source_entity_id
                .as_ref()
                .is_some_and(|source| source.id == "face_1")
    }));
}

#[test]
fn recovery_queue_reports_partial_boundary_source_face_provenance() {
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces.push(boundary_face(
        "extra_boundary_face",
        ["1", "0", "2"],
        "other",
    ));

    let queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("partial source-face provenance should be reported before recovery");

    assert_eq!(queue.evidence.status, StageEvidenceStatus::Failed);
    assert_eq!(queue.evidence.entity_counts["missing_items"], 1);
    assert_eq!(queue.evidence.entity_counts["missing_source_face_items"], 1);
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_provenance_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_topology_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_boundary_face_items"],
        1
    );
    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::SourceFace
            && item.status == TetrahedronRecoveryStatus::Missing
            && item.source_face_topology == Some(TetrahedronSourceFaceTopology::BoundaryFace)
            && item.source_face_node_ids
                == Some([
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                ])
            && item
                .source_entity_id
                .as_ref()
                .is_some_and(|source| source.id == "face_1")
    }));
}

#[test]
fn recovery_queue_reports_missing_source_face_present_as_volume_face() {
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces.remove(0);

    let queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("volume-face source faces should be reported as recovery evidence");

    assert_eq!(queue.evidence.status, StageEvidenceStatus::Failed);
    assert_eq!(queue.evidence.entity_counts["missing_source_face_items"], 1);
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_provenance_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_topology_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_boundary_face_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_volume_face_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_absent_face_items"],
        0
    );
    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::SourceFace
            && item.status == TetrahedronRecoveryStatus::Missing
            && item.source_face_topology == Some(TetrahedronSourceFaceTopology::VolumeFace)
            && item.source_face_node_ids
                == Some([
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                ])
            && item
                .source_entity_id
                .as_ref()
                .is_some_and(|source| source.id == "face_1")
    }));
}

#[test]
fn recovery_queue_rejects_invalid_protected_edge_before_recovery() {
    let mut plc = tetrahedron_plc();
    plc.nodes.push(plc_node("4", [2.0, 2.0, 2.0]));
    plc.protected_edges[0].node_ids[1] = entity(MeshingStage::ProtectedBoundaryComplex, "4");

    assert!(matches!(
        build_recovery_queue_from_plc(&plc, &tetrahedron_mesh()),
        Err(TetrahedronRecoveryError::InvalidProtectedBoundaryComplex { .. })
    ));
}

#[test]
fn recovery_queue_reports_missing_source_edge() {
    let mut plc = tetrahedron_plc();
    plc.protected_edges[0].node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        entity(MeshingStage::ProtectedBoundaryComplex, "2"),
    ];
    plc.protected_edges[0].source_edge_id = entity(MeshingStage::CurveMesh, "edge_2");
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[0].node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        entity(MeshingStage::ProtectedBoundaryComplex, "3"),
    ];
    mesh.boundary_faces[3].node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "2"),
        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        entity(MeshingStage::ProtectedBoundaryComplex, "3"),
    ];

    let queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("missing source edges should be reported as recovery evidence");

    assert_eq!(queue.evidence.status, StageEvidenceStatus::Failed);
    assert_eq!(queue.evidence.entity_counts["missing_source_edge_items"], 1);
    assert_eq!(
        queue.evidence.entity_counts["missing_source_edge_topology_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_edge_provenance_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_edge_volume_edge_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_edge_absent_edge_items"],
        0
    );
    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::SourceEdge
            && item.status == TetrahedronRecoveryStatus::Missing
            && item.protected_edge_topology == Some(TetrahedronProtectedEdgeTopology::VolumeEdge)
            && item
                .source_entity_id
                .as_ref()
                .is_some_and(|source| source.id == "edge_2")
    }));
}

#[test]
fn recovery_queue_reports_missing_source_edge_absent_from_volume_edges() {
    let mut plc = tetrahedron_plc();
    plc.protected_edges[0].node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        entity(MeshingStage::ProtectedBoundaryComplex, "2"),
    ];
    plc.protected_edges[0].source_edge_id = entity(MeshingStage::CurveMesh, "edge_2");
    let mut mesh = tetrahedron_mesh();
    mesh.nodes.push(tetrahedron_node(
        entity(MeshingStage::ProtectedBoundaryComplex, "4"),
        [2.0, 0.0, 0.0],
    ));
    mesh.elements[0].node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        entity(MeshingStage::ProtectedBoundaryComplex, "3"),
        entity(MeshingStage::ProtectedBoundaryComplex, "4"),
    ];
    mesh.boundary_faces[0].node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        entity(MeshingStage::ProtectedBoundaryComplex, "3"),
    ];
    mesh.boundary_faces[3].node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "2"),
        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        entity(MeshingStage::ProtectedBoundaryComplex, "3"),
    ];

    let queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("absent volume source edges should be reported as recovery evidence");

    assert_eq!(queue.evidence.entity_counts["missing_source_edge_items"], 1);
    assert_eq!(
        queue.evidence.entity_counts["missing_source_edge_topology_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_edge_volume_edge_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_edge_absent_edge_items"],
        1
    );
    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::SourceEdge
            && item.status == TetrahedronRecoveryStatus::Missing
            && item.protected_edge_topology == Some(TetrahedronProtectedEdgeTopology::Absent)
            && item.protected_edge_node_ids
                == Some([
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                ])
    }));
}

#[test]
fn recovery_queue_reports_missing_material_interface() {
    let mut mesh = tetrahedron_mesh();
    mesh.elements[0].material_region_id = "other_body".to_string();

    let queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("missing material interfaces should be reported as recovery evidence");

    assert_eq!(queue.evidence.status, StageEvidenceStatus::Failed);
    assert_eq!(queue.evidence.entity_counts["missing_items"], 1);
    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_items"],
        1
    );
    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::MaterialInterface
            && item.status == TetrahedronRecoveryStatus::Missing
            && item.material_interface_id.as_deref() == Some("solid_body")
    }));
}

#[test]
fn recovery_queue_rejects_open_plc_even_when_summary_claims_ready() {
    let mut plc = tetrahedron_plc();
    plc.facets.pop();

    assert!(matches!(
        build_recovery_queue_from_plc(&plc, &tetrahedron_mesh()),
        Err(TetrahedronRecoveryError::InvalidProtectedBoundaryComplex { .. })
    ));
}

fn assert_incomplete_recovery(
    error: TetrahedronRecoveryError,
    expected_missing_item_count: usize,
    expected_missing_source_face_item_count: usize,
    expected_missing_source_edge_item_count: usize,
    expected_missing_material_interface_item_count: usize,
) -> StageEvidence {
    let TetrahedronRecoveryError::IncompleteRecovery {
        missing_item_count,
        missing_source_face_item_count,
        missing_source_edge_item_count,
        missing_material_interface_item_count,
        recovery_evidence,
    } = error
    else {
        panic!("expected incomplete recovery error, got {error:?}");
    };

    assert_eq!(missing_item_count, expected_missing_item_count);
    assert_eq!(
        missing_source_face_item_count,
        expected_missing_source_face_item_count
    );
    assert_eq!(
        missing_source_edge_item_count,
        expected_missing_source_edge_item_count
    );
    assert_eq!(
        missing_material_interface_item_count,
        expected_missing_material_interface_item_count
    );
    recovery_evidence
}

fn tetrahedron_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "tetrahedron_plc".to_string(),
        nodes: vec![
            plc_node("0", [0.0, 0.0, 0.0]),
            plc_node("1", [1.0, 0.0, 0.0]),
            plc_node("2", [0.0, 1.0, 0.0]),
            plc_node("3", [0.0, 0.0, 1.0]),
        ],
        facets: vec![
            facet("facet_1", ["0", "2", "1"], "face_1"),
            facet("facet_2", ["0", "1", "3"], "face_2"),
            facet("facet_3", ["1", "2", "3"], "face_3"),
            facet("facet_4", ["2", "0", "3"], "face_4"),
        ],
        protected_edges: vec![PlcProtectedEdge {
            edge_id: entity(MeshingStage::ProtectedBoundaryComplex, "plc_edge_1"),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
            ],
            source_edge_id: entity(MeshingStage::CurveMesh, "edge_1"),
        }],
        validation: PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
    }
}

fn tetrahedron_mesh() -> TetrahedronMesh {
    TetrahedronMesh {
        mesh_id: "tetrahedron".to_string(),
        nodes: vec![
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                [0.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                [1.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                [0.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                [0.0, 0.0, 1.0],
            ),
        ],
        elements: vec![Tetrahedron4Element {
            element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_1"),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            ],
            material_region_id: "solid_body".to_string(),
        }],
        boundary_faces: vec![
            boundary_face("facet_1", ["0", "2", "1"], "face_1"),
            boundary_face("facet_2", ["0", "1", "3"], "face_2"),
            boundary_face("facet_3", ["1", "2", "3"], "face_3"),
            boundary_face("facet_4", ["2", "0", "3"], "face_4"),
        ],
        recovery_complete: false,
        quality_optimized: false,
        evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
    }
}

fn boundary_diagonal_flip_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "boundary_diagonal_flip_plc".to_string(),
        nodes: vec![
            plc_node("0", [0.0, 0.0, 0.0]),
            plc_node("1", [1.0, 0.0, 0.0]),
            plc_node("2", [0.0, 1.0, 0.0]),
            plc_node("3", [1.0, 1.0, 0.0]),
            plc_node("4", [0.5, 0.5, 1.0]),
        ],
        facets: vec![
            facet("facet_1", ["0", "1", "2"], "face_1"),
            facet("facet_2", ["0", "3", "1"], "face_2"),
            facet("facet_3", ["0", "2", "4"], "face_3"),
            facet("facet_4", ["0", "4", "3"], "face_4"),
            facet("facet_5", ["1", "3", "4"], "face_5"),
            facet("facet_6", ["1", "4", "2"], "face_6"),
        ],
        protected_edges: vec![PlcProtectedEdge {
            edge_id: entity(MeshingStage::ProtectedBoundaryComplex, "plc_edge_1"),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
            ],
            source_edge_id: entity(MeshingStage::CurveMesh, "edge_1"),
        }],
        validation: PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
    }
}

fn source_face_boundary_diagonal_flip_plc() -> ProtectedBoundaryComplex {
    let mut plc = boundary_diagonal_flip_plc();
    plc.complex_id = "source_face_boundary_diagonal_flip_plc".to_string();
    plc.protected_edges = Vec::new();
    plc
}

fn boundary_diagonal_flip_tetrahedron_mesh() -> TetrahedronMesh {
    TetrahedronMesh {
        mesh_id: "boundary_diagonal_flip_tetrahedron".to_string(),
        nodes: vec![
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                [0.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                [1.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                [0.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                [1.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                [0.5, 0.5, 1.0],
            ),
        ],
        elements: vec![
            Tetrahedron4Element {
                element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_1"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                ],
                material_region_id: "solid_body".to_string(),
            },
            Tetrahedron4Element {
                element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_2"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                ],
                material_region_id: "solid_body".to_string(),
            },
        ],
        boundary_faces: vec![
            boundary_face("old_facet_1", ["0", "2", "3"], "old_face_1"),
            boundary_face("old_facet_2", ["1", "3", "2"], "old_face_2"),
            boundary_face("facet_3", ["0", "2", "4"], "face_3"),
            boundary_face("facet_4", ["0", "4", "3"], "face_4"),
            boundary_face("facet_5", ["1", "3", "4"], "face_5"),
            boundary_face("facet_6", ["1", "4", "2"], "face_6"),
        ],
        recovery_complete: false,
        quality_optimized: false,
        evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
    }
}

fn two_region_bipyramid_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "two_region_bipyramid_plc".to_string(),
        nodes: vec![
            plc_node("0", [0.0, 0.0, 0.0]),
            plc_node("1", [1.0, 0.0, 0.0]),
            plc_node("2", [0.0, 1.0, 0.0]),
            plc_node("3", [1.0, 1.0, 0.0]),
            plc_node("4", [0.5, 0.5, 1.0]),
        ],
        facets: vec![
            facet_with_material("facet_a_1", ["0", "2", "3"], "face_a_1", "region_a"),
            facet_with_material("facet_a_2", ["0", "4", "2"], "face_a_2", "region_a"),
            facet_with_material("facet_a_3", ["0", "3", "4"], "face_a_3", "region_a"),
            facet_with_material("facet_b_1", ["1", "3", "2"], "face_b_1", "region_b"),
            facet_with_material("facet_b_2", ["1", "4", "3"], "face_b_2", "region_b"),
            facet_with_material("facet_b_3", ["1", "2", "4"], "face_b_3", "region_b"),
        ],
        protected_edges: Vec::new(),
        validation: PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
    }
}

fn two_region_bipyramid_plc_with_region_a_protected_edge() -> ProtectedBoundaryComplex {
    let mut plc = two_region_bipyramid_plc();
    plc.complex_id = "two_region_bipyramid_with_protected_edge_plc".to_string();
    plc.protected_edges = vec![PlcProtectedEdge {
        edge_id: entity(
            MeshingStage::ProtectedBoundaryComplex,
            "plc_edge_region_a_0_2",
        ),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, "0"),
            entity(MeshingStage::ProtectedBoundaryComplex, "2"),
        ],
        source_edge_id: entity(MeshingStage::CurveMesh, "edge_region_a_0_2"),
    }];
    plc
}

fn two_region_bipyramid_tetrahedron_mesh() -> TetrahedronMesh {
    TetrahedronMesh {
        mesh_id: "two_region_bipyramid_tetrahedron".to_string(),
        nodes: vec![
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                [0.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                [1.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                [0.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                [1.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                [0.5, 0.5, 1.0],
            ),
        ],
        elements: vec![
            Tetrahedron4Element {
                element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_a"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                ],
                material_region_id: "region_a".to_string(),
            },
            Tetrahedron4Element {
                element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_b"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                ],
                material_region_id: "region_b".to_string(),
            },
        ],
        boundary_faces: vec![
            boundary_face("facet_a_1", ["0", "2", "3"], "face_a_1"),
            boundary_face("facet_a_2", ["0", "4", "2"], "face_a_2"),
            boundary_face("facet_a_3", ["0", "3", "4"], "face_a_3"),
            boundary_face("facet_b_1", ["1", "3", "2"], "face_b_1"),
            boundary_face("facet_b_2", ["1", "4", "3"], "face_b_2"),
            boundary_face("facet_b_3", ["1", "2", "4"], "face_b_3"),
        ],
        recovery_complete: false,
        quality_optimized: false,
        evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
    }
}

fn two_element_material_partition_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "two_element_material_partition_plc".to_string(),
        nodes: vec![
            plc_node("0", [0.0, 0.0, 0.0]),
            plc_node("1", [1.0, 0.0, 0.0]),
            plc_node("2", [0.0, 1.0, 0.0]),
            plc_node("3", [1.0, 1.0, 0.0]),
            plc_node("4", [0.5, 0.5, 1.0]),
        ],
        facets: vec![
            facet_with_material("facet_a_1", ["0", "1", "2"], "face_a_1", "region_a"),
            facet_with_material("facet_a_2", ["0", "3", "1"], "face_a_2", "region_a"),
            facet_with_material("facet_a_3", ["0", "2", "4"], "face_a_3", "region_a"),
            facet_with_material("facet_a_4", ["0", "4", "3"], "face_a_4", "region_a"),
            facet_with_material("facet_a_5", ["1", "3", "4"], "face_a_5", "region_a"),
            facet_with_material("facet_a_6", ["1", "4", "2"], "face_a_6", "region_a"),
        ],
        protected_edges: Vec::new(),
        validation: PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
    }
}

fn two_element_material_partition_seed_mesh() -> TetrahedronMesh {
    TetrahedronMesh {
        mesh_id: "two_element_material_partition_seed".to_string(),
        nodes: vec![
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                [0.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                [1.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                [0.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                [1.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                [0.5, 0.5, 1.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "5"),
                [3.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "6"),
                [4.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "7"),
                [3.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "8"),
                [3.0, 0.0, 1.0],
            ),
        ],
        elements: vec![Tetrahedron4Element {
            element_id: entity(MeshingStage::TetrahedronMesh, "support_tetrahedron"),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, "5"),
                entity(MeshingStage::ProtectedBoundaryComplex, "6"),
                entity(MeshingStage::ProtectedBoundaryComplex, "7"),
                entity(MeshingStage::ProtectedBoundaryComplex, "8"),
            ],
            material_region_id: "unrelated_region".to_string(),
        }],
        boundary_faces: Vec::new(),
        recovery_complete: false,
        quality_optimized: false,
        evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
    }
}

fn add_unclassified_region_a_boundary_neighbor(mesh: &mut TetrahedronMesh) {
    mesh.nodes.push(tetrahedron_node(
        entity(MeshingStage::ProtectedBoundaryComplex, "5"),
        [0.5, 0.5, -1.0],
    ));
    mesh.elements.push(Tetrahedron4Element {
        element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_unclassified"),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, "0"),
            entity(MeshingStage::ProtectedBoundaryComplex, "2"),
            entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            entity(MeshingStage::ProtectedBoundaryComplex, "5"),
        ],
        material_region_id: "unclassified".to_string(),
    });
}

fn replace_region_b_with_unclassified_region_a_interior_neighbor(mesh: &mut TetrahedronMesh) {
    mesh.nodes.push(tetrahedron_node(
        entity(MeshingStage::ProtectedBoundaryComplex, "5"),
        [0.5, 0.5, -1.0],
    ));
    mesh.elements[1] = Tetrahedron4Element {
        element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_unclassified"),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, "2"),
            entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            entity(MeshingStage::ProtectedBoundaryComplex, "4"),
            entity(MeshingStage::ProtectedBoundaryComplex, "5"),
        ],
        material_region_id: "unclassified".to_string(),
    };
    mesh.boundary_faces
        .retain(|face| face.source_face_id.id.starts_with("face_a"));
}

fn interior_material_interface_propagation_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "interior_material_interface_propagation_plc".to_string(),
        nodes: vec![
            plc_node("0", [0.0, 0.0, 0.0]),
            plc_node("1", [1.0, 0.0, 0.0]),
            plc_node("2", [0.0, 1.0, 0.0]),
            plc_node("3", [0.0, 0.0, 1.0]),
            plc_node("4", [0.0, 0.0, -1.0]),
            plc_node("5", [2.0, 0.0, 0.0]),
            plc_node("6", [2.0, 1.0, 0.0]),
            plc_node("7", [2.0, 0.0, 1.0]),
        ],
        facets: vec![
            facet_with_material("facet_a", ["0", "1", "2"], "face_a", "region_a"),
            facet_with_material("facet_b", ["5", "6", "7"], "face_b", "region_b"),
        ],
        protected_edges: Vec::new(),
        validation: PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
    }
}

fn interior_material_interface_propagation_mesh() -> TetrahedronMesh {
    TetrahedronMesh {
        mesh_id: "interior_material_interface_propagation_tetrahedron".to_string(),
        nodes: vec![
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                [0.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                [1.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                [0.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                [0.0, 0.0, 1.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                [0.0, 0.0, -1.0],
            ),
        ],
        elements: vec![
            Tetrahedron4Element {
                element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_seed"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                ],
                material_region_id: "unclassified".to_string(),
            },
            Tetrahedron4Element {
                element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_interior"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                ],
                material_region_id: "unclassified".to_string(),
            },
        ],
        boundary_faces: Vec::new(),
        recovery_complete: false,
        quality_optimized: false,
        evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
    }
}

fn plc_node(id: &str, coordinates_m: [f64; 3]) -> PlcNode {
    PlcNode {
        node_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
        coordinates_m,
    }
}

fn facet(id: &str, node_ids: [&str; 3], source_face_id: &str) -> PlcFacet {
    facet_with_material(id, node_ids, source_face_id, "solid_body")
}

fn facet_with_material(
    id: &str,
    node_ids: [&str; 3],
    source_face_id: &str,
    material_interface_id: &str,
) -> PlcFacet {
    PlcFacet {
        facet_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[0]),
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[1]),
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[2]),
        ],
        source_face_id: entity(MeshingStage::SurfaceMesh, source_face_id),
        material_interface_ids: vec![material_interface_id.to_string()],
    }
}

fn boundary_face(id: &str, node_ids: [&str; 3], source_face_id: &str) -> TetrahedronBoundaryFace {
    let node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, node_ids[0]),
        entity(MeshingStage::ProtectedBoundaryComplex, node_ids[1]),
        entity(MeshingStage::ProtectedBoundaryComplex, node_ids[2]),
    ];
    TetrahedronBoundaryFace {
        face_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
        source_edge_ids: source_edge_ids(node_ids.clone()),
        node_ids,
        source_face_id: entity(MeshingStage::SurfaceMesh, source_face_id),
    }
}

fn tetrahedron_node(node_id: TopologyEntityId, coordinates_m: [f64; 3]) -> TetrahedronMeshNode {
    TetrahedronMeshNode {
        node_id,
        coordinates_m,
    }
}

fn sorted_face_ids(mut node_ids: [TopologyEntityId; 3]) -> [TopologyEntityId; 3] {
    node_ids.sort();
    node_ids
}

fn source_edge_ids(node_ids: [TopologyEntityId; 3]) -> [Option<TopologyEntityId>; 3] {
    [
        source_edge_id_for_edge([node_ids[0].clone(), node_ids[1].clone()]),
        source_edge_id_for_edge([node_ids[1].clone(), node_ids[2].clone()]),
        source_edge_id_for_edge([node_ids[2].clone(), node_ids[0].clone()]),
    ]
}

fn source_edge_id_for_edge(mut node_ids: [TopologyEntityId; 2]) -> Option<TopologyEntityId> {
    node_ids.sort();
    (node_ids
        == [
            entity(MeshingStage::ProtectedBoundaryComplex, "0"),
            entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        ])
    .then(|| entity(MeshingStage::CurveMesh, "edge_1"))
}

fn entity(stage: MeshingStage, id: &str) -> TopologyEntityId {
    TopologyEntityId {
        stage,
        id: id.to_string(),
    }
}
