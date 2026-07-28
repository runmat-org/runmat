use super::*;
use runmat_meshing_core::contracts::{
    MeshingStage, PlcFacet, PlcNode, PlcProtectedEdge, PlcProtectedEdgeCadCurveBoundary,
    PlcValidationSummary, StageEvidence, StageEvidenceStatus, Tetrahedron4Element,
    TetrahedronBoundaryFace, TetrahedronMeshNode, TopologyEntityId,
};

mod boundary_leaks;
mod fixtures;
use fixtures::*;
mod input_validation;
mod material_interfaces;
mod source_edge_recovery;
mod source_face_recovery;

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
fn recovery_queue_classifies_recovered_cad_curve_source_edges() {
    let queue = build_recovery_queue_from_plc(&cad_curve_tetrahedron_plc(), &tetrahedron_mesh())
        .expect("matching Tetrahedron mesh should recover CAD curve PLC constraints");

    assert_eq!(
        queue.evidence.entity_counts["cad_curve_source_edge_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["recovered_cad_curve_source_edge_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_cad_curve_source_edge_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_cad_curve_source_edge_topology_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_cad_curve_source_edge_provenance_items"],
        0
    );
}

#[test]
fn recovery_queue_classifies_missing_cad_curve_source_edge_provenance() {
    let mut mesh = tetrahedron_mesh();
    for boundary_face in &mut mesh.boundary_faces {
        boundary_face.source_edge_ids = [None, None, None];
    }

    let queue = build_recovery_queue_from_plc(&cad_curve_tetrahedron_plc(), &mesh)
        .expect("boundary topology with missing CAD curve provenance should build a queue");

    assert_eq!(
        queue.evidence.entity_counts["cad_curve_source_edge_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["recovered_cad_curve_source_edge_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_cad_curve_source_edge_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_cad_curve_source_edge_topology_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_cad_curve_source_edge_provenance_items"],
        1
    );
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
fn recovery_stage_result_rejects_duplicate_boundary_face_topology() {
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces.push(boundary_face(
        "extra_boundary_face",
        ["1", "0", "2"],
        "other",
    ));

    assert_eq!(
        recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh),
        Err(
            TetrahedronRecoveryError::DuplicateTetrahedronBoundaryFaceTopology {
                face_id: entity(
                    MeshingStage::ProtectedBoundaryComplex,
                    "extra_boundary_face"
                ),
                existing_face_id: entity(MeshingStage::ProtectedBoundaryComplex, "facet_1"),
            }
        )
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
    mesh.boundary_faces = vec![
        boundary_face("facet_3", ["1", "2", "3"], "face_3"),
        boundary_face("generated_1_2_4", ["1", "2", "4"], "generated_face_1_2_4"),
        boundary_face("generated_1_3_4", ["1", "3", "4"], "generated_face_1_3_4"),
        boundary_face("generated_2_3_4", ["2", "3", "4"], "generated_face_2_3_4"),
    ];

    let recovery_evidence = assert_incomplete_recovery(
        recover_tetrahedron_mesh_from_plc(&plc, mesh).expect_err("missing face should fail"),
        4,
        3,
        1,
        0,
    );
    assert_eq!(
        recovery_evidence.entity_counts["missing_source_face_absent_face_items"],
        3
    );
    assert_eq!(
        recovery_evidence.entity_counts["missing_source_edge_absent_edge_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["removed_unsupported_boundary_faces"],
        0
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
fn recovery_queue_accepts_subdivided_boundary_source_face_coverage() {
    let mut mesh = tetrahedron_mesh();
    let split_node_id = entity(MeshingStage::TetrahedronMesh, "split_edge_1_2");
    mesh.nodes
        .push(tetrahedron_node(split_node_id.clone(), [0.5, 0.5, 0.0]));
    mesh.elements = vec![
        Tetrahedron4Element {
            element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_1a"),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                split_node_id.clone(),
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            ],
            material_region_id: "solid_body".to_string(),
        },
        Tetrahedron4Element {
            element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_1b"),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                split_node_id.clone(),
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            ],
            material_region_id: "solid_body".to_string(),
        },
    ];
    mesh.boundary_faces = vec![
        boundary_face_from_ids(
            "child_face_1a",
            [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                split_node_id.clone(),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
            ],
            "face_1",
            [None, None, None],
        ),
        boundary_face_from_ids(
            "child_face_1b",
            [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                split_node_id.clone(),
            ],
            "face_1",
            [None, None, None],
        ),
        boundary_face("facet_2", ["0", "1", "3"], "face_2"),
        boundary_face_from_ids(
            "child_face_3a",
            [
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                split_node_id.clone(),
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            ],
            "face_3",
            [None, None, None],
        ),
        boundary_face_from_ids(
            "child_face_3b",
            [
                split_node_id,
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            ],
            "face_3",
            [None, None, None],
        ),
        boundary_face("facet_4", ["2", "0", "3"], "face_4"),
    ];

    let mut plc = tetrahedron_plc();
    plc.protected_edges.clear();
    let queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("subdivided boundary source faces should be accepted as recovered coverage");

    assert_eq!(queue.evidence.status, StageEvidenceStatus::Complete);
    assert_eq!(queue.evidence.entity_counts["missing_items"], 0);
    assert_eq!(queue.evidence.entity_counts["source_face_items"], 4);
    assert_eq!(queue.evidence.entity_counts["missing_source_face_items"], 0);
}

#[test]
fn recovery_queue_reports_partial_boundary_source_face_provenance() {
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[0].source_face_id = entity(MeshingStage::SurfaceMesh, "other");

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
    mesh.boundary_faces.retain(|face| {
        !(face
            .node_ids
            .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "0"))
            && face
                .node_ids
                .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "2")))
    });

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
        [2.0, 1.0, 0.0],
    ));
    mesh.elements[0].node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        entity(MeshingStage::ProtectedBoundaryComplex, "3"),
        entity(MeshingStage::ProtectedBoundaryComplex, "4"),
    ];
    mesh.boundary_faces = vec![
        boundary_face("facet_2", ["0", "1", "3"], "face_2"),
        boundary_face("generated_0_1_4", ["0", "1", "4"], "generated_face_0_1_4"),
        boundary_face("generated_0_3_4", ["0", "3", "4"], "generated_face_0_3_4"),
        boundary_face("generated_1_3_4", ["1", "3", "4"], "generated_face_1_3_4"),
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
