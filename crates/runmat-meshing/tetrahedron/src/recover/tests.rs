use super::*;
use runmat_meshing_core::contracts::{
    PlcFacet, PlcNode, PlcProtectedEdge, PlcValidationSummary, Tetrahedron4Element,
    TetrahedronBoundaryFace, TetrahedronMeshNode, TopologyEntityId,
};

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
        result.recovery_queue.evidence.entity_counts["missing_items"],
        0
    );
}

#[test]
fn recovery_stage_result_recovers_missing_exterior_boundary_face_before_audit() {
    let mut mesh = tetrahedron_mesh();
    let missing_face = mesh.boundary_faces.remove(2);

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
        result.recovery_queue.evidence.entity_counts["recovered_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_source_edge_items"],
        0
    );
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

    assert_eq!(
        recover_tetrahedron_mesh_from_plc(&plc, mesh),
        Err(TetrahedronRecoveryError::IncompleteRecovery {
            missing_item_count: 1,
            missing_source_face_item_count: 0,
            missing_source_edge_item_count: 0,
            missing_material_interface_item_count: 1,
        })
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
        result.recovery_queue.evidence.entity_counts["missing_material_interface_items"],
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
    mesh.elements[0].node_ids[0] = entity(MeshingStage::ProtectedBoundaryComplex, "4");

    assert_eq!(
        recover_tetrahedron_mesh_from_plc(&plc, mesh),
        Err(TetrahedronRecoveryError::IncompleteRecovery {
            missing_item_count: 1,
            missing_source_face_item_count: 1,
            missing_source_edge_item_count: 0,
            missing_material_interface_item_count: 0,
        })
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
    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::SourceFace
            && item.status == TetrahedronRecoveryStatus::Missing
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
