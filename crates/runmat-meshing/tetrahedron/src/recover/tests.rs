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
fn recovery_stage_result_keeps_mesh_unrecovered_when_queue_has_missing_items() {
    let plc = tetrahedron_plc();
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[0].source_face_id = entity(MeshingStage::SurfaceMesh, "other");

    let result = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("missing source faces should be reported as recovery evidence");

    assert!(!result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.recovery_queue.evidence.status,
        StageEvidenceStatus::Failed
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_items"],
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
    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::SourceEdge
            && item.status == TetrahedronRecoveryStatus::Missing
            && item
                .source_entity_id
                .as_ref()
                .is_some_and(|source| source.id == "edge_2")
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

fn plc_node(id: &str, coordinates_m: [f64; 3]) -> PlcNode {
    PlcNode {
        node_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
        coordinates_m,
    }
}

fn facet(id: &str, node_ids: [&str; 3], source_face_id: &str) -> PlcFacet {
    PlcFacet {
        facet_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[0]),
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[1]),
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[2]),
        ],
        source_face_id: entity(MeshingStage::SurfaceMesh, source_face_id),
        material_interface_ids: vec!["solid_body".to_string()],
    }
}

fn boundary_face(id: &str, node_ids: [&str; 3], source_face_id: &str) -> TetrahedronBoundaryFace {
    TetrahedronBoundaryFace {
        face_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[0]),
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[1]),
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[2]),
        ],
        source_face_id: entity(MeshingStage::SurfaceMesh, source_face_id),
    }
}

fn tetrahedron_node(node_id: TopologyEntityId, coordinates_m: [f64; 3]) -> TetrahedronMeshNode {
    TetrahedronMeshNode {
        node_id,
        coordinates_m,
    }
}

fn entity(stage: MeshingStage, id: &str) -> TopologyEntityId {
    TopologyEntityId {
        stage,
        id: id.to_string(),
    }
}
