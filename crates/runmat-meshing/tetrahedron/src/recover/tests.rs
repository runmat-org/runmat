use super::*;
use runmat_meshing_core::contracts::{
    PlcFacet, PlcNode, PlcProtectedEdge, PlcValidationSummary, Tetrahedron4Element,
    TetrahedronBoundaryFace, TetrahedronMeshNode, TopologyEntityId,
};

#[test]
fn builds_recovery_queue_for_recovered_plc_constraints() {
    let queue =
        build_recovery_queue_from_plc(&single_facet_plc(), &single_facet_tetrahedron_mesh())
            .expect("matching Tetrahedron mesh should recover PLC constraints");

    assert_eq!(queue.items.len(), 3);
    assert_eq!(queue.evidence.stage, MeshingStage::ConstraintRecovery);
    assert_eq!(queue.evidence.entity_counts["source_face_items"], 1);
    assert_eq!(queue.evidence.entity_counts["source_edge_items"], 1);
    assert_eq!(queue.evidence.entity_counts["material_interface_items"], 1);
    assert!(queue
        .items
        .iter()
        .all(|item| item.status == TetrahedronRecoveryStatus::Recovered));
}

#[test]
fn recovery_queue_rejects_missing_source_face() {
    let mut mesh = single_facet_tetrahedron_mesh();
    mesh.boundary_faces[0].source_face_id = entity(MeshingStage::SurfaceMesh, "other");

    assert_eq!(
        build_recovery_queue_from_plc(&single_facet_plc(), &mesh),
        Err(TetrahedronRecoveryError::MissingSourceFaceRecovery {
            face_id: "face_1".to_string()
        })
    );
}

#[test]
fn recovery_queue_rejects_missing_source_edge() {
    let mut plc = single_facet_plc();
    plc.protected_edges[0].node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        entity(MeshingStage::ProtectedBoundaryComplex, "3"),
    ];

    assert_eq!(
        build_recovery_queue_from_plc(&plc, &single_facet_tetrahedron_mesh()),
        Err(TetrahedronRecoveryError::MissingSourceEdgeRecovery {
            edge_id: "edge_1".to_string()
        })
    );
}

#[test]
fn recovery_queue_rejects_missing_material_interface() {
    let mut mesh = single_facet_tetrahedron_mesh();
    mesh.elements[0].material_region_id = "other_body".to_string();

    assert_eq!(
        build_recovery_queue_from_plc(&single_facet_plc(), &mesh),
        Err(TetrahedronRecoveryError::MissingMaterialInterfaceRecovery {
            material_interface_id: "solid_body".to_string()
        })
    );
}

fn single_facet_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "single_facet_plc".to_string(),
        nodes: vec![
            plc_node("0", [0.0, 0.0, 0.0]),
            plc_node("1", [1.0, 0.0, 0.0]),
            plc_node("2", [0.0, 1.0, 0.0]),
        ],
        facets: vec![PlcFacet {
            facet_id: entity(MeshingStage::ProtectedBoundaryComplex, "facet_1"),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
            ],
            source_face_id: entity(MeshingStage::SurfaceMesh, "face_1"),
            material_interface_ids: vec!["solid_body".to_string()],
        }],
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

fn single_facet_tetrahedron_mesh() -> TetrahedronMesh {
    TetrahedronMesh {
        mesh_id: "single_facet_tetrahedron".to_string(),
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
            tetrahedron_node(entity(MeshingStage::TetrahedronMesh, "3"), [0.0, 0.0, 1.0]),
        ],
        elements: vec![Tetrahedron4Element {
            element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_1"),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                entity(MeshingStage::TetrahedronMesh, "3"),
            ],
            material_region_id: "solid_body".to_string(),
        }],
        boundary_faces: vec![TetrahedronBoundaryFace {
            face_id: entity(MeshingStage::ProtectedBoundaryComplex, "facet_1"),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
            ],
            source_face_id: entity(MeshingStage::SurfaceMesh, "face_1"),
        }],
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
