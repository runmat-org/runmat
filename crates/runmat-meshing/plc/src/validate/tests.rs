use super::*;
use runmat_meshing_core::contracts::{
    MeshingStage, PlcFacet, PlcNode, PlcProtectedEdge, ProtectedBoundaryComplex, StageEvidence,
};

#[test]
fn validates_closed_manifold_plc() {
    let summary = validate_protected_boundary_complex(&tetrahedron_plc())
        .expect("closed manifold PLC should validate");

    assert!(summary.valid_for_volume_meshing());
}

#[test]
fn rejects_not_volume_ready_validation_summary() {
    let mut plc = tetrahedron_plc();
    plc.validation.watertight = false;

    assert_eq!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::ValidationSummaryNotVolumeReady {
            summary: plc.validation,
        })
    );
}

#[test]
fn rejects_open_boundary_edge() {
    let mut plc = tetrahedron_plc();
    plc.facets.pop();

    assert!(matches!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::OpenBoundaryEdge { .. })
    ));
}

#[test]
fn rejects_inconsistent_boundary_edge_orientation() {
    let mut plc = tetrahedron_plc();
    plc.facets[0].node_ids = [entity("0"), entity("1"), entity("2")];

    assert!(matches!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::InconsistentBoundaryEdgeOrientation { .. })
    ));
}

#[test]
fn rejects_facet_that_references_unknown_node() {
    let mut plc = tetrahedron_plc();
    plc.facets[0].node_ids[0] = entity("missing");

    assert!(matches!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::FacetReferencesUnknownNode { .. })
    ));
}

#[test]
fn rejects_facet_with_empty_source_face_id() {
    let mut plc = tetrahedron_plc();
    plc.facets[0].source_face_id = source_face("");

    assert!(matches!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::FacetHasEmptySourceFaceId { .. })
    ));
}

#[test]
fn rejects_facet_with_non_surface_source_face_id() {
    let mut plc = tetrahedron_plc();
    plc.facets[0].source_face_id = entity("source_face");

    assert!(matches!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::FacetSourceFaceStageMismatch { .. })
    ));
}

#[test]
fn rejects_empty_material_interface_id() {
    let mut plc = tetrahedron_plc();
    plc.facets[0].material_interface_ids = vec!["".to_string()];

    assert!(matches!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::FacetHasEmptyMaterialInterfaceId { .. })
    ));
}

#[test]
fn rejects_repeated_material_interface_id_on_facet() {
    let mut plc = tetrahedron_plc();
    plc.facets[0].material_interface_ids = vec!["body".to_string(), "body".to_string()];

    assert!(matches!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::FacetHasRepeatedMaterialInterfaceId { .. })
    ));
}

#[test]
fn rejects_unreferenced_node() {
    let mut plc = tetrahedron_plc();
    plc.nodes.push(node("4", [2.0, 2.0, 2.0]));

    assert!(matches!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::UnreferencedNode { .. })
    ));
}

#[test]
fn rejects_protected_edge_that_references_unknown_node() {
    let mut plc = tetrahedron_plc();
    plc.protected_edges.push(PlcProtectedEdge {
        edge_id: entity("edge_missing"),
        node_ids: [entity("0"), entity("missing")],
        source_edge_id: source_edge("source_edge"),
    });

    assert!(matches!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::ProtectedEdgeReferencesUnknownNode { .. })
    ));
}

#[test]
fn rejects_protected_edge_that_is_not_a_boundary_edge() {
    let mut plc = octahedron_plc();
    plc.protected_edges.push(PlcProtectedEdge {
        edge_id: entity("pole_to_pole"),
        node_ids: [entity("0"), entity("5")],
        source_edge_id: source_edge("source_edge"),
    });

    assert!(matches!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::ProtectedEdgeNotOnBoundary { .. })
    ));
}

#[test]
fn rejects_protected_edge_with_empty_source_edge_id() {
    let mut plc = tetrahedron_plc();
    plc.protected_edges.push(PlcProtectedEdge {
        edge_id: entity("edge_empty_source"),
        node_ids: [entity("0"), entity("1")],
        source_edge_id: source_edge(""),
    });

    assert!(matches!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::ProtectedEdgeHasEmptySourceEdgeId { .. })
    ));
}

#[test]
fn rejects_protected_edge_with_non_curve_source_edge_id() {
    let mut plc = tetrahedron_plc();
    plc.protected_edges.push(PlcProtectedEdge {
        edge_id: entity("edge_wrong_stage_source"),
        node_ids: [entity("0"), entity("1")],
        source_edge_id: entity("source_edge"),
    });

    assert!(matches!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::ProtectedEdgeSourceEdgeStageMismatch { .. })
    ));
}

#[test]
fn rejects_disconnected_boundary_components_until_shell_nesting_is_classified() {
    let plc = disconnected_tetrahedra_plc();

    assert!(matches!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::DisconnectedBoundaryComponents { component_count: 2 })
    ));
}

fn tetrahedron_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "tetrahedron_plc".to_string(),
        nodes: vec![
            node("0", [0.0, 0.0, 0.0]),
            node("1", [1.0, 0.0, 0.0]),
            node("2", [0.0, 1.0, 0.0]),
            node("3", [0.0, 0.0, 1.0]),
        ],
        facets: vec![
            facet("f0", ["0", "2", "1"]),
            facet("f1", ["0", "1", "3"]),
            facet("f2", ["1", "2", "3"]),
            facet("f3", ["2", "0", "3"]),
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

fn disconnected_tetrahedra_plc() -> ProtectedBoundaryComplex {
    let mut plc = tetrahedron_plc();
    plc.complex_id = "disconnected_tetrahedra_plc".to_string();
    plc.nodes.extend([
        node("10", [3.0, 0.0, 0.0]),
        node("11", [4.0, 0.0, 0.0]),
        node("12", [3.0, 1.0, 0.0]),
        node("13", [3.0, 0.0, 1.0]),
    ]);
    plc.facets.extend([
        facet("f10", ["10", "12", "11"]),
        facet("f11", ["10", "11", "13"]),
        facet("f12", ["11", "12", "13"]),
        facet("f13", ["12", "10", "13"]),
    ]);
    plc
}

fn octahedron_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "octahedron_plc".to_string(),
        nodes: vec![
            node("0", [0.0, 0.0, 1.0]),
            node("1", [1.0, 0.0, 0.0]),
            node("2", [0.0, 1.0, 0.0]),
            node("3", [-1.0, 0.0, 0.0]),
            node("4", [0.0, -1.0, 0.0]),
            node("5", [0.0, 0.0, -1.0]),
        ],
        facets: vec![
            facet("f0", ["0", "1", "2"]),
            facet("f1", ["0", "2", "3"]),
            facet("f2", ["0", "3", "4"]),
            facet("f3", ["0", "4", "1"]),
            facet("f4", ["5", "2", "1"]),
            facet("f5", ["5", "3", "2"]),
            facet("f6", ["5", "4", "3"]),
            facet("f7", ["5", "1", "4"]),
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

fn node(id: &str, coordinates_m: [f64; 3]) -> PlcNode {
    PlcNode {
        node_id: entity(id),
        coordinates_m,
    }
}

fn facet(id: &str, node_ids: [&str; 3]) -> PlcFacet {
    PlcFacet {
        facet_id: entity(id),
        node_ids: [
            entity(node_ids[0]),
            entity(node_ids[1]),
            entity(node_ids[2]),
        ],
        source_face_id: source_face(id),
        material_interface_ids: Vec::new(),
    }
}

fn entity(id: &str) -> TopologyEntityId {
    TopologyEntityId {
        stage: MeshingStage::ProtectedBoundaryComplex,
        id: id.to_string(),
    }
}

fn source_edge(id: &str) -> TopologyEntityId {
    TopologyEntityId {
        stage: MeshingStage::CurveMesh,
        id: id.to_string(),
    }
}

fn source_face(id: &str) -> TopologyEntityId {
    TopologyEntityId {
        stage: MeshingStage::SurfaceMesh,
        id: id.to_string(),
    }
}
