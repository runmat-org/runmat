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
fn reports_boundary_component_evidence_for_closed_shell() {
    let report = classify_boundary_components(&tetrahedron_plc());
    let shell_classification = classify_shell_nesting(&report);

    assert_eq!(
        report,
        PlcBoundaryComponentReport {
            component_count: 1,
            referenced_node_count: 4,
            min_component_node_count: 4,
            max_component_node_count: 4,
        }
    );
    assert_eq!(
        shell_classification,
        PlcShellClassificationReport {
            shell_nesting_classified: true,
            outer_shell_count: 1,
            nested_shell_count: 0,
            max_nesting_depth: 0,
        }
    );
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
fn rejects_node_with_non_plc_stage() {
    let mut plc = tetrahedron_plc();
    plc.nodes[0].node_id = source_face("wrong_stage_node");

    assert_eq!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::PlcEntityStageMismatch {
            entity_id: source_face("wrong_stage_node"),
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
fn rejects_facet_id_with_non_plc_stage() {
    let mut plc = tetrahedron_plc();
    plc.facets[0].facet_id = source_face("wrong_stage_facet");

    assert_eq!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::PlcEntityStageMismatch {
            entity_id: source_face("wrong_stage_facet"),
        })
    );
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
fn rejects_facet_node_with_non_plc_stage() {
    let mut plc = tetrahedron_plc();
    plc.facets[0].node_ids[0] = source_face("wrong_stage_node");

    assert_eq!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::PlcEntityStageMismatch {
            entity_id: source_face("wrong_stage_node"),
        })
    );
}

#[test]
fn rejects_duplicate_facet_id() {
    let mut plc = tetrahedron_plc();
    plc.facets[1].facet_id = entity("f0");

    assert_eq!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::DuplicateFacet {
            facet_id: entity("f0"),
        })
    );
}

#[test]
fn rejects_duplicate_boundary_facet() {
    let mut plc = tetrahedron_plc();
    plc.facets[1].node_ids = [entity("1"), entity("2"), entity("0")];

    assert_eq!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::DuplicateBoundaryFacet {
            first_facet_id: entity("f0"),
            second_facet_id: entity("f1"),
            node_ids: [entity("0"), entity("1"), entity("2")],
        })
    );
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
fn rejects_protected_edge_id_with_non_plc_stage() {
    let mut plc = tetrahedron_plc();
    plc.protected_edges.push(PlcProtectedEdge {
        edge_id: source_edge("wrong_stage_edge"),
        node_ids: [entity("0"), entity("1")],
        source_edge_id: source_edge("source_edge_01"),
    });

    assert_eq!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::PlcEntityStageMismatch {
            entity_id: source_edge("wrong_stage_edge"),
        })
    );
}

#[test]
fn rejects_protected_edge_node_with_non_plc_stage() {
    let mut plc = tetrahedron_plc();
    plc.protected_edges.push(PlcProtectedEdge {
        edge_id: entity("edge_wrong_stage_node"),
        node_ids: [entity("0"), source_face("wrong_stage_node")],
        source_edge_id: source_edge("source_edge_01"),
    });

    assert_eq!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::PlcEntityStageMismatch {
            entity_id: source_face("wrong_stage_node"),
        })
    );
}

#[test]
fn rejects_duplicate_protected_edge_id() {
    let mut plc = tetrahedron_plc();
    plc.protected_edges.push(PlcProtectedEdge {
        edge_id: entity("edge_duplicate"),
        node_ids: [entity("0"), entity("1")],
        source_edge_id: source_edge("source_edge_01"),
    });
    plc.protected_edges.push(PlcProtectedEdge {
        edge_id: entity("edge_duplicate"),
        node_ids: [entity("0"), entity("2")],
        source_edge_id: source_edge("source_edge_02"),
    });

    assert_eq!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::DuplicateProtectedEdge {
            edge_id: entity("edge_duplicate"),
        })
    );
}

#[test]
fn rejects_duplicate_protected_boundary_segment() {
    let mut plc = tetrahedron_plc();
    plc.protected_edges.push(PlcProtectedEdge {
        edge_id: entity("edge_01_a"),
        node_ids: [entity("0"), entity("1")],
        source_edge_id: source_edge("source_edge_01_a"),
    });
    plc.protected_edges.push(PlcProtectedEdge {
        edge_id: entity("edge_01_b"),
        node_ids: [entity("1"), entity("0")],
        source_edge_id: source_edge("source_edge_01_b"),
    });

    assert_eq!(
        validate_protected_boundary_complex(&plc),
        Err(PlcValidationError::DuplicateProtectedBoundarySegment {
            first_edge_id: entity("edge_01_a"),
            second_edge_id: entity("edge_01_b"),
            node_ids: [entity("0"), entity("1")],
        })
    );
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
    let report = classify_boundary_components(&plc);

    assert_eq!(report.component_count, 2);
    assert_eq!(report.min_component_node_count, 4);
    assert_eq!(report.max_component_node_count, 4);
    assert_eq!(
        classify_shell_nesting(&report),
        PlcShellClassificationReport {
            shell_nesting_classified: false,
            outer_shell_count: 0,
            nested_shell_count: 0,
            max_nesting_depth: 0,
        }
    );

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
