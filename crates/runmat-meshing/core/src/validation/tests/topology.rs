use super::*;
use fixtures::*;

#[test]
fn accepts_face_connected_volume_components_within_budget() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.nodes.push(AnalysisMeshNode {
        node_id: 5,
        coordinates_m: [0.0, 0.0, -1.0],
        provenance: Vec::new(),
    });
    mesh.volume_elements.push(AnalysisVolumeElement {
        element_id: "e2".to_string(),
        kind: VolumeElementKind::Tetrahedron4,
        node_ids: vec![1, 3, 2, 5],
        material_region_id: "mat_region".to_string(),
        provenance: Vec::new(),
    });

    assert_eq!(volume_component_count(&mesh), 1);
    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            max_volume_component_count: Some(1),
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("face-connected tetrahedra should remain one volume component");
}

#[test]
fn rejects_unintended_isolated_volume_components() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.nodes.extend([
        AnalysisMeshNode {
            node_id: 5,
            coordinates_m: [10.0, 0.0, 0.0],
            provenance: Vec::new(),
        },
        AnalysisMeshNode {
            node_id: 6,
            coordinates_m: [11.0, 0.0, 0.0],
            provenance: Vec::new(),
        },
        AnalysisMeshNode {
            node_id: 7,
            coordinates_m: [10.0, 1.0, 0.0],
            provenance: Vec::new(),
        },
        AnalysisMeshNode {
            node_id: 8,
            coordinates_m: [10.0, 0.0, 1.0],
            provenance: Vec::new(),
        },
    ]);
    mesh.volume_elements.push(AnalysisVolumeElement {
        element_id: "e2".to_string(),
        kind: VolumeElementKind::Tetrahedron4,
        node_ids: vec![5, 6, 7, 8],
        material_region_id: "mat_region".to_string(),
        provenance: Vec::new(),
    });

    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            max_volume_component_count: Some(1),
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("isolated volume component should fail");

    assert_eq!(
        err,
        AnalysisMeshValidationError::VolumeComponentCountExceeded {
            component_count: 2,
            max_component_count: 1,
        }
    );
}

#[test]
fn rejects_unsupported_element_kind_until_assembly_exists() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.volume_elements[0].kind = VolumeElementKind::Hex8;
    mesh.volume_elements[0].node_ids = vec![1, 2, 3, 4, 1, 2, 3, 4];
    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("unsupported element kind should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::UnsupportedVolumeElementKind {
            element_id: "e1".to_string()
        }
    );
}

#[test]
fn rejects_missing_material_coverage() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.volume_elements[0].material_region_id.clear();
    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("missing material region should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingMaterialRegion {
            element_id: "e1".to_string()
        }
    );
}

#[test]
fn rejects_unclassified_material_ownership() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.volume_elements[0].material_region_id = "unclassified".to_string();
    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("unresolved material ownership should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::UnclassifiedMaterialRegion {
            element_id: "e1".to_string()
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "unclassified_material_region"
    );
}

#[test]
fn rejects_unmapped_boundary_nodes() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.boundary_faces[0].node_ids = vec![1, 2, 99];
    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("unknown boundary node should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::UnknownBoundaryFaceNode {
            face_id: "f1".to_string(),
            node_id: 99
        }
    );
}

#[test]
fn rejects_unmapped_boundary_edge_nodes() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.boundary_edges = vec![AnalysisBoundaryEdge {
        edge_id: "edge1".to_string(),
        node_ids: [1, 99],
        adjacent_boundary_face_ids: vec!["f1".to_string()],
        region_ids: Vec::new(),
        provenance: Vec::new(),
    }];
    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("unknown boundary edge node should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::UnknownBoundaryEdgeNode {
            edge_id: "edge1".to_string(),
            node_id: 99
        }
    );
}

#[test]
fn rejects_boundary_edge_adjacent_to_unknown_face() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.boundary_edges = vec![AnalysisBoundaryEdge {
        edge_id: "edge1".to_string(),
        node_ids: [1, 2],
        adjacent_boundary_face_ids: vec!["missing_face".to_string()],
        region_ids: Vec::new(),
        provenance: Vec::new(),
    }];
    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("unknown boundary edge adjacent face should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::UnknownBoundaryEdgeAdjacentFace {
            edge_id: "edge1".to_string(),
            face_id: "missing_face".to_string()
        }
    );
}
