use super::*;
use crate::{
    contracts::{AnalysisBoundaryEdge, AnalysisMeshNode, AnalysisVolumeElement, VolumeElementKind},
    quality::ElementQuality,
};

mod fixtures;

use fixtures::*;

#[test]
fn accepts_minimal_valid_tetrahedron4_mesh() {
    let mesh = valid_tetrahedron_mesh();
    validate_analysis_mesh(&mesh, QualityThresholds::default()).expect("mesh should validate");
}

#[test]
fn validation_options_round_trip_with_required_regions() {
    let options = AnalysisMeshValidationOptions {
        required_boundary_region_ids: vec!["fixed".to_string(), "loaded".to_string()],
        required_material_region_ids: vec!["solid".to_string()],
        max_volume_element_count: Some(42),
        coverage_sample_points_m: vec![[0.25, 0.25, 0.25]],
        min_coverage_sample_ratio: 0.75,
        require_no_unrecovered_tetrahedron_components: true,
        ..AnalysisMeshValidationOptions::default()
    };

    let encoded = serde_json::to_value(&options).expect("validation options should serialize");
    let decoded: AnalysisMeshValidationOptions =
        serde_json::from_value(encoded).expect("validation options should deserialize");

    assert_eq!(decoded, options);
}

#[test]
fn rejects_empty_volume_elements() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.volume_elements.clear();
    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("empty volume elements should fail");
    assert_eq!(err, AnalysisMeshValidationError::EmptyVolumeElements);
}

#[test]
fn rejects_mesh_that_exceeds_element_budget() {
    let mesh = valid_tetrahedron_mesh();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            max_volume_element_count: Some(0),
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("element budget overrun should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::ElementBudgetExceeded {
            element_count: 1,
            max_element_count: 0,
        }
    );
}

#[test]
fn rejects_unrecovered_tetrahedron_components_recovery_when_policy_requires_strict_recovery() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_unrecovered_component_count = 1;

    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            require_no_unrecovered_tetrahedron_components: true,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("strict recovery policy should reject unrecovered Tetrahedron components evidence");

    assert_eq!(
        err,
        AnalysisMeshValidationError::UnrecoveredTetrahedronComponentsPresent { component_count: 1 }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "unrecovered_tetrahedron_components_present"
    );
}

#[test]
fn rejects_unrepaired_exact_quality_when_policy_requires_strict_recovery() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_exact_quality_unrepaired_boundary_adjacent_count = 2;
    mesh.backend
        .tetrahedron_exact_quality_unrepaired_node_adjacent_count = 4;
    mesh.backend
        .tetrahedron_exact_quality_unrepaired_interior_seed_count = 3;
    mesh.backend
        .tetrahedron_exact_quality_unrepaired_edge_star_count = 5;

    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            require_no_unrepaired_exact_quality: true,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("strict recovery policy should reject unrepaired exact-quality evidence");

    assert_eq!(
        err,
        AnalysisMeshValidationError::UnrepairedExactQualityPresent {
            total_count: 5,
            general_cavity_count: 0,
            boundary_adjacent_count: 2,
            node_adjacent_count: 4,
            interior_seed_count: 3,
            edge_star_count: 5,
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "unrepaired_exact_quality_present"
    );
}

#[test]
fn rejects_unrepaired_general_cavity_exact_quality_when_policy_requires_strict_recovery() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_exact_quality_unrepaired_total_count = 1;
    mesh.backend
        .tetrahedron_exact_quality_unrepaired_general_cavity_count = 1;

    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            require_no_unrepaired_exact_quality: true,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("strict recovery policy should reject unclassified cavity evidence");

    assert_eq!(
        err,
        AnalysisMeshValidationError::UnrepairedExactQualityPresent {
            total_count: 1,
            general_cavity_count: 1,
            boundary_adjacent_count: 0,
            node_adjacent_count: 0,
            interior_seed_count: 0,
            edge_star_count: 0,
        }
    );
}

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
fn rejects_missing_boundary_edge_recovery_when_required() {
    let mesh = valid_tetrahedron_mesh();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            min_boundary_edge_recovery_ratio: 1.0,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("missing boundary edge recovery should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::BoundaryEdgeRecoveryFailed {
            recovery_ratio: "0.000000".to_string(),
            required_ratio: "1.000000".to_string(),
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

#[test]
fn rejects_quality_threshold_failures() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.quality.min_scaled_jacobian = 0.01;
    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("low jacobian should fail");
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "quality_threshold_failed"
    );
    assert_eq!(
        err,
        AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "min_scaled_jacobian".to_string()
        }
    );
}

#[test]
fn rejects_exact_quality_threshold_failures() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.quality.min_exact_scaled_jacobian = 0.01;
    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("low exact jacobian should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "min_exact_scaled_jacobian".to_string()
        }
    );
}

#[test]
fn rejects_element_exact_quality_threshold_failures() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.quality.elements.push(ElementQuality {
        element_id: "e1".to_string(),
        scaled_jacobian: 0.8,
        exact_scaled_jacobian: 0.01,
        aspect_ratio: 1.0,
        volume_m3: 1.0 / 6.0,
    });
    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("low element exact jacobian should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "element_exact_scaled_jacobian".to_string()
        }
    );
}

#[test]
fn rejects_boundary_projection_quality_threshold_failures() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.quality.max_boundary_projection_error_m = 2.0e-6;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("boundary projection error should fail");

    assert_eq!(
        err,
        AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "max_boundary_projection_error_m".to_string()
        }
    );
}

#[test]
fn rejects_mesh_that_underfills_expected_bounds() {
    let mesh = valid_tetrahedron_mesh();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            expected_bounds_m: Some([[0.0, 0.0, 0.0], [4.0, 1.0, 1.0]]),
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("mesh should fail bounds coverage");
    assert_eq!(
        err,
        AnalysisMeshValidationError::BoundsCoverageFailed {
            axis: 0,
            coverage_ratio: "0.250000".to_string(),
            required_ratio: "0.900000".to_string(),
        }
    );
}

#[test]
fn rejects_mesh_that_underfills_expected_volume() {
    let mesh = valid_tetrahedron_mesh();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            expected_volume_m3: Some(1.0),
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("mesh should fail volume coverage");
    assert_eq!(
        err,
        AnalysisMeshValidationError::VolumeCoverageFailed {
            coverage_ratio: "0.166667".to_string(),
            required_ratio: "0.900000".to_string(),
        }
    );
}

#[test]
fn rejects_uncovered_interior_coverage_samples() {
    let mesh = valid_tetrahedron_mesh();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            coverage_sample_points_m: vec![[0.1, 0.1, 0.1], [2.0, 2.0, 2.0]],
            min_coverage_sample_ratio: 1.0,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("uncovered interior coverage sample should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::CoverageSampleFailed {
            coverage_ratio: "0.500000".to_string(),
            required_ratio: "1.000000".to_string(),
        }
    );
}

#[test]
fn accepts_covered_interior_coverage_samples() {
    let mesh = valid_tetrahedron_mesh();
    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            coverage_sample_points_m: vec![[0.1, 0.1, 0.1]],
            min_coverage_sample_ratio: 1.0,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("covered interior coverage sample should pass");
}

#[test]
fn rejects_nearby_uncovered_samples_for_small_tetrahedra() {
    let mut mesh = valid_tetrahedron_mesh();
    for node in &mut mesh.nodes {
        for coordinate in &mut node.coordinates_m {
            *coordinate *= 1.0e-3;
        }
    }
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            coverage_sample_points_m: vec![[1.01e-3, 1.0e-6, 1.0e-6]],
            min_coverage_sample_ratio: 1.0,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("sample outside a small tetrahedron should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::CoverageSampleFailed {
            coverage_ratio: "0.000000".to_string(),
            required_ratio: "1.000000".to_string(),
        }
    );
}

#[test]
fn rejects_mesh_that_underfills_expected_boundary_area() {
    let mesh = valid_tetrahedron_mesh();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            expected_boundary_area_m2: Some(2.0),
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("mesh should fail boundary area coverage");
    assert_eq!(
        err,
        AnalysisMeshValidationError::BoundaryAreaCoverageFailed {
            area_ratio: "0.250000".to_string(),
            required_ratio: "0.900000".to_string(),
        }
    );
}

#[test]
fn rejects_unrecovered_boundary_faces_when_required() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.boundary_faces[0].adjacent_volume_element_ids.clear();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            min_boundary_face_recovery_ratio: 1.0,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("missing boundary recovery should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::BoundaryFaceRecoveryFailed {
            recovery_ratio: "0.000000".to_string(),
            required_ratio: "1.000000".to_string(),
        }
    );
}

#[test]
fn rejects_missing_required_boundary_region() {
    let mesh = valid_tetrahedron_mesh();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_boundary_region_ids: vec!["loaded".to_string()],
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("missing boundary region should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingRequiredBoundaryRegion {
            region_id: "loaded".to_string()
        }
    );
}

#[test]
fn rejects_required_boundary_region_without_recovered_face() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.boundary_faces[0].adjacent_volume_element_ids.clear();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_boundary_region_ids: vec!["fixed".to_string()],
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("unrecovered boundary region should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingRequiredBoundaryRegionRecovery {
            region_id: "fixed".to_string()
        }
    );
}

#[test]
fn rejects_missing_required_material_region() {
    let mesh = valid_tetrahedron_mesh();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_material_region_ids: vec!["rib".to_string()],
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("missing material region should fail");
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "missing_required_material_region"
    );
    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingRequiredMaterialRegion {
            region_id: "rib".to_string()
        }
    );
}

#[test]
fn rejects_required_material_region_without_positive_volume() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.nodes[3].coordinates_m = mesh.nodes[0].coordinates_m;
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_material_region_ids: vec!["mat_region".to_string()],
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("zero-volume material region should fail");
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "missing_required_material_region_coverage"
    );
    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingRequiredMaterialRegionCoverage {
            region_id: "mat_region".to_string()
        }
    );
}
