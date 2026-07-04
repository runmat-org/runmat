use super::*;
use fixtures::*;

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
