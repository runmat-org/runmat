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
fn rejects_rolled_back_material_interface_partition_recovery() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_rolled_back_absent_material_partition_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_rolled_back_absent_material_partition_element_count = 2;
    mesh.backend
        .tetrahedron_rolled_back_absent_material_partition_boundary_face_count = 3;

    let err = validate_analysis_mesh_with_options(&mesh, AnalysisMeshValidationOptions::default())
        .expect_err("rolled-back material-interface partition recovery should fail readiness");

    assert_eq!(
        err,
        AnalysisMeshValidationError::RolledBackMaterialInterfacePartitionRecoveryPresent {
            recovery_item_count: 1,
            element_count: 2,
            boundary_face_count: 3,
            post_insertion_audit_rejection_count: 0,
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "rolled_back_material_interface_partition_recovery_present"
    );
}

#[test]
fn rejects_material_interface_partition_post_insertion_audit_rejection() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_rejected_absent_material_partition_post_insertion_audit_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("post-insertion audit rejection should fail readiness");

    assert_eq!(
        err,
        AnalysisMeshValidationError::RolledBackMaterialInterfacePartitionRecoveryPresent {
            recovery_item_count: 0,
            element_count: 0,
            boundary_face_count: 0,
            post_insertion_audit_rejection_count: 1,
        }
    );
}

#[test]
fn rejects_incomplete_tetrahedron_recovery() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_missing_recovery_item_count = 3;
    mesh.backend
        .tetrahedron_missing_source_face_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_material_interface_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("remaining Tetrahedron recovery items should fail readiness");

    assert_eq!(
        err,
        AnalysisMeshValidationError::IncompleteTetrahedronRecoveryPresent {
            missing_item_count: 3,
            missing_source_face_item_count: 1,
            missing_source_edge_item_count: 1,
            missing_material_interface_item_count: 1,
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "incomplete_tetrahedron_recovery_present"
    );
}

#[test]
fn rejects_incomplete_tetrahedron_recovery_from_aggregate_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_missing_recovery_item_count = 4;
    mesh.backend
        .tetrahedron_missing_source_face_recovery_item_count = 1;

    let err = validate_analysis_mesh_with_options(&mesh, AnalysisMeshValidationOptions::default())
        .expect_err("aggregate missing recovery evidence should fail readiness");

    assert_eq!(
        err,
        AnalysisMeshValidationError::IncompleteTetrahedronRecoveryPresent {
            missing_item_count: 4,
            missing_source_face_item_count: 1,
            missing_source_edge_item_count: 0,
            missing_material_interface_item_count: 0,
        }
    );
}

#[test]
fn rejects_recovered_source_edge_count_that_exceeds_typed_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_boundary_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_recovered_boundary_edge_source_edge_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("recovered source-edge evidence cannot exceed typed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryEvidence {
            family: "boundary_edge_source_edge".to_string(),
            recovered_count: 2,
            input_count: 1,
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "inconsistent_tetrahedron_recovery_evidence"
    );
}

#[test]
fn rejects_recovered_source_face_count_that_exceeds_typed_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_face_source_face_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_recovered_volume_face_source_face_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("recovered source-face evidence cannot exceed typed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryEvidence {
            family: "volume_face_source_face".to_string(),
            recovered_count: 2,
            input_count: 1,
        }
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
