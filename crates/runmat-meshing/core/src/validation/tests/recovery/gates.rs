use super::*;

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
