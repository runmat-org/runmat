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
    mesh.backend.tetrahedron_recovery_item_count = 3;
    mesh.backend.tetrahedron_missing_recovery_item_count = 3;
    mesh.backend.tetrahedron_source_face_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_source_face_recovery_item_count = 1;
    mesh.backend.tetrahedron_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_material_interface_recovery_item_count = 1;
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
        .expect_err("inconsistent aggregate missing recovery evidence should fail readiness");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "missing_items".to_string(),
            aggregate_count: 4,
            typed_count: 1,
        }
    );
}

#[test]
fn rejects_recovery_item_aggregate_that_does_not_match_typed_inputs() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_recovery_item_count = 1;
    mesh.backend.tetrahedron_source_face_recovery_item_count = 1;
    mesh.backend.tetrahedron_source_edge_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("aggregate recovery items must match typed recovery inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "recovery_items".to_string(),
            aggregate_count: 1,
            typed_count: 2,
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "inconsistent_tetrahedron_recovery_aggregate_evidence"
    );
}

#[test]
fn rejects_recovery_status_aggregate_that_does_not_match_total_items() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_recovery_item_count = 2;
    mesh.backend.tetrahedron_source_face_recovery_item_count = 2;
    mesh.backend.tetrahedron_recovered_item_count = 2;
    mesh.backend
        .tetrahedron_recovered_source_face_recovery_item_count = 2;
    mesh.backend.tetrahedron_missing_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_source_face_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("recovered plus missing recovery statuses must match total items");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "recovery_status_items".to_string(),
            aggregate_count: 2,
            typed_count: 3,
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
fn rejects_recovered_cad_curve_source_edge_count_that_exceeds_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_cad_curve_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_recovered_cad_curve_source_edge_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("recovered CAD curve source-edge evidence cannot exceed CAD-backed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryEvidence {
            family: "cad_curve_source_edge".to_string(),
            recovered_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_cad_curve_interior_edge_source_edge_count_that_exceeds_interior_edge_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_interior_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_cad_curve_interior_edge_source_edge_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("CAD curve interior-edge evidence cannot exceed interior-edge inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "cad_curve_interior_edge_source_edge".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_recovered_cad_curve_interior_edge_source_edge_count_that_exceeds_cad_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_interior_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_cad_curve_interior_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_recovered_cad_curve_interior_edge_source_edge_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("recovered CAD curve interior-edge evidence cannot exceed CAD inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryEvidence {
            family: "cad_curve_interior_edge_source_edge".to_string(),
            recovered_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_attempted_source_edge_split_refill_count_that_exceeds_volume_and_interior_inputs() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_source_edge_split_refill_item_count = 2;
    mesh.backend
        .tetrahedron_accepted_source_edge_split_refill_candidate_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("split/refill attempts cannot exceed volume and interior source-edge inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "attempted_source_edge_split_refill".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_source_edge_split_refill_status_count_that_does_not_match_attempted_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_source_edge_split_refill_item_count = 2;
    mesh.backend
        .tetrahedron_accepted_source_edge_split_refill_candidate_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("accepted plus rejected split/refill counts must match attempts");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "source_edge_split_refill_status_items".to_string(),
            aggregate_count: 2,
            typed_count: 1,
        }
    );
}

#[test]
fn rejects_cad_curve_source_edge_split_refill_count_that_exceeds_attempted_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_source_edge_split_refill_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_cad_curve_source_edge_split_refill_item_count = 1;
    mesh.backend
        .tetrahedron_accepted_source_edge_split_refill_candidate_item_count = 1;
    mesh.backend
        .tetrahedron_accepted_cad_curve_source_edge_split_refill_candidate_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("CAD-backed accepted split/refill candidates cannot exceed CAD attempts");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "accepted_cad_curve_source_edge_split_refill_candidate".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_cad_curve_source_edge_status_count_that_does_not_match_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_cad_curve_source_edge_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_recovered_cad_curve_source_edge_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("CAD curve source-edge status counts must reconcile with input count");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "cad_curve_source_edge_status_items".to_string(),
            aggregate_count: 2,
            typed_count: 1,
        }
    );
}

#[test]
fn rejects_cad_curve_source_edge_missing_reason_count_that_does_not_match_missing_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_cad_curve_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_cad_curve_source_edge_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("CAD curve source-edge missing reasons must reconcile with missing count");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "cad_curve_source_edge_missing_reason_items".to_string(),
            aggregate_count: 1,
            typed_count: 0,
        }
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
fn rejects_recovered_material_interface_count_that_exceeds_typed_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_material_interface_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_recovered_material_interface_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("recovered material-interface evidence cannot exceed typed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryEvidence {
            family: "material_interface".to_string(),
            recovered_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_recovered_absent_material_partition_count_that_exceeds_typed_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_missing_material_interface_absent_partition_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_recovered_absent_partition_material_interface_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("recovered absent material partition evidence cannot exceed typed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryEvidence {
            family: "absent_partition_material_interface".to_string(),
            recovered_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_attempted_absent_source_edge_count_that_exceeds_typed_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_absent_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_absent_source_edge_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("attempted absent source-edge evidence cannot exceed typed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "attempted_absent_source_edge".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "inconsistent_tetrahedron_recovery_item_evidence"
    );
}

#[test]
fn rejects_cad_curve_absent_source_edge_reconnection_count_that_exceeds_attempted_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_absent_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_absent_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_cad_curve_absent_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_reconnected_cad_curve_absent_source_edge_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default()).expect_err(
        "reconnected CAD curve absent source-edge evidence cannot exceed CAD-backed attempts",
    );

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "reconnected_cad_curve_absent_source_edge".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn allows_two_protected_edge_boundary_face_restoration_attempts_per_volume_edge_input() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count = 2;

    validate_analysis_mesh(&mesh, Default::default())
        .expect("one manifold protected-edge input can restore two adjacent boundary faces");
}

#[test]
fn rejects_protected_edge_boundary_face_restoration_attempts_beyond_manifold_edge_bound() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count = 3;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("protected-edge boundary restoration attempts are bounded by adjacent facets");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "attempted_protected_edge_boundary_face_restoration".to_string(),
            item_count: 3,
            input_count: 2,
        }
    );
}

#[test]
fn rejects_rejected_protected_edge_boundary_face_restoration_count_that_exceeds_attempted_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count = 2;
    mesh.backend
        .tetrahedron_rejected_protected_edge_boundary_face_restoration_item_count = 3;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("rejected protected-edge restoration count cannot exceed attempts");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "rejected_protected_edge_boundary_face_restoration".to_string(),
            item_count: 3,
            input_count: 2,
        }
    );
}

#[test]
fn rejects_cad_curve_protected_edge_boundary_face_restoration_count_that_exceeds_attempted_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_cad_curve_protected_edge_boundary_face_restoration_item_count = 1;
    mesh.backend
        .tetrahedron_recovered_cad_curve_protected_edge_boundary_face_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default()).expect_err(
        "recovered CAD curve protected-edge boundary-face evidence cannot exceed CAD attempts",
    );

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "recovered_cad_curve_protected_edge_boundary_face".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_rejected_absent_source_edge_count_that_exceeds_attempted_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_absent_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_absent_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_rejected_absent_source_edge_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("rejected absent source-edge evidence cannot exceed attempts");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "rejected_absent_source_edge".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_attempted_volume_face_source_face_restoration_count_that_exceeds_typed_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_face_source_face_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_volume_face_source_face_boundary_restoration_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("attempted source-face boundary restoration cannot exceed typed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "attempted_volume_face_source_face_boundary_restoration".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_attempted_material_interface_count_that_exceeds_typed_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_material_interface_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_material_interface_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("attempted material-interface evidence cannot exceed typed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "attempted_material_interface".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_attempted_absent_material_partition_count_that_exceeds_typed_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_missing_material_interface_absent_partition_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_absent_material_partition_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("attempted absent material partition evidence cannot exceed typed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "attempted_absent_material_partition".to_string(),
            item_count: 2,
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
