use super::*;

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
fn rejects_applied_source_edge_split_refill_count_that_exceeds_accepted_candidates() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_source_edge_split_refill_item_count = 1;
    mesh.backend
        .tetrahedron_accepted_source_edge_split_refill_candidate_item_count = 1;
    mesh.backend
        .tetrahedron_applied_source_edge_split_refill_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("applied split/refill edits cannot exceed accepted candidates");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "applied_source_edge_split_refill".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_cad_curve_applied_source_edge_split_refill_count_that_exceeds_cad_accepted_count() {
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
        .tetrahedron_accepted_cad_curve_source_edge_split_refill_candidate_item_count = 1;
    mesh.backend
        .tetrahedron_post_repair_attempted_source_edge_split_refill_item_count = 1;
    mesh.backend
        .tetrahedron_applied_source_edge_split_refill_item_count = 1;
    mesh.backend
        .tetrahedron_applied_cad_curve_source_edge_split_refill_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("CAD-backed applied split/refill edits cannot exceed CAD accepted candidates");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "applied_cad_curve_source_edge_split_refill".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_post_repair_source_edge_split_refill_attempt_count_that_exceeds_accepted_candidates() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_source_edge_split_refill_item_count = 1;
    mesh.backend
        .tetrahedron_accepted_source_edge_split_refill_candidate_item_count = 1;
    mesh.backend
        .tetrahedron_post_repair_attempted_source_edge_split_refill_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("post-repair split/refill attempts cannot exceed accepted candidates");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "post_repair_attempted_source_edge_split_refill".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_applied_source_edge_split_refill_count_that_exceeds_post_repair_attempts() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_source_edge_split_refill_item_count = 2;
    mesh.backend
        .tetrahedron_accepted_source_edge_split_refill_candidate_item_count = 2;
    mesh.backend
        .tetrahedron_post_repair_attempted_source_edge_split_refill_item_count = 1;
    mesh.backend
        .tetrahedron_applied_source_edge_split_refill_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("applied split/refill edits cannot exceed post-repair attempts");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "applied_source_edge_split_refill".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_post_repair_source_edge_split_refill_status_count_that_does_not_match_attempted_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_source_edge_split_refill_item_count = 2;
    mesh.backend
        .tetrahedron_accepted_source_edge_split_refill_candidate_item_count = 2;
    mesh.backend
        .tetrahedron_post_repair_attempted_source_edge_split_refill_item_count = 2;
    mesh.backend
        .tetrahedron_applied_source_edge_split_refill_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("applied plus rejected post-repair split/refill counts must match attempts");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "post_repair_source_edge_split_refill_status_items".to_string(),
            aggregate_count: 2,
            typed_count: 1,
        }
    );
}

#[test]
fn rejects_post_repair_cad_curve_source_edge_split_refill_status_count_that_does_not_match_attempted_count(
) {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_source_edge_split_refill_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_cad_curve_source_edge_split_refill_item_count = 1;
    mesh.backend
        .tetrahedron_accepted_source_edge_split_refill_candidate_item_count = 2;
    mesh.backend
        .tetrahedron_accepted_cad_curve_source_edge_split_refill_candidate_item_count = 1;
    mesh.backend
        .tetrahedron_post_repair_attempted_source_edge_split_refill_item_count = 2;
    mesh.backend
        .tetrahedron_post_repair_attempted_cad_curve_source_edge_split_refill_item_count = 1;
    mesh.backend
        .tetrahedron_applied_source_edge_split_refill_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default()).expect_err(
        "CAD applied plus rejected post-repair split/refill counts must match CAD attempts",
    );

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "post_repair_cad_curve_source_edge_split_refill_status_items".to_string(),
            aggregate_count: 1,
            typed_count: 0,
        }
    );
}
