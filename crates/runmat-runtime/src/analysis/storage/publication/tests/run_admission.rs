use super::*;

#[test]
fn run_ceiling_rejection_reports_quality_and_components_without_staging_payloads() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    // The preparation owner's own phase charges are the run's staging requirement.
    let mut required = 0u64;
    crate::analysis::storage::preparation::prepare_run_artifact_with_admission(
        &run,
        "2026-01-01T00:00:00Z".into(),
        &mut |bytes| {
            required += bytes;
            Ok(())
        },
    )
    .unwrap();
    let builder = AnalysisPublicationBuilder::new(
        Digest::sha256(b"run-admission-failure-facts"),
        required - 1,
    )
    .unwrap();
    let error = builder.stage_run(&run).unwrap_err();
    let AnalysisPublicationError::ResourceLimit { message } = error else {
        panic!("expected the unchanged resource rejection")
    };
    assert!(message.contains("run staging facts:"));
    assert!(message.contains(&format!(
        "numerically_acceptable={}",
        run.numerically_acceptable
    )));
    assert!(message.contains("retained_modes=0"));
    assert!(message.contains("modal_nodes=0"));
    assert!(message.contains("modal_projection_bytes="));
    assert!(message.contains("render_projection_bytes="));
    assert!(message.contains(&format!("additional={required}")));
    assert!(builder.store.runs.read().unwrap().is_empty());
    assert!(builder.store.chunks.read().unwrap().is_empty());
    assert!(!*builder.admission.lock().unwrap());
}
