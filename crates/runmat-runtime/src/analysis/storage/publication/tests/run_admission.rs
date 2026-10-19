use super::*;

#[test]
fn run_ceiling_rejection_reports_quality_and_components_without_staging_payloads() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let required = super::super::resident(&run, 128 << 20).unwrap() * 2;
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
