use super::*;

fn request() -> AnalysisStudyStepRequest {
    let model = sample_model();
    let procedure_step = model.sequence.steps[0].clone();
    AnalysisStudyStepRequest {
        execution: AnalysisExecutionStep {
            step_id: procedure_step.step_id.clone(),
            backend: ComputeBackend::Gpu,
            accelerator_admission: runmat_analysis_core::AcceleratorAdmissionPolicy::Required,
            solver_options: AnalysisSolverOptions::defaults_for(&procedure_step).unwrap(),
        },
        model,
        procedure_step,
        solver_mesh_artifact_path: None,
        solver_mesh: None,
        require_accelerator: true,
        context: OperationContext::new(Some("original-trace".into()), None),
    }
}

#[test]
fn owned_step_keeps_original_backend_policy_and_options_and_requires_admitted_scope() {
    let original = request();
    let bytes = serde_json::to_vec(&original).unwrap();
    let roundtrip: AnalysisStudyStepRequest = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(roundtrip.execution, original.execution);
    assert_eq!(roundtrip.model, original.model);
    assert!(roundtrip.require_accelerator);
    let error = analysis_execute_study_step(roundtrip).unwrap_err();
    assert_eq!(error.error_code, "RM.FEA.SignInRequired");
}

#[test]
fn owned_dispatch_preserves_cfd_typed_refusal_and_rejects_foreign_step() {
    let _host = analysis_test_guard();
    let mut request = request();
    request.execution.solver_options =
        AnalysisSolverOptions::IncompressibleFlow(Default::default());
    let scope = fea_run_admission::FeaRunScope::admit_with(
        fea_run_admission::FeaRunRequest::local(
            fea_run_admission::FeaRunKind::Solve,
            1000,
            runmat_execution::Digest::sha256(b"owned-step-test"),
        )
        .unwrap(),
        fea_run_admission::entrypoints::test_admission().as_ref(),
    )
    .unwrap();
    let error = scope
        .run_sync(|| analysis_execute_study_step(request.clone()))
        .unwrap_err();
    assert_eq!(error.error_code, "RM.FEA.RUN_CFD.AUTHORED_STUDY_REQUIRED");
    request.execution.step_id = ProcedureStepId("different_step".into());
    let error = scope
        .run_sync(|| analysis_execute_study_step(request))
        .unwrap_err();
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_STUDY.SEQUENCE_CONTRACT_FAILED"
    );
    scope
        .settle(fea_run_admission::FeaRunOutcome::Failed)
        .unwrap();
}
