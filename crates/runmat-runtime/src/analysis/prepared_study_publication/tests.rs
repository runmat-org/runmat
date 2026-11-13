use super::super::storage::AnalysisArtifactStore;
use super::*;
use runmat_analysis_core::{
    AssessmentFunctionEvaluation, AssessmentFunctionOutcome, AssessmentResult, AssessmentStatus,
};
use std::sync::Arc;
struct Receipt {
    id: String,
    valid: bool,
}
impl AssessmentExecutionReceipt for Receipt {
    fn assessment_id(&self) -> &str {
        &self.id
    }
    fn program_identity(&self) -> Digest {
        Digest::sha256("admitted-program")
    }
    fn validate(
        &self,
        _: &PublishedStudyAssessment,
    ) -> Result<AssessmentFunctionEvaluation, String> {
        if !self.valid {
            return Err("substituted evidence".into());
        }
        Ok(AssessmentFunctionEvaluation::Completed(
            AssessmentFunctionOutcome {
                evaluated: true,
                accepted: false,
            },
        ))
    }
}
fn fixture() -> (
    PreparedStudyPublication,
    Arc<super::super::storage::InMemoryAnalysisArtifactStore>,
) {
    let (compiled, mut published, mut definition) = super::super::assessment::tests::fixture();
    definition.rule = runmat_analysis_core::AssessmentRule::Function {
        function: runmat_analysis_core::AssessmentFunctionReference {
            module_path: "verify.m".into(),
            symbol: "verify".into(),
            source_digest: [1; 32],
        },
        parameters: Vec::new(),
    };
    published.assessments = vec![PublishedStudyAssessment {
        result: AssessmentResult {
            assessment_id: definition.assessment_id.clone(),
            output_id: definition.output_id.clone(),
            status: AssessmentStatus::Indeterminate,
            measured: Some(
                runmat_analysis_core::AssessmentQuantity::try_from_dimension(
                    runmat_types::PhysicalDimension::LENGTH,
                    None,
                    0.002,
                )
                .expect("finite length assessment quantity"),
            ),
            issue: Some(AssessmentIssue::FunctionExecutionUnavailable),
        },
        definition,
        study_definition: compiled.definition_digest().unwrap(),
        run_id: Some("immutable-run".into()),
        evidence: None,
        function_execution_identity: None,
        function_execution_diagnostic: None,
    }];
    let store = Arc::new(super::super::storage::InMemoryAnalysisArtifactStore::new());
    let mut run = super::super::analysis_run_linear_static_with_options(
        &super::super::tests::sample_model(),
        runmat_analysis_fea::ComputeBackend::Cpu,
        Default::default(),
        crate::operations::OperationContext::new(None, None),
    )
    .unwrap()
    .data;
    run.run_id = "immutable-run".into();
    let step = runmat_analysis_core::ProcedureStep::structural_static("step_default_static");
    published.steps = vec![super::super::AnalysisStudyStepRunData {
        ordinal: 0,
        execution: super::super::AnalysisExecutionStep {
            step_id: step.step_id.clone(),
            backend: runmat_analysis_fea::ComputeBackend::Cpu,
            accelerator_admission: Default::default(),
            solver_options: super::super::AnalysisSolverOptions::defaults_for(&step).unwrap(),
        },
        procedure_step: step,
        identity: run.run.identity.clone(),
        run_operation: super::super::AnalysisOperation::LinearStatic.name().into(),
        run_op_version: super::super::AnalysisOperation::LinearStatic
            .version()
            .into(),
        run_id: run.run_id.clone(),
        run_status: run.run_status,
        numerically_acceptable: run.numerically_acceptable,
        solver_convergence: run.solver_convergence,
        result_quality: run.result_quality,
        quality_reasons: run.quality_reasons.clone(),
        provenance: run.provenance.clone(),
        numerical_result: super::super::AnalysisStudyStepNumericalResultData::Standard,
        evidence_artifact_path: String::new(),
    }];
    let definition = compiled.definition_digest().unwrap();
    let staging =
        super::super::storage::AnalysisPublicationBuilder::new(definition, 128 * 1024 * 1024)
            .unwrap();
    staging.stage_run(&run).unwrap();
    let manifest =
        serde_json::json!({"study_definition_digest": definition, "public_run": published});
    (
        PreparedStudyPublication {
            completion: DeferredStudyRun {
                published: OperationEnvelope::new(
                    "test-publication",
                    "1",
                    &crate::operations::OperationContext::new(None, None),
                    published,
                ),
                staging,
                store: store.clone(),
                definition,
            },
            manifest,
        },
        store,
    )
}

#[test]
fn invalid_receipt_sets_never_persist_a_partial_manifest() {
    let _guard = super::super::tests::analysis_test_guard();
    for receipts in [
        vec![],
        vec![Receipt {
            id: "other".into(),
            valid: true,
        }],
        vec![Receipt {
            id: "deflection".into(),
            valid: false,
        }],
        vec![
            Receipt {
                id: "deflection".into(),
                valid: true,
            },
            Receipt {
                id: "deflection".into(),
                valid: true,
            },
        ],
    ] {
        let (publication, persisted) = fixture();
        assert!(publication.complete(receipts).is_err());
        assert!(persisted.list_runs().unwrap().is_empty());
    }
}
#[test]
fn complete_receipts_publish_once_with_audited_engineering_failure() {
    let _guard = super::super::tests::analysis_test_guard();
    let (publication, persisted) = fixture();
    let completed = publication
        .complete(vec![Receipt {
            id: "deflection".into(),
            valid: true,
        }])
        .unwrap();
    assert_eq!(persisted.list_runs().unwrap().len(), 1);
    assert_eq!(
        completed.data.assessments[0].result.status,
        AssessmentStatus::Failed
    );
    assert_eq!(
        completed.data.assessments[0].function_execution_identity,
        Some(Digest::sha256("admitted-program"))
    );
    assert_eq!(
        completed.data.result_manifest_artifact_path.as_deref(),
        None
    );
    assert_eq!(
        completed.data.run_status,
        super::super::RunStatus::Completed
    );
}
#[test]
fn cancellation_after_function_admission_publishes_no_children() {
    let _guard = super::super::tests::analysis_test_guard();
    let (publication, store) = fixture();
    let _interrupt = crate::interrupt::replace_interrupt(Some(
        runmat_execution::CancellationToken::from_shared(Arc::new(
            std::sync::atomic::AtomicBool::new(true),
        )),
    ));
    assert!(publication
        .complete(vec![Receipt {
            id: "deflection".into(),
            valid: true
        }])
        .is_err());
    assert!(store.list_runs().unwrap().is_empty());
}

#[test]
fn changing_the_global_backend_cannot_redirect_a_prepared_commit() {
    let _guard = super::super::tests::analysis_test_guard();
    let (publication, original) = fixture();
    let previous = super::super::storage::capture_artifact_store().unwrap();
    let replacement = Arc::new(super::super::storage::InMemoryAnalysisArtifactStore::new());
    super::super::storage::set_artifact_store_for_tests(replacement.clone());
    let completed = publication.complete(vec![Receipt {
        id: "deflection".into(),
        valid: true,
    }]);
    super::super::storage::set_artifact_store_for_tests(previous);
    completed.unwrap();
    assert_eq!(original.list_runs().unwrap().len(), 1);
    assert!(replacement.list_runs().unwrap().is_empty());
}

mod preparation;
