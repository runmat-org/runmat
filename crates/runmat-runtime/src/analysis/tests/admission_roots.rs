//! Local FEA roots require lease admission before numerical/provider effects.
use super::*;
use crate::analysis::fea_run_admission::{
    FeaRunAdmission, FeaRunAdmissionContext, FeaRunDenied, FeaRunPermit, FeaRunRequest,
};

#[derive(Default)]
struct DenyingHost(Mutex<Vec<FeaRunRequest>>);
impl FeaRunAdmission for DenyingHost {
    fn admit(&self, request: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
        self.0.lock().unwrap().push(request.clone());
        Err(FeaRunDenied::SignInRequired)
    }
}

fn assert_billing_denied(error: OperationErrorEnvelope, root: &str) {
    assert_eq!(
        error.error_code, "RM.FEA.SignInRequired",
        "{root}: {}",
        error.message
    );
}

type ModelRoot = fn(
    &AnalysisModel,
    ComputeBackend,
    OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope>;

#[test]
fn local_model_roots_require_admission_after_pure_domain_validation() {
    let _host_state = analysis_test_guard();
    let host = Arc::new(DenyingHost::default());
    let context = FeaRunAdmissionContext::new(host.clone());
    let model = sample_model();
    let roots: [(&str, ModelRoot); 14] = [
        ("linear", analysis_run_linear_static_op),
        ("linear options", |m, b, c| {
            analysis_run_linear_static_with_options(m, b, AnalysisRunOptions::default(), c)
        }),
        ("modal", analysis_run_modal_op),
        ("modal options", |m, b, c| {
            analysis_run_modal_with_options_op(m, b, AnalysisModalRunOptions::default(), c)
        }),
        ("acoustic", analysis_run_acoustic_op),
        ("acoustic options", |m, b, c| {
            analysis_run_acoustic_with_options_op(m, b, AnalysisAcousticRunOptions::default(), c)
        }),
        ("transient", analysis_run_transient_op),
        ("transient options", |m, b, c| {
            analysis_run_transient_with_options_op(m, b, AnalysisTransientRunOptions::default(), c)
        }),
        ("thermal", analysis_run_thermal_op),
        ("thermal options", |m, b, c| {
            analysis_run_thermal_with_options_op(m, b, AnalysisThermalRunOptions::default(), c)
        }),
        ("nonlinear", analysis_run_nonlinear_op),
        ("nonlinear options", |m, b, c| {
            analysis_run_nonlinear_with_options_op(m, b, AnalysisNonlinearRunOptions::default(), c)
        }),
        ("cfd", analysis_run_cfd_op),
        ("cfd options", |m, b, c| {
            analysis_run_cfd_with_options_op(m, b, AnalysisCfdRunOptions::default(), c)
        }),
    ];
    for (name, execute) in roots {
        let error = context
            .run_sync(|| {
                execute(
                    &model,
                    ComputeBackend::Cpu,
                    OperationContext::new(None, None),
                )
            })
            .expect_err("cancelled or incompatible model must not complete");
        assert!(
            matches!(
                error.error_type,
                OperationErrorType::Capacity | OperationErrorType::Validation
            ),
            "{name}: local admission must preserve cancellation/domain guards: {}",
            error.error_code
        );
        if error.error_type != OperationErrorType::Validation {
            assert_billing_denied(error, name);
        }
    }
    let requests = host.0.lock().unwrap();
    assert!(
        !requests.is_empty(),
        "valid local public roots must require the installed lease provider"
    );
    let admitted = requests.len();
    drop(requests);

    // Retired reduced coupled-flow model roots remain validation-only.
    for error in context.run_sync(|| {
        [
            analysis_run_cht_op(&model, OperationContext::new(None, None)).unwrap_err(),
            analysis_run_fsi_op(&model, OperationContext::new(None, None)).unwrap_err(),
        ]
    }) {
        assert!(error.error_code.ends_with(".AUTHORED_STUDY_REQUIRED"));
        assert_eq!(error.error_type, OperationErrorType::Validation);
    }
    assert_eq!(host.0.lock().unwrap().len(), admitted);
}

struct RejectExecution;
impl StructuralLinearExecutor for RejectExecution {
    fn execute(
        &self,
        _: &runmat_analysis_fea::execution::StructuralLinearWorkload,
        _: &runmat_meshing_core::SolverMeshArtifact,
    ) -> Result<runmat_analysis_fea::execution::StructuralLinearWorkloadResult, String> {
        Err("local numerical executor rejected the workload".into())
    }
}
impl StructuralStudyPlacementExecutor for RejectExecution {
    fn allocated_cpu_millicores(&self, _: usize) -> u32 {
        1000
    }
    fn execute<'a>(
        &'a self,
        _: &'a [StructuralStudyPlacementEntry<'a>],
    ) -> StructuralStudyPlacementFuture<'a> {
        Box::pin(async { Err("local placement executor rejected the workload".into()) })
    }
}

#[test]
fn local_study_placement_and_meshing_refuse_before_provider_effects() {
    let _host_state = analysis_test_guard();
    let meshing_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls = Arc::clone(&meshing_calls);
    let _meshing = crate::meshing::replace_meshing_execution_provider(Some(Arc::new(move |_| {
        calls.fetch_add(1, Ordering::SeqCst);
        Err("offline local meshing provider reached".into())
    })))
    .unwrap();
    let host = Arc::new(DenyingHost::default());
    let context = FeaRunAdmissionContext::new(host.clone());
    // Bind the model to a canonical source identity before provider admission.
    // The rejecting fixture never opens source bytes or produces a mesh.
    let mut geometry = closed_cube_geometry_asset();
    geometry.source.sha256 = hex::encode(
        runmat_execution::Digest::sha256(b"offline local meshing admission fixture").bytes(),
    );
    let study = linear_static_study_spec_for_geometry(geometry);
    let sweep = AnalysisStudySweepSpec {
        sweep_id: "denied_roots".into(),
        studies: vec![study.clone()],
        // This cohort exercises terminal refusals, not collected sweep outcomes.
        fail_fast: true,
    };
    let operation_context = || OperationContext::new(None, None);
    for (name, error) in context.run_sync(|| {
        [
            (
                "study",
                analysis_run_study_op(&study, operation_context()).unwrap_err(),
            ),
            (
                "study executor",
                analysis_run_study_with_structural_executor_op(
                    &study,
                    operation_context(),
                    &RejectExecution,
                )
                .unwrap_err(),
            ),
            (
                "study sweep",
                analysis_run_study_sweep_op(&sweep, operation_context()).unwrap_err(),
            ),
        ]
    }) {
        assert_billing_denied(error, name);
    }
    assert_eq!(meshing_calls.load(Ordering::SeqCst), 0);
    let placement_errors = pollster::block_on(context.run_future(async {
        [
            analysis_execute_structural_study_with_placement(
                &study,
                operation_context(),
                &RejectExecution,
            )
            .await
            .unwrap_err(),
            analysis_execute_structural_sweep_with_placement(
                &sweep,
                operation_context(),
                &RejectExecution,
            )
            .await
            .unwrap_err(),
        ]
    }));
    for error in placement_errors {
        assert_billing_denied(error, "placement");
    }
    let error = context
        .run_sync(|| {
            crate::meshing::execute_meshing(&crate::meshing::MeshingExecutionRequest {
                source_path: "must-not-be-read.step".into(),
                source_units: None,
                expected_source_sha256: [1; 32],
                settings: runmat_meshing_core::MeshingRequestSettings::default(),
                contact_definitions: Vec::new(),
            })
        })
        .unwrap_err();
    assert!(matches!(
        error,
        crate::meshing::MeshingExecutionError::Admission(FeaRunDenied::SignInRequired)
    ));
    let requests = host.0.lock().unwrap();
    assert_eq!(
        requests.len(),
        6,
        "each denied public root is admitted before work"
    );
    assert_eq!(meshing_calls.load(Ordering::SeqCst), 0);
}
