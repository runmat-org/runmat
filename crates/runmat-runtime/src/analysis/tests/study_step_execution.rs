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

#[test]
fn owned_study_sweep_commits_mesh_results_and_evidence_on_async_only_storage() {
    struct NumericalExecutor {
        reject_step: bool,
    }
    impl AnalysisStudyStepExecutor for NumericalExecutor {
        fn allocated_cpu_millicores(&self) -> u64 {
            1000
        }
        fn execute_step<'a>(
            &'a self,
            mut request: AnalysisStudyStepRequest,
        ) -> AnalysisStudyStepFuture<'a> {
            Box::pin(async move {
                assert!(storage::scoped_artifact_provider().is_none());
                // Suspend at the actual numerical boundary. No private result
                // transaction or scoped filesystem authority may cross it.
                let mut yielded = false;
                futures::future::poll_fn(move |cx| {
                    if std::mem::replace(&mut yielded, true) {
                        std::task::Poll::Ready(())
                    } else {
                        cx.waker().wake_by_ref();
                        std::task::Poll::Pending
                    }
                })
                .await;
                assert!(storage::scoped_artifact_provider().is_none());
                assert!(request.solver_mesh.is_some());
                if self.reject_step {
                    request.execution.step_id = ProcedureStepId("foreign_step".into());
                }
                analysis_execute_study_step(request)
            })
        }
    }

    let _host = analysis_test_guard();
    let provider = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::new());
    let _provider = runmat_filesystem::replace_provider(provider.clone());
    let root = PathBuf::from("/owned-async-study");
    let _config = scoped_fea_artifact_roots(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .unwrap();
    assert_eq!(
        runmat_filesystem::FsProvider::workspace_access(provider.as_ref(), &root, true, 1024)
            .err()
            .unwrap()
            .kind(),
        std::io::ErrorKind::Unsupported
    );

    let mesh = ready_minimal_analysis_mesh();
    let evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&mesh);
    let _mesher = crate::meshing::replace_meshing_execution_provider(Some(std::sync::Arc::new(
        move |_request: &crate::meshing::MeshingExecutionRequest| {
            Ok(crate::meshing::MeshingExecutionOutput {
                artifact: mesh.clone(),
                evidence: evidence.clone(),
            })
        },
    )))
    .unwrap();
    let mut study = sample_linear_static_study_spec();
    study.geometry.source.sha256 = "ab".repeat(32);
    let canonical = |target: &mut RegionTarget| {
        if matches!(target.region_id.as_str(), "root" | "tip") {
            target.region_id = persistent_face(&target.region_id)
                .analysis_selection_id()
                .unwrap();
        }
    };
    for constraint in &mut study.model.constraints {
        constraint.visit_targets_mut(canonical);
    }
    for action in &mut study.model.excitations {
        if let ExcitationTarget::Region(target) = &mut action.excitation_mut().target {
            canonical(target);
        }
    }
    let sweep = AnalysisStudySweepSpec {
        sweep_id: "owned_async_sweep".into(),
        studies: vec![study],
        fail_fast: true,
    };
    let context = OperationContext::new(
        Some("owned-async-trace".into()),
        Some("owned-async-request".into()),
    );
    let result = futures::executor::block_on(analysis_execute_study_sweep_with_step_executor(
        &sweep,
        context.clone(),
        &NumericalExecutor { reject_step: false },
    ))
    .unwrap();
    assert_eq!(result.data.success_count, 1);
    let run = &result.data.run_entries[0].result;
    for path in [
        run.solver_mesh_artifact_path.as_ref().unwrap(),
        run.meshing_evidence_artifact_path.as_ref().unwrap(),
        &run.evidence_artifact_path,
        &result.data.evidence_artifact_path,
    ] {
        assert!(!provider
            .read_project_path(PathBuf::from(path))
            .unwrap()
            .is_empty());
    }
    let persisted = futures::executor::block_on(storage::with_artifact_access(false, |store| {
        store.load_run(&run.steps[0].run_id)
    }))
    .unwrap()
    .unwrap()
    .unwrap();
    assert_eq!(persisted.run_id, run.steps[0].run_id);
    assert!(storage::scoped_artifact_provider().is_none());

    // Aggregate run.json is intentionally mutable: a replacement stages through
    // the same private provider, without an immutable
    // collision or a write escaping to the process-wide provider.
    let summary = PathBuf::from(&result.data.evidence_artifact_path);
    let original = provider.read_project_path(summary.clone()).unwrap();
    let mut replacement = original.clone();
    replacement.push(b'\n');
    futures::executor::block_on(storage::with_artifact_access(true, |_| {
        super::super::evidence_filesystem::write_captured_evidence(&summary, replacement.clone())
    }))
    .unwrap()
    .unwrap();
    assert_eq!(provider.read_project_path(summary).unwrap(), replacement);

    let error = futures::executor::block_on(analysis_execute_study_with_step_executor(
        &sweep.studies[0],
        context,
        &NumericalExecutor { reject_step: true },
    ))
    .unwrap_err();
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_STUDY.SEQUENCE_CONTRACT_FAILED"
    );
    assert_eq!(error.trace_id.as_deref(), Some("owned-async-trace"));
    assert_eq!(error.request_id.as_deref(), Some("owned-async-request"));
    assert!(storage::scoped_artifact_provider().is_none());
}

/// A structural study with no solver mesh, no meshing settings and no explicit beam/shell
/// elements: what `fea.study(id, geometry, model)` builds without the `Mesh` option.
fn unmeshed_modal_study_spec() -> AnalysisStudySpec {
    let mut spec = sample_linear_static_study_spec();
    let procedure_step = ProcedureStep::structural_modal("step_default_modal");
    spec.model.sequence =
        AnalysisSequence::from_steps("unmeshed_modal", vec![procedure_step.clone()]);
    spec.model.signals.clear();
    spec.model.excitations.clear();
    spec.model.excitation_cases.clear();
    spec.execution = AnalysisExecutionPlan {
        steps: vec![AnalysisExecutionStep {
            step_id: procedure_step.step_id.clone(),
            backend: ComputeBackend::Cpu,
            accelerator_admission: Default::default(),
            solver_options: AnalysisSolverOptions::defaults_for(&procedure_step).unwrap(),
        }],
    };
    spec.meshing_settings = None;
    spec.solver_mesh_artifact_path = None;
    assert!(spec
        .model
        .structural
        .as_ref()
        .is_none_or(|structural| structural.elements.is_empty()));
    spec
}

#[test]
fn unmeshed_structural_study_is_refused_before_any_step_by_every_host() {
    struct UnreachableExecutor;
    impl AnalysisStudyStepExecutor for UnreachableExecutor {
        fn allocated_cpu_millicores(&self) -> u64 {
            1000
        }
        fn execute_step<'a>(&'a self, _: AnalysisStudyStepRequest) -> AnalysisStudyStepFuture<'a> {
            panic!("an unmeshed structural study must not reach a step executor")
        }
    }
    let _guard = analysis_test_guard();
    let spec = unmeshed_modal_study_spec();

    let in_process = analysis_run_study_op(&spec, OperationContext::new(None, None))
        .expect_err("the in-process host must refuse an unmeshed structural study");
    let owned = futures::executor::block_on(analysis_execute_study_with_step_executor(
        &spec,
        OperationContext::new(None, None),
        &UnreachableExecutor,
    ))
    .expect_err("the step-executor host must refuse an unmeshed structural study");
    let Err(native) =
        analysis_prepare_structural_study_op(&spec, OperationContext::new(None, None))
    else {
        panic!("the native host must refuse an unmeshed structural study");
    };
    for error in [in_process, owned, native] {
        assert_eq!(error.error_code, "RM.FEA.RUN_STUDY.PORTABLE_MESH_REQUIRED");
        assert_eq!(
            error.message,
            "portable structural study execution requires a canonical solver mesh"
        );
    }
}
