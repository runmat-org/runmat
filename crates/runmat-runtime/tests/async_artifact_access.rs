//! Exercise canonical publication and field readers over an async-only provider.
//! Numerical values are fixture data; this test performs no mesh or solve.
use runmat_analysis_core::*;
use runmat_analysis_fea::{
    admission::*, ComputeBackend, ExecutionBackend, FeaRunResult, SolverMethod,
    SolverPreconditioner,
};
use runmat_filesystem::{FsProvider, MemoryFsProvider};
use runmat_runtime::analysis::{storage::*, *};
use std::{path::Path, sync::Arc};

fn fixture() -> AnalysisRunResult {
    let registered =
        registered_formulations(PhysicsFamily::Thermal, AnalysisProcedure::Static).remove(0);
    let identity = admit_formulation(FormulationAdmissionRequest {
        family: PhysicsFamily::Thermal,
        procedure: AnalysisProcedure::Static,
        solution_space: SolutionSpace::Direct,
        realization: registered.realization().clone(),
    })
    .unwrap()
    .into_identity();
    let solver_backend =
        runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference;
    AnalysisRunResult {
        observation_scalars: vec![],
        run_id: "run_async_storage".into(),
        run: FeaRunResult {
            identity,
            requested_backend: ComputeBackend::Cpu,
            execution_backend: ExecutionBackend::Cpu,
            solver_backend,
            solver_device_apply_k_ratio: 0.0,
            solver_method: SolverMethod::AssembledPcg,
            preconditioner: SolverPreconditioner::None,
            solver_host_sync_count: 0,
            accelerator_evidence: vec![],
            diagnostics: vec![],
            spatial_projection_evidence: vec![],
            structural_static_convergence: None,
            structural_static_balance: None,
            structural_constraint_result: None,
            structural_connector_result: None,
            observations: vec![],
            observation_histories: vec![],
            series_reductions: vec![],
            fields: vec![AnalysisField::host_f64(
                "temperature",
                runmat_analysis_core::AnalysisFieldLayout::scalar(
                    runmat_analysis_core::ObservationQuantity::Temperature,
                    runmat_analysis_core::FieldLocation::Node,
                ),
                vec![3],
                vec![0.0, 0.25, 0.5],
            )],
        },
        render_topology: None,
        modal_results: None,
        buckling_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        model_validity: QualityGate::Pass,
        solver_convergence: QualityGate::Pass,
        result_quality: QualityGate::Pass,
        run_status: RunStatus::Completed,
        numerically_acceptable: true,
        quality_reasons: vec![],
        provenance: RunProvenance {
            requested_backend: ComputeBackend::Cpu,
            execution_backend: ExecutionBackend::Cpu,
            solver_backend,
            solver_device_apply_k_ratio: 0.0,
            solver_host_sync_count: 0,
            precision_mode: "f64".into(),
            deterministic_mode: true,
            solver_method: SolverMethod::AssembledPcg,
            preconditioner: SolverPreconditioner::None,
            quality_policy: "exploratory".into(),
            fallback_events: vec![],
            effective_memory_ceiling_bytes: None,
            static_solve_path: None,
        },
    }
}

#[test]
fn canonical_publication_reopens_and_failed_operations_do_not_publish() {
    futures::executor::block_on(async {
        let provider = Arc::new(MemoryFsProvider::new());
        let _guard = runmat_filesystem::replace_provider(provider.clone());
        // The async transaction owns one tree, including study evidence.
        // Do not inherit a separately configured operator evidence root.
        let root = std::path::PathBuf::from(".artifacts");
        configure_fea_runtime(FeaRuntimeConfig {
            artifact_root: Some(root.clone()),
            study_artifact_root: Some(root.join("studies")),
            ..FeaRuntimeConfig::default()
        })
        .unwrap();
        configure_artifact_store(AnalysisArtifactStoreConfig::Filesystem { root }).unwrap();
        let run = fixture();
        let definition = runmat_execution::Digest::sha256(b"async-artifact-fixture");
        let receipt = with_artifact_access(true, |store| {
            let builder = AnalysisPublicationBuilder::new(definition, 16 << 20)?;
            builder.stage_run(&run)?;
            let prepared = builder.finish(serde_json::json!({"study_definition_digest":definition,"public_run":{"steps":[{"run_id":run.run_id}]}}))?;
            store.commit_publication(prepared)
        }).await.unwrap().unwrap();
        assert!(receipt.manifest_artifact_path.is_some());
        with_artifact_access(false, |_| -> Result<(), String> {
            assert!(run_attempts(&run.run_id)?.is_empty());
            assert_eq!(resolve_run_attempt(&run.run_id)?, run.run_id);
            let result = analysis_results_by_run_id_op(
                &run.run_id,
                AnalysisResultsQuery::metadata_only(),
                runmat_runtime::operations::OperationContext::new(None, None),
            )
            .map_err(|error| error.message)?;
            assert_eq!(result.data.field_descriptors.len(), 1);
            Ok(())
        })
        .await
        .unwrap()
        .unwrap();
        assert!(resolve_run_attempt(&run.run_id).is_err());
        let field = with_artifact_access(false, |store| {
            store.load_field_page(
                &run.run_id,
                &AnalysisFieldSelection::new("temperature", None),
                AnalysisFieldRequestOptions::default(),
            )
        })
        .await
        .unwrap()
        .unwrap();
        let AnalysisFieldPageLookup::Page(page) = field else {
            panic!("published field is not readable after reopen")
        };
        assert_eq!(
            page.values,
            AnalysisFieldPageValues::Real(vec![0.0, 0.25, 0.5])
        );
        let loaded = with_artifact_access(false, |store| store.load_run(&run.run_id))
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            loaded.run.fields[0].as_host_f64().unwrap(),
            &[0.0, 0.25, 0.5]
        );

        let mut failed = fixture();
        failed.run_id = "run_discarded".into();
        let failure = with_artifact_access(true, |store| -> Result<(), String> {
            store.persist_run(&failed)?;
            Err("operation rejected before commit".into())
        })
        .await
        .unwrap();
        assert!(failure.is_err());
        assert!(
            with_artifact_access(false, |store| store.load_run(&failed.run_id))
                .await
                .unwrap()
                .unwrap()
                .is_none()
        );
        // The global provider remains the original async authority throughout.
        let snapshot = provider
            .capture_workspace(runmat_filesystem::workspace::WorkspaceCaptureRequest {
                root: "/.artifacts".into(),
                excluded_subtrees: vec![],
                max_files: 1000,
                max_bytes: 16 << 20,
            })
            .await
            .unwrap();
        assert!(snapshot.files.len() > 1);
        if let Ok(path) = std::env::var("RUNMAT_ASYNC_ARTIFACT_FIXTURE") {
            std::fs::write(path, serde_json::to_vec(&snapshot).unwrap()).unwrap();
        }
        assert!(provider
            .read(Path::new("/.artifacts/runs/run_discarded.json"))
            .await
            .is_err());
    });
}
