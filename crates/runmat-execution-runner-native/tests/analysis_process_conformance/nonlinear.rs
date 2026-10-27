//! Real scheduled native J2 phases and authenticated runtime publication, using original meshes.
use super::*;
use runmat_analysis_execution::{
    prepare_nonlinear_solid_study_root, NonlinearSolidStudyPreparation,
};
use runmat_execution_artifact::cache::{CacheExport, FilesystemObjectStore};
#[path = "../support/nonlinear.rs"]
mod support;

pub(super) async fn run() {
    for quadratic in [false, true] {
        let (workload, mesh) = support::fixture(quadratic);
        let product = runmat_analysis_fea::study::plan_nonlinear_solid_study_product(
            support::compiled(&workload, &mesh),
            &mesh,
            workload.controls.clone(),
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let mut config = NativeExecutionConfig::for_current_executable().unwrap();
        config.store_root = directory.path().join("session");
        config.max_workers = 1;
        config.worker_memory_bytes = workload.model.sequence.steps[0]
            .accuracy
            .resources
            .resident_byte_budget();
        config.worker_scratch_bytes = config.worker_memory_bytes;
        config.worker_cpu_millicores = 1000;
        config
            .worker_capabilities
            .insert(Capability::Custom(format!(
                "runmat.analysis.host:{}",
                runmat_analysis_execution::ANALYSIS_HOST_ABI
            )));
        let session = NativeProgramSession::new(config).unwrap();
        let source = session.object_source();
        let limits = ObjectInventoryLimits::default();
        let policy = runmat_analysis_execution::AnalysisOutputPolicy::new(
            runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES,
            limits,
        )
        .unwrap();
        let access = session.artifact_access().clone();
        let memory = workload.model.sequence.steps[0]
            .accuracy
            .resources
            .resident_byte_budget();
        let preparation = NonlinearSolidStudyPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            product,
            mesh: mesh.clone(),
            accepted_block_limit: 9,
            frozen_project_digest: None,
            program_revision: revision(),
            artifact_access: access.clone(),
            output_policy: policy,
        };
        let mut resources = analysis_resources();
        resources.memory_bytes = memory;
        resources.scratch_bytes = memory;
        resources.max_artifact_bytes = limits.max_total_bytes;
        let completed = runmat_execution_runner_native::execute_nonlinear_solid_study_locally(
            session,
            preparation,
            runmat_execution_runner_native::NonlinearSolidStudyExecutionPolicy {
                resources,
                retry: RetryPolicy::IdempotentInfrastructure,
                deadline_unix_millis: None,
                priority: 0,
            },
        )
        .await
        .unwrap();
        assert_eq!(completed.closures.len(), 7);
        assert_eq!(completed.terminal_usage.evidence.attempts.len(), 7);
        let prepared_root = prepare_nonlinear_solid_study_root(&completed).unwrap();
        let mut retained =
            FilesystemObjectStore::open(directory.path().join("retained"), limits.max_object_bytes)
                .unwrap();
        for reference in &prepared_root.catalog.objects {
            let bytes = prepared_root
                .objects
                .iter()
                .find(|o| o.descriptor.digest == reference.logical_digest)
                .map(|o| o.bytes.clone())
                .unwrap_or_else(|| {
                    source
                        .read_verified(reference.logical_digest)
                        .unwrap()
                        .unwrap()
                });
            retained
                .write_verified(
                    &runmat_execution_artifact::LogicalObject::new(
                        runmat_execution_artifact::ObjectNamespace::ResultValue,
                        "original-j2-runtime",
                        "application/octet-stream",
                        bytes,
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        let expected = completed.terminal_block.block.result.accepted_state.clone();
        drop(completed);
        drop(source);
        let (dto, objects) =
            runmat_execution_runner_native::nonlinear_solid_study_completion_from_source(
                retained,
                prepared_root.root.clone(),
                prepared_root.root.logical_digest,
                access,
                limits,
                memory,
            )
            .unwrap();
        objects.authenticate(&dto, memory).unwrap();
        assert_eq!(dto.terminal_block.result.accepted_state, expected);
        let prefix = objects.materialize_block(&dto, 0, 0, memory).unwrap();
        assert_eq!(prefix.load_factors.len(), 9);
        let alpha = (5.6 * 0.9 - 5.0) / 30.0;
        assert!(prefix.accepted_state.point_history().all(|(_, _, p)| (p
            .equivalent_plastic_strain
            - alpha)
            .abs()
            < 2e-9));
        let terminal = objects.materialize_block(&dto, 1, 0, memory).unwrap();
        assert_eq!(
            terminal.displacement_snapshots[0].field_id,
            runmat_analysis_fea::contracts::fea_nonlinear_displacement_field_id(9)
        );
        assert_eq!(terminal.accepted_state, expected);
        let mut original_factors = prefix.load_factors.clone();
        original_factors.extend_from_slice(&terminal.load_factors);
        let accepted_count = original_factors.len();
        let mut tampered = dto.clone();
        tampered.block_count += 1;
        assert!(objects.authenticate(&tampered, memory).is_err());
        assert!(objects.materialize_block(&dto, 0, memory, memory).is_err());
        let evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&mesh);
        runmat_runtime::analysis::storage::configure_artifact_store(
            runmat_runtime::analysis::storage::AnalysisArtifactStoreConfig::Filesystem {
                root: directory.path().join("published"),
            },
        )
        .unwrap();
        let definition = dto.compiled.definition_digest().unwrap();
        let published = runmat_runtime::analysis::publish_nonlinear_solid_study_result(
            &dto,
            &objects,
            &mesh,
            &evidence,
            definition,
            runmat_runtime::operations::OperationContext::new(None, None),
            runmat_runtime::analysis::AnalysisStudyCompletionIdentity::new(format!(
                "j2-runtime:{}",
                dto.root.logical_digest
            ))
            .unwrap(),
        )
        .unwrap();
        assert!(published.data.result_manifest_artifact_path.is_some());
        let stored =
            runmat_runtime::analysis::storage::load_run_result(&published.data.steps[0].run_id)
                .unwrap()
                .unwrap();
        let history = stored.nonlinear_results.unwrap();
        assert_eq!(history.load_factors, original_factors);
        assert_eq!(history.load_factors.last(), Some(&1.0));
        for (index, field) in history.displacement_snapshots.iter().enumerate() {
            assert_eq!(
                field.field_id,
                runmat_analysis_fea::contracts::fea_nonlinear_displacement_field_id(index)
            );
        }
        assert_eq!(history.displacement_snapshots.len(), accepted_count);
        assert_eq!(history.plastic_strain_snapshots.len(), accepted_count);
        assert_eq!(
            history.equivalent_plastic_strain_snapshots.len(),
            accepted_count
        );
        let final_alpha = history
            .equivalent_plastic_strain_snapshots
            .last()
            .unwrap()
            .as_host_f64()
            .unwrap();
        assert!(final_alpha.iter().all(|v| (v - 0.02).abs() < 2e-9));
        println!("original J2 scheduled/native/runtime cold witness PASS quadratic={quadratic}");
    }
}
