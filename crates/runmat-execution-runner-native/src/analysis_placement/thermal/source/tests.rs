use super::*;
use crate::analysis_thermal_prestress::{
    execute_with_pool,
    tests::{pool::RecordingPool, preparation},
};
use crate::StructuralSequenceExecutionPolicy;
use runmat_execution::{resource::ResourceRequest, task::RetryPolicy};
use runmat_meshing_core::ElementOrder;

#[derive(Clone)]
struct DownloadedThermalObjects(BTreeMap<Digest, Vec<u8>>);

impl CacheImport for DownloadedThermalObjects {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.get(&digest).cloned())
    }
}

#[test]
fn runtime_thermal_source_replays_real_completion_and_authenticates_all_catalog_bytes() {
    run(false);
}

#[test]
fn runtime_thermal_publication_rejects_unprojected_authored_observations() {
    run(true);
}

fn run(with_observation: bool) {
    let directory = tempfile::tempdir().unwrap();
    let mut pool = RecordingPool::new(directory.path());
    let mut prepared = preparation(1, ElementOrder::Tet4);
    if with_observation {
        use runmat_analysis_core::{ObservationRequest, PartialAccessPolicy, SamplingPolicy};
        use runmat_analysis_fea::study::{
            plan_structural_study, StructuralNumericalPlanningPolicy, StructuralStudyPlanningPolicy,
        };
        let ObservationRequest::History(mut observation) =
            crate::analysis_thermal::tests::history_finalization::histories(
                PartialAccessPolicy::ManifestAndChunks,
            )
            .remove(0)
        else {
            unreachable!()
        };
        observation.sampling = SamplingPolicy::Final;
        observation.reduction = None;
        prepared.compiled.model.sequence.steps[0].observations =
            vec![ObservationRequest::Field(observation)];
        prepared.structural.authored_study_digest = prepared.compiled.definition_digest().unwrap();
        prepared.structural.workloads = plan_structural_study(
            &prepared.compiled,
            StructuralStudyPlanningPolicy {
                backend: runmat_analysis_fea::ComputeBackend::Cpu,
                formulation_requirement: prepared.compiled.formulation_requirement,
                numerics: StructuralNumericalPlanningPolicy::production(),
            },
        )
        .unwrap();
    }
    let compiled = prepared.compiled.clone();
    let mesh = prepared.structural.mesh.clone();
    let workloads = prepared.structural.workloads.clone();
    let frozen_project_digest = prepared.structural.frozen_project_digest;
    let prefix = workloads.prestress_prefix().unwrap();
    let preload_step_id = prefix.preload.step_id.clone();
    let modal_step_id = prefix.consumer_step_ids[0].clone();
    // Leave the async executor before synchronous publication, avoiding a retained
    // large debug test future beneath the publication call. Keep the default test stack.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let completed = runtime
        .block_on(Box::pin(execute_with_pool(
            &mut pool,
            prepared,
            StructuralSequenceExecutionPolicy {
                resources: ResourceRequest {
                    cpu_millicores: 1000,
                    memory_bytes: 64 * 1024 * 1024,
                    scratch_bytes: 0,
                    max_wall_millis: 60_000,
                    max_artifact_bytes: 64 * 1024 * 1024,
                    max_egress_bytes: 0,
                    max_relay_bytes: 0,
                    accelerators: Vec::new(),
                    required_capabilities: Default::default(),
                },
                retry: RetryPolicy::IdempotentInfrastructure,
                deadline_unix_millis: None,
                priority: 0,
            },
        )))
        .unwrap();
    let task_count = pool.tasks.len();
    let root = prepare_thermal_study_root(&completed.thermal).unwrap();
    let source = RetainedThermalSource {
        completed: completed.thermal,
        store: pool.store.clone(),
        objects: root
            .objects
            .into_iter()
            .map(|object| (object.descriptor.digest, object.bytes))
            .collect(),
    };
    let dto = completion(&source.completed).unwrap();
    source.authenticate(&dto).unwrap();
    assert!(!dto.field_partitions.is_empty());
    assert_eq!(
        pool.tasks.len(),
        task_count,
        "authentication must not execute numerical tasks"
    );
    for reference in &dto.catalog.objects {
        assert_eq!(
            PublishedObjectSource::read_verified(&source, reference)
                .unwrap()
                .len() as u64,
            reference.encoded_length
        );
    }
    let mut forged = dto.clone();
    forged.host_workload_digest = forged.workload.identity().unwrap().digest();
    assert!(source.authenticate(&forged).is_err());
    let mut forged = dto.clone();
    forged.states.swap(0, 1);
    assert!(source.authenticate(&forged).is_err());
    let mut forged = dto.clone();
    forged.field_partitions.clear();
    assert!(source.authenticate(&forged).is_err());
    let mut forged = dto.clone();
    forged.catalog.objects.pop();
    assert!(source.authenticate(&forged).is_err());
    let mut forged = dto.clone();
    forged.root.logical_digest = Digest::sha256(b"foreign thermal root");
    assert!(source.authenticate(&forged).is_err());
    let mut reference = dto.root.clone();
    reference.encoded_length += 1;
    assert!(PublishedObjectSource::read_verified(&source, &reference).is_err());
    assert!(convert(source.completed.clone(), pool.store.clone()).is_ok());
    let empty_directory = tempfile::tempdir().unwrap();
    let empty_store =
        FilesystemObjectStore::open(empty_directory.path(), 64 * 1024 * 1024).unwrap();
    assert!(
        convert(source.completed.clone(), empty_store).is_err(),
        "retained metadata must not substitute for missing thermal object bytes"
    );
    let downloaded = DownloadedThermalObjects(
        dto.catalog
            .objects
            .iter()
            .map(|reference| {
                (
                    reference.logical_digest,
                    PublishedObjectSource::read_verified(&source, reference).unwrap(),
                )
            })
            .collect(),
    );
    let mut missing_root = downloaded.clone();
    missing_root.0.remove(&dto.root.logical_digest);
    assert!(thermal_study_completion_from_source(source.completed.clone(), missing_root).is_err());
    let mut missing_descendant = downloaded.clone();
    let descendant = dto
        .catalog
        .objects
        .iter()
        .find(|reference| reference.id != dto.root.id)
        .unwrap();
    missing_descendant.0.remove(&descendant.logical_digest);
    assert!(
        thermal_study_completion_from_source(source.completed.clone(), missing_descendant).is_err()
    );
    let mut foreign_reference = dto.root.clone();
    foreign_reference.authorization_scope.push_str("-foreign");
    assert!(PublishedObjectSource::read_verified(&source, &foreign_reference).is_err());
    // Public conversion and publication use only the downloaded declared closure, not the
    // originating worker/session store or fallback synthetic root bytes.
    let imported = runmat_analysis_execution::import_thermal_study_root(
        &downloaded,
        &dto.root,
        dto.root.logical_digest,
        &source.completed.result.host.artifact_access,
        crate::NativeAnalysisHostLimits::default().inventory,
    )
    .unwrap();
    let (thermal, thermal_source) =
        thermal_study_completion_from_source(imported, downloaded).unwrap();
    assert_eq!(thermal, dto);
    let crate::ThermalPrestressStructuralCompletion::Modal(modal) = completed.structural else {
        panic!("expected genuine modal completion");
    };
    use runmat_runtime::analysis::{
        StructuralStudyNumericalResult, StructuralStudyPrestressedModalNumericalResult,
        StructuralStudyResult,
    };
    let numerical = StructuralStudyNumericalResult::ThermalPrestress {
        thermal: Box::new(thermal),
        thermal_source,
        structural: Box::new(StructuralStudyNumericalResult::StructuralPrestressedModal(
            Box::new(StructuralStudyPrestressedModalNumericalResult {
                frozen_project_digest,
                preload_step_id,
                modal_step_id,
                preload_result: modal.preload.result,
                preload_solver_result_identity: modal.preload.identity,
                preload_numerical_receipt: modal.preload.numerical_receipt,
                prestress_state: modal.prestress_state.state,
                modal_result: modal.modal.result,
                modal_numerical_receipt: modal.modal.numerical_receipt,
            }),
        )),
    };
    // Explicit fixture evidence tests the typed boundary, not exact-CAD meshing qualification.
    let evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&mesh);
    let result = StructuralStudyResult::new(compiled, mesh, evidence, workloads, numerical);
    if with_observation {
        let error = result
            .err()
            .expect("unprojected observations must fail closed");
        assert!(
            error
                .to_string()
                .contains("does not yet project authored observations"),
            "{error}"
        );
        assert_eq!(pool.tasks.len(), task_count);
        return;
    }
    let result = result.unwrap();
    let published = runmat_runtime::analysis::publish_structural_study_result(
        &result,
        runmat_runtime::operations::OperationContext::new(
            Some("thermal-prefix-test".into()),
            Some("publication".into()),
        ),
        runmat_runtime::analysis::StructuralStudyCompletionIdentity::new(format!(
            "thermal-prefix-{}",
            directory.path().display()
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(published.data.steps.len(), 3);
    assert!(published.data.result_manifest_artifact_path.is_some());
    let thermal_run =
        runmat_runtime::analysis::storage::load_run_result(&published.data.steps[0].run_id)
            .unwrap()
            .unwrap();
    assert!(thermal_run.run.fields.is_empty());
    let histories = thermal_run.thermal_results.unwrap();
    assert_eq!(
        histories.temperature_snapshots.len(),
        dto.manifest.fields.len()
    );
    assert_eq!(
        histories.temperature_gradient_snapshots.len(),
        dto.manifest.fields.len()
    );
    assert_eq!(
        histories.heat_flux_snapshots.len(),
        dto.manifest.fields.len()
    );
    assert_eq!(
        histories.energy_storage_rate_snapshots.len(),
        dto.manifest.fields.len()
    );
    assert_eq!(
        published.data.states.len(),
        result
            .compiled()
            .model
            .sequence
            .steps
            .iter()
            .map(|step| step.state_outputs.len())
            .sum::<usize>()
    );
    for (published_state, retained) in published.data.states.iter().zip(&dto.states) {
        assert_eq!(published_state.identity, retained.identity);
    }
    assert_eq!(
        pool.tasks.len(),
        task_count,
        "publication must not solve again"
    );
}
