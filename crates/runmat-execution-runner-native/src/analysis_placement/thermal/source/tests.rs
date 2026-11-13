use super::*;
use crate::analysis_thermal_prestress::{
    execute_with_pool,
    tests::{pool::RecordingPool, preparation},
};
use crate::StructuralSequenceExecutionPolicy;
use runmat_execution::{resource::ResourceRequest, task::RetryPolicy};
use runmat_meshing_core::ElementOrder;
mod observation_queries;
mod standalone_publication;

#[derive(Clone)]
struct DownloadedThermalObjects(BTreeMap<Digest, Vec<u8>>);

impl CacheImport for DownloadedThermalObjects {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.get(&digest).cloned())
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        if self
            .0
            .get(&digest)
            .is_some_and(|bytes| bytes.len() as u64 > maximum)
        {
            return Err(runmat_execution_artifact::ArtifactError::Limit(
                "downloaded thermal object exceeds caller read allowance".into(),
            ));
        }
        Ok(self.0.get(&digest).cloned())
    }
}

/// Provider causes, deliberately with misleading text to prevent string classification.
struct RejectingThermalObjects {
    capacity: bool,
}
impl CacheImport for RejectingThermalObjects {
    fn read_verified(&self, _: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Err(if self.capacity {
            runmat_execution_artifact::ArtifactError::Limit("opaque provider rejection".into())
        } else {
            runmat_execution_artifact::ArtifactError::Identity("memory limit exceeded".into())
        })
    }
    fn read_verified_bounded(&self, digest: Digest, _: u64) -> ArtifactResult<Option<Vec<u8>>> {
        self.read_verified(digest)
    }
}

#[test]
fn runtime_thermal_source_replays_real_completion_and_authenticates_all_catalog_bytes() {
    run(None, ElementOrder::Tet4);
}

#[test]
fn runtime_thermal_publication_preserves_authored_observations_and_selective_histories() {
    for policy in [
        runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks,
        runmat_analysis_core::PartialAccessPolicy::WholeArtifact,
    ] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            run(Some(policy), order);
        }
    }
}

fn run(observation_policy: Option<runmat_analysis_core::PartialAccessPolicy>, order: ElementOrder) {
    let directory = tempfile::tempdir().unwrap();
    let mut pool = RecordingPool::new(directory.path());
    let mut prepared = preparation(1, order);
    if let Some(policy) = observation_policy {
        use runmat_analysis_core::{
            IndependentDomain, ObservationId, ObservationRequest, ProcedureStep, SamplingPolicy,
        };
        use runmat_analysis_fea::study::{
            plan_structural_study, StructuralNumericalPlanningPolicy, StructuralStudyPlanningPolicy,
        };
        let mut histories = crate::analysis_thermal::tests::history_finalization::histories(policy);
        let ObservationRequest::History(mut observation) = histories[0].clone() else {
            unreachable!()
        };
        observation.sampling = SamplingPolicy::Final;
        observation.reduction = None;
        observation.observation_id = ObservationId("temperature-final".into());
        histories.push(ObservationRequest::Field(observation));
        let step = &mut prepared.compiled.model.sequence.steps[0];
        let mut transient = ProcedureStep::thermal_transient(step.step_id.0.clone());
        transient.state_outputs = step.state_outputs.clone();
        transient.independent_domain = IndependentDomain::TimeSeconds {
            origin: 0.0,
            start: 0.0,
            end: 2.0,
            interval_count: 2,
        };
        transient.accuracy.resources.maximum_samples = 3;
        transient.observations = histories;
        *step = transient;
        prepared.compiled.properties = runmat_analysis_fea::properties::compile_property_map(
            &prepared.compiled.model,
            Some(&prepared.structural.mesh),
        )
        .unwrap();
        prepared.structural.authored_study_digest = prepared.compiled.definition_digest().unwrap();
        prepared.structural.workloads = plan_structural_study(
            &prepared.compiled,
            StructuralStudyPlanningPolicy {
                backend: runmat_analysis_fea::ComputeBackend::Cpu,
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
        inventory_limits: completed.thermal.output_policy().inventory_limits(),
        catalog: root.catalog,
        completed: completed.thermal,
        store: pool.store.clone(),
        objects: root
            .objects
            .into_iter()
            .map(|object| (object.descriptor.digest, object.bytes))
            .collect(),
    };
    let dto = completion(&source.completed).unwrap();
    assert!(matches!(
        source.publication_workspace(&dto, 0),
        Err(runmat_runtime::analysis::storage::AnalysisPublicationError::ResourceLimit { .. })
    ));
    let workspace = source.publication_workspace(&dto, 128 << 20).unwrap();
    assert!(workspace > source.completed.retained_storage_bytes(128 << 20).unwrap());
    println!("original thermal source requested publication workspace={workspace} bytes");
    let mut outside = dto.clone();
    outside.catalog.objects.pop();
    assert!(source.authenticate_admitted(&outside, workspace).is_err());
    assert!(source.authenticate_admitted(&dto, workspace - 1).is_err());
    source.authenticate_admitted(&dto, workspace).unwrap();
    authentication::validate_completion(&source.completed, &dto).unwrap();
    for substitute in 0..6 {
        let mut forged = dto.clone();
        match substitute {
            0 => forged.field_partitions.swap(0, 1),
            1 => forged
                .field_partitions
                .push(forged.field_partitions[0].clone()),
            2 => forged.field_partitions[0].source_indices[0] += 1,
            3 => forged.frozen_project_digest = Some(Digest::sha256(b"foreign project")),
            4 => forged.compiled.module_path.push_str("foreign"),
            _ => forged.field_partitions[0].schema_version += 1,
        }
        assert!(authentication::validate_completion(&source.completed, &forged).is_err());
    }
    assert!(matches!(
        CacheImport::read_verified_bounded(&source, dto.root.logical_digest, 0),
        Err(runmat_execution_artifact::ArtifactError::Limit(_))
    ));
    // Independently replay through a failing original provider. Capacity survives replay;
    // an identity failure with capacity-looking text must remain a contract failure.
    for capacity in [true, false] {
        let rejecting = RetainedThermalSource {
            inventory_limits: source.inventory_limits,
            completed: source.completed.clone(),
            catalog: source.catalog.clone(),
            objects: source.objects.clone(),
            store: RejectingThermalObjects { capacity },
        };
        assert_eq!(
            matches!(
                rejecting.authenticate_original(&dto).unwrap_err(),
                runmat_runtime::analysis::storage::AnalysisPublicationError::ResourceLimit { .. }
            ),
            capacity,
        );
        let descendant = dto
            .catalog
            .objects
            .iter()
            .find(|reference| !rejecting.objects.contains_key(&reference.logical_digest))
            .unwrap();
        assert_eq!(
            matches!(
                rejecting.read_catalog_object(descendant).unwrap_err(),
                runmat_execution_artifact::ArtifactError::Limit(_)
            ),
            capacity,
        );
    }
    source.authenticate(&dto).unwrap();
    if observation_policy.is_some() {
        observation_queries::check(&source, &dto);
    }
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
    for metadata in 0..2 {
        let mut reference = dto.root.clone();
        match metadata {
            0 => reference.media_type.push_str("foreign"),
            _ => reference.value_schema.push_str("foreign"),
        }
        assert!(PublishedObjectSource::read_verified(&source, &reference).is_err());
    }
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
        thermal_source: thermal_source.clone(),
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
    let result =
        StructuralStudyResult::new(compiled, mesh, evidence, workloads, numerical).unwrap();
    let published = runmat_runtime::analysis::publish_structural_study_result(
        &result,
        runmat_runtime::operations::OperationContext::new(
            Some("thermal-prefix-test".into()),
            Some("publication".into()),
        ),
        runmat_runtime::analysis::AnalysisStudyCompletionIdentity::new(format!(
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
    if observation_policy.is_some() {
        observation_queries::check_persistence(
            &published.data.steps[0].run_id,
            &thermal_run,
            &dto,
            &thermal_source,
            &source,
        );
        assert_eq!(thermal_run.run.observations.len(), 1);
        assert_eq!(thermal_run.run.series_reductions.len(), 1);
        assert!(
            thermal_run.run.observation_histories.is_empty(),
            "histories stay catalog-backed"
        );
    }
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
