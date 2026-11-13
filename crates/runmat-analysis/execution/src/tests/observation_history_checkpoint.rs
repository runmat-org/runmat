use super::*;

#[test]
fn result_inventory_deduplicates_content_shared_between_outputs() {
    let first = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Field,
        digest("shared-output-field"),
        b"shared immutable output chunk",
        256,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let second = prepare_analysis_objects(
        AnalysisNumericalObjectKind::History,
        digest("shared-output-history"),
        b"shared immutable output chunk",
        256,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let mut references = first
        .unique_references(&access(), ValueRefKind::ResultObject)
        .unwrap();
    references.extend(
        second
            .unique_references(&access(), ValueRefKind::ResultObject)
            .unwrap(),
    );
    assert_eq!(references.len(), 4);
    crate::object::retain_unique_references(&mut references);
    assert_eq!(references.len(), 3);
    assert_ne!(references[0].id, references[2].id);
}

struct FailingObjects {
    successful_writes_remaining: usize,
}

impl runmat_execution_artifact::cache::CacheExport for FailingObjects {
    fn write_verified(
        &mut self,
        object: &runmat_execution_artifact::LogicalObject,
    ) -> runmat_execution_artifact::ArtifactResult<()> {
        object.validate()?;
        if self.successful_writes_remaining == 0 {
            return Err(runmat_execution_artifact::ArtifactError::Invalid(
                "injected checkpoint publication failure".into(),
            ));
        }
        self.successful_writes_remaining -= 1;
        Ok(())
    }
}

#[test]
fn high_rate_history_checkpoints_write_only_new_chunks_and_resume_exactly() {
    use runmat_analysis_fea::results::field::{
        RecoveredField, RecoveredFieldCatalog, RecoveredFieldDefinition,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    };
    use runmat_analysis_fea::results::observation::{
        evaluate_observation_sample, resolve_observation_plan, BoundedObservationHistory,
        ObservationLimits,
    };
    use runmat_analysis_fea::topology::AnalysisProjection;
    use runmat_meshing_core::ElementOrder;

    let sample_count = 32_usize;
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let definition = RecoveredFieldDefinition::new(
        "displacement.high_rate",
        runmat_analysis_core::ObservationQuantity::Displacement,
        runmat_analysis_core::FieldLocation::Node,
        runmat_analysis_fea::results::field::RecoveredFieldEntityDomain::Nodes,
        runmat_analysis_core::ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut request = structural_displacement_observation();
    let spec = request.spec_mut().unwrap();
    spec.observation_id = runmat_analysis_core::ObservationId("high_rate_history".into());
    spec.sampling = runmat_analysis_core::SamplingPolicy::TimeSeconds(
        (0..sample_count).map(|index| index as f64).collect(),
    );
    spec.storage.retention = runmat_analysis_core::RetentionPolicy::LatestSamples { count: 2 };
    let mut step = ProcedureStep::structural_transient("high-rate-history");
    step.independent_domain = runmat_analysis_core::IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: (sample_count - 1) as f64,
        interval_count: sample_count - 1,
    };
    step.observations = vec![runmat_analysis_core::ObservationRequest::History(
        spec.clone(),
    )];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let observation_id = runmat_analysis_core::ObservationId("high_rate_history".into());
    let result_at = |index: usize| {
        let coordinate = RecoveredFieldSample::TimeSeconds(index as f64);
        let field = RecoveredField::new(
            definition.clone(),
            coordinate,
            runmat_analysis_core::AnalysisField::host_f64(
                definition.field_id(),
                definition.layout(),
                vec![projection.nodes.len(), 3],
                vec![index as f64 + 1.0; projection.nodes.len() * 3],
            ),
        )
        .unwrap();
        let fields = RecoveredFieldCatalog::new(&projection, vec![field]).unwrap();
        evaluate_observation_sample(&plan, &observation_id, &projection, &fields, coordinate)
            .unwrap()
    };

    let limits = ObjectInventoryLimits::default();
    let artifact_access = access();
    let mut store = MemoryObjects::default();
    let mut history = BoundedObservationHistory::new(&plan, &observation_id).unwrap();
    let mut publisher = ObservationHistoryCheckpointPublisher::new(256, limits).unwrap();
    let mut midpoint = None;
    for index in 0..sample_count / 2 {
        history
            .push(
                RecoveredFieldSample::TimeSeconds(index as f64),
                result_at(index),
            )
            .unwrap();
        if index == 0 {
            let mut failing_publisher =
                ObservationHistoryCheckpointPublisher::new(256, limits).unwrap();
            let mut failing_store = FailingObjects {
                successful_writes_remaining: 1,
            };
            assert!(failing_publisher
                .publish(&mut failing_store, &history, &artifact_access)
                .is_err());
            assert_eq!(failing_publisher.predecessor(), None);
        }
        let published = publisher
            .publish(&mut store, &history, &artifact_access)
            .unwrap();
        assert!(published.objects.manifest.partitions.len() <= 3);
        if index > 0 {
            assert!(published.written_object_count < published.objects.objects.len());
            assert!(
                published.written_bytes
                    < published
                        .objects
                        .objects
                        .iter()
                        .map(|object| object.descriptor.encoded_length)
                        .sum::<u64>()
            );
        }
        midpoint = Some(published);
    }
    let midpoint = midpoint.unwrap();
    let imported = import_observation_history_checkpoint(
        &store,
        ObservationHistoryCheckpointImport {
            root: &midpoint.root,
            expected_identity: midpoint.checkpoint.manifest().identity,
            expected_payload_digest: midpoint.objects.manifest.payload_digest,
            access: &artifact_access,
            limits,
        },
    )
    .unwrap();
    let midpoint_identity = imported.checkpoint.manifest().identity;

    let next_index = sample_count / 2;
    let next_result = result_at(next_index);
    let next_result_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationResult,
        next_result.logical_identity.digest(),
        &next_result.canonical_bytes().unwrap(),
        256,
        ObjectNamespace::ResultValue,
        limits,
    )
    .unwrap();
    let (study_identity, step_identity, execution_plan) = execution_identities();
    let source_input = AnalysisWorkloadInput {
        kind: next_result_objects.manifest.kind,
        logical_identity: next_result_objects.manifest.logical_identity,
        payload_digest: next_result_objects.manifest.payload_digest,
        manifest_digest: next_result_objects.root.digest,
        manifest_encoded_length: next_result_objects.root.encoded_length,
        reference_kind: ValueRefKind::ResultObject,
    };
    let invalid_source_host = AnalysisHostWorkload::new(
        AnalysisWorkloadRequest::new(
            study_identity,
            step_identity,
            execution_plan,
            AnalysisWorkKind::Solve,
            vec![source_input.clone()],
            vec![AnalysisNumericalObjectKind::ObservationResult],
            BTreeSet::from([AnalysisCapabilityRequirement::Host {
                abi: ANALYSIS_HOST_ABI.into(),
            }]),
        )
        .unwrap(),
        artifact_access.clone(),
        crate::AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default()).unwrap(),
    )
    .unwrap();
    let source_host = AnalysisHostWorkload::new(
        AnalysisWorkloadRequest::new(
            study_identity,
            step_identity,
            execution_plan,
            AnalysisWorkKind::ReduceObservationPartitions,
            vec![source_input],
            vec![AnalysisNumericalObjectKind::ObservationResult],
            BTreeSet::from([AnalysisCapabilityRequirement::Host {
                abi: ANALYSIS_HOST_ABI.into(),
            }]),
        )
        .unwrap(),
        artifact_access.clone(),
        crate::AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default()).unwrap(),
    )
    .unwrap();
    let imported_result = ImportedReducedObservation {
        result: next_result.clone(),
        objects: next_result_objects,
        producer: source_host.workload.identity,
    };
    assert!(
        prepare_observation_history_checkpoint_task(ObservationHistoryCheckpointPreparation {
            source_host: invalid_source_host,
            plan: plan.clone(),
            observation_id: observation_id.clone(),
            sample: imported_result.clone(),
            previous: Some(midpoint.objects.clone()),
            program_revision: revision(),
            chunk_bytes: 256,
            inventory_limits: limits,
        })
        .is_err()
    );
    let scheduled =
        prepare_observation_history_checkpoint_task(ObservationHistoryCheckpointPreparation {
            source_host,
            plan: plan.clone(),
            observation_id: observation_id.clone(),
            sample: imported_result,
            previous: Some(midpoint.objects.clone()),
            program_revision: revision(),
            chunk_bytes: 256,
            inventory_limits: limits,
        })
        .unwrap();
    assert_eq!(scheduled.request.sample_ordinal, (sample_count / 2) as u64);
    assert_eq!(
        scheduled.request.sample_coordinate,
        RecoveredFieldSample::TimeSeconds((sample_count / 2) as f64)
    );
    let mut wrong_ordinal = scheduled.request.clone();
    wrong_ordinal.sample_ordinal += 1;
    assert!(wrong_ordinal.validate().is_err());
    scheduled.program.write_to(&mut store).unwrap();
    let expected = scheduled
        .request
        .advance(Some(midpoint.checkpoint.clone()), next_result)
        .unwrap();
    let scheduled_completion = execute_serial_analysis(
        &scheduled.program.request,
        &mut store,
        &HistoryCheckpointKernel(expected),
        256,
        limits,
    )
    .unwrap();
    let scheduled_response =
        AnalysisHostResponse::completed(&scheduled.host, &scheduled_completion)
            .unwrap()
            .program_response();
    let scheduled_import =
        import_observation_history_checkpoint_task(&store, scheduled_response, &scheduled, limits)
            .unwrap();
    assert_eq!(
        scheduled_import.checkpoint.manifest().predecessor,
        Some(midpoint_identity)
    );

    let durable_root = std::env::temp_dir().join(format!(
        "runmat-history-checkpoint-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    {
        let mut durable = runmat_execution_artifact::cache::FilesystemObjectStore::open(
            &durable_root,
            limits.max_object_bytes,
        )
        .unwrap();
        for object in &midpoint.objects.objects {
            runmat_execution_artifact::cache::CacheExport::write_verified(&mut durable, object)
                .unwrap();
        }
    }
    let mut durable = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &durable_root,
        limits.max_object_bytes,
    )
    .unwrap();
    let durable_imported = import_observation_history_checkpoint(
        &durable,
        ObservationHistoryCheckpointImport {
            root: &midpoint.root,
            expected_identity: midpoint_identity,
            expected_payload_digest: midpoint.objects.manifest.payload_digest,
            access: &artifact_access,
            limits,
        },
    )
    .unwrap();
    let mut durable_history = BoundedObservationHistory::resume(
        &plan,
        &observation_id,
        durable_imported.checkpoint.clone(),
    )
    .unwrap();
    durable_history
        .push(
            RecoveredFieldSample::TimeSeconds((sample_count / 2) as f64),
            result_at(sample_count / 2),
        )
        .unwrap();
    let mut durable_publisher =
        ObservationHistoryCheckpointPublisher::resume_from(&durable_imported, 256, limits).unwrap();
    let durable_next = durable_publisher
        .publish(&mut durable, &durable_history, &artifact_access)
        .unwrap();
    assert_eq!(
        durable_next.checkpoint.manifest().predecessor,
        Some(midpoint_identity)
    );
    drop(durable);
    let durable = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        &durable_root,
        limits.max_object_bytes,
    )
    .unwrap();
    import_observation_history_checkpoint(
        &durable,
        ObservationHistoryCheckpointImport {
            root: &durable_next.root,
            expected_identity: durable_next.checkpoint.manifest().identity,
            expected_payload_digest: durable_next.objects.manifest.payload_digest,
            access: &artifact_access,
            limits,
        },
    )
    .unwrap();
    std::fs::remove_dir_all(&durable_root).unwrap();

    let mut history =
        BoundedObservationHistory::resume(&plan, &observation_id, imported.checkpoint.clone())
            .unwrap();
    let mut publisher =
        ObservationHistoryCheckpointPublisher::resume_from(&imported, 256, limits).unwrap();
    let mut final_publication = None;
    for index in sample_count / 2..sample_count {
        history
            .push(
                RecoveredFieldSample::TimeSeconds(index as f64),
                result_at(index),
            )
            .unwrap();
        let published = publisher
            .publish(&mut store, &history, &artifact_access)
            .unwrap();
        if index == sample_count / 2 {
            assert_eq!(
                published.checkpoint.manifest().predecessor,
                Some(midpoint_identity)
            );
        }
        assert!(published.written_object_count < published.objects.objects.len());
        final_publication = Some(published);
    }
    let final_publication = final_publication.unwrap();
    assert_eq!(final_publication.checkpoint.samples().len(), 2);

    let completed = publisher
        .finish_and_publish(&mut store, history, &artifact_access)
        .unwrap();
    assert!(completed.written_object_count < completed.objects.objects.len());
    assert!(
        completed.written_bytes
            < completed
                .objects
                .objects
                .iter()
                .map(|object| object.descriptor.encoded_length)
                .sum::<u64>()
    );
    let restarted = completed.history.clone();
    let imported_history = import_observation_history(
        &store,
        ObservationHistoryImport {
            root: &completed.root,
            expected_identity: completed.history.manifest().identity,
            expected_payload_digest: completed.objects.manifest.payload_digest,
            partial_access: runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks,
            access: &artifact_access,
            limits,
        },
    )
    .unwrap();
    let selected = import_observation_history_sample(
        &store,
        &imported_history,
        RecoveredFieldSample::TimeSeconds((sample_count - 1) as f64),
        limits,
    )
    .unwrap();
    assert_eq!(
        selected.coordinate,
        RecoveredFieldSample::TimeSeconds((sample_count - 1) as f64)
    );
    let mut uninterrupted = BoundedObservationHistory::new(&plan, &observation_id).unwrap();
    for index in 0..sample_count {
        uninterrupted
            .push(
                RecoveredFieldSample::TimeSeconds(index as f64),
                result_at(index),
            )
            .unwrap();
    }
    assert_eq!(restarted, uninterrupted.finish().unwrap());
}

struct HistoryCheckpointKernel(
    runmat_analysis_fea::results::observation::ObservationHistoryCheckpoint,
);

impl AnalysisKernel for HistoryCheckpointKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(
            invocation.kind,
            AnalysisWorkKind::CheckpointObservationHistory
        );
        let mut partitions = Vec::with_capacity(self.0.samples().len() + 1);
        partitions.push(AnalysisKernelOutputPartition {
            logical_identity: self.0.manifest().identity.digest(),
            payload: self
                .0
                .manifest()
                .canonical_bytes()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        });
        partitions.extend(
            self.0
                .samples()
                .iter()
                .map(|sample| {
                    Ok(AnalysisKernelOutputPartition {
                        logical_identity: sample.identity.digest(),
                        payload: sample
                            .canonical_bytes()
                            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
                    })
                })
                .collect::<AnalysisBridgeResult<Vec<_>>>()?,
        );
        Ok(vec![AnalysisKernelOutput::partitioned(
            AnalysisNumericalObjectKind::ObservationHistoryCheckpoint,
            self.0.manifest().identity.digest(),
            partitions,
        )])
    }
}
