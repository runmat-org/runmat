//! Actual frequency coordinator outputs consumed from a reopened, producer-free store.
use super::*;
use runmat_analysis_core::{
    ExtremaTiePolicy, ObservationComponent, ObservationId, ObservationRequest, ReductionPolicy,
    SamplingPolicy, SeriesReduction, SpatialReduction,
};
use runmat_analysis_execution::{import_analysis_partition, AnalysisPartitionImport};
use runmat_analysis_fea::results::observation::{ObservationResult, SeriesReductionResult};
use runmat_execution::resource::ResourceRequest;
use runmat_execution::task::RetryPolicy;
use runmat_execution_artifact::cache::FilesystemObjectStore;

#[path = "cold_history/publication.rs"]
mod publication;

struct SubstitutedChunk<'a> {
    source: &'a FilesystemObjectStore,
    digest: Digest,
}
impl CacheImport for SubstitutedChunk<'_> {
    fn read_verified(
        &self,
        digest: Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        let mut bytes = self.source.read_verified(digest)?;
        if digest == self.digest {
            if let Some(bytes) = &mut bytes {
                bytes[0] ^= 1;
            }
        }
        Ok(bytes)
    }
}

#[test]
fn original_frequency_scalars_and_series_survive_producer_free_catalog_replay() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        check(order, None);
    }
}

fn check(order: ElementOrder, publication_root: Option<&std::path::Path>) {
    let (mut model, _) = fixture_model_and_mesh();
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
    let mut step = ProcedureStep::structural_harmonic("frequency");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
    let ObservationRequest::Field(mut scalar) =
        super::super::tests::structural_displacement_observation()
    else {
        unreachable!()
    };
    scalar.observation_id = ObservationId("displacement_peak".into());
    scalar.component = ObservationComponent::Magnitude;
    scalar.sampling = SamplingPolicy::FrequencyHertz(vec![20.0, 40.0]);
    scalar.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Maximum,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    let mut history = scalar.clone();
    history.observation_id = ObservationId("displacement_peak_history".into());
    history.reduction.as_mut().unwrap().series = SeriesReduction::Maximum;
    scalar.sampling = SamplingPolicy::Final;
    step.observations = vec![
        ObservationRequest::Reduction(scalar),
        ObservationRequest::History(history),
    ];
    model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "cold-frequency-history",
        vec![step.clone()],
    );
    let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model,
        step_id: step.step_id,
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[40.0], &[]).unwrap(),
        solver: StructuralFrequencySolverPlan::Direct {
            policy: DirectHarmonicPolicy::default(),
        },
        prestress_transition: None,
    })
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let mut pool =
        crate::analysis_thermal_prestress::tests::pool::RecordingPool::new(directory.path());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let completed = runtime
        .block_on(Box::pin(crate::analysis_frequency::execute_with_pool(
            &mut pool,
            crate::StructuralFrequencyExecutionPreparation {
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
                workload: workload.clone(),
                mesh: mesh.clone(),
                prestress_state: None,
                modal_state: None,
                program_revision: revision(),
                artifact_access: access(),
                chunk_bytes: 1024,
                inventory_limits: ObjectInventoryLimits::default(),
                observation_limits: Default::default(),
            },
            crate::StructuralSequenceExecutionPolicy {
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
            Vec::new(),
        )))
        .unwrap();
    let (catalog, roots) = completed.published_artifacts(&access()).unwrap();
    let task_count = pool.tasks.len();
    assert!(
        task_count > 2,
        "the complete coordinator must execute recovery and observations"
    );
    let originals = completed
        .observation_samples
        .iter()
        .map(|sample| {
            let observations = sample.observations.as_ref().unwrap();
            observations
                .manifest
                .results
                .iter()
                .map(|entry| {
                    let partition = import_analysis_partition(
                        &pool,
                        AnalysisPartitionImport {
                            root: &observations.output.root,
                            expected_kind: AnalysisNumericalObjectKind::Observation,
                            expected_logical_identity: observations.output.logical_identity,
                            expected_payload_digest: observations.output.payload_digest,
                            partition_identity: entry.result.digest(),
                            access: &observations.access,
                            limits: ObjectInventoryLimits::default(),
                        },
                    )
                    .unwrap();
                    ObservationResult::from_canonical_bytes(&partition.payload()).unwrap()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        pool.tasks.len(),
        task_count,
        "reading original evidence must not solve again"
    );
    // The cold consumer has no executor or task pool. Reopen only original immutable bytes.
    drop(pool);
    drop(runtime);
    let cold = FilesystemObjectStore::open(directory.path(), 64 * 1024 * 1024).unwrap();
    let mut original_scalars = Vec::new();
    for (ordinal, (sample, originals)) in completed
        .observation_samples
        .iter()
        .zip(&originals)
        .enumerate()
    {
        let observations = sample.observations.as_ref().unwrap();
        assert_eq!(observations.manifest.results.len(), ordinal + 1);
        for (entry, original) in observations.manifest.results.iter().zip(originals) {
            let request = |limits| AnalysisPartitionImport {
                root: &observations.output.root,
                expected_kind: AnalysisNumericalObjectKind::Observation,
                expected_logical_identity: observations.output.logical_identity,
                expected_payload_digest: observations.output.payload_digest,
                partition_identity: entry.result.digest(),
                access: &observations.access,
                limits,
            };
            let insufficient = ObjectInventoryLimits {
                max_total_bytes: 1,
                ..ObjectInventoryLimits::default()
            };
            assert!(import_analysis_partition(&cold, request(insufficient)).is_err());
            let partition =
                import_analysis_partition(&cold, request(ObjectInventoryLimits::default()))
                    .unwrap();
            let corrupt = SubstitutedChunk {
                source: &cold,
                digest: partition.manifest.chunks[partition.partition.first_chunk_ordinal as usize]
                    .digest,
            };
            assert!(
                import_analysis_partition(&corrupt, request(ObjectInventoryLimits::default()))
                    .is_err()
            );
            let decoded = ObservationResult::from_canonical_bytes(&partition.payload()).unwrap();
            entry.validate_result(&decoded).unwrap();
            assert_eq!(&decoded, original);
            assert_eq!(decoded.sample, sample.sample);
            original_scalars.extend(
                runmat_analysis_fea::results::assessment::scalar_observation_evidence(&decoded),
            );
        }
    }
    assert_eq!(original_scalars.len(), 3);
    assert_eq!(completed.histories.len(), 1);
    let finalization = completed.histories[0].finalization.as_ref().unwrap();
    let original_series = finalization.result.series_reduction.as_ref().unwrap();
    let objects = finalization.result.series_objects.as_ref().unwrap();
    let root = objects
        .root_reference(
            &access(),
            runmat_execution::value::ValueRefKind::ResultObject,
        )
        .unwrap();
    let imported = runmat_analysis_execution::import_analysis_objects(
        &cold,
        &root,
        AnalysisNumericalObjectKind::SeriesReductionResult,
        objects.manifest.logical_identity,
        objects.manifest.payload_digest,
        &access(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let decoded = SeriesReductionResult::from_canonical_bytes(&imported.payload()).unwrap();
    assert_eq!(&decoded, original_series);
    let series = runmat_analysis_fea::results::assessment::scalar_series_evidence(&decoded)
        .expect("the original scalar maximum must remain available for assessment");
    let governing = original_scalars
        .iter()
        .filter(|scalar| scalar.observation_id.0 == "displacement_peak_history")
        .max_by(|left, right| left.value.total_cmp(&right.value))
        .unwrap();
    assert_eq!(series.value, governing.value);
    assert_eq!(series.sample, governing.sample);
    assert_eq!(series.entity, governing.entity);
    let original_pages = publication_root.map(|_| original_field_pages(&completed, &cold));
    let source = runmat_execution_artifact::ArtifactPublishedObjectSource::new(cold);
    let numerical = runmat_runtime::analysis::StructuralStudyFrequencyNumericalResult::new(
        None,
        workload,
        completed.result.result,
        completed.publication,
        catalog,
        roots,
        source,
    )
    .unwrap();
    if let Some(root) = publication_root {
        publication::publish(
            root,
            order,
            mesh,
            numerical,
            original_scalars,
            series,
            original_pages.unwrap(),
        );
    }
}

fn original_field_pages(
    completed: &crate::CompletedStructuralFrequencyStudy,
    source: &FilesystemObjectStore,
) -> Vec<runmat_runtime::analysis::AnalysisFieldPageValues> {
    use runmat_analysis_fea::results::field::{
        RecoveredFieldNumericRepresentation, RecoveredFieldPartition,
    };
    completed
        .entries
        .iter()
        .map(|entry| {
            let fields = &entry.recovered_fields;
            let field = &fields.manifest.fields[0];
            let mut values = Vec::new();
            for descriptor in &field.partitions {
                let partition = import_analysis_partition(
                    source,
                    AnalysisPartitionImport {
                        root: &fields.output.root,
                        expected_kind: AnalysisNumericalObjectKind::Field,
                        expected_logical_identity: fields.output.logical_identity,
                        expected_payload_digest: fields.output.payload_digest,
                        partition_identity: descriptor.partition.digest(),
                        access: &access(),
                        limits: ObjectInventoryLimits::default(),
                    },
                )
                .unwrap();
                let decoded =
                    RecoveredFieldPartition::from_canonical_bytes(&partition.payload()).unwrap();
                assert_eq!(decoded.definition, field.definition);
                values.extend_from_slice(decoded.storage.as_host_f64().unwrap());
            }
            match field.definition.numeric_representation() {
                RecoveredFieldNumericRepresentation::Real => {
                    runmat_runtime::analysis::AnalysisFieldPageValues::Real(values)
                }
                RecoveredFieldNumericRepresentation::ComplexInterleaved => {
                    runmat_runtime::analysis::AnalysisFieldPageValues::ComplexInterleaved(values)
                }
            }
        })
        .collect()
}
