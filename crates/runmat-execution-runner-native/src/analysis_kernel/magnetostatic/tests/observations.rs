//! Native sampled observations from reopened publications, never a recovery/solve on read.
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    execute_serial_analysis, import_analysis_partition, import_magnetic_recovered_fields,
    import_magnetostatic_result, import_observation_partition_contribution,
    import_reduced_observation, prepare_magnetic_field_recovery, prepare_observation_fanout,
    prepare_observation_reduction, AnalysisHostResponse, AnalysisPartitionImport,
    MagneticFieldRecoveryPreparation, ObservationPreparation, ObservationPublishedSource,
    ObservationReductionPreparation, ObservationWorkloadSource, PreparedAnalysisProgram,
};
use runmat_analysis_fea::{
    execution::{FieldRecoveryRequest, SequenceIdentity},
    results::{
        field::{
            RecoveredFieldPartition, RecoveredFieldSample, SampledVolumeFieldTopology,
            VolumeFieldSamplingRule,
        },
        observation::{ObservationEntity, ObservationLimits},
    },
};
use runmat_execution_artifact::{
    cache::{CacheImport, FilesystemObjectStore},
    ArtifactResult, ProgramExecutionResponse,
};
use std::{cell::RefCell, collections::BTreeSet};

struct RecordingStore {
    store: FilesystemObjectStore,
    reads: RefCell<BTreeSet<Digest>>,
}
impl CacheImport for RecordingStore {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.reads.borrow_mut().insert(digest);
        self.store.read_verified(digest)
    }
}

fn spec(
    id: &str,
    quantity: ObservationQuantity,
    component: ObservationComponent,
    reduction: Option<SpatialReduction>,
) -> ObservationRequest {
    let edge = quantity == ObservationQuantity::MagneticEdgeCirculation;
    let location = if edge {
        FieldLocation::Edge
    } else {
        FieldLocation::IntegrationPoint
    };
    let spec = ObservationSpec {
        observation_id: ObservationId(id.into()),
        quantity,
        value_shape: if edge {
            ObservationValueShape::Scalar
        } else {
            ObservationValueShape::Vector3
        },
        target: ObservationTarget::WholeModel,
        source_location: location,
        output_location: location,
        recovery: RecoveryPolicy::Native,
        component,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Real,
        sampling: SamplingPolicy::Final,
        reduction: reduction.map(|spatial| ReductionPolicy {
            spatial,
            series: SeriesReduction::None,
            extrema_ties: ExtremaTiePolicy::FirstCanonical,
        }),
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        cost: ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 1_000_000,
            peak_resident_bytes: 16 * 1024 * 1024,
            stored_bytes: 16 * 1024 * 1024,
            transfer_bytes: 16 * 1024 * 1024,
            collective_count: 0,
        }),
    };
    if reduction.is_some() {
        ObservationRequest::Reduction(spec)
    } else {
        ObservationRequest::Field(spec)
    }
}

fn execute(
    store: &mut FilesystemObjectStore,
    program: &PreparedAnalysisProgram,
    host: &runmat_analysis_execution::AnalysisHostWorkload,
    policy: AnalysisOutputPolicy,
) -> ProgramExecutionResponse {
    program.write_to(store).unwrap();
    let completed = execute_serial_analysis(
        &program.request,
        store,
        &NativeAnalysisKernel,
        policy.chunk_bytes(),
        policy.inventory_limits(),
    )
    .unwrap();
    AnalysisHostResponse::completed(host, &completed)
        .unwrap()
        .program_response()
}

fn source_input(
    task: &PreparedMagnetostaticTask,
    kind: AnalysisNumericalObjectKind,
) -> runmat_analysis_execution::PreparedAnalysisObjects {
    task.inputs
        .iter()
        .find(|input| input.manifest.kind == kind)
        .unwrap()
        .clone()
}

fn replace(
    inputs: &mut [runmat_analysis_execution::PreparedAnalysisObjects],
    kind: AnalysisNumericalObjectKind,
    replacement: runmat_analysis_execution::PreparedAnalysisObjects,
) {
    *inputs
        .iter_mut()
        .find(|input| input.manifest.kind == kind)
        .unwrap() = replacement;
}

#[test]
fn magnetic_authored_native_observations_are_admitted_before_solve() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (mut workload, mesh, _) = fixture::task(order, 2, 1.0, 1.0, 1024);
        let ObservationRequest::Field(native) = spec(
            "B",
            ObservationQuantity::MagneticFluxDensity,
            ObservationComponent::All,
            None,
        ) else {
            panic!("native field specification")
        };
        workload.model.sequence.steps[0].observations =
            vec![ObservationRequest::Field(native.clone())];
        assert!(workload.formulation(&mesh).is_ok());
        let original = workload.identity().unwrap();
        let mut wrong_quantity = native.clone();
        wrong_quantity.quantity = ObservationQuantity::ElectricField;
        let mut wrong_location = native.clone();
        wrong_location.source_location = FieldLocation::Edge;
        wrong_location.output_location = FieldLocation::Edge;
        let mut unknown_selector = native.clone();
        unknown_selector.target =
            ObservationTarget::Spatial(ExcitationTarget::Region(RegionTarget::volume(
                runmat_meshing_core::fixtures::fixture_region("missing-magnetic-volume"),
            )));
        let mut understated_cost = native;
        understated_cost.cost = ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 1,
            peak_resident_bytes: 1,
            stored_bytes: 1,
            transfer_bytes: 1,
            collective_count: 0,
        });
        for invalid in [
            wrong_quantity,
            wrong_location,
            unknown_selector,
            understated_cost,
        ] {
            let mut changed = workload.clone();
            changed.model.sequence.steps[0].observations = vec![ObservationRequest::Field(invalid)];
            // This is newly authored, canonically reidentified intent, not a stale result/request
            // substitution. The source model stays intact when pre-solve admission rejects it.
            let changed_identity = changed.identity().unwrap();
            assert_ne!(changed_identity, original);
            assert!(changed.formulation(&mesh).is_err());
        }
        assert_eq!(workload.identity().unwrap(), original);
    }
}

#[test]
fn magnetic_native_fields_reopen_selective_blocks_and_shared_observation_workers_preserve_values() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (mut workload, mesh, original) = fixture::task(order, 2, 1.0, 1.0, 1024);
        workload.model.sequence.steps[0].observations = vec![
            spec(
                "B",
                ObservationQuantity::MagneticFluxDensity,
                ObservationComponent::All,
                None,
            ),
            spec(
                "By",
                ObservationQuantity::MagneticFluxDensity,
                ObservationComponent::Y,
                None,
            ),
            spec(
                "Hnorm",
                ObservationQuantity::MagneticFieldIntensity,
                ObservationComponent::Magnitude,
                None,
            ),
            spec(
                "edge_min",
                ObservationQuantity::MagneticEdgeCirculation,
                ObservationComponent::All,
                Some(SpatialReduction::Minimum),
            ),
            spec(
                "edge_max",
                ObservationQuantity::MagneticEdgeCirculation,
                ObservationComponent::All,
                Some(SpatialReduction::Maximum),
            ),
        ];
        let task = prepare_magnetostatic_task(MagnetostaticTaskPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            authored_study_digest: Digest::sha256(b"native magnetic authored observations"),
            frozen_project_digest: None,
            program_revision: original.program.request.recipe.program_revision.clone(),
            artifact_access: original.host.artifact_access.clone(),
            workload: workload.clone(),
            mesh: mesh.clone(),
            output_policy: original.output_policy,
            required_capabilities: Default::default(),
        })
        .unwrap();
        let limits = task.output_policy.inventory_limits();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("published");
        let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let response = execute(&mut store, &task.program, &task.host, task.output_policy);
        let solved = import_magnetostatic_result(&store, response, &task, limits).unwrap();
        let retained = solved.result;
        let sampling = retained.sampling_topology().clone();
        let recovery = prepare_magnetic_field_recovery(MagneticFieldRecoveryPreparation {
            source_host: task.host.clone(),
            workload: source_input(&task, AnalysisNumericalObjectKind::MagnetostaticWorkload),
            mesh: source_input(&task, AnalysisNumericalObjectKind::Mesh),
            source_result: solved.objects.clone(),
            program_revision: task.program.request.recipe.program_revision.clone(),
            output_policy: task.output_policy,
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        })
        .unwrap();
        let invoke = |inputs: &[runmat_analysis_execution::PreparedAnalysisObjects], policy| {
            NativeAnalysisKernel.execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::RecoverField,
                host: &recovery.host,
                inputs,
                output_policy: policy,
            })
        };
        let mut tight = limits;
        tight.max_objects = 1;
        let error = runmat_analysis_execution::build_magnetic_recovery_metadata_resource_plan(
            &recovery.request,
            &recovery.inputs,
            AnalysisOutputPolicy::new(recovery.output_policy.chunk_bytes(), tight).unwrap(),
            runmat_analysis_fea::planning::resource::ByteCount(0),
            runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("partition inventory exceeds output policy"));
        let mut missing = recovery.inputs.clone();
        missing.retain(|input| input.manifest.kind != AnalysisNumericalObjectKind::SolverResult);
        assert!(invoke(&missing, recovery.output_policy).is_err());
        assert!(invoke(
            &recovery.inputs,
            AnalysisOutputPolicy::new(8192, limits).unwrap()
        )
        .is_err());
        let (_, _, foreign_task) = fixture::task(order, 2, -2.0, 1.0, 1024);
        let foreign_result = result(&foreign_task);
        let foreign_result_objects = prepare_analysis_objects(
            AnalysisNumericalObjectKind::SolverResult,
            foreign_result.source_result_identity().unwrap().digest(),
            &foreign_result.canonical_bytes().unwrap(),
            1024,
            ObjectNamespace::ResultValue,
            limits,
        )
        .unwrap();
        let mut foreign = recovery.inputs.clone();
        replace(
            &mut foreign,
            AnalysisNumericalObjectKind::SolverResult,
            foreign_result_objects,
        );
        assert!(invoke(&foreign, recovery.output_policy).is_err());
        let foreign_request = FieldRecoveryRequest::new(
            recovery.request.producer,
            runmat_analysis_fea::execution::FieldRecoverySource::SolverResult(
                foreign_result.source_result_identity().unwrap(),
            ),
            recovery.request.topology,
            recovery.request.plan.clone(),
        )
        .unwrap();
        let mut foreign = recovery.inputs.clone();
        replace(
            &mut foreign,
            AnalysisNumericalObjectKind::FieldRecoveryRequest,
            prepare_analysis_objects(
                AnalysisNumericalObjectKind::FieldRecoveryRequest,
                foreign_request.identity().digest(),
                &foreign_request.canonical_bytes().unwrap(),
                1024,
                ObjectNamespace::InputValue,
                limits,
            )
            .unwrap(),
        );
        assert!(invoke(&foreign, recovery.output_policy).is_err());
        let mut foreign = recovery.inputs.clone();
        replace(
            &mut foreign,
            AnalysisNumericalObjectKind::MagnetostaticWorkload,
            source_input(
                &foreign_task,
                AnalysisNumericalObjectKind::MagnetostaticWorkload,
            ),
        );
        assert!(invoke(&foreign, recovery.output_policy).is_err());
        let foreign_order = if order == ElementOrder::Tet4 {
            ElementOrder::Tet10
        } else {
            ElementOrder::Tet4
        };
        let (_, foreign_mesh, foreign_order_task) = fixture::task(foreign_order, 2, 1.0, 1.0, 1024);
        let mut foreign = recovery.inputs.clone();
        replace(
            &mut foreign,
            AnalysisNumericalObjectKind::Mesh,
            source_input(&foreign_order_task, AnalysisNumericalObjectKind::Mesh),
        );
        assert!(invoke(&foreign, recovery.output_policy).is_err());
        let mut foreign = recovery.inputs.clone();
        replace(
            &mut foreign,
            AnalysisNumericalObjectKind::ResourcePlan,
            source_input(&task, AnalysisNumericalObjectKind::ResourcePlan),
        );
        assert!(invoke(&foreign, recovery.output_policy).is_err());
        let wrong_sampling = SampledVolumeFieldTopology::derive(
            &mesh,
            VolumeFieldSamplingRule::TetrahedronFourPoint,
        )
        .unwrap();
        let wrong_request = FieldRecoveryRequest::new(
            recovery.request.producer,
            recovery.request.source,
            wrong_sampling.identity().unwrap(),
            recovery.request.plan.clone(),
        );
        if let Ok(request) = wrong_request {
            let objects = prepare_analysis_objects(
                AnalysisNumericalObjectKind::FieldRecoveryRequest,
                request.identity().digest(),
                &request.canonical_bytes().unwrap(),
                1024,
                ObjectNamespace::InputValue,
                limits,
            )
            .unwrap();
            let mut foreign = recovery.inputs.clone();
            replace(
                &mut foreign,
                AnalysisNumericalObjectKind::FieldRecoveryRequest,
                objects,
            );
            assert!(invoke(&foreign, recovery.output_policy).is_err());
        }
        let response = execute(
            &mut store,
            &recovery.program,
            &recovery.host,
            recovery.output_policy,
        );
        drop(store);
        let cold = RecordingStore {
            store: FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap(),
            reads: RefCell::new(BTreeSet::new()),
        };
        let fields = import_magnetic_recovered_fields(&cold, response, &recovery, limits).unwrap();
        assert_eq!(
            fields.manifest.recovery_request,
            recovery.request.identity()
        );
        // Catalog-only reopening must not gather the field value partitions.
        let data_chunks = fields
            .catalog_objects
            .manifest
            .partitions
            .iter()
            .skip(1)
            .flat_map(|partition| {
                fields
                    .catalog_objects
                    .manifest
                    .chunks
                    .iter()
                    .skip(partition.first_chunk_ordinal as usize)
                    .take(partition.chunk_count as usize)
            })
            .map(|chunk| chunk.digest)
            .collect::<BTreeSet<_>>();
        assert!(cold.reads.borrow().is_disjoint(&data_chunks));
        // Expected values are the retained native fields themselves, by field identity: the
        // integration-point samples and their element averages alike.
        let native = retained
            .recover_native_fields_from_result(
                &retained.native_field_request_from_result().unwrap(),
            )
            .unwrap();
        for entry in &fields.manifest.fields {
            let expected = native
                .fields()
                .iter()
                .find(|field| field.storage().field_id == entry.definition.field_id())
                .expect("published field is a native field")
                .storage()
                .as_host_f64()
                .unwrap()
                .to_vec();
            let width =
                if entry.definition.quantity() == ObservationQuantity::MagneticEdgeCirculation {
                    1
                } else {
                    3
                };
            let mut read_count = 0;
            for descriptor in &entry.partitions {
                let imported = import_analysis_partition(
                    &cold,
                    AnalysisPartitionImport {
                        root: &fields.output.root,
                        expected_kind: AnalysisNumericalObjectKind::Field,
                        expected_logical_identity: fields.output.logical_identity,
                        expected_payload_digest: fields.output.payload_digest,
                        partition_identity: descriptor.partition.digest(),
                        access: &recovery.host.artifact_access,
                        limits,
                    },
                )
                .unwrap();
                let partition =
                    RecoveredFieldPartition::from_canonical_bytes(&imported.payload()).unwrap();
                for (offset, &index) in partition.source_indices.iter().enumerate() {
                    assert_eq!(
                        &partition.storage.as_host_f64().unwrap()
                            [offset * width..(offset + 1) * width],
                        &expected[index * width..(index + 1) * width]
                    );
                    read_count += 1;
                }
            }
            assert_eq!(read_count * width, expected.len());
        }
        let preparation =
            |sampling: SampledVolumeFieldTopology,
             mesh_objects: runmat_analysis_execution::PreparedAnalysisObjects| {
                ObservationPreparation {
                    source_host: recovery.host.clone(),
                    source: ObservationPublishedSource::SampledRecovery {
                        recovery_request: recovery.request.clone(),
                        fields: fields.clone(),
                        sampling,
                    },
                    source_workload: ObservationWorkloadSource::Magnetostatic(source_input(
                        &task,
                        AnalysisNumericalObjectKind::MagnetostaticWorkload,
                    )),
                    mesh: Some(mesh_objects),
                    accelerator_requests: Vec::new(),
                    sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence)
                        .unwrap(),
                    step: workload.model.sequence.steps[0].clone(),
                    limits: ObservationLimits::default(),
                    program_revision: task.program.request.recipe.program_revision.clone(),
                    chunk_bytes: 1024,
                    inventory_limits: limits,
                    local_execution:
                        runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
                }
            };
        assert!(prepare_observation_fanout(
            &cold,
            preparation(
                wrong_sampling,
                source_input(&task, AnalysisNumericalObjectKind::Mesh)
            ),
            RecoveredFieldSample::Final,
            None
        )
        .is_err());
        assert!(prepare_observation_fanout(
            &cold,
            preparation(
                sampling.clone(),
                source_input(&foreign_order_task, AnalysisNumericalObjectKind::Mesh)
            ),
            RecoveredFieldSample::Final,
            None
        )
        .is_err());
        assert!(sampling.validate_for_mesh(&foreign_mesh).is_err());
        let mut foreign_preparation = preparation(
            sampling.clone(),
            source_input(&task, AnalysisNumericalObjectKind::Mesh),
        );
        foreign_preparation.source_workload =
            ObservationWorkloadSource::Magnetostatic(source_input(
                &foreign_task,
                AnalysisNumericalObjectKind::MagnetostaticWorkload,
            ));
        assert!(prepare_observation_fanout(
            &cold,
            foreign_preparation,
            RecoveredFieldSample::Final,
            None
        )
        .is_err());
        cold.reads.borrow_mut().clear();
        let fanout = prepare_observation_fanout(
            &cold,
            preparation(
                sampling.clone(),
                source_input(&task, AnalysisNumericalObjectKind::Mesh),
            ),
            RecoveredFieldSample::Final,
            None,
        )
        .unwrap();
        // A is published and verified above, but no observation requests it: its value blocks
        // must remain unread while B/H/edge observations admit their selected canonical blocks.
        let potential = fields
            .manifest
            .fields
            .iter()
            .find(|entry| {
                entry.definition.quantity() == ObservationQuantity::MagneticVectorPotential
            })
            .unwrap();
        for block in &potential.partitions {
            let partition = fields
                .catalog_objects
                .manifest
                .partitions
                .iter()
                .find(|p| p.logical_identity == block.partition.digest())
                .unwrap();
            for chunk in fields
                .catalog_objects
                .manifest
                .chunks
                .iter()
                .skip(partition.first_chunk_ordinal as usize)
                .take(partition.chunk_count as usize)
            {
                assert!(!cold.reads.borrow().contains(&chunk.digest));
            }
        }
        assert!(!cold.reads.borrow().is_disjoint(&data_chunks));
        drop(cold);
        let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        for observation in &fanout.source_request.plan.observations {
            let selected = fanout
                .partitions
                .iter()
                .filter(|partition| {
                    partition.partition_plan.observation_id == observation.spec.observation_id
                })
                .collect::<Vec<_>>();
            let mut contributions = Vec::new();
            for partition in &selected {
                let mut foreign = partition.inputs.clone();
                replace(
                    &mut foreign,
                    AnalysisNumericalObjectKind::ResourcePlan,
                    source_input(&task, AnalysisNumericalObjectKind::ResourcePlan),
                );
                assert!(NativeAnalysisKernel
                    .execute(AnalysisKernelInvocation {
                        kind: AnalysisWorkKind::EvaluateObservationPartition,
                        host: &partition.host,
                        inputs: &foreign,
                        output_policy: task.output_policy
                    })
                    .is_err());
                let response = execute(
                    &mut store,
                    &partition.program,
                    &partition.host,
                    task.output_policy,
                );
                let reopened = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
                contributions.push(
                    import_observation_partition_contribution(
                        &reopened, response, partition, limits,
                    )
                    .unwrap(),
                );
            }
            let reduction = prepare_observation_reduction(ObservationReductionPreparation {
                source_host: selected[0].host.clone(),
                source_request: fanout.source_request.clone(),
                partition_plan: selected[0].partition_plan.clone(),
                contributions,
                program_revision: task.program.request.recipe.program_revision.clone(),
                chunk_bytes: 1024,
                inventory_limits: limits,
            })
            .unwrap();
            let response = execute(
                &mut store,
                &reduction.program,
                &reduction.host,
                task.output_policy,
            );
            let reopened = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
            let actual = import_reduced_observation(&reopened, response, &reduction, limits)
                .unwrap()
                .result;
            let values = actual.field.as_host_f64().unwrap();
            match observation.spec.observation_id.0.as_str() {
                "B" => {
                    let expected = retained
                        .response()
                        .flux_density_t
                        .canonical_values()
                        .clone();
                    assert_eq!(values, expected);
                    assert_eq!(
                        actual.entities,
                        (0..sampling.integration_sample_count())
                            .map(|i| ObservationEntity::IntegrationPoint(
                                sampling.integration_sample_identity(i).unwrap()
                            ))
                            .collect::<Vec<_>>()
                    );
                }
                "By" => assert_eq!(
                    values,
                    (0..retained.response().sample_count())
                        .map(|i| retained.response().flux_density(i)[1])
                        .collect::<Vec<_>>()
                ),
                "Hnorm" => assert_eq!(
                    values,
                    (0..retained.response().sample_count())
                        .map(|i| retained
                            .response()
                            .magnetic_field(i)
                            .iter()
                            .map(|v| v * v)
                            .sum::<f64>()
                            .sqrt())
                        .collect::<Vec<_>>()
                ),
                "edge_min" | "edge_max" => {
                    let expected = if observation.spec.observation_id.0 == "edge_min" {
                        retained
                            .response()
                            .edge_circulations_wb
                            .canonical_values()
                            .iter()
                            .copied()
                            .reduce(f64::min)
                            .unwrap()
                    } else {
                        retained
                            .response()
                            .edge_circulations_wb
                            .canonical_values()
                            .iter()
                            .copied()
                            .reduce(f64::max)
                            .unwrap()
                    };
                    assert_eq!(values, &[expected]);
                    assert!(matches!(
                        actual.entities.as_slice(),
                        [ObservationEntity::OrientedEdge(_)]
                    ));
                }
                other => panic!("unexpected observation {other}"),
            }
        }
    }
}
