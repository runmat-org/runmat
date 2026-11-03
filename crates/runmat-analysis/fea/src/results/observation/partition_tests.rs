use runmat_analysis_core::{
    AnalysisField, ComplexRepresentation, CompressionPolicy, CoordinateFrame, ExtremaTiePolicy,
    FieldLocation, ObservationComponent, ObservationCostEstimate, ObservationEvent, ObservationId,
    ObservationPurpose, ObservationQuantity, ObservationRequest, ObservationSpec,
    ObservationStoragePolicy, ObservationTarget, ObservationValueShape, PartialAccessPolicy,
    Position, ProcedureStep, RecoveryPolicy, ReductionPolicy, RetentionPolicy, SamplingPolicy,
    SeriesReduction, SignConvention, SignalPrecision, SolverObservationEvent, SpatialReduction,
};
use runmat_execution::Digest;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

use crate::{
    execution::{SequenceIdentity, SolverResultIdentity, StepIdentity},
    results::field::{
        RecoveredField, RecoveredFieldCatalog, RecoveredFieldDefinition, RecoveredFieldDerivation,
        RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation,
        RecoveredFieldPartitionSet, RecoveredFieldSample,
    },
    topology::AnalysisProjection,
};

use super::{
    distributed::{combine_observation_partitions, evaluate_partitioned_observation},
    evaluate_solid_observation_partition, prepare_solid_observation_field_partition,
    topology::ObservationTopology,
    BoundedObservationHistory, ObservationEntity, ObservationEventAttribution,
    ObservationExecutionPolicy, ObservationLimits, ObservationPartitionContribution,
    ObservationPartitionPlan, SolverEventAttribution, SolverEventSource,
};

fn integration_point_stress_definition() -> RecoveredFieldDefinition {
    RecoveredFieldDefinition::new(
        crate::contracts::FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
        ObservationQuantity::Stress,
        FieldLocation::IntegrationPoint,
        RecoveredFieldEntityDomain::VolumeIntegrationPoints,
        ObservationValueShape::SymmetricTensor3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap()
}

fn definition() -> RecoveredFieldDefinition {
    RecoveredFieldDefinition::new(
        "temperature.partitioned",
        ObservationQuantity::Temperature,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap()
}

pub(super) fn spec(id: &str, target: ObservationTarget) -> ObservationSpec {
    ObservationSpec {
        observation_id: ObservationId(id.into()),
        quantity: ObservationQuantity::Temperature,
        value_shape: ObservationValueShape::Scalar,
        target,
        source_location: FieldLocation::Node,
        output_location: FieldLocation::Node,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Real,
        sampling: SamplingPolicy::Final,
        reduction: None,
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        cost: runmat_analysis_core::ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 100_000,
            peak_resident_bytes: 1_000_000,
            stored_bytes: 1_000_000,
            transfer_bytes: 1_000_000,
            collective_count: 0,
        }),
    }
}

pub(super) fn prepared(
    request: ObservationRequest,
) -> (
    AnalysisProjection,
    super::ResolvedObservationPlan,
    RecoveredField,
) {
    let projection =
        AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
            .unwrap();
    prepared_on(projection, request)
}

fn prepared_on(
    projection: AnalysisProjection,
    request: ObservationRequest,
) -> (
    AnalysisProjection,
    super::ResolvedObservationPlan,
    RecoveredField,
) {
    let definition = definition();
    let mut step = ProcedureStep::thermal_transient("thermal");
    step.independent_domain = runmat_analysis_core::IndependentDomain::Static;
    step.observations = vec![request];
    let plan = super::resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let values = projection
        .nodes
        .iter()
        .map(|node| node.coordinates_m.canonical_value().iter().sum())
        .collect::<Vec<_>>();
    let field = RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64("temperature.partitioned", definition.layout(), vec![values.len()], values),
    )
    .unwrap();
    let catalog = RecoveredFieldCatalog::new(&projection, vec![field.clone()]).unwrap();
    assert_eq!(
        catalog
            .get(field.definition().identity(), RecoveredFieldSample::Final)
            .unwrap(),
        &field
    );
    (projection, plan, field)
}

#[test]
fn canonical_partition_plan_covers_a_selected_field_once() {
    let (projection, plan, field) = prepared(ObservationRequest::Field(spec(
        "field",
        ObservationTarget::WholeModel,
    )));
    let partition_plan = ObservationPartitionPlan::derive(
        &plan,
        &ObservationId("field".into()),
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
    )
    .unwrap();
    assert_eq!(partition_plan.partitions.len(), 3);
    assert_eq!(
        partition_plan.resource_cost.stored_bytes,
        plan.observations[0].resolved_cost.stored_bytes * 3
    );
    assert!(
        partition_plan.resource_cost.transfer_bytes
            > plan.observations[0].resolved_cost.transfer_bytes
    );
    assert_eq!(partition_plan.resource_cost.collective_count, 0);
    assert_eq!(
        partition_plan
            .partitions
            .iter()
            .map(|partition| partition.target.entity_count())
            .sum::<usize>(),
        projection.nodes.len()
    );
    let decoded =
        ObservationPartitionPlan::from_canonical_bytes(&partition_plan.canonical_bytes().unwrap())
            .unwrap();
    decoded.validate_for(&plan).unwrap();
    assert_eq!(decoded, partition_plan);

    let topology = ObservationTopology::from_projection(&projection).unwrap();
    let single = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::serial(),
        None,
    )
    .unwrap();
    let split = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
        None,
    )
    .unwrap();
    assert_eq!(single.logical_identity, split.logical_identity);
    assert_ne!(
        single.partition_evidence.identity,
        split.partition_evidence.identity
    );
}

#[test]
fn integration_point_partitions_reconstruct_exact_entities_and_values() {
    let projection =
        AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
            .unwrap();
    let definition = integration_point_stress_definition();
    let mut request = spec("integration_point_stress", ObservationTarget::WholeModel);
    request.quantity = ObservationQuantity::Stress;
    request.value_shape = ObservationValueShape::SymmetricTensor3;
    request.source_location = FieldLocation::IntegrationPoint;
    request.output_location = FieldLocation::IntegrationPoint;
    let mut step = ProcedureStep::structural_static("static");
    step.observations = vec![ObservationRequest::Field(request)];
    let plan = super::resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
    let field = RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64(
            crate::contracts::FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS, definition.layout(),
            vec![4, 6],
            values.clone(),
        ),
    )
    .unwrap();
    let topology = ObservationTopology::from_projection(&projection).unwrap();

    let serial = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::serial(),
        None,
    )
    .unwrap();
    let partitioned = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
        None,
    )
    .unwrap();

    assert_eq!(partitioned.logical_identity, serial.logical_identity);
    assert_eq!(partitioned.field.as_host_f64().unwrap(), values);
    assert_eq!(partitioned.entities, serial.entities);
    assert_eq!(partitioned.entities.len(), 4);
    assert!(partitioned
        .entities
        .iter()
        .all(|entity| matches!(entity, ObservationEntity::IntegrationPoint(_))));
}

#[test]
fn path_interpolation_is_equivalent_across_partition_layouts() {
    let mut path = spec(
        "path",
        ObservationTarget::Path {
            positions: vec![
                Position::new([0.1, 0.1, 0.1]),
                Position::new([0.2, 0.3, 0.1]),
                Position::new([0.05, 0.15, 0.4]),
            ],
        },
    );
    path.output_location = FieldLocation::Point;
    path.recovery = RecoveryPolicy::ShapeFunctionInterpolation;
    let (projection, plan, field) = prepared(ObservationRequest::Field(path));
    let topology = ObservationTopology::from_projection(&projection).unwrap();
    let serial = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::serial(),
        None,
    )
    .unwrap();
    let partitioned = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
        None,
    )
    .unwrap();
    assert_eq!(serial.logical_identity, partitioned.logical_identity);
    assert_eq!(serial.entities, partitioned.entities);
    assert_eq!(serial.field, partitioned.field);
    assert_ne!(
        serial.partition_evidence.identity,
        partitioned.partition_evidence.identity
    );
}

#[test]
fn materialized_volume_weighted_recovery_is_partition_equivalent() {
    let projection =
        AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
            .unwrap();
    let definition = RecoveredFieldDefinition::new_derived(
        "structural.nodal_von_mises",
        ObservationQuantity::VonMisesStress,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
        RecoveredFieldDerivation::LocationTransfer {
            source_location: FieldLocation::Element,
            recovery: RecoveryPolicy::VolumeWeightedAverage,
        },
    )
    .unwrap();
    let mut observation = spec("nodal-von-mises", ObservationTarget::WholeModel);
    observation.quantity = ObservationQuantity::VonMisesStress;
    observation.source_location = FieldLocation::Element;
    observation.output_location = FieldLocation::Node;
    observation.recovery = RecoveryPolicy::VolumeWeightedAverage;
    let mut step = ProcedureStep::structural_static("structural");
    step.observations = vec![ObservationRequest::Field(observation)];
    let plan = super::resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let values = (0..projection.nodes.len())
        .map(|index| index as f64 + 1.0)
        .collect::<Vec<_>>();
    let field = RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64("structural.nodal_von_mises", definition.layout(), vec![values.len()], values),
    )
    .unwrap();
    let topology = ObservationTopology::from_projection(&projection).unwrap();
    let serial = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::serial(),
        None,
    )
    .unwrap();
    let partitioned = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
        None,
    )
    .unwrap();

    assert_eq!(serial.logical_identity, partitioned.logical_identity);
    assert_eq!(serial.entities, partitioned.entities);
    assert_eq!(serial.field, partitioned.field);
    assert_ne!(
        serial.partition_evidence.identity,
        partitioned.partition_evidence.identity
    );
}

#[test]
fn spatial_extrema_retain_canonical_locations_and_tie_policy_across_partitions() {
    for ties in [
        ExtremaTiePolicy::FirstCanonical,
        ExtremaTiePolicy::AllLocations,
    ] {
        let mut maximum = spec("maximum", ObservationTarget::WholeModel);
        maximum.reduction = Some(ReductionPolicy {
            spatial: SpatialReduction::Maximum,
            series: SeriesReduction::None,
            extrema_ties: ties,
        });
        let (projection, plan, field) = prepared(ObservationRequest::Reduction(maximum));
        let topology = ObservationTopology::from_projection(&projection).unwrap();
        let serial = evaluate_partitioned_observation(
            &plan,
            &plan.observations[0],
            &field,
            &topology,
            ObservationExecutionPolicy::serial(),
            None,
        )
        .unwrap();
        let partitioned = evaluate_partitioned_observation(
            &plan,
            &plan.observations[0],
            &field,
            &topology,
            ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(serial.logical_identity, partitioned.logical_identity);
        assert_eq!(serial.field, partitioned.field);
        assert_eq!(serial.entities, partitioned.entities);
        assert!(serial
            .field
            .as_host_f64()
            .unwrap()
            .iter()
            .all(|value| (*value - 1.0).abs() < 1.0e-12));
        assert_eq!(
            serial.entities.len(),
            if ties == ExtremaTiePolicy::FirstCanonical {
                1
            } else {
                6
            }
        );
    }
}

#[test]
fn path_extrema_return_physical_sample_locations() {
    let mut maximum = spec(
        "path_maximum",
        ObservationTarget::Path {
            positions: vec![
                Position::new([0.0, 0.0, 0.0]),
                Position::new([1.0, 0.0, 0.0]),
                Position::new([0.0, 1.0, 0.0]),
            ],
        },
    );
    maximum.output_location = FieldLocation::Point;
    maximum.recovery = RecoveryPolicy::ShapeFunctionInterpolation;
    maximum.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Maximum,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::AllLocations,
    });
    let (projection, plan, field) = prepared(ObservationRequest::Reduction(maximum));
    let topology = ObservationTopology::from_projection(&projection).unwrap();
    let result = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
        None,
    )
    .unwrap();
    assert!(result
        .field
        .as_host_f64()
        .unwrap()
        .iter()
        .all(|value| (*value - 1.0).abs() < 1.0e-12));
    assert_eq!(result.entities.len(), 2);
    assert!(matches!(
        result.entities.as_slice(),
        [
            super::ObservationEntity::Point { ordinal: 1, .. },
            super::ObservationEntity::Point { ordinal: 2, .. }
        ]
    ));
    super::publish_observation(&plan, &result, 1024).unwrap();
}

// witness: FEA-SHARED-OBSERVATIONS-001/bounded_partitioned_evaluation @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Partitioned boundary reduction matches single-partition evaluation.
#[test]
fn partitioned_boundary_reduction_matches_single_partition_evaluation() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let shared_face = mesh.topology.boundary_faces[0].provenance[0].clone();
    for face in &mut mesh.topology.boundary_faces {
        face.provenance = vec![shared_face.clone()];
    }
    mesh.seal_canonical_digest().unwrap();
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let face_selector = projection.boundary_faces[0].provenance[0]
        .analysis_selection_id()
        .unwrap();
    let mut observation = spec(
        "boundary_average",
        ObservationTarget::Spatial(runmat_analysis_core::ExcitationTarget::Region(
            runmat_analysis_core::RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&(face_selector)),
            ),
        )),
    );
    observation.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Average,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    observation.output_location = FieldLocation::Global;
    let (projection, plan, field) =
        prepared_on(projection, ObservationRequest::Reduction(observation));
    let topology = ObservationTopology::from_projection(&projection).unwrap();
    let resolved = &plan.observations[0];
    let single = evaluate_partitioned_observation(
        &plan,
        resolved,
        &field,
        &topology,
        ObservationExecutionPolicy::serial(),
        None,
    )
    .unwrap();
    let split = evaluate_partitioned_observation(
        &plan,
        resolved,
        &field,
        &topology,
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
        None,
    )
    .unwrap();
    let split_plan = ObservationPartitionPlan::derive(
        &plan,
        &resolved.spec.observation_id,
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
    )
    .unwrap();
    assert_eq!(split_plan.resource_cost.collective_count, 2);
    assert!(split_plan.resource_cost.transfer_bytes > resolved.resolved_cost.transfer_bytes);
    assert_eq!(single.entities, split.entities);
    assert_eq!(single.field.shape, split.field.shape);
    for (single, split) in single
        .field
        .as_host_f64()
        .unwrap()
        .iter()
        .zip(split.field.as_host_f64().unwrap())
    {
        assert!((single - split).abs() <= 1.0e-14);
    }
}

#[test]
fn partitioned_history_restart_preserves_placement_neutral_result_identity() {
    for retention in [
        RetentionPolicy::Canonical,
        RetentionPolicy::LatestSamples { count: 1 },
    ] {
        let projection =
            AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
                .unwrap();
        let definition = definition();
        let mut history_spec = spec("partitioned_history", ObservationTarget::WholeModel);
        history_spec.sampling = SamplingPolicy::TimeSeconds(vec![0.0, 1.0]);
        history_spec.storage.retention = retention;
        let mut step = ProcedureStep::thermal_transient("partitioned-history");
        step.independent_domain = runmat_analysis_core::IndependentDomain::TimeSeconds {
            origin: 0.0,
            start: 0.0,
            end: 1.0,
            interval_count: 10,
        };
        step.observations = vec![ObservationRequest::History(history_spec)];
        let plan = super::resolve_observation_plan(
            &step,
            &projection,
            std::slice::from_ref(&definition),
            ObservationLimits::default(),
        )
        .unwrap();
        let observation_id = ObservationId("partitioned_history".into());
        let topology = ObservationTopology::from_projection(&projection).unwrap();
        let result_at = |time: f64, policy: ObservationExecutionPolicy| {
            let coordinate = RecoveredFieldSample::TimeSeconds(time);
            let values = projection
                .nodes
                .iter()
                .map(|node| node.coordinates_m.canonical_value().iter().sum::<f64>() + time)
                .collect::<Vec<_>>();
            let field = RecoveredField::new(
                definition.clone(),
                coordinate,
                AnalysisField::host_f64("temperature.partitioned", definition.layout(), vec![values.len()], values),
            )
            .unwrap();
            evaluate_partitioned_observation(
                &plan,
                &plan.observations[0],
                &field,
                &topology,
                policy,
                None,
            )
            .unwrap()
        };
        let split_policy = ObservationExecutionPolicy::canonical(
            std::num::NonZeroU32::new(3).expect("nonzero partition count"),
        );
        let mut partitioned = BoundedObservationHistory::new(&plan, &observation_id).unwrap();
        let first = result_at(0.0, split_policy);
        assert!(first.partition_evidence.contributions.len() > 1);
        let serial_first = result_at(0.0, ObservationExecutionPolicy::serial());
        assert_eq!(first.logical_identity, serial_first.logical_identity);
        let sequence = SequenceIdentity::new(Digest::sha256("checkpoint sequence")).unwrap();
        let producer = StepIdentity::new(sequence, Digest::sha256("checkpoint producer")).unwrap();
        let request_for = |result: &super::ObservationResult| {
            crate::execution::ObservationHistoryCheckpointRequest::new(
                plan.clone(),
                producer,
                observation_id.clone(),
                None,
                0,
                result,
            )
            .unwrap()
        };
        let split_request = request_for(&first);
        let serial_request = request_for(&serial_first);
        assert_ne!(split_request.identity, serial_request.identity);
        split_request.validate_advance_inputs(None, &first).unwrap();
        assert!(split_request
            .validate_advance_inputs(None, &serial_first)
            .is_err());
        assert!(serial_request
            .validate_advance_inputs(None, &first)
            .is_err());
        let mut legacy = split_request.clone();
        legacy.schema_version = 1;
        assert!(legacy.validate().is_err());
        partitioned
            .push(RecoveredFieldSample::TimeSeconds(0.0), first)
            .unwrap();
        let checkpoint = partitioned.checkpoint(None).unwrap();
        let checkpoint_identity = checkpoint.manifest().identity;
        let mut partitioned =
            BoundedObservationHistory::resume(&plan, &observation_id, checkpoint).unwrap();
        partitioned
            .push(
                RecoveredFieldSample::TimeSeconds(1.0),
                result_at(1.0, split_policy),
            )
            .unwrap();
        let continued = partitioned.checkpoint(Some(checkpoint_identity)).unwrap();
        assert_eq!(continued.manifest().predecessor, Some(checkpoint_identity));
        let partitioned_checkpoint = continued;
        let partitioned = partitioned.finish().unwrap();

        let mut serial = BoundedObservationHistory::new(&plan, &observation_id).unwrap();
        for time in [0.0, 1.0] {
            serial
                .push(
                    RecoveredFieldSample::TimeSeconds(time),
                    result_at(time, ObservationExecutionPolicy::serial()),
                )
                .unwrap();
        }
        let serial_checkpoint = serial.checkpoint(None).unwrap();
        assert_ne!(
            partitioned_checkpoint
                .manifest()
                .observed_execution_prefix_digest,
            serial_checkpoint
                .manifest()
                .observed_execution_prefix_digest
        );
        assert_ne!(
            partitioned_checkpoint.manifest().identity,
            serial_checkpoint.manifest().identity
        );
        let mut substituted = partitioned_checkpoint.samples().to_vec();
        let RecoveredFieldSample::TimeSeconds(time) = substituted[0].coordinate else {
            panic!("expected time coordinate")
        };
        substituted[0].result = result_at(time, ObservationExecutionPolicy::serial());
        assert_eq!(
            substituted[0].identity,
            partitioned_checkpoint.samples()[0].identity
        );
        assert!(
            super::ObservationHistoryCheckpoint::from_manifest_and_samples(
                partitioned_checkpoint.manifest().clone(),
                substituted
            )
            .is_err()
        );
        let resumed = BoundedObservationHistory::resume(
            &plan,
            &observation_id,
            partitioned_checkpoint.clone(),
        )
        .unwrap();
        assert_eq!(
            resumed
                .checkpoint(partitioned_checkpoint.manifest().predecessor)
                .unwrap(),
            partitioned_checkpoint
        );
        let serial = serial.finish().unwrap();
        assert_eq!(partitioned.manifest().identity, serial.manifest().identity);
        assert_eq!(
            partitioned.manifest().observed_prefix_digest,
            serial.manifest().observed_prefix_digest
        );
    }
}

pub(super) fn checkpoint_test_evidence(time: f64) -> super::ObservationPartitionEvidence {
    let projection =
        AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
            .unwrap();
    let definition = definition();
    let mut requested = spec("temperature", ObservationTarget::WholeModel);
    requested.sampling = SamplingPolicy::TimeSeconds(vec![0.0, 1.0, 2.0]);
    let mut step = ProcedureStep::thermal_transient("checkpoint");
    step.independent_domain = runmat_analysis_core::IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 2.0,
        interval_count: 2,
    };
    step.observations = vec![ObservationRequest::History(requested)];
    let plan = super::resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let field = RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::TimeSeconds(time),
        AnalysisField::host_f64(
            "temperature.partitioned", definition.layout(),
            vec![projection.nodes.len()],
            vec![time; projection.nodes.len()],
        ),
    )
    .unwrap();
    evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &ObservationTopology::from_projection(&projection).unwrap(),
        ObservationExecutionPolicy::serial(),
        None,
    )
    .unwrap()
    .partition_evidence
}

#[test]
fn pressure_resultant_and_offset_moment_match_the_divergence_theorem() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let boundary = mesh.topology.boundary_faces[0].provenance[0].clone();
    for face in &mut mesh.topology.boundary_faces {
        face.provenance = vec![boundary.clone()];
    }
    mesh.seal_canonical_digest().unwrap();
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let selector = boundary.analysis_selection_id().unwrap();
    let definition = RecoveredFieldDefinition::new(
        "contact.pressure",
        ObservationQuantity::ContactPressure,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let target = ObservationTarget::Spatial(runmat_analysis_core::ExcitationTarget::Region(
        runmat_analysis_core::RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_boundary_selection(&(selector)),
        ),
    ));
    let mut resultant = spec("pressure_resultant", target.clone());
    resultant.quantity = ObservationQuantity::ContactPressure;
    resultant.output_location = FieldLocation::Global;
    resultant.sign_convention = SignConvention::OutwardPositive;
    resultant.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Resultant,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    let about = Position::new([0.0, 1.0, 0.0]);
    let mut moment = resultant.clone();
    moment.observation_id = ObservationId("pressure_moment".into());
    moment.reduction.as_mut().unwrap().spatial = SpatialReduction::Moment { about };
    let mut step = ProcedureStep::structural_nonlinear("pressure-resultants");
    step.observations = vec![
        ObservationRequest::Reduction(resultant),
        ObservationRequest::Reduction(moment),
    ];
    let plan = super::resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let values = projection
        .nodes
        .iter()
        .map(|node| node.coordinates_m.canonical_value()[0])
        .collect::<Vec<_>>();
    let field = RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64("contact.pressure", definition.layout(), vec![values.len()], values),
    )
    .unwrap();
    let topology = ObservationTopology::from_projection(&projection).unwrap();
    let policy = ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap());
    let resultant = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        policy,
        None,
    )
    .unwrap();
    let moment = evaluate_partitioned_observation(
        &plan,
        &plan.observations[1],
        &field,
        &topology,
        policy,
        None,
    )
    .unwrap();

    let volume = projection
        .elements
        .iter()
        .map(|element| element.signed_volume_m3)
        .sum::<f64>();
    let centroid = projection.elements[0].centroid_m;
    let expected_resultant = [volume, 0.0, 0.0];
    let expected_moment = [
        0.0,
        (centroid[2] - about.canonical_value()[2]) * volume,
        -(centroid[1] - about.canonical_value()[1]) * volume,
    ];
    assert_eq!(resultant.value_shape, ObservationValueShape::Vector3);
    assert_eq!(moment.value_shape, ObservationValueShape::Vector3);
    assert!(resultant.entities.is_empty() && moment.entities.is_empty());
    for (actual, expected) in resultant
        .field
        .as_host_f64()
        .unwrap()
        .iter()
        .zip(expected_resultant)
    {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    for (actual, expected) in moment
        .field
        .as_host_f64()
        .unwrap()
        .iter()
        .zip(expected_moment)
    {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
}

#[test]
fn complex_volume_resultant_and_moment_preserve_the_complete_phasor() {
    let projection =
        AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
            .unwrap();
    let definition = RecoveredFieldDefinition::new(
        "magnetic_flux_density.phasor",
        ObservationQuantity::MagneticFluxDensity,
        FieldLocation::Element,
        RecoveredFieldEntityDomain::VolumeElements,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::ComplexInterleaved,
    )
    .unwrap();
    let mut resultant = spec("flux_resultant", ObservationTarget::WholeModel);
    resultant.quantity = ObservationQuantity::MagneticFluxDensity;
    resultant.value_shape = ObservationValueShape::Vector3;
    resultant.source_location = FieldLocation::Element;
    resultant.output_location = FieldLocation::Global;
    resultant.complex_representation = ComplexRepresentation::Complex;
    resultant.storage.precision = SignalPrecision::Complex128;
    resultant.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Resultant,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    let about = Position::new([0.25, -0.5, 1.0]);
    let mut moment = resultant.clone();
    moment.observation_id = ObservationId("flux_moment".into());
    moment.reduction.as_mut().unwrap().spatial = SpatialReduction::Moment { about };
    let mut invalid_magnitude = resultant.clone();
    invalid_magnitude.observation_id = ObservationId("flux_magnitude_resultant".into());
    invalid_magnitude.complex_representation = ComplexRepresentation::Magnitude;
    invalid_magnitude.storage.precision = SignalPrecision::Real64;

    let mut invalid_step = ProcedureStep::electromagnetic_harmonic("invalid-flux-resultant");
    invalid_step.observations = vec![ObservationRequest::Reduction(invalid_magnitude)];
    let error = super::resolve_observation_plan(
        &invalid_step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("complete complex phasor"),
        "{error}"
    );

    let mut step = ProcedureStep::electromagnetic_harmonic("complex-flux-resultants");
    step.observations = vec![
        ObservationRequest::Reduction(resultant),
        ObservationRequest::Reduction(moment),
    ];
    let plan = super::resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let phasor = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let field = RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64(
            "magnetic_flux_density.phasor", definition.layout(),
            vec![projection.elements.len(), 3, 2],
            phasor.repeat(projection.elements.len()),
        ),
    )
    .unwrap();
    let topology = ObservationTopology::from_projection(&projection).unwrap();
    let volume = projection
        .elements
        .iter()
        .map(|element| element.signed_volume_m3)
        .sum::<f64>();
    let expected_resultant = phasor.map(|value| value * volume);
    let mut expected_moment = [0.0; 6];
    for element in &projection.elements {
        let arm = [
            element.centroid_m[0] - about.canonical_value()[0],
            element.centroid_m[1] - about.canonical_value()[1],
            element.centroid_m[2] - about.canonical_value()[2],
        ];
        for (offset, force) in [
            [phasor[0], phasor[2], phasor[4]],
            [phasor[1], phasor[3], phasor[5]],
        ]
        .into_iter()
        .enumerate()
        {
            let moment = [
                arm[1] * force[2] - arm[2] * force[1],
                arm[2] * force[0] - arm[0] * force[2],
                arm[0] * force[1] - arm[1] * force[0],
            ];
            expected_moment[offset] += moment[0] * element.signed_volume_m3;
            expected_moment[2 + offset] += moment[1] * element.signed_volume_m3;
            expected_moment[4 + offset] += moment[2] * element.signed_volume_m3;
        }
    }
    for (observation, expected) in plan
        .observations
        .iter()
        .zip([expected_resultant, expected_moment])
    {
        let serial = evaluate_partitioned_observation(
            &plan,
            observation,
            &field,
            &topology,
            ObservationExecutionPolicy::serial(),
            None,
        )
        .unwrap();
        let split = evaluate_partitioned_observation(
            &plan,
            observation,
            &field,
            &topology,
            ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(serial.field.shape, vec![3, 2]);
        assert_eq!(serial.logical_identity, split.logical_identity);
        for ((actual, split), expected) in serial
            .field
            .as_host_f64()
            .unwrap()
            .iter()
            .zip(split.field.as_host_f64().unwrap())
            .zip(expected)
        {
            assert!((actual - expected).abs() <= 1.0e-12);
            assert!((split - expected).abs() <= 1.0e-12);
        }
    }
}

#[test]
fn boundary_stress_resultant_uses_cauchy_traction_and_rejects_ambiguous_pressure_sign() {
    let projection =
        AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
            .unwrap();
    let face = &projection.boundary_faces[0];
    let target = ObservationTarget::Spatial(runmat_analysis_core::ExcitationTarget::Region(
        runmat_analysis_core::RegionTarget::boundary(
            face.provenance[0].analysis_selection_id().unwrap(),
        ),
    ));
    let definition = RecoveredFieldDefinition::new(
        "stress.traction",
        ObservationQuantity::Stress,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::SymmetricTensor3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut observation = spec("stress_resultant", target.clone());
    observation.quantity = ObservationQuantity::Stress;
    observation.value_shape = ObservationValueShape::SymmetricTensor3;
    observation.output_location = FieldLocation::Global;
    observation.sign_convention = SignConvention::TensionPositive;
    observation.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Resultant,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    let mut step = ProcedureStep::structural_static("stress-resultant");
    step.observations = vec![ObservationRequest::Reduction(observation)];
    let plan = super::resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let tensor = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let field = RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64(
            "stress.traction", definition.layout(),
            vec![projection.nodes.len(), 6],
            tensor.repeat(projection.nodes.len()),
        ),
    )
    .unwrap();
    let result = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &ObservationTopology::from_projection(&projection).unwrap(),
        ObservationExecutionPolicy::serial(),
        None,
    )
    .unwrap();
    let area = face.oriented_area_vector_m2;
    let expected = [
        tensor[0] * area[0] + tensor[3] * area[1] + tensor[5] * area[2],
        tensor[3] * area[0] + tensor[1] * area[1] + tensor[4] * area[2],
        tensor[5] * area[0] + tensor[4] * area[1] + tensor[2] * area[2],
    ];
    for (actual, expected) in result.field.as_host_f64().unwrap().iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }

    let pressure_definition = RecoveredFieldDefinition::new(
        "pressure.ambiguous",
        ObservationQuantity::ContactPressure,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut pressure = spec("ambiguous_pressure", target);
    pressure.quantity = ObservationQuantity::ContactPressure;
    pressure.output_location = FieldLocation::Global;
    pressure.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Resultant,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    let mut step = ProcedureStep::structural_nonlinear("ambiguous-pressure");
    step.observations = vec![ObservationRequest::Reduction(pressure)];
    assert!(super::resolve_observation_plan(
        &step,
        &projection,
        &[pressure_definition],
        ObservationLimits::default(),
    )
    .unwrap_err()
    .to_string()
    .contains("explicitly oriented boundary pressure"));
}

#[test]
fn complex_boundary_pressure_resultant_is_partition_equivalent() {
    let projection =
        AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
            .unwrap();
    let face = &projection.boundary_faces[0];
    let target = ObservationTarget::Spatial(runmat_analysis_core::ExcitationTarget::Region(
        runmat_analysis_core::RegionTarget::boundary(
            face.provenance[0].analysis_selection_id().unwrap(),
        ),
    ));
    let definition = RecoveredFieldDefinition::new(
        "pressure.phasor",
        ObservationQuantity::AcousticPressure,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::ComplexInterleaved,
    )
    .unwrap();
    let mut observation = spec("pressure_phasor_resultant", target);
    observation.quantity = ObservationQuantity::AcousticPressure;
    observation.output_location = FieldLocation::Global;
    observation.sign_convention = SignConvention::OutwardPositive;
    observation.complex_representation = ComplexRepresentation::Complex;
    observation.storage.precision = SignalPrecision::Complex128;
    observation.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Resultant,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    let mut step = ProcedureStep::acoustic_harmonic("complex-pressure-resultant");
    step.observations = vec![ObservationRequest::Reduction(observation)];
    let plan = super::resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let pressure = [2.0, -3.0];
    let field = RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64(
            "pressure.phasor", definition.layout(),
            vec![projection.nodes.len(), 2],
            pressure.repeat(projection.nodes.len()),
        ),
    )
    .unwrap();
    let topology = ObservationTopology::from_projection(&projection).unwrap();
    let serial = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::serial(),
        None,
    )
    .unwrap();
    let split = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
        None,
    )
    .unwrap();
    let area = face.oriented_area_vector_m2;
    let expected = [
        pressure[0] * area[0],
        pressure[1] * area[0],
        pressure[0] * area[1],
        pressure[1] * area[1],
        pressure[0] * area[2],
        pressure[1] * area[2],
    ];
    assert_eq!(serial.field.shape, vec![3, 2]);
    assert_eq!(serial.logical_identity, split.logical_identity);
    for ((actual, split), expected) in serial
        .field
        .as_host_f64()
        .unwrap()
        .iter()
        .zip(split.field.as_host_f64().unwrap())
        .zip(expected)
    {
        assert!((actual - expected).abs() <= 1.0e-12);
        assert!((split - expected).abs() <= 1.0e-12);
    }
}

#[test]
fn vector_l2_norm_is_scalar_and_partition_equivalent() {
    let projection =
        AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
            .unwrap();
    let definition = RecoveredFieldDefinition::new(
        "displacement.norm",
        ObservationQuantity::Displacement,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut observation = spec("displacement_norm", ObservationTarget::WholeModel);
    observation.quantity = ObservationQuantity::Displacement;
    observation.value_shape = ObservationValueShape::Vector3;
    observation.output_location = FieldLocation::Global;
    observation.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::EuclideanNorm,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    let mut step = ProcedureStep::structural_static("vector-norm");
    step.observations = vec![ObservationRequest::Reduction(observation)];
    let plan = super::resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let field = RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64(
            "displacement.norm", definition.layout(),
            vec![projection.nodes.len(), 3],
            [3.0, 4.0, 0.0].repeat(projection.nodes.len()),
        ),
    )
    .unwrap();
    let topology = ObservationTopology::from_projection(&projection).unwrap();
    let serial = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::serial(),
        None,
    )
    .unwrap();
    let split = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
        None,
    )
    .unwrap();
    let volume = projection
        .elements
        .iter()
        .map(|element| element.signed_volume_m3)
        .sum::<f64>();
    let expected = 5.0 * volume.sqrt();
    assert_eq!(serial.value_shape, ObservationValueShape::Scalar);
    assert!((serial.field.as_host_f64().unwrap()[0] - expected).abs() <= 1.0e-12);
    assert!((split.field.as_host_f64().unwrap()[0] - expected).abs() <= 1.0e-12);
    assert_eq!(serial.logical_identity, split.logical_identity);
}

#[test]
fn nodal_reaction_resultant_and_moment_are_discrete_partition_equivalent_sums() {
    let projection =
        AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
            .unwrap();
    let definition = RecoveredFieldDefinition::new(
        "reaction.force",
        ObservationQuantity::ReactionForce,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut resultant = spec("reaction_resultant", ObservationTarget::WholeModel);
    resultant.quantity = ObservationQuantity::ReactionForce;
    resultant.value_shape = ObservationValueShape::Vector3;
    resultant.output_location = FieldLocation::Global;
    resultant.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Resultant,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    let about = Position::new([0.25, -0.5, 1.0]);
    let mut moment = resultant.clone();
    moment.observation_id = ObservationId("reaction_moment".into());
    moment.reduction.as_mut().unwrap().spatial = SpatialReduction::Moment { about };
    let mut step = ProcedureStep::structural_static("reaction-reductions");
    step.observations = vec![
        ObservationRequest::Reduction(resultant),
        ObservationRequest::Reduction(moment),
    ];
    let plan = super::resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let mut values = Vec::with_capacity(projection.nodes.len() * 3);
    let mut expected_resultant = [0.0; 3];
    let mut expected_moment = [0.0; 3];
    for node in &projection.nodes {
        let force = [
            node.coordinates_m.canonical_value()[0] + 1.0,
            2.0 * node.coordinates_m.canonical_value()[1] - 0.5,
            -node.coordinates_m.canonical_value()[2] + 0.25,
        ];
        values.extend(force);
        for axis in 0..3 {
            expected_resultant[axis] += force[axis];
        }
        let arm = [
            node.coordinates_m.canonical_value()[0] - about.canonical_value()[0],
            node.coordinates_m.canonical_value()[1] - about.canonical_value()[1],
            node.coordinates_m.canonical_value()[2] - about.canonical_value()[2],
        ];
        expected_moment[0] += arm[1] * force[2] - arm[2] * force[1];
        expected_moment[1] += arm[2] * force[0] - arm[0] * force[2];
        expected_moment[2] += arm[0] * force[1] - arm[1] * force[0];
    }
    let field = RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64("reaction.force", definition.layout(), vec![projection.nodes.len(), 3], values),
    )
    .unwrap();
    let topology = ObservationTopology::from_projection(&projection).unwrap();
    let serial_policy = ObservationExecutionPolicy::serial();
    let split_policy = ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(4).unwrap());
    for (observation, expected) in plan
        .observations
        .iter()
        .zip([expected_resultant, expected_moment])
    {
        let serial = evaluate_partitioned_observation(
            &plan,
            observation,
            &field,
            &topology,
            serial_policy,
            None,
        )
        .unwrap();
        let split = evaluate_partitioned_observation(
            &plan,
            observation,
            &field,
            &topology,
            split_policy,
            None,
        )
        .unwrap();
        assert_eq!(serial.value_shape, ObservationValueShape::Vector3);
        assert_eq!(serial.logical_identity, split.logical_identity);
        for ((serial, split), expected) in serial
            .field
            .as_host_f64()
            .unwrap()
            .iter()
            .zip(split.field.as_host_f64().unwrap())
            .zip(expected)
        {
            assert!((serial - expected).abs() <= 1.0e-12);
            assert!((split - expected).abs() <= 1.0e-12);
        }
    }

    let mut constraint_resultant = spec(
        "constraint_reaction",
        ObservationTarget::Constraint {
            constraint_id: "fixed-end".into(),
        },
    );
    constraint_resultant.quantity = ObservationQuantity::ReactionForce;
    constraint_resultant.value_shape = ObservationValueShape::Vector3;
    constraint_resultant.output_location = FieldLocation::Global;
    constraint_resultant.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Resultant,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    let mut step = ProcedureStep::structural_static("constraint-reaction");
    step.observations = vec![ObservationRequest::Reduction(constraint_resultant)];
    assert!(super::resolve_observation_plan(
        &step,
        &projection,
        &[definition],
        ObservationLimits::default(),
    )
    .unwrap_err()
    .to_string()
    .contains("canonical executed constraint plan"));
}

#[test]
fn reduction_rejects_missing_duplicate_and_substituted_contributions() {
    let (projection, plan, field) = prepared(ObservationRequest::Field(spec(
        "field",
        ObservationTarget::WholeModel,
    )));
    let partition_plan = ObservationPartitionPlan::derive(
        &plan,
        &ObservationId("field".into()),
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
    )
    .unwrap();
    let contributions = partition_plan
        .partitions
        .iter()
        .map(|partition| {
            let source = prepare_solid_observation_field_partition(
                &plan,
                &partition_plan,
                partition,
                &projection,
                &field,
            )
            .unwrap();
            assert!(source.source_indices().count() < projection.nodes.len());
            let source = RecoveredFieldPartitionSet::from_canonical_bytes(
                &source.canonical_bytes().unwrap(),
            )
            .unwrap();
            let contribution = evaluate_solid_observation_partition(
                &plan,
                &partition_plan,
                partition,
                &projection,
                &source,
                None,
            )
            .unwrap();
            ObservationPartitionContribution::from_canonical_bytes(
                &contribution.canonical_bytes().unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    assert!(
        combine_observation_partitions(&plan, &partition_plan, contributions[..2].to_vec(),)
            .is_err()
    );
    let mut duplicate = contributions.clone();
    duplicate[2] = duplicate[1].clone();
    assert!(combine_observation_partitions(&plan, &partition_plan, duplicate).is_err());

    // Tampered partition plans cannot be sealed, so `validate_for` never sees one.
    let mut substituted_plan = partition_plan.clone().into_contents();
    substituted_plan.observation_id = ObservationId("other".into());
    assert!(ObservationPartitionPlan::try_from(substituted_plan).is_err());

    let mut substituted_cost = partition_plan.clone().into_contents();
    substituted_cost.resource_cost.transfer_bytes += 1;
    assert!(ObservationPartitionPlan::try_from(substituted_cost.clone()).is_err());
    assert!(serde_json::from_value::<ObservationPartitionPlan>(
        serde_json::to_value(&substituted_cost).unwrap()
    )
    .is_err());
}

#[test]
fn partitioned_event_evaluation_binds_exact_attribution_across_workers() {
    let projection =
        AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
            .unwrap();
    let definition = definition();
    let sample = RecoveredFieldSample::TimeSeconds(0.5);
    let mut request = spec("first_yield_temperature", ObservationTarget::WholeModel);
    request.sampling =
        SamplingPolicy::Event(ObservationEvent::Solver(SolverObservationEvent::FirstYield));
    request.cost =
        runmat_analysis_core::ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 1_000_000,
            peak_resident_bytes: 10_000_000,
            stored_bytes: 10_000_000,
            transfer_bytes: 10_000_000,
            collective_count: 0,
        });
    let mut step = ProcedureStep::thermal_transient("partitioned-event");
    step.accuracy.resources.maximum_samples = 4;
    step.independent_domain = runmat_analysis_core::IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 1.0,
        interval_count: 10,
    };
    step.observations = vec![ObservationRequest::History(request)];
    let plan = super::resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let values = projection
        .nodes
        .iter()
        .map(|node| node.coordinates_m.canonical_value().iter().sum())
        .collect::<Vec<_>>();
    let field = RecoveredField::new(
        definition.clone(),
        sample,
        AnalysisField::host_f64("temperature.partitioned", definition.layout(), vec![values.len()], values),
    )
    .unwrap();
    let sequence = SequenceIdentity::new(Digest::sha256("partition-event-sequence")).unwrap();
    let producer = StepIdentity::new(sequence, Digest::sha256("partition-event-step")).unwrap();
    let source =
        SolverResultIdentity::from_step_result(producer, Digest::sha256("partition-event-result"))
            .unwrap();
    let attribution = ObservationEventAttribution::Solver(SolverEventAttribution {
        event: SolverObservationEvent::FirstYield,
        sample,
        occurrence_index: 0,
        source: SolverEventSource::CompletedResult { result: source },
    });
    let topology = ObservationTopology::from_projection(&projection).unwrap();
    let single = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::serial(),
        Some(attribution),
    )
    .unwrap();
    let split_policy = ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap());
    let split = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        split_policy,
        Some(attribution),
    )
    .unwrap();
    assert_eq!(single.logical_identity, split.logical_identity);
    assert_eq!(
        split.partition_evidence.event_attribution,
        Some(attribution)
    );

    let partition_plan = ObservationPartitionPlan::derive(
        &plan,
        &ObservationId("first_yield_temperature".into()),
        split_policy,
    )
    .unwrap();
    let mut contributions = partition_plan
        .partitions
        .iter()
        .map(|partition| {
            let source = prepare_solid_observation_field_partition(
                &plan,
                &partition_plan,
                partition,
                &projection,
                &field,
            )
            .unwrap();
            let contribution = evaluate_solid_observation_partition(
                &plan,
                &partition_plan,
                partition,
                &projection,
                &source,
                Some(attribution),
            )
            .unwrap();
            ObservationPartitionContribution::from_canonical_bytes(
                &contribution.canonical_bytes().unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let substituted_attribution = ObservationEventAttribution::Solver(SolverEventAttribution {
        event: SolverObservationEvent::FirstYield,
        sample,
        occurrence_index: 0,
        source: SolverEventSource::CompletedResult {
            result: SolverResultIdentity::from_step_result(
                producer,
                Digest::sha256("substituted-partition-event-result"),
            )
            .unwrap(),
        },
    });
    contributions[0].event_attribution = Some(substituted_attribution);
    assert!(contributions[0].validate().is_err());
    assert!(combine_observation_partitions(&plan, &partition_plan, contributions).is_err());
}
