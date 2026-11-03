use runmat_analysis_core::{
    AnalysisField, ComplexRepresentation, CompressionPolicy, CoordinateFrame, FieldLocation,
    IndependentDomain, ObservationComponent, ObservationCostEstimate, ObservationEvent,
    ObservationId, ObservationPurpose, ObservationQuantity, ObservationRequest, ObservationSpec,
    ObservationStoragePolicy, ObservationTarget, ObservationValueShape, PartialAccessPolicy,
    ProcedureStep, RecoveryPolicy, RetentionPolicy, SamplingPolicy, SignConvention,
    SignalPrecision, SolverObservationEvent,
};
use runmat_execution::Digest;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

use crate::{
    execution::{
        FieldRecoveryRequest, FieldRecoverySource, ObservationEvaluationRequest, SequenceIdentity,
        SolverResultIdentity, StepIdentity,
    },
    physics::structural::topology::CanonicalStructuralTopologyIdentity,
    results::{
        field::{
            RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinition,
            RecoveredFieldEntityCounts, RecoveredFieldEntityDomain,
            RecoveredFieldNumericRepresentation, RecoveredFieldSample,
        },
        observation::{
            prepare_solid_observation_field_partition, resolve_observation_plan,
            ObservationEventAttribution, ObservationExecutionPolicy, ObservationLimits,
            ObservationPartitionPlan, SolverEventAttribution, SolverEventSource,
        },
    },
    topology::AnalysisProjection,
};

use super::validate_event_attribution;

#[test]
fn portable_partition_event_requires_the_exact_completed_result() {
    let projection =
        AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet4)).unwrap();
    let definition = RecoveredFieldDefinition::new(
        "temperature.event",
        ObservationQuantity::Temperature,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let sample = RecoveredFieldSample::TimeSeconds(0.5);
    let field = RecoveredField::new(
        definition.clone(),
        sample,
        AnalysisField::host_f64(
            "temperature.event", definition.layout(),
            vec![projection.nodes.len()],
            vec![300.0; projection.nodes.len()],
        ),
    )
    .unwrap();
    let mut step = ProcedureStep::thermal_transient("portable-event");
    step.independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 1.0,
        interval_count: 10,
    };
    step.accuracy.resources.maximum_samples = 4;
    step.observations = vec![ObservationRequest::History(ObservationSpec {
        observation_id: ObservationId("first_yield".into()),
        quantity: ObservationQuantity::Temperature,
        value_shape: ObservationValueShape::Scalar,
        target: ObservationTarget::WholeModel,
        source_location: FieldLocation::Node,
        output_location: FieldLocation::Node,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Real,
        sampling: SamplingPolicy::Event(ObservationEvent::Solver(
            SolverObservationEvent::FirstYield,
        )),
        reduction: None,
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        cost: runmat_analysis_core::ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 1_000_000,
            peak_resident_bytes: 10_000_000,
            stored_bytes: 10_000_000,
            transfer_bytes: 10_000_000,
            collective_count: 0,
        }),
    })];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let sequence = SequenceIdentity::new(Digest::sha256("portable-event-sequence")).unwrap();
    let producer = StepIdentity::new(sequence, Digest::sha256("portable-event-step")).unwrap();
    let source_result =
        SolverResultIdentity::from_step_result(producer, Digest::sha256("portable-event-result"))
            .unwrap();
    let topology = CanonicalStructuralTopologyIdentity::Explicit {
        structural_topology_digest: projection.projection_digest,
    };
    let recovery = FieldRecoveryRequest::new(
        producer,
        FieldRecoverySource::SolverResult(source_result),
        crate::execution::FieldTopologyIdentity::Structural { topology },
        crate::execution::FieldRecoveryPlan::default(),
    )
    .unwrap();
    let artifact = RecoveredFieldArtifact::new(
        recovery.identity(),
        FieldRecoverySource::SolverResult(source_result),
        projection.projection_digest,
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: projection.nodes.len(),
            volume_elements: projection.elements.len(),
            volume_integration_points: projection.elements.len() * 4,
            oriented_edges: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        },
        vec![field.clone()],
    )
    .unwrap();
    let selection = crate::results::observation::ObservationFieldSelection::structural(
        &recovery,
        &artifact.manifest().unwrap(),
        vec![field.identity()],
    )
    .unwrap();
    let request = ObservationEvaluationRequest::new(selection, plan).unwrap();
    let partition_plan = ObservationPartitionPlan::derive(
        &request.plan,
        &ObservationId("first_yield".into()),
        ObservationExecutionPolicy::serial(),
    )
    .unwrap();
    let partition = &partition_plan.partitions[0];
    let source_fields = prepare_solid_observation_field_partition(
        &request.plan,
        &partition_plan,
        partition,
        &projection,
        &field,
    )
    .unwrap();
    let attribution = ObservationEventAttribution::Solver(SolverEventAttribution {
        event: SolverObservationEvent::FirstYield,
        sample,
        occurrence_index: 0,
        source: SolverEventSource::CompletedResult {
            result: source_result,
        },
    });
    validate_event_attribution(&request, &partition_plan, &source_fields, Some(attribution))
        .unwrap();

    let evaluation = super::ObservationPartitionEvaluationRequest::new(
        &request,
        &partition_plan,
        partition,
        &source_fields,
        Some(attribution),
    )
    .unwrap();
    assert_eq!(
        evaluation.identity(),
        super::ObservationPartitionEvaluationRequest::published_identity(
            &request,
            &partition_plan,
            partition,
            source_fields.identity,
            Some(attribution),
        )
        .unwrap()
    );
    let mut noncanonical = partition.clone();
    noncanonical.ordinal += 1;
    assert!(
        super::ObservationPartitionEvaluationRequest::published_identity(
            &request,
            &partition_plan,
            &noncanonical,
            source_fields.identity,
            Some(attribution),
        )
        .is_err()
    );
    let contribution = crate::results::observation::evaluate_solid_observation_partition(
        &request.plan,
        &partition_plan,
        partition,
        &projection,
        &source_fields,
        Some(attribution),
    )
    .unwrap();
    let reduction = super::ObservationPartitionReductionRequest::new(
        &request,
        &partition_plan,
        std::slice::from_ref(&contribution),
    )
    .unwrap();
    assert_eq!(
        reduction.identity(),
        super::ObservationPartitionReductionRequest::published_identity(
            &request,
            &partition_plan,
            &[contribution.identity],
        )
        .unwrap()
    );
    assert!(
        super::ObservationPartitionReductionRequest::published_identity(
            &request,
            &partition_plan,
            &[],
        )
        .is_err()
    );
    let split_plan = ObservationPartitionPlan::derive(
        &request.plan,
        &ObservationId("first_yield".into()),
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(2).unwrap()),
    )
    .unwrap();
    assert_eq!(split_plan.partitions.len(), 2);
    assert!(
        super::ObservationPartitionReductionRequest::published_identity(
            &request,
            &split_plan,
            &[contribution.identity, contribution.identity],
        )
        .is_err()
    );

    let substituted = ObservationEventAttribution::Solver(SolverEventAttribution {
        event: SolverObservationEvent::FirstYield,
        sample,
        occurrence_index: 0,
        source: SolverEventSource::CompletedResult {
            result: SolverResultIdentity::from_step_result(
                producer,
                Digest::sha256("substituted-portable-event-result"),
            )
            .unwrap(),
        },
    });
    assert!(validate_event_attribution(
        &request,
        &partition_plan,
        &source_fields,
        Some(substituted),
    )
    .is_err());
    assert!(validate_event_attribution(&request, &partition_plan, &source_fields, None,).is_err());
}
