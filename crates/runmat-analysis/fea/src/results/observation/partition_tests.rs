use runmat_analysis_core::{
    AnalysisField, ComplexRepresentation, CompressionPolicy, CoordinateFrame, ExtremaTiePolicy,
    FieldLocation, ObservationComponent, ObservationCostEstimate, ObservationId,
    ObservationPurpose, ObservationQuantity, ObservationRequest, ObservationSpec,
    ObservationStoragePolicy, ObservationTarget, ObservationValueShape, PartialAccessPolicy,
    ProcedureStep, RecoveryPolicy, ReductionPolicy, RetentionPolicy, SamplingPolicy,
    SeriesReduction, SignConvention, SignalPrecision, SpatialReduction,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

use crate::{
    results::field::{
        RecoveredField, RecoveredFieldCatalog, RecoveredFieldDefinition,
        RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation,
        RecoveredFieldPartitionSet, RecoveredFieldSample,
    },
    topology::AnalysisProjection,
};

use super::{
    distributed::{combine_observation_partitions, evaluate_partitioned_observation},
    evaluate_solid_observation_partition, prepare_solid_observation_field_partition,
    topology::StructuralObservationTopology,
    ObservationExecutionPolicy, ObservationLimits, ObservationPartitionContribution,
    ObservationPartitionPlan,
};

fn definition() -> RecoveredFieldDefinition {
    RecoveredFieldDefinition::new(
        "temperature.partitioned",
        ObservationQuantity::Temperature,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::StructuralNodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap()
}

fn spec(id: &str, target: ObservationTarget) -> ObservationSpec {
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
        cost: ObservationCostEstimate {
            operation_count: 100_000,
            peak_resident_bytes: 1_000_000,
            stored_bytes: 1_000_000,
            transfer_bytes: 1_000_000,
            collective_count: 0,
        },
    }
}

fn prepared(
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
        .map(|node| node.coordinates_m.iter().sum())
        .collect::<Vec<_>>();
    let field = RecoveredField::new(
        definition,
        RecoveredFieldSample::Final,
        AnalysisField::host_f64("temperature.partitioned", vec![values.len()], values),
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

    let topology = StructuralObservationTopology::from_projection(&projection);
    let single = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::serial(),
    )
    .unwrap();
    let split = evaluate_partitioned_observation(
        &plan,
        &plan.observations[0],
        &field,
        &topology,
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
    )
    .unwrap();
    assert_eq!(single.logical_identity, split.logical_identity);
    assert_ne!(
        single.partition_evidence.identity,
        split.partition_evidence.identity
    );
}

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
            runmat_analysis_core::RegionTarget::boundary(face_selector),
        )),
    );
    observation.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Average,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    let (projection, plan, field) =
        prepared_on(projection, ObservationRequest::Reduction(observation));
    let topology = StructuralObservationTopology::from_projection(&projection);
    let resolved = &plan.observations[0];
    let single = evaluate_partitioned_observation(
        &plan,
        resolved,
        &field,
        &topology,
        ObservationExecutionPolicy::serial(),
    )
    .unwrap();
    let split = evaluate_partitioned_observation(
        &plan,
        resolved,
        &field,
        &topology,
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
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

    let mut substituted_plan = partition_plan.clone();
    substituted_plan.observation_id = ObservationId("other".into());
    assert!(substituted_plan.validate_for(&plan).is_err());

    let mut substituted_cost = partition_plan.clone();
    substituted_cost.resource_cost.transfer_bytes += 1;
    assert!(substituted_cost.validate_for(&plan).is_err());
    assert!(substituted_cost.canonical_bytes().is_err());
}
