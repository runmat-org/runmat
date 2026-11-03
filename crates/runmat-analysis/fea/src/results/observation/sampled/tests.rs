//! Independent storage/selection witnesses, not magnetic discretization qualification.
use super::*;
use crate::{
    physics::magnetic::test_support::authored_model,
    results::field::{
        RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation, RecoveredFieldSample,
        VolumeFieldSamplingRule,
    },
    results::observation::{
        distributed::combine_observation_partitions, partition_tests::spec, ObservationEntity,
        ObservationExecutionPolicy,
    },
};
use runmat_analysis_core::{
    AnalysisField, ExcitationTarget, ExtremaTiePolicy, FieldLocation, ObservationComponent,
    ObservationId, ObservationQuantity, ObservationSpec, ObservationTarget, ObservationValueShape,
    Position, ReductionPolicy, RegionTarget, SeriesReduction, TargetCardinality,
};
use runmat_meshing_core::{
    fixtures::{quadratic_slab_mesh, slab_mesh},
    ElementOrder,
};
use std::num::NonZeroU32;

fn mesh(order: ElementOrder) -> SolverMeshArtifact {
    match order {
        ElementOrder::Tet4 => slab_mesh(1),
        ElementOrder::Tet10 => quadratic_slab_mesh(1),
    }
}

fn native_field(sampling: &SampledVolumeFieldTopology, edge: bool) -> RecoveredField {
    let (id, quantity, location, domain, shape, count) = if edge {
        (
            "magnetic.edge_circulation",
            ObservationQuantity::MagneticEdgeCirculation,
            FieldLocation::Edge,
            RecoveredFieldEntityDomain::OrientedEdges,
            ObservationValueShape::Scalar,
            sampling.oriented_edges().len(),
        )
    } else {
        (
            "magnetic.flux_density",
            ObservationQuantity::MagneticFluxDensity,
            FieldLocation::IntegrationPoint,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints,
            ObservationValueShape::Vector3,
            sampling.integration_sample_count(),
        )
    };
    let definition = RecoveredFieldDefinition::new(
        id,
        quantity,
        location,
        domain,
        shape,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let values = if edge {
        (0..count).map(|i| i as f64 - count as f64 / 2.0).collect()
    } else {
        (0..count)
            .flat_map(|i| [i as f64, -(i as f64), 2.0])
            .collect()
    };
    let storage_shape = if edge { vec![count] } else { vec![count, 3] };
    RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64(id, definition.layout(), storage_shape, values),
    )
    .unwrap()
}

fn native_spec(field: &RecoveredField) -> ObservationSpec {
    let mut request = spec("sampled", ObservationTarget::WholeModel);
    request.quantity = field.definition().quantity();
    request.value_shape = field.definition().value_shape().clone();
    request.source_location = field.definition().location();
    request.output_location = request.source_location;
    request
}

fn step(request: ObservationRequest) -> ProcedureStep {
    let mut step = authored_model(0.0).sequence.steps.remove(0);
    step.observations = vec![request];
    step
}

fn plan(
    step: &ProcedureStep,
    sampling: &SampledVolumeFieldTopology,
    mesh: &SolverMeshArtifact,
    field: &RecoveredField,
) -> ResolvedObservationPlan {
    resolve_sampled_observation_plan(
        step,
        sampling,
        mesh,
        std::slice::from_ref(field.definition()),
        ObservationLimits::default(),
    )
    .unwrap()
}

fn serial(
    plan: &ResolvedObservationPlan,
    sampling: &SampledVolumeFieldTopology,
    mesh: &SolverMeshArtifact,
    field: &RecoveredField,
) -> ObservationResult {
    let catalog = RecoveredFieldCatalog::new_for_topology(
        sampling.identity().unwrap().discretization_digest(),
        sampling.entity_counts(),
        vec![field.clone()],
    )
    .unwrap();
    evaluate_sampled_observations(plan, sampling, mesh, &catalog)
        .unwrap()
        .remove(0)
}

fn split(
    plan: &ResolvedObservationPlan,
    sampling: &SampledVolumeFieldTopology,
    mesh: &SolverMeshArtifact,
    field: &RecoveredField,
) -> ObservationResult {
    let partition_plan = ObservationPartitionPlan::derive(
        plan,
        &ObservationId("sampled".into()),
        ObservationExecutionPolicy::canonical(NonZeroU32::new(3).unwrap()),
    )
    .unwrap();
    let mut contributions = Vec::new();
    for partition in &partition_plan.partitions {
        let indices = sampled_observation_partition_source_indices(
            plan,
            &partition_plan,
            partition,
            sampling,
            mesh,
        )
        .unwrap();
        let selected = prepare_sampled_observation_field_partition(
            plan,
            &partition_plan,
            partition,
            sampling,
            mesh,
            field,
        )
        .unwrap();
        assert_eq!(
            selected
                .partitions
                .iter()
                .flat_map(|p| p.source_indices.iter().copied())
                .collect::<Vec<_>>(),
            indices
        );
        assert!(indices.len() < field.storage().shape[0]);
        let cold =
            RecoveredFieldPartitionSet::from_canonical_bytes(&selected.canonical_bytes().unwrap())
                .unwrap();
        assert_eq!(cold, selected);
        let contribution = evaluate_sampled_observation_partition(
            plan,
            &partition_plan,
            partition,
            sampling,
            mesh,
            &cold,
        )
        .unwrap();
        let contribution = ObservationPartitionContribution::from_canonical_bytes(
            &contribution.canonical_bytes().unwrap(),
        )
        .unwrap();
        contributions.push(contribution);
    }
    contributions.reverse();
    combine_observation_partitions(plan, &partition_plan, contributions).unwrap()
}

#[test]
fn sampled_native_fields_keep_exact_samples_edges_values_and_partition_independence() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = mesh(order);
        for rule in [
            VolumeFieldSamplingRule::TetrahedronFourPoint,
            VolumeFieldSamplingRule::TetrahedronKeastFifteen,
        ] {
            let sampling = SampledVolumeFieldTopology::derive(&mesh, rule).unwrap();
            for edge in [false, true] {
                let field = native_field(&sampling, edge);
                let plan = plan(
                    &step(ObservationRequest::Field(native_spec(&field))),
                    &sampling,
                    &mesh,
                    &field,
                );
                let full = serial(&plan, &sampling, &mesh, &field);
                assert_eq!(full.field.as_host_f64(), field.storage().as_host_f64());
                let expected = if edge {
                    sampling
                        .oriented_edges()
                        .iter()
                        .map(|e| ObservationEntity::OrientedEdge(e.identity))
                        .collect::<Vec<_>>()
                } else {
                    (0..sampling.integration_sample_count())
                        .map(|i| {
                            ObservationEntity::IntegrationPoint(
                                sampling.integration_sample_identity(i).unwrap(),
                            )
                        })
                        .collect()
                };
                assert_eq!(full.entities, expected);
                let partitioned = split(&plan, &sampling, &mesh, &field);
                assert_eq!(partitioned.logical_identity, full.logical_identity);
                assert_eq!(partitioned.entities, full.entities);
                assert_eq!(partitioned.field, full.field);
                assert_ne!(
                    partitioned.partition_evidence.identity,
                    full.partition_evidence.identity
                );
            }
        }
    }
}

#[test]
fn sampled_component_magnitude_and_signed_edge_extrema_are_native() {
    let mesh = mesh(ElementOrder::Tet10);
    let sampling =
        SampledVolumeFieldTopology::derive(&mesh, VolumeFieldSamplingRule::TetrahedronKeastFifteen)
            .unwrap();
    let vector = native_field(&sampling, false);
    for component in [ObservationComponent::X, ObservationComponent::Magnitude] {
        let mut request = native_spec(&vector);
        request.component = component;
        let plan = plan(
            &step(ObservationRequest::Field(request)),
            &sampling,
            &mesh,
            &vector,
        );
        let full = serial(&plan, &sampling, &mesh, &vector);
        let values = full.field.as_host_f64().unwrap();
        for (i, value) in values.iter().enumerate() {
            let expected = if component == ObservationComponent::X {
                i as f64
            } else {
                (2.0 * (i as f64).powi(2) + 4.0).sqrt()
            };
            assert_eq!(*value, expected);
        }
        assert_eq!(
            split(&plan, &sampling, &mesh, &vector).logical_identity,
            full.logical_identity
        );
    }
    let edge = native_field(&sampling, true);
    let mut request = native_spec(&edge);
    request.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Maximum,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::AllLocations,
    });
    let plan = plan(
        &step(ObservationRequest::Reduction(request)),
        &sampling,
        &mesh,
        &edge,
    );
    let maximum = serial(&plan, &sampling, &mesh, &edge);
    assert_eq!(
        maximum.field.as_host_f64().unwrap(),
        vec![
            (sampling.oriented_edges().len() - 1) as f64
                - sampling.oriented_edges().len() as f64 / 2.0
        ]
    );
    assert_eq!(
        maximum.entities,
        vec![ObservationEntity::OrientedEdge(
            sampling.oriented_edges().last().unwrap().identity
        )]
    );
    assert_eq!(
        split(&plan, &sampling, &mesh, &edge).logical_identity,
        maximum.logical_identity
    );
    let mut vector_reinterpretation = native_spec(&edge);
    vector_reinterpretation.component = ObservationComponent::X;
    assert!(resolve_sampled_observation_plan(
        &step(ObservationRequest::Field(vector_reinterpretation)),
        &sampling,
        &mesh,
        std::slice::from_ref(edge.definition()),
        ObservationLimits::default()
    )
    .is_err());
}

#[test]
fn sampled_region_cardinality_counts_parents_but_limits_count_actual_samples() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = mesh(order);
        for rule in [
            VolumeFieldSamplingRule::TetrahedronFourPoint,
            VolumeFieldSamplingRule::TetrahedronKeastFifteen,
        ] {
            let sampling = SampledVolumeFieldTopology::derive(&mesh, rule).unwrap();
            let field = native_field(&sampling, false);
            let mut request = native_spec(&field);
            let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
            let selector = projection.elements[0]
                .region_id
                .analysis_selection_id()
                .unwrap();
            request.target = ObservationTarget::Spatial(ExcitationTarget::Region(
                RegionTarget::volume(selector),
            ));
            let region_plan = plan(
                &step(ObservationRequest::Field(request.clone())),
                &sampling,
                &mesh,
                &field,
            );
            assert_eq!(
                serial(&region_plan, &sampling, &mesh, &field)
                    .entities
                    .len(),
                sampling.integration_sample_count()
            );
            let ObservationTarget::Spatial(ExcitationTarget::Region(region)) = &mut request.target
            else {
                unreachable!()
            };
            region.cardinality = TargetCardinality::One;
            assert!(resolve_sampled_observation_plan(
                &step(ObservationRequest::Field(request)),
                &sampling,
                &mesh,
                std::slice::from_ref(field.definition()),
                ObservationLimits::default()
            )
            .is_err());
            let limits = ObservationLimits {
                maximum_selected_entities: sampling.integration_sample_count() - 1,
                ..ObservationLimits::default()
            };
            assert!(resolve_sampled_observation_plan(
                &step(ObservationRequest::Field(native_spec(&field))),
                &sampling,
                &mesh,
                std::slice::from_ref(field.definition()),
                limits
            )
            .is_err());
        }
    }
}

#[test]
fn sampled_foreign_layout_mesh_weighted_requests_probes_and_histories_fail_closed() {
    let mesh = mesh(ElementOrder::Tet4);
    let sampling =
        SampledVolumeFieldTopology::derive(&mesh, VolumeFieldSamplingRule::TetrahedronKeastFifteen)
            .unwrap();
    let field = native_field(&sampling, false);
    let plan = plan(
        &step(ObservationRequest::Field(native_spec(&field))),
        &sampling,
        &mesh,
        &field,
    );
    let foreign =
        SampledVolumeFieldTopology::derive(&mesh, VolumeFieldSamplingRule::TetrahedronFourPoint)
            .unwrap();
    let original_catalog = RecoveredFieldCatalog::new_for_topology(
        plan.topology_digest,
        sampling.entity_counts(),
        vec![field.clone()],
    )
    .unwrap();
    assert!(evaluate_sampled_observations(&plan, &foreign, &mesh, &original_catalog).is_err());
    assert!(
        evaluate_sampled_observations(&plan, &sampling, &slab_mesh(2), &original_catalog).is_err()
    );
    let partition_plan = ObservationPartitionPlan::derive(
        &plan,
        &ObservationId("sampled".into()),
        ObservationExecutionPolicy::serial(),
    )
    .unwrap();
    assert!(sampled_observation_partition_source_indices(
        &plan,
        &partition_plan,
        &partition_plan.partitions[0],
        &foreign,
        &mesh
    )
    .is_err());
    let mut malformed = serde_json::to_value(&sampling).unwrap();
    malformed["data"]["element_corner_nodes"][0][0] = serde_json::json!(u32::MAX);
    assert!(serde_json::from_value::<SampledVolumeFieldTopology>(malformed).is_err());
    let edge = native_field(&sampling, true);
    let mut weighted = native_spec(&edge);
    weighted.output_location = FieldLocation::Global;
    weighted.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Integral,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::AllLocations,
    });
    let mut probe = native_spec(&field);
    probe.target = ObservationTarget::Point {
        position: Position::new([0.1; 3]),
    };
    probe.output_location = FieldLocation::Point;
    for (request, source) in [
        (ObservationRequest::Reduction(weighted), &edge),
        (ObservationRequest::Probe(probe), &field),
        (ObservationRequest::History(native_spec(&field)), &field),
    ] {
        assert!(resolve_sampled_observation_plan(
            &step(request),
            &sampling,
            &mesh,
            std::slice::from_ref(source.definition()),
            ObservationLimits::default()
        )
        .is_err());
    }
}
