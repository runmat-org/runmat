use runmat_analysis_core::{
    AccuracyObservationEvent, AccuracyPlan, ComplexRepresentation, CompressionPolicy,
    CoordinateFrame, ExtremaTiePolicy, FieldLocation, ObservationComponent,
    ObservationCostEstimate, ObservationEvent, ObservationId, ObservationPurpose,
    ObservationQuantity, ObservationRequest, ObservationSpec, ObservationStoragePolicy,
    ObservationTarget, ObservationValueShape, PartialAccessPolicy, Position, ProcedureStep,
    RecoveryPolicy, ReductionPolicy, RetentionPolicy, SamplingPolicy, SeriesObservationEvent,
    SeriesReduction, SeriesWeighting, SignConvention, SignalPrecision, SolverObservationEvent,
    SpatialReduction,
};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, PersistentEntityId,
    PersistentEntityKind, StableDigest,
};

use crate::{
    connectors::{
        CanonicalConnectorEntry, CanonicalConnectorNode, CanonicalStructuralConnectorPlan,
    },
    execution::{
        AccuracyIdentity, AnalysisResultIdentity, AnalysisWorkKind, AnalysisWorkloadIdentity,
        ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity,
        FieldRecoveryRequest, FieldRecoverySource, NumericalPlanIdentity,
        ObservationAssemblyRequest, OperatorIdentity, ProblemIdentity, SequenceIdentity,
        SolverResultIdentity, StepIdentity, StudyIdentity,
    },
    fixtures::{fixture_model, FixtureId},
    planning::resource::{Count, ObservationResourceFacts},
    results::field::{
        RecoveredField, RecoveredFieldArtifact, RecoveredFieldCatalog, RecoveredFieldDefinition,
        RecoveredFieldDerivation, RecoveredFieldEntityCounts, RecoveredFieldEntityDomain,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample, RecoveredFieldSelection,
    },
    topology::AnalysisProjection,
};

use super::*;

fn scalar_definition() -> RecoveredFieldDefinition {
    RecoveredFieldDefinition::new(
        "temperature.recovered",
        ObservationQuantity::Temperature,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::StructuralNodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap()
}

fn complex_pressure_definition() -> RecoveredFieldDefinition {
    RecoveredFieldDefinition::new(
        "acoustic.pressure.recovered",
        ObservationQuantity::AcousticPressure,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::StructuralNodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::ComplexInterleaved,
    )
    .unwrap()
}

fn catalog(
    projection: &AnalysisProjection,
    definition: &RecoveredFieldDefinition,
    shape: Vec<usize>,
    values: Vec<f64>,
) -> RecoveredFieldCatalog {
    catalog_at_sample(
        projection,
        definition,
        RecoveredFieldSample::Final,
        shape,
        values,
    )
}

fn catalog_at_sample(
    projection: &AnalysisProjection,
    definition: &RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    shape: Vec<usize>,
    values: Vec<f64>,
) -> RecoveredFieldCatalog {
    let storage =
        runmat_analysis_core::AnalysisField::host_f64(definition.field_id(), shape, values);
    RecoveredFieldCatalog::new(
        projection,
        vec![RecoveredField::new(definition.clone(), sample, storage).unwrap()],
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
        cost: runmat_analysis_core::ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 100_000,
            peak_resident_bytes: 1_000_000,
            stored_bytes: 1_000_000,
            transfer_bytes: 1_000_000,
            collective_count: 0,
        }),
    }
}

fn thermal_step(requests: Vec<ObservationRequest>) -> ProcedureStep {
    let mut step = ProcedureStep::thermal_transient("thermal");
    step.independent_domain = runmat_analysis_core::IndependentDomain::Static;
    step.observations = requests;
    step
}

fn complex_pressure_history_spec(
    id: &str,
    representation: ComplexRepresentation,
) -> ObservationSpec {
    let mut spec = spec(id, ObservationTarget::WholeModel);
    spec.quantity = ObservationQuantity::AcousticPressure;
    spec.complex_representation = representation;
    spec.sampling = SamplingPolicy::FrequencyHertz(vec![100.0, 200.0]);
    spec.storage.precision = if representation == ComplexRepresentation::Complex {
        SignalPrecision::Complex128
    } else {
        SignalPrecision::Real64
    };
    spec.cost = runmat_analysis_core::ObservationCostPolicy::UpperBound(ObservationCostEstimate {
        operation_count: 1_000_000,
        peak_resident_bytes: 10_000_000,
        stored_bytes: 10_000_000,
        transfer_bytes: 10_000_000,
        collective_count: 0,
    });
    spec
}

fn projection(order: ElementOrder) -> AnalysisProjection {
    AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(order)).unwrap()
}

#[test]
fn constraint_reaction_resultant_uses_canonical_group_ownership() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for face in &mut mesh.topology.boundary_faces {
        face.provenance.push(PersistentEntityId {
            kind: PersistentEntityKind::Face,
            source_topology_id: "root".into(),
            assembly_path: vec!["root".into()],
        });
        face.provenance.sort();
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let constraint_plan = crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
        &model,
        Some(&mesh),
    )
    .unwrap();
    assert_eq!(constraint_plan.groups.len(), 1);

    let mut request = spec(
        "root-reaction",
        ObservationTarget::Constraint {
            constraint_id: "bc_root".into(),
        },
    );
    request.quantity = ObservationQuantity::ReactionForce;
    request.value_shape = ObservationValueShape::Vector3;
    request.source_location = FieldLocation::Constraint;
    request.output_location = FieldLocation::Global;
    request.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Resultant,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    let mut moment_request = request.clone();
    moment_request.observation_id = ObservationId("root-reaction-moment".into());
    moment_request.quantity = ObservationQuantity::ReactionMoment;
    model.sequence.steps[0].observations = vec![
        ObservationRequest::Reduction(request),
        ObservationRequest::Reduction(moment_request),
    ];
    let force_definition = RecoveredFieldDefinition::new(
        "structural.constraint_reaction_force",
        ObservationQuantity::ReactionForce,
        FieldLocation::Constraint,
        RecoveredFieldEntityDomain::ConstraintGroups,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let moment_definition = RecoveredFieldDefinition::new(
        "structural.constraint_reaction_moment_about_global_origin",
        ObservationQuantity::ReactionMoment,
        FieldLocation::Constraint,
        RecoveredFieldEntityDomain::ConstraintGroups,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let plan = resolve_structural_observation_plan(
        &model.sequence.steps[0],
        &model,
        Some(&mesh),
        &constraint_plan,
        &[force_definition.clone(), moment_definition.clone()],
        ObservationLimits::default(),
    )
    .unwrap();
    assert!(matches!(
        &plan.observations[0].target,
        ResolvedObservationTarget::ConstraintGroups { indices, identities }
            if indices == &[0]
                && identities == &[StableDigest::from_bytes(
                    *constraint_plan.groups[0].source_digest.bytes()
                )]
    ));
    let force_field = RecoveredField::new(
        force_definition,
        RecoveredFieldSample::Final,
        runmat_analysis_core::AnalysisField::host_f64(
            "structural.constraint_reaction_force",
            vec![1, 3],
            vec![125.0, -250.0, 375.0],
        ),
    )
    .unwrap();
    let moment_field = RecoveredField::new(
        moment_definition,
        RecoveredFieldSample::Final,
        runmat_analysis_core::AnalysisField::host_f64(
            "structural.constraint_reaction_moment_about_global_origin",
            vec![1, 3],
            vec![12.5, -25.0, 37.5],
        ),
    )
    .unwrap();
    let catalog = RecoveredFieldCatalog::new_for_topology(
        plan.topology_digest,
        RecoveredFieldEntityCounts {
            structural_nodes: mesh.topology.nodes.len(),
            solid_elements: mesh.topology.volume_elements.len(),
            solid_integration_points: mesh.topology.volume_elements.len() * 4,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 1,
        },
        vec![force_field, moment_field],
    )
    .unwrap();
    let results = evaluate_structural_observations(&plan, &model, Some(&mesh), &catalog).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].field.as_host_f64().unwrap(),
        &[125.0, -250.0, 375.0]
    );
    assert_eq!(
        results[1].field.as_host_f64().unwrap(),
        &[12.5, -25.0, 37.5]
    );
    assert!(results[0].entities.is_empty());

    model.sequence.steps[0].observations[0]
        .spec_mut()
        .unwrap()
        .target = ObservationTarget::Constraint {
        constraint_id: "missing".into(),
    };
    assert!(resolve_structural_observation_plan(
        &model.sequence.steps[0],
        &model,
        Some(&mesh),
        &constraint_plan,
        &crate::post::fields::structural_observable_field_definitions().unwrap(),
        ObservationLimits::default(),
    )
    .is_err());
}

#[test]
fn connector_observation_binds_exact_identity_and_partition_source() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for face in &mut mesh.topology.boundary_faces {
        face.provenance.push(PersistentEntityId {
            kind: PersistentEntityKind::Face,
            source_topology_id: "root".into(),
            assembly_path: vec!["root".into()],
        });
        face.provenance.sort();
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let constraint_plan = crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
        &model,
        Some(&mesh),
    )
    .unwrap();
    let source_digest = Digest::sha256(b"connector-source");
    let connector_plan = CanonicalStructuralConnectorPlan::new(
        constraint_plan.topology,
        mesh.topology.nodes.len(),
        Digest::sha256(b"authored-connectors"),
        vec![CanonicalConnectorEntry::IsotropicGroundSpring {
            connector_id: "mount".into(),
            source_digest,
            target: CanonicalConnectorNode {
                node_index: 0,
                node_identity: mesh.topology.nodes[0].stable_identity,
            },
            stiffness_newtons_per_meter: 1_000.0,
        }],
    )
    .unwrap();
    let definition = RecoveredFieldDefinition::new(
        crate::contracts::FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE,
        ObservationQuantity::ReactionForce,
        FieldLocation::Connector,
        RecoveredFieldEntityDomain::Connectors,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut request = spec(
        "mount-force",
        ObservationTarget::Connector {
            connector_id: "mount".into(),
        },
    );
    request.quantity = ObservationQuantity::ReactionForce;
    request.value_shape = ObservationValueShape::Vector3;
    request.source_location = FieldLocation::Connector;
    request.output_location = FieldLocation::Connector;
    model.sequence.steps[0].observations = vec![ObservationRequest::Probe(request)];

    let plan = resolve_structural_observation_plan_with_connectors(
        &model.sequence.steps[0],
        &model,
        Some(&mesh),
        &constraint_plan,
        &connector_plan,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let connector_identity = StableDigest::from_bytes(*source_digest.bytes());
    assert!(matches!(
        &plan.observations[0].target,
        ResolvedObservationTarget::Connectors { indices, identities }
            if indices == &[0] && identities == &[connector_identity]
    ));
    let field = RecoveredField::new(
        definition,
        RecoveredFieldSample::Final,
        runmat_analysis_core::AnalysisField::host_f64(
            crate::contracts::FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE,
            vec![1, 3],
            vec![10.0, -20.0, 30.0],
        ),
    )
    .unwrap();
    let catalog = RecoveredFieldCatalog::new_for_topology(
        plan.topology_digest,
        RecoveredFieldEntityCounts {
            structural_nodes: mesh.topology.nodes.len(),
            solid_elements: mesh.topology.volume_elements.len(),
            solid_integration_points: mesh.topology.volume_elements.len() * 4,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 1,
            constraint_groups: constraint_plan.groups.len(),
        },
        vec![field.clone()],
    )
    .unwrap();
    let results = evaluate_structural_observations(&plan, &model, Some(&mesh), &catalog).unwrap();
    assert_eq!(
        results[0].field.as_host_f64().unwrap(),
        &[10.0, -20.0, 30.0]
    );
    assert_eq!(
        results[0].entities,
        vec![ObservationEntity::Connector(connector_identity)]
    );

    let partition_plan = ObservationPartitionPlan::derive(
        &plan,
        &ObservationId("mount-force".into()),
        ObservationExecutionPolicy::serial(),
    )
    .unwrap();
    assert_eq!(
        structural_observation_partition_source_indices(
            &plan,
            &partition_plan,
            &partition_plan.partitions[0],
            &model,
            Some(&mesh),
        )
        .unwrap(),
        vec![0]
    );
    let partition = prepare_structural_observation_field_partition(
        &plan,
        &partition_plan,
        &partition_plan.partitions[0],
        &model,
        Some(&mesh),
        &field,
    )
    .unwrap();
    let contribution = evaluate_structural_observation_partition(
        &plan,
        &partition_plan,
        &partition_plan.partitions[0],
        &model,
        Some(&mesh),
        &partition,
        None,
    )
    .unwrap();
    let partitioned =
        reduce_observation_partitions(&plan, &partition_plan, vec![contribution]).unwrap();
    assert_eq!(partitioned.logical_identity, results[0].logical_identity);

    model.sequence.steps[0].observations[0]
        .spec_mut()
        .unwrap()
        .target = ObservationTarget::Connector {
        connector_id: "missing".into(),
    };
    assert!(resolve_structural_observation_plan_with_connectors(
        &model.sequence.steps[0],
        &model,
        Some(&mesh),
        &constraint_plan,
        &connector_plan,
        std::slice::from_ref(&field.definition().clone()),
        ObservationLimits::default(),
    )
    .is_err());
}

fn completed_scalar_series(
    operation: SeriesReduction,
    ties: ExtremaTiePolicy,
    retention: RetentionPolicy,
    times: &[f64],
    values: &[f64],
) -> Result<(ResolvedObservationPlan, ObservationHistory), ObservationError> {
    assert_eq!(times.len(), values.len());
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let mut request = spec(
        "series",
        ObservationTarget::Point {
            position: Position::new(projection.nodes[0].coordinates_m),
        },
    );
    request.sampling = SamplingPolicy::TimeSeconds(times.to_vec());
    request.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::None,
        series: operation,
        extrema_ties: ties,
    });
    request.storage.retention = retention;
    request.cost =
        runmat_analysis_core::ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 1_000_000,
            peak_resident_bytes: 10_000_000,
            stored_bytes: 10_000_000,
            transfer_bytes: 10_000_000,
            collective_count: 0,
        });
    let mut step = ProcedureStep::thermal_transient("series-step");
    step.independent_domain = runmat_analysis_core::IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: times[0],
        end: *times.last().unwrap(),
        interval_count: times.len() - 1,
    };
    step.observations = vec![ObservationRequest::History(request)];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )?;
    let observation_id = ObservationId("series".into());
    let mut builder = BoundedObservationHistory::new(&plan, &observation_id)?;
    for (&time, &value) in times.iter().zip(values) {
        let sample = RecoveredFieldSample::TimeSeconds(time);
        let fields = catalog_at_sample(
            &projection,
            &definition,
            sample,
            vec![4],
            vec![value, 0.0, 0.0, 0.0],
        );
        let result =
            evaluate_observation_sample(&plan, &observation_id, &projection, &fields, sample)?;
        builder.push(sample, result)?;
    }
    Ok((plan, builder.finish()?))
}

#[test]
fn automatic_cost_policy_is_resolved_before_execution() {
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let mut request = spec("automatic-cost", ObservationTarget::WholeModel);
    request.cost = runmat_analysis_core::ObservationCostPolicy::Automatic;
    let step = thermal_step(vec![ObservationRequest::Field(request)]);

    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();

    assert!(plan.observations[0].resolved_cost.operation_count > 0);
    assert!(plan.observations[0].resolved_cost.stored_bytes > 0);
}

#[test]
fn materialized_volume_weighted_field_binds_by_recovery_provenance() {
    let projection = projection(ElementOrder::Tet4);
    let definition = RecoveredFieldDefinition::new_derived(
        "structural.nodal_von_mises",
        ObservationQuantity::VonMisesStress,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::StructuralNodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
        RecoveredFieldDerivation::LocationTransfer {
            source_location: FieldLocation::Element,
            recovery: RecoveryPolicy::VolumeWeightedAverage,
        },
    )
    .unwrap();
    let mut request = spec("nodal-von-mises", ObservationTarget::WholeModel);
    request.quantity = ObservationQuantity::VonMisesStress;
    request.source_location = FieldLocation::Element;
    request.output_location = FieldLocation::Node;
    request.recovery = RecoveryPolicy::VolumeWeightedAverage;
    request.cost = runmat_analysis_core::ObservationCostPolicy::Automatic;
    let mut step = ProcedureStep::structural_static("static");
    step.observations = vec![ObservationRequest::Field(request)];

    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let fields = catalog(
        &projection,
        &definition,
        vec![4],
        vec![10.0, 20.0, 30.0, 40.0],
    );
    let results = evaluate_observations(&plan, &projection, &fields).unwrap();

    assert_eq!(
        results[0].field.as_host_f64().unwrap(),
        &[10.0, 20.0, 30.0, 40.0]
    );
}

#[test]
fn field_and_exact_probe_use_typed_source_and_canonical_entities() {
    let projection = projection(ElementOrder::Tet4);
    let node = &projection.nodes[2];
    let field_request = ObservationRequest::Field(spec("field", ObservationTarget::WholeModel));
    let probe_request = ObservationRequest::Probe(spec(
        "probe",
        ObservationTarget::Point {
            position: Position::new(node.coordinates_m),
        },
    ));
    let step = thermal_step(vec![field_request, probe_request]);
    let definition = scalar_definition();
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let fields = catalog(
        &projection,
        &definition,
        vec![4],
        vec![10.0, 20.0, 30.0, 40.0],
    );
    let results = evaluate_observations(&plan, &projection, &fields).unwrap();

    assert_eq!(
        results[0].field.as_host_f64().unwrap(),
        &[10.0, 20.0, 30.0, 40.0]
    );
    assert_eq!(results[0].entities.len(), 4);
    assert_eq!(results[1].field.as_host_f64().unwrap(), &[30.0]);
    assert_eq!(
        results[1].entities,
        vec![ObservationEntity::Node(node.stable_identity)]
    );
    let decoded =
        ObservationResult::from_canonical_bytes(&results[1].canonical_bytes().unwrap()).unwrap();
    assert_eq!(decoded, results[1]);
}

#[test]
fn integration_point_field_retains_canonical_point_entities() {
    let projection = projection(ElementOrder::Tet10);
    let definition = RecoveredFieldDefinition::new(
        crate::contracts::FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
        ObservationQuantity::Stress,
        FieldLocation::IntegrationPoint,
        RecoveredFieldEntityDomain::SolidIntegrationPoints,
        ObservationValueShape::SymmetricTensor3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut request = spec("point_stress", ObservationTarget::WholeModel);
    request.quantity = ObservationQuantity::Stress;
    request.value_shape = ObservationValueShape::SymmetricTensor3;
    request.source_location = FieldLocation::IntegrationPoint;
    request.output_location = FieldLocation::IntegrationPoint;
    let mut step = ProcedureStep::structural_static("static");
    step.observations = vec![ObservationRequest::Field(request)];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
    let fields = catalog(&projection, &definition, vec![4, 6], values.clone());

    let result = evaluate_observations(&plan, &projection, &fields).unwrap();

    assert_eq!(result[0].field.as_host_f64().unwrap(), values);
    assert_eq!(result[0].entities.len(), 4);
    assert!(result[0]
        .entities
        .iter()
        .all(|entity| matches!(entity, ObservationEntity::IntegrationPoint(_))));
    assert!(result[0].entities.windows(2).all(|pair| pair[0] != pair[1]));
}

#[test]
fn integration_point_region_cardinality_applies_to_parent_elements() {
    let projection = projection(ElementOrder::Tet10);
    let definition = RecoveredFieldDefinition::new(
        crate::contracts::FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
        ObservationQuantity::Stress,
        FieldLocation::IntegrationPoint,
        RecoveredFieldEntityDomain::SolidIntegrationPoints,
        ObservationValueShape::SymmetricTensor3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let selector = projection.elements[0]
        .region_id
        .analysis_selection_id()
        .unwrap();
    let mut region = runmat_analysis_core::RegionTarget::volume(selector);
    region.cardinality = runmat_analysis_core::TargetCardinality::One;
    let mut request = spec(
        "point_stress",
        ObservationTarget::Spatial(runmat_analysis_core::ExcitationTarget::Region(region)),
    );
    request.quantity = ObservationQuantity::Stress;
    request.value_shape = ObservationValueShape::SymmetricTensor3;
    request.source_location = FieldLocation::IntegrationPoint;
    request.output_location = FieldLocation::IntegrationPoint;
    let mut step = ProcedureStep::structural_static("static");
    step.observations = vec![ObservationRequest::Field(request)];

    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();

    let ResolvedObservationTarget::Elements {
        domain,
        indices,
        identities,
    } = &plan.observations[0].target
    else {
        panic!("integration-point field did not resolve to discrete entities");
    };
    assert_eq!(*domain, RecoveredFieldEntityDomain::SolidIntegrationPoints);
    assert_eq!(indices, &[0, 1, 2, 3]);
    assert_eq!(identities.len(), 4);
}

#[test]
fn von_mises_projection_keeps_stress_as_its_source_quantity() {
    let projection = projection(ElementOrder::Tet4);
    let definition = RecoveredFieldDefinition::new(
        "stress.recovered",
        ObservationQuantity::Stress,
        FieldLocation::Element,
        RecoveredFieldEntityDomain::SolidElements,
        ObservationValueShape::SymmetricTensor3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut request = spec("von_mises", ObservationTarget::WholeModel);
    request.quantity = ObservationQuantity::Stress;
    request.value_shape = ObservationValueShape::SymmetricTensor3;
    request.source_location = FieldLocation::Element;
    request.output_location = FieldLocation::Element;
    request.component = ObservationComponent::VonMises;
    request.cost = runmat_analysis_core::ObservationCostPolicy::Automatic;
    let mut step = ProcedureStep::structural_static("static");
    step.observations = vec![ObservationRequest::Field(request)];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let fields = catalog(
        &projection,
        &definition,
        vec![1, 6],
        vec![100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    let result = evaluate_observations(&plan, &projection, &fields).unwrap();

    assert_eq!(result[0].field.as_host_f64().unwrap(), &[100.0]);
    assert_eq!(
        plan.observations[0].spec.result_quantity(),
        runmat_analysis_core::ObservationResultQuantity::Physical(
            ObservationQuantity::VonMisesStress
        )
    );
}

#[test]
fn principal_projections_execute_through_the_canonical_observation_plan() {
    let projection = projection(ElementOrder::Tet4);
    let definition = RecoveredFieldDefinition::new(
        "stress.recovered",
        ObservationQuantity::Stress,
        FieldLocation::Element,
        RecoveredFieldEntityDomain::SolidElements,
        ObservationValueShape::SymmetricTensor3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut maximum = spec("maximum_principal", ObservationTarget::WholeModel);
    maximum.quantity = ObservationQuantity::Stress;
    maximum.value_shape = ObservationValueShape::SymmetricTensor3;
    maximum.source_location = FieldLocation::Element;
    maximum.output_location = FieldLocation::Element;
    maximum.component = ObservationComponent::MaximumPrincipal;
    maximum.cost = runmat_analysis_core::ObservationCostPolicy::Automatic;
    let mut minimum = maximum.clone();
    minimum.observation_id = ObservationId("minimum_principal".into());
    minimum.component = ObservationComponent::MinimumPrincipal;
    let mut step = ProcedureStep::structural_static("static");
    step.observations = vec![
        ObservationRequest::Field(maximum),
        ObservationRequest::Field(minimum),
    ];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let fields = catalog(
        &projection,
        &definition,
        vec![1, 6],
        vec![2.0, 2.0, 1.0, 1.0, 0.0, 0.0],
    );
    let result = evaluate_observations(&plan, &projection, &fields).unwrap();

    assert!((result[0].field.as_host_f64().unwrap()[0] - 3.0).abs() < 1e-12);
    assert!((result[1].field.as_host_f64().unwrap()[0] - 1.0).abs() < 1e-12);
}

#[test]
fn path_interpolation_reproduces_linear_fields_for_tet4_and_tet10() {
    let positions = [[0.1, 0.1, 0.1], [0.2, 0.3, 0.1], [0.05, 0.15, 0.4]];
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection = projection(order);
        let definition = scalar_definition();
        let mut path = spec(
            "temperature_path",
            ObservationTarget::Path {
                positions: positions.into_iter().map(Position::new).collect(),
            },
        );
        path.output_location = FieldLocation::Point;
        path.recovery = RecoveryPolicy::ShapeFunctionInterpolation;
        let step = thermal_step(vec![ObservationRequest::Field(path)]);
        let plan = resolve_observation_plan(
            &step,
            &projection,
            std::slice::from_ref(&definition),
            ObservationLimits::default(),
        )
        .unwrap();
        let nodal_values = projection
            .nodes
            .iter()
            .map(|node| {
                let [x, y, z] = node.coordinates_m;
                10.0 + 2.0 * x - 3.0 * y + 4.0 * z
            })
            .collect::<Vec<_>>();
        let fields = catalog(
            &projection,
            &definition,
            vec![nodal_values.len()],
            nodal_values,
        );
        let result = evaluate_observations(&plan, &projection, &fields)
            .unwrap()
            .remove(0);
        let expected = positions.map(|[x, y, z]| 10.0 + 2.0 * x - 3.0 * y + 4.0 * z);
        for (actual, expected) in result.field.as_host_f64().unwrap().iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-11);
        }
        assert_eq!(result.location, FieldLocation::Point);
        assert_eq!(result.entities.len(), positions.len());
        for (ordinal, entity) in result.entities.iter().enumerate() {
            let ObservationEntity::Point {
                ordinal: actual_ordinal,
                position_m,
                ..
            } = entity
            else {
                panic!("path result must retain physical point locations")
            };
            assert_eq!(*actual_ordinal, ordinal as u64);
            assert_eq!(*position_m, positions[ordinal]);
        }
        let decoded =
            ObservationResult::from_canonical_bytes(&result.canonical_bytes().unwrap()).unwrap();
        assert_eq!(decoded, result);
    }
}

#[test]
fn path_interpolation_rejects_points_outside_the_admitted_volume() {
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let mut path = spec(
        "invalid_path",
        ObservationTarget::Path {
            positions: vec![
                Position::new([0.1, 0.1, 0.1]),
                Position::new([2.0, 0.0, 0.0]),
            ],
        },
    );
    path.output_location = FieldLocation::Point;
    path.recovery = RecoveryPolicy::ShapeFunctionInterpolation;
    let error = resolve_observation_plan(
        &thermal_step(vec![ObservationRequest::Field(path)]),
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("outside the admitted solid topology"));
}

#[test]
fn volume_and_boundary_reductions_use_finite_element_quadrature() {
    let projection = projection(ElementOrder::Tet10);
    let linear_values = projection
        .nodes
        .iter()
        .map(|node| node.coordinates_m.iter().sum())
        .collect::<Vec<f64>>();
    let definition = scalar_definition();
    let mut volume = spec("volume_average", ObservationTarget::WholeModel);
    volume.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Average,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    volume.output_location = FieldLocation::Global;
    let face_selector = projection.boundary_faces[0].provenance[0]
        .analysis_selection_id()
        .unwrap();
    let expected_boundary_average = projection.boundary_faces[0].centroid_m.iter().sum::<f64>();
    let mut boundary = spec(
        "boundary_average",
        ObservationTarget::Spatial(runmat_analysis_core::ExcitationTarget::Region(
            runmat_analysis_core::RegionTarget::boundary(face_selector),
        )),
    );
    boundary.reduction = volume.reduction.clone();
    boundary.output_location = FieldLocation::Global;
    let step = thermal_step(vec![
        ObservationRequest::Reduction(volume),
        ObservationRequest::Reduction(boundary),
    ]);
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let fields = catalog(
        &projection,
        &definition,
        vec![linear_values.len()],
        linear_values,
    );
    let results = evaluate_observations(&plan, &projection, &fields).unwrap();

    assert!((results[0].field.as_host_f64().unwrap()[0] - 0.75).abs() < 1.0e-12);
    assert!(
        (results[1].field.as_host_f64().unwrap()[0] - expected_boundary_average).abs() < 1.0e-12
    );
}

#[test]
fn partial_access_policy_controls_chunk_shape_without_changing_result() {
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let step = thermal_step(vec![ObservationRequest::Field(spec(
        "temperature",
        ObservationTarget::WholeModel,
    ))]);
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let fields = catalog(&projection, &definition, vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let mut result = evaluate_observations(&plan, &projection, &fields)
        .unwrap()
        .remove(0);
    let chunked = publish_observation(&plan, &result, 16).unwrap();
    assert!(chunked.chunks.len() > 1);
    chunked.revalidate(16).unwrap();

    let mut whole_step = step.clone();
    whole_step.observations[0]
        .spec_mut()
        .unwrap()
        .storage
        .partial_access = PartialAccessPolicy::WholeArtifact;
    let whole_plan = resolve_observation_plan(
        &whole_step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    result = evaluate_observations(&whole_plan, &projection, &fields)
        .unwrap()
        .remove(0);
    let whole = publish_observation(&whole_plan, &result, 16).unwrap();
    assert_eq!(whole.chunks.len(), 1);
    whole.revalidate(whole.chunks[0].len()).unwrap();
}

#[test]
fn bounded_history_retains_latest_samples_with_independent_identities() {
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let mut history = spec("temperature_history", ObservationTarget::WholeModel);
    history.sampling = SamplingPolicy::TimeSeconds(vec![0.0, 1.0, 2.0]);
    history.storage.retention = RetentionPolicy::LatestSamples { count: 2 };
    history.cost =
        runmat_analysis_core::ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 1_000_000,
            peak_resident_bytes: 10_000_000,
            stored_bytes: 10_000_000,
            transfer_bytes: 10_000_000,
            collective_count: 0,
        });
    let mut step = ProcedureStep::thermal_transient("thermal-history");
    step.independent_domain = runmat_analysis_core::IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 2.0,
        interval_count: 20,
    };
    step.observations = vec![ObservationRequest::History(history)];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let resource_facts =
        ObservationResourceFacts::from_plan(&plan, ObservationExecutionPolicy::scalable()).unwrap();
    assert_eq!(resource_facts.maximum_samples_per_observation, Count(3));
    assert_eq!(resource_facts.produced_sample_count, Count(3));
    assert_eq!(resource_facts.retained_sample_count, Count(2));
    let observation_id = ObservationId("temperature_history".into());
    let mut builder = BoundedObservationHistory::new(&plan, &observation_id).unwrap();
    let mut result_identities = Vec::new();
    for time in [0.0, 1.0, 2.0] {
        let sample = RecoveredFieldSample::TimeSeconds(time);
        let fields = catalog_at_sample(&projection, &definition, sample, vec![4], vec![1.0; 4]);
        let result =
            evaluate_observation_sample(&plan, &observation_id, &projection, &fields, sample)
                .unwrap();
        result_identities.push(result.logical_identity);
        builder.push(sample, result).unwrap();
    }
    let history = builder.finish().unwrap();
    assert_eq!(history.manifest().observed_sample_count, 3);
    assert_eq!(history.manifest().maximum_retained_samples, 2);
    assert_eq!(
        history
            .samples()
            .iter()
            .map(|sample| sample.coordinate)
            .collect::<Vec<_>>(),
        vec![
            RecoveredFieldSample::TimeSeconds(1.0),
            RecoveredFieldSample::TimeSeconds(2.0),
        ]
    );
    assert_eq!(
        history.samples()[0].result.logical_identity,
        result_identities[1]
    );
    assert_ne!(result_identities[0], result_identities[1]);
    assert_ne!(history.samples()[0].identity, history.samples()[1].identity);
    let decoded_manifest = ObservationHistoryManifest::from_canonical_bytes(
        &history.manifest().canonical_bytes().unwrap(),
    )
    .unwrap();
    let decoded_samples = history
        .samples()
        .iter()
        .map(|sample| {
            ObservationHistorySample::from_canonical_bytes(&sample.canonical_bytes().unwrap())
                .unwrap()
        })
        .collect();
    assert_eq!(
        ObservationHistory::from_manifest_and_samples(decoded_manifest, decoded_samples).unwrap(),
        history
    );
    let mut invalid_manifest = serde_json::to_value(history.manifest()).unwrap();
    invalid_manifest["maximum_retained_samples"] = serde_json::json!(3);
    assert!(serde_json::from_value::<ObservationHistoryManifest>(invalid_manifest).is_err());
}

#[test]
fn history_checkpoint_restart_preserves_exact_lineage_without_gaps_or_duplicates() {
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let mut history = spec("temperature_restart", ObservationTarget::WholeModel);
    history.sampling = SamplingPolicy::TimeSeconds(vec![0.0, 1.0, 2.0]);
    history.storage.retention = RetentionPolicy::LatestSamples { count: 2 };
    let mut step = ProcedureStep::thermal_transient("thermal-restart");
    step.independent_domain = runmat_analysis_core::IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 2.0,
        interval_count: 20,
    };
    step.observations = vec![ObservationRequest::History(history)];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let observation_id = ObservationId("temperature_restart".into());
    let result_at = |time: f64| {
        let sample = RecoveredFieldSample::TimeSeconds(time);
        let fields = catalog_at_sample(
            &projection,
            &definition,
            sample,
            vec![4],
            vec![time + 1.0; 4],
        );
        evaluate_observation_sample(&plan, &observation_id, &projection, &fields, sample).unwrap()
    };

    let mut uninterrupted = BoundedObservationHistory::new(&plan, &observation_id).unwrap();
    for time in [0.0, 1.0, 2.0] {
        uninterrupted
            .push(RecoveredFieldSample::TimeSeconds(time), result_at(time))
            .unwrap();
    }
    let uninterrupted = uninterrupted.finish().unwrap();

    let mut first_attempt = BoundedObservationHistory::new(&plan, &observation_id).unwrap();
    assert!(first_attempt.checkpoint(None).is_err());
    first_attempt
        .push(RecoveredFieldSample::TimeSeconds(0.0), result_at(0.0))
        .unwrap();
    let first_checkpoint = first_attempt.checkpoint(None).unwrap();
    let first_checkpoint = ObservationHistoryCheckpoint::from_canonical_bytes(
        &first_checkpoint.canonical_bytes().unwrap(),
    )
    .unwrap();
    let first_identity = first_checkpoint.manifest().identity;

    let mut second_attempt =
        BoundedObservationHistory::resume(&plan, &observation_id, first_checkpoint).unwrap();
    assert!(second_attempt
        .push(RecoveredFieldSample::TimeSeconds(0.0), result_at(0.0))
        .is_err());
    assert!(second_attempt
        .push(RecoveredFieldSample::TimeSeconds(2.0), result_at(2.0))
        .is_err());
    second_attempt
        .push(RecoveredFieldSample::TimeSeconds(1.0), result_at(1.0))
        .unwrap();
    let second_checkpoint = second_attempt.checkpoint(Some(first_identity)).unwrap();
    assert_eq!(
        second_checkpoint.manifest().predecessor,
        Some(first_identity)
    );
    let second_identity = second_checkpoint.manifest().identity;

    let mut final_attempt =
        BoundedObservationHistory::resume(&plan, &observation_id, second_checkpoint).unwrap();
    final_attempt
        .push(RecoveredFieldSample::TimeSeconds(2.0), result_at(2.0))
        .unwrap();
    let final_checkpoint = final_attempt
        .checkpoint(Some(second_identity))
        .expect("latest retention remains checkpointable after eviction");
    assert_eq!(
        final_checkpoint.manifest().predecessor,
        Some(second_identity)
    );
    assert_eq!(final_checkpoint.samples().len(), 2);
    let mut invalid_manifest = serde_json::to_value(final_checkpoint.manifest()).unwrap();
    invalid_manifest["retained_prefix_digest"] =
        serde_json::to_value(Digest::sha256("substituted-retained-prefix")).unwrap();
    assert!(
        serde_json::from_value::<ObservationHistoryCheckpointManifest>(invalid_manifest).is_err()
    );
    let restarted = final_attempt.finish().unwrap();
    assert_eq!(restarted, uninterrupted);

    let mut stale_plan = plan.clone();
    stale_plan.identity = ResolvedObservationPlanIdentity(Digest::sha256("stale-plan"));
    assert!(
        BoundedObservationHistory::resume(&stale_plan, &observation_id, final_checkpoint).is_err()
    );
}

#[test]
fn series_statistics_make_nonuniform_weighting_explicit() {
    let times = [0.0, 1.0, 3.0];
    let values = [0.0, 2.0, 2.0];
    let (uniform_plan, uniform_history) = completed_scalar_series(
        SeriesReduction::Mean {
            weighting: SeriesWeighting::UniformSamples,
        },
        ExtremaTiePolicy::FirstCanonical,
        RetentionPolicy::Canonical,
        &times,
        &values,
    )
    .unwrap();
    let uniform = reduce_observation_history(
        &uniform_plan,
        &uniform_history,
        ObservationExecutionPolicy::serial(),
    )
    .unwrap();
    let SeriesReductionPayload::Values { field, extrema } = &uniform.payload else {
        panic!("mean must produce one value field")
    };
    assert!(extrema.is_empty());
    assert!((field.as_host_f64().unwrap()[0] - 4.0 / 3.0).abs() < 1.0e-14);
    let serial_resources =
        ObservationResourceFacts::from_plan(&uniform_plan, ObservationExecutionPolicy::serial())
            .unwrap();
    let scalable_resources =
        ObservationResourceFacts::from_plan(&uniform_plan, ObservationExecutionPolicy::scalable())
            .unwrap();
    assert_eq!(serial_resources.collective_count, Count(0));
    assert_eq!(scalable_resources.collective_count, Count(2));

    let (trapezoidal_plan, trapezoidal_history) = completed_scalar_series(
        SeriesReduction::Mean {
            weighting: SeriesWeighting::TrapezoidalIndependentDomain,
        },
        ExtremaTiePolicy::FirstCanonical,
        RetentionPolicy::Canonical,
        &times,
        &values,
    )
    .unwrap();
    let serial = reduce_observation_history(
        &trapezoidal_plan,
        &trapezoidal_history,
        ObservationExecutionPolicy::serial(),
    )
    .unwrap();
    let split = reduce_observation_history(
        &trapezoidal_plan,
        &trapezoidal_history,
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
    )
    .unwrap();
    assert_eq!(serial.identity, split.identity);
    assert_ne!(serial.evidence, split.evidence);
    let SeriesReductionPayload::Values { field, .. } = serial.payload else {
        panic!("mean must produce one value field")
    };
    assert!((field.as_host_f64().unwrap()[0] - 5.0 / 3.0).abs() < 1.0e-14);
    assert_eq!(
        SeriesReductionResult::from_canonical_bytes(&split.canonical_bytes().unwrap()).unwrap(),
        split
    );
}

#[test]
fn series_extrema_and_envelope_retain_samples_entities_and_exact_ties() {
    let (plan, history) = completed_scalar_series(
        SeriesReduction::Envelope,
        ExtremaTiePolicy::AllLocations,
        RetentionPolicy::Canonical,
        &[0.0, 1.0, 2.0],
        &[1.0, 3.0, 3.0],
    )
    .unwrap();
    let result = reduce_observation_history(
        &plan,
        &history,
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(2).unwrap()),
    )
    .unwrap();
    let SeriesReductionPayload::Envelope {
        minimum,
        maximum,
        minima,
        maxima,
    } = &result.payload
    else {
        panic!("envelope must produce minimum and maximum fields")
    };
    assert_eq!(minimum.as_host_f64().unwrap(), &[1.0]);
    assert_eq!(maximum.as_host_f64().unwrap(), &[3.0]);
    assert_eq!(minima.len(), 1);
    assert_eq!(maxima.len(), 2);
    assert!(minima[0].entity.is_some());
    assert_eq!(
        maxima[0].source.coordinate,
        RecoveredFieldSample::TimeSeconds(1.0)
    );
    assert_eq!(
        maxima[1].source.coordinate,
        RecoveredFieldSample::TimeSeconds(2.0)
    );
    result.validate_for(&plan, &history).unwrap();

    let mut substituted = result.clone();
    substituted.evidence.contributions[0].values[0] += 1.0;
    assert!(substituted.validate_for(&plan, &history).is_err());
}

#[test]
fn every_scalar_series_operation_has_defined_numerical_semantics() {
    let cases = [
        (SeriesReduction::Minimum, -2.0, 1usize),
        (SeriesReduction::Maximum, 2.0, 1usize),
        (SeriesReduction::AbsoluteMaximum, 2.0, 2usize),
        (
            SeriesReduction::RootMeanSquare {
                weighting: SeriesWeighting::UniformSamples,
            },
            (8.0_f64 / 3.0).sqrt(),
            0usize,
        ),
        (
            SeriesReduction::Variance {
                weighting: SeriesWeighting::UniformSamples,
            },
            8.0 / 3.0,
            0usize,
        ),
    ];
    for (operation, expected, expected_extrema) in cases {
        let (plan, history) = completed_scalar_series(
            operation,
            ExtremaTiePolicy::AllLocations,
            RetentionPolicy::Canonical,
            &[0.0, 1.0, 2.0],
            &[-2.0, 0.0, 2.0],
        )
        .unwrap();
        let result =
            reduce_observation_history(&plan, &history, ObservationExecutionPolicy::scalable())
                .unwrap();
        let SeriesReductionPayload::Values { field, extrema } = result.payload else {
            panic!("non-envelope reduction must produce one value field")
        };
        assert!((field.as_host_f64().unwrap()[0] - expected).abs() < 1.0e-14);
        assert_eq!(extrema.len(), expected_extrema);
    }

    let (plan, history) = completed_scalar_series(
        SeriesReduction::Variance {
            weighting: SeriesWeighting::UniformSamples,
        },
        ExtremaTiePolicy::FirstCanonical,
        RetentionPolicy::Canonical,
        &[0.0, 1.0, 2.0],
        &[1.0e12 - 1.0, 1.0e12, 1.0e12 + 1.0],
    )
    .unwrap();
    let stable =
        reduce_observation_history(&plan, &history, ObservationExecutionPolicy::serial()).unwrap();
    let SeriesReductionPayload::Values { field, .. } = stable.payload else {
        panic!("variance must produce one value field")
    };
    assert!((field.as_host_f64().unwrap()[0] - 2.0 / 3.0).abs() < 1.0e-14);
}

#[test]
fn series_reduction_rejects_incomplete_retention() {
    assert!(completed_scalar_series(
        SeriesReduction::RootMeanSquare {
            weighting: SeriesWeighting::UniformSamples,
        },
        ExtremaTiePolicy::FirstCanonical,
        RetentionPolicy::LatestSamples { count: 2 },
        &[0.0, 1.0, 2.0],
        &[1.0, 2.0, 3.0],
    )
    .is_err());
}

#[test]
fn typed_event_histories_preserve_source_attribution_and_ordering() {
    let projection = projection(ElementOrder::Tet4);
    let definition = RecoveredFieldDefinition::new(
        "structural.displacement.event",
        ObservationQuantity::Displacement,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::StructuralNodes,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let event = ObservationEvent::Solver(SolverObservationEvent::FirstYield);
    let mut request = spec("first_yield_displacement", ObservationTarget::WholeModel);
    request.quantity = ObservationQuantity::Displacement;
    request.value_shape = ObservationValueShape::Vector3;
    request.sampling = SamplingPolicy::Event(event);
    request.cost =
        runmat_analysis_core::ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 1_000_000,
            peak_resident_bytes: 10_000_000,
            stored_bytes: 10_000_000,
            transfer_bytes: 10_000_000,
            collective_count: 0,
        });
    let mut step = ProcedureStep::structural_nonlinear("event-history");
    step.accuracy.resources.maximum_samples = 4;
    step.observations = vec![ObservationRequest::History(request)];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let observation_id = ObservationId("first_yield_displacement".into());
    let sample = RecoveredFieldSample::LoadFactor(0.75);
    let fields = catalog_at_sample(&projection, &definition, sample, vec![4, 3], vec![2.0; 12]);
    let sequence = SequenceIdentity::new(Digest::sha256("event-sequence")).unwrap();
    let producer = StepIdentity::new(sequence, Digest::sha256("event-step")).unwrap();
    let source_result =
        SolverResultIdentity::from_step_result(producer, Digest::sha256("event-result")).unwrap();
    let attribution = ObservationEventAttribution::Solver(SolverEventAttribution {
        event: SolverObservationEvent::FirstYield,
        sample,
        occurrence_index: 0,
        source: SolverEventSource::CompletedResult {
            result: source_result,
        },
    });

    assert!(
        evaluate_observation_sample(&plan, &observation_id, &projection, &fields, sample,).is_err()
    );
    let event_sample = evaluate_observation_event_sample(
        &plan,
        &observation_id,
        &projection,
        &fields,
        attribution,
    )
    .unwrap();
    let substituted_source =
        SolverResultIdentity::from_step_result(producer, Digest::sha256("other-event-result"))
            .unwrap();
    let mut substituted_sample = event_sample.clone();
    substituted_sample.event_attribution = Some(ObservationEventAttribution::Solver(
        SolverEventAttribution {
            event: SolverObservationEvent::FirstYield,
            sample,
            occurrence_index: 0,
            source: SolverEventSource::CompletedResult {
                result: substituted_source,
            },
        },
    ));
    assert!(substituted_sample.validate().is_err());
    let mut builder = BoundedObservationHistory::new(&plan, &observation_id).unwrap();
    assert!(builder.push(sample, event_sample.result.clone()).is_err());
    builder.push_event(event_sample).unwrap();
    let history = builder.finish().unwrap();
    assert_eq!(history.samples()[0].event_attribution, Some(attribution));
    assert_eq!(
        ObservationHistory::from_canonical_bytes(&history.canonical_bytes().unwrap()).unwrap(),
        history
    );

    let wrong_event = ObservationEventAttribution::Solver(SolverEventAttribution {
        event: SolverObservationEvent::ContactStatusChange,
        sample,
        occurrence_index: 0,
        source: SolverEventSource::CompletedResult {
            result: source_result,
        },
    });
    assert!(evaluate_observation_event_sample(
        &plan,
        &observation_id,
        &projection,
        &fields,
        wrong_event,
    )
    .is_err());
    let invalid_source = ObservationEventAttribution::Solver(SolverEventAttribution {
        event: SolverObservationEvent::ConvergenceFailure,
        sample,
        occurrence_index: 0,
        source: SolverEventSource::CompletedResult {
            result: source_result,
        },
    });
    assert!(invalid_source.validate().is_err());
    let repeated_first_yield = ObservationEventAttribution::Solver(SolverEventAttribution {
        event: SolverObservationEvent::FirstYield,
        sample,
        occurrence_index: 1,
        source: SolverEventSource::CompletedResult {
            result: source_result,
        },
    });
    assert!(repeated_first_yield.validate().is_err());

    let mut contact_request = spec("contact_changes", ObservationTarget::WholeModel);
    contact_request.quantity = ObservationQuantity::Displacement;
    contact_request.value_shape = ObservationValueShape::Vector3;
    contact_request.sampling = SamplingPolicy::Event(ObservationEvent::Solver(
        SolverObservationEvent::ContactStatusChange,
    ));
    contact_request.cost =
        runmat_analysis_core::ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 1_000_000,
            peak_resident_bytes: 10_000_000,
            stored_bytes: 10_000_000,
            transfer_bytes: 10_000_000,
            collective_count: 0,
        });
    let mut contact_step = ProcedureStep::structural_nonlinear("contact-events");
    contact_step.accuracy.resources.maximum_samples = 4;
    contact_step.observations = vec![ObservationRequest::History(contact_request)];
    let contact_plan = resolve_observation_plan(
        &contact_step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let contact_id = ObservationId("contact_changes".into());
    let mut contact_builder = BoundedObservationHistory::new(&contact_plan, &contact_id).unwrap();
    for occurrence_index in [0, 1] {
        let attribution = ObservationEventAttribution::Solver(SolverEventAttribution {
            event: SolverObservationEvent::ContactStatusChange,
            sample,
            occurrence_index,
            source: SolverEventSource::CompletedResult {
                result: source_result,
            },
        });
        let event_sample = evaluate_observation_event_sample(
            &contact_plan,
            &contact_id,
            &projection,
            &fields,
            attribution,
        )
        .unwrap();
        contact_builder.push_event(event_sample).unwrap();
    }
    let contact_history = contact_builder.finish().unwrap();
    assert_ne!(
        contact_history.samples()[0].identity,
        contact_history.samples()[1].identity
    );
    assert_eq!(
        contact_history.samples()[1]
            .event_attribution
            .unwrap()
            .occurrence_index(),
        1
    );

    let series_attribution = ObservationEventAttribution::Series(SeriesEventAttribution {
        event: SeriesObservationEvent::PeakResponse,
        sample,
        source_history: history.manifest().identity,
    });
    assert_eq!(
        series_attribution.event(),
        ObservationEvent::Series(SeriesObservationEvent::PeakResponse)
    );
    series_attribution.validate().unwrap();
    let accuracy = crate::results::accuracy::evaluate_accuracy(
        &AccuracyPlan::experimental_baseline("event-accuracy"),
        &[],
    )
    .unwrap();
    let accuracy_attribution = ObservationEventAttribution::Accuracy(AccuracyEventAttribution {
        event: AccuracyObservationEvent::CriterionCrossing,
        sample,
        source_report: accuracy.identity(),
    });
    assert_eq!(
        accuracy_attribution.event(),
        ObservationEvent::Accuracy(AccuracyObservationEvent::CriterionCrossing)
    );
    accuracy_attribution.validate().unwrap();
}

#[test]
fn event_histories_may_finish_empty_but_cannot_request_series_statistics() {
    let projection = projection(ElementOrder::Tet4);
    let definition = RecoveredFieldDefinition::new(
        "structural.displacement.optional_event",
        ObservationQuantity::Displacement,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::StructuralNodes,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut request = spec("optional_first_yield", ObservationTarget::WholeModel);
    request.quantity = ObservationQuantity::Displacement;
    request.value_shape = ObservationValueShape::Vector3;
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
    let mut step = ProcedureStep::structural_nonlinear("optional-event-history");
    step.accuracy.resources.maximum_samples = 4;
    step.observations = vec![ObservationRequest::History(request.clone())];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let history =
        BoundedObservationHistory::new(&plan, &ObservationId("optional_first_yield".into()))
            .unwrap()
            .finish()
            .unwrap();
    assert!(history.samples().is_empty());
    assert_eq!(
        ObservationHistory::from_canonical_bytes(&history.canonical_bytes().unwrap()).unwrap(),
        history
    );

    request.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::None,
        series: SeriesReduction::Mean {
            weighting: SeriesWeighting::UniformSamples,
        },
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    step.observations = vec![ObservationRequest::History(request)];
    assert!(resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .is_err());
}

#[test]
fn complex_frequency_histories_preserve_exact_phasors_and_derived_views() {
    let projection = projection(ElementOrder::Tet4);
    let definition = complex_pressure_definition();
    let representations = [
        ComplexRepresentation::Real,
        ComplexRepresentation::Imaginary,
        ComplexRepresentation::Magnitude,
        ComplexRepresentation::PhaseRadians,
        ComplexRepresentation::Complex,
    ];
    let mut step = ProcedureStep::acoustic_harmonic("acoustic-frequency-history");
    step.independent_domain =
        runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![100.0, 200.0]);
    step.observations = representations
        .iter()
        .enumerate()
        .map(|(index, representation)| {
            ObservationRequest::History(complex_pressure_history_spec(
                &format!("pressure_{index}"),
                *representation,
            ))
        })
        .collect();
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let mut builders = plan
        .observations
        .iter()
        .map(|observation| {
            BoundedObservationHistory::new(&plan, &observation.spec.observation_id).unwrap()
        })
        .collect::<Vec<_>>();

    for frequency in [100.0, 200.0] {
        let scale = frequency / 100.0;
        let values = [3.0, 4.0, 0.0, 2.0, -3.0, 0.0, 0.0, -4.0]
            .into_iter()
            .map(|value| value * scale)
            .collect::<Vec<_>>();
        let sample = RecoveredFieldSample::FrequencyHertz(frequency);
        let fields = catalog_at_sample(&projection, &definition, sample, vec![4, 2], values);
        for (observation, builder) in plan.observations.iter().zip(&mut builders) {
            let result = evaluate_observation_sample(
                &plan,
                &observation.spec.observation_id,
                &projection,
                &fields,
                sample,
            )
            .unwrap();
            builder.push(sample, result).unwrap();
        }
    }

    let histories = builders
        .into_iter()
        .map(|builder| builder.finish().unwrap())
        .collect::<Vec<_>>();
    let final_values = histories
        .iter()
        .map(|history| {
            let decoded =
                ObservationHistory::from_canonical_bytes(&history.canonical_bytes().unwrap())
                    .unwrap();
            assert_eq!(&decoded, history);
            assert_eq!(
                history.manifest().complex_representation,
                history.samples()[0].result.complex_representation
            );
            history.samples()[0]
                .result
                .field
                .as_host_f64()
                .unwrap()
                .to_vec()
        })
        .collect::<Vec<_>>();
    assert_eq!(final_values[0], vec![3.0, 0.0, -3.0, 0.0]);
    assert_eq!(final_values[1], vec![4.0, 2.0, 0.0, -4.0]);
    assert_eq!(final_values[2], vec![5.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        final_values[3],
        vec![
            4.0_f64.atan2(3.0),
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            -std::f64::consts::FRAC_PI_2,
        ]
    );
    assert_eq!(
        final_values[4],
        vec![3.0, 4.0, 0.0, 2.0, -3.0, 0.0, 0.0, -4.0]
    );
    assert_eq!(histories[4].samples()[0].result.field.shape, vec![4, 2]);
    assert_eq!(
        histories[4]
            .samples()
            .iter()
            .map(|sample| sample.coordinate)
            .collect::<Vec<_>>(),
        vec![
            RecoveredFieldSample::FrequencyHertz(100.0),
            RecoveredFieldSample::FrequencyHertz(200.0),
        ]
    );
}

#[test]
fn resolved_frequency_schedule_is_the_single_sample_admission_boundary() {
    let projection = projection(ElementOrder::Tet4);
    let definition = complex_pressure_definition();
    let mut step = ProcedureStep::acoustic_harmonic("frequency-sample-admission");
    step.independent_domain =
        runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![100.0, 150.0, 200.0]);
    step.observations = vec![ObservationRequest::History(complex_pressure_history_spec(
        "selected_frequencies",
        ComplexRepresentation::Complex,
    ))];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let observation = &plan.observations[0];

    assert!(resolved_observation_accepts_sample(
        &plan,
        observation,
        RecoveredFieldSample::FrequencyHertz(100.0),
        None,
    )
    .unwrap());
    assert!(!resolved_observation_accepts_sample(
        &plan,
        observation,
        RecoveredFieldSample::FrequencyHertz(150.0),
        None,
    )
    .unwrap());
    assert!(resolved_observation_accepts_sample(
        &plan,
        observation,
        RecoveredFieldSample::Final,
        None,
    )
    .is_err());
}

#[test]
fn observation_assembly_requires_the_exact_subset_scheduled_at_one_frequency() {
    let projection = projection(ElementOrder::Tet4);
    let definition = complex_pressure_definition();
    let mut first =
        complex_pressure_history_spec("first_frequency_set", ComplexRepresentation::Complex);
    first.sampling = SamplingPolicy::FrequencyHertz(vec![100.0, 200.0]);
    let mut second =
        complex_pressure_history_spec("second_frequency_set", ComplexRepresentation::Complex);
    second.sampling = SamplingPolicy::FrequencyHertz(vec![150.0]);
    let mut step = ProcedureStep::acoustic_harmonic("frequency-assembly-admission");
    step.independent_domain =
        runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![100.0, 150.0, 200.0]);
    step.observations = vec![
        ObservationRequest::History(first),
        ObservationRequest::History(second),
    ];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let fields_at = |frequency| {
        catalog_at_sample(
            &projection,
            &definition,
            RecoveredFieldSample::FrequencyHertz(frequency),
            vec![4, 2],
            vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
        )
    };
    let sequence = SequenceIdentity::new(Digest::sha256("frequency-assembly-sequence")).unwrap();
    let producer = StepIdentity::new(sequence, Digest::sha256("frequency-assembly-step")).unwrap();
    let source_result =
        SolverResultIdentity::from_step_result(producer, Digest::sha256("frequency-result"))
            .unwrap();
    let topology =
        crate::physics::structural::topology::CanonicalStructuralTopologyIdentity::Explicit {
            structural_topology_digest: projection.projection_digest,
        };
    let recovery = FieldRecoveryRequest::new(
        producer,
        FieldRecoverySource::SolverResult(source_result),
        topology,
        crate::execution::FieldRecoveryPlan::default(),
    )
    .unwrap();
    let source_catalog = fields_at(100.0);
    let source_field = source_catalog.fields()[0].clone();
    let artifact = RecoveredFieldArtifact::new(
        recovery.identity(),
        FieldRecoverySource::SolverResult(source_result),
        projection.projection_digest,
        RecoveredFieldEntityCounts {
            structural_nodes: projection.nodes.len(),
            solid_elements: projection.elements.len(),
            solid_integration_points: projection.elements.len() * 4,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        },
        vec![source_field.clone()],
    )
    .unwrap();
    let selection =
        RecoveredFieldSelection::new(artifact.manifest().unwrap(), vec![source_field.identity()])
            .unwrap();
    let source = crate::execution::ObservationEvaluationRequest::new(
        producer,
        &selection,
        topology,
        plan.clone(),
    )
    .unwrap();
    let first_result = evaluate_observation_sample(
        &plan,
        &ObservationId("first_frequency_set".into()),
        &projection,
        &fields_at(100.0),
        RecoveredFieldSample::FrequencyHertz(100.0),
    )
    .unwrap();
    let second_result = evaluate_observation_sample(
        &plan,
        &ObservationId("second_frequency_set".into()),
        &projection,
        &fields_at(150.0),
        RecoveredFieldSample::FrequencyHertz(150.0),
    )
    .unwrap();
    assert!(ObservationAssemblyRequest::new(&source, std::slice::from_ref(&first_result)).is_ok());
    assert!(ObservationAssemblyRequest::new(&source, &[]).is_err());
    assert!(ObservationAssemblyRequest::new(&source, &[first_result, second_result]).is_err());
}

#[test]
fn complex_observation_precision_and_real_source_mismatch_fail_closed() {
    let projection = projection(ElementOrder::Tet4);
    let complex_definition = complex_pressure_definition();
    let mut wrong_precision =
        complex_pressure_history_spec("complex_wrong_precision", ComplexRepresentation::Complex);
    wrong_precision.storage.precision = SignalPrecision::Real64;
    let mut step = ProcedureStep::acoustic_harmonic("complex-wrong-precision");
    step.independent_domain =
        runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![100.0, 200.0]);
    step.observations = vec![ObservationRequest::History(wrong_precision)];
    assert!(resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&complex_definition),
        ObservationLimits::default(),
    )
    .unwrap_err()
    .to_string()
    .contains("precision"));

    let real_definition = RecoveredFieldDefinition::new(
        "acoustic.pressure.real-only",
        ObservationQuantity::AcousticPressure,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::StructuralNodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    step.observations = vec![ObservationRequest::History(complex_pressure_history_spec(
        "imaginary_from_real",
        ComplexRepresentation::Imaginary,
    ))];
    assert!(resolve_observation_plan(
        &step,
        &projection,
        &[real_definition],
        ObservationLimits::default(),
    )
    .unwrap_err()
    .to_string()
    .contains("real recovered field"));
}

#[test]
fn complex_vector_component_projection_uses_logical_components() {
    let projection = projection(ElementOrder::Tet4);
    let definition = RecoveredFieldDefinition::new(
        "acoustic.velocity.recovered",
        ObservationQuantity::AcousticParticleVelocity,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::StructuralNodes,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::ComplexInterleaved,
    )
    .unwrap();
    let mut y_component =
        complex_pressure_history_spec("velocity_y_complex", ComplexRepresentation::Complex);
    y_component.quantity = ObservationQuantity::AcousticParticleVelocity;
    y_component.value_shape = ObservationValueShape::Vector3;
    y_component.component = ObservationComponent::Y;
    let mut magnitude =
        complex_pressure_history_spec("velocity_magnitude", ComplexRepresentation::Magnitude);
    magnitude.quantity = ObservationQuantity::AcousticParticleVelocity;
    magnitude.value_shape = ObservationValueShape::Vector3;
    magnitude.component = ObservationComponent::Magnitude;
    let mut step = ProcedureStep::acoustic_harmonic("complex-vector-components");
    step.independent_domain =
        runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![100.0, 200.0]);
    step.observations = vec![
        ObservationRequest::History(y_component),
        ObservationRequest::History(magnitude),
    ];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let source_node = [3.0, 4.0, 0.0, 12.0, 0.0, 0.0];
    let values = source_node.repeat(4);
    let sample = RecoveredFieldSample::FrequencyHertz(100.0);
    let fields = catalog_at_sample(&projection, &definition, sample, vec![4, 3, 2], values);

    let y = evaluate_observation_sample(
        &plan,
        &ObservationId("velocity_y_complex".into()),
        &projection,
        &fields,
        sample,
    )
    .unwrap();
    assert_eq!(y.value_shape, ObservationValueShape::Scalar);
    assert_eq!(y.field.shape, vec![4, 2]);
    assert_eq!(
        y.field.as_host_f64().unwrap(),
        &[0.0, 12.0, 0.0, 12.0, 0.0, 12.0, 0.0, 12.0]
    );
    let magnitude = evaluate_observation_sample(
        &plan,
        &ObservationId("velocity_magnitude".into()),
        &projection,
        &fields,
        sample,
    )
    .unwrap();
    assert_eq!(magnitude.field.shape, vec![4]);
    assert_eq!(magnitude.field.as_host_f64().unwrap(), &[13.0; 4]);
}

#[test]
fn study_parameter_history_keeps_parameter_identity_and_exact_values() {
    let projection = projection(ElementOrder::Tet4);
    let definition = RecoveredFieldDefinition::new(
        "structural.displacement.parameter-study",
        ObservationQuantity::Displacement,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::StructuralNodes,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut request = spec("parameter_history", ObservationTarget::WholeModel);
    request.quantity = ObservationQuantity::Displacement;
    request.value_shape = ObservationValueShape::Vector3;
    request.sampling = SamplingPolicy::StudyParameterValues(vec![0.01, 0.05]);
    request.cost =
        runmat_analysis_core::ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 1_000_000,
            peak_resident_bytes: 10_000_000,
            stored_bytes: 10_000_000,
            transfer_bytes: 10_000_000,
            collective_count: 0,
        });
    let mut step = ProcedureStep::structural_modal("parameter-study");
    step.procedure = runmat_analysis_core::AnalysisProcedure::Adaptive;
    step.independent_domain = runmat_analysis_core::IndependentDomain::StudyParameter {
        parameter_id: "damping_ratio".into(),
        canonical_values: vec![0.01, 0.02, 0.05],
    };
    step.observations = vec![ObservationRequest::History(request)];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let observation_id = ObservationId("parameter_history".into());
    let mut builder = BoundedObservationHistory::new(&plan, &observation_id).unwrap();
    for value in [0.01, 0.05] {
        let sample = RecoveredFieldSample::StudyParameterValue(value);
        let fields = catalog_at_sample(
            &projection,
            &definition,
            sample,
            vec![4, 3],
            vec![value; 12],
        );
        let result =
            evaluate_observation_sample(&plan, &observation_id, &projection, &fields, sample)
                .unwrap();
        builder.push(sample, result).unwrap();
    }
    let history = builder.finish().unwrap();
    assert_eq!(
        history.manifest().independent_domain,
        step.independent_domain
    );
    assert_eq!(
        history
            .samples()
            .iter()
            .map(|sample| sample.coordinate)
            .collect::<Vec<_>>(),
        vec![
            RecoveredFieldSample::StudyParameterValue(0.01),
            RecoveredFieldSample::StudyParameterValue(0.05),
        ]
    );
    assert_eq!(
        ObservationHistory::from_canonical_bytes(&history.canonical_bytes().unwrap()).unwrap(),
        history
    );
}

#[test]
fn history_bounds_and_coordinates_fail_closed_before_publication() {
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let mut unbounded = spec("unbounded", ObservationTarget::WholeModel);
    unbounded.sampling = SamplingPolicy::EveryAcceptedIncrement;
    unbounded.storage.retention = RetentionPolicy::Canonical;
    let mut step = ProcedureStep::thermal_transient("unbounded-history");
    step.observations = vec![ObservationRequest::History(unbounded)];
    assert!(resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap_err()
    .to_string()
    .contains("cost upper bound understates"));

    let mut explicit = spec("explicit", ObservationTarget::WholeModel);
    explicit.sampling = SamplingPolicy::TimeSeconds(vec![0.0, 1.0]);
    explicit.storage.retention = RetentionPolicy::ExplicitSamples { maximum_count: 1 };
    step.observations = vec![ObservationRequest::History(explicit)];
    assert!(resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .is_err());

    let mut latest = spec("latest", ObservationTarget::WholeModel);
    latest.sampling = SamplingPolicy::TimeSeconds(vec![0.0, 1.0]);
    latest.storage.retention = RetentionPolicy::LatestSamples { count: 1 };
    step.observations = vec![ObservationRequest::History(latest)];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let observation_id = ObservationId("latest".into());
    let sample = RecoveredFieldSample::TimeSeconds(0.5);
    let fields = catalog_at_sample(&projection, &definition, sample, vec![4], vec![1.0; 4]);
    assert!(
        evaluate_observation_sample(&plan, &observation_id, &projection, &fields, sample).is_err()
    );
}

#[test]
fn invalid_and_unbounded_requests_fail_before_field_evaluation() {
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let mut unsupported = spec("history", ObservationTarget::WholeModel);
    unsupported.sampling = SamplingPolicy::EveryAcceptedIncrement;
    let step = thermal_step(vec![ObservationRequest::History(unsupported)]);
    assert!(resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap_err()
    .to_string()
    .contains("sampling policy is invalid"));

    let step = thermal_step(vec![ObservationRequest::Field(spec(
        "too_large",
        ObservationTarget::WholeModel,
    ))]);
    let limits = ObservationLimits {
        maximum_values: 1,
        ..ObservationLimits::default()
    };
    assert!(matches!(
        resolve_observation_plan(&step, &projection, &[definition], limits),
        Err(ObservationError::Limit("observation value count"))
    ));
}

#[test]
fn explicit_structural_nodes_resolve_and_evaluate_without_a_solver_mesh() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let structural = model.structural.as_ref().unwrap();
    let node_count = structural.nodes.len();
    let beam_count = structural
        .elements
        .iter()
        .filter(|element| {
            matches!(
                element.kind,
                runmat_analysis_core::StructuralElementKind::Beam(_)
            )
        })
        .count();
    let shell_count = structural.elements.len() - beam_count;
    let definition = RecoveredFieldDefinition::new(
        "structural.displacement",
        ObservationQuantity::Displacement,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::StructuralNodes,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut request = spec("explicit-displacement", ObservationTarget::WholeModel);
    request.quantity = ObservationQuantity::Displacement;
    request.value_shape = ObservationValueShape::Vector3;
    model.sequence.steps[0].observations = vec![ObservationRequest::Field(request)];
    let constraint_plan =
        crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(&model, None)
            .unwrap();
    let step = &model.sequence.steps[0];
    let plan = resolve_structural_observation_plan(
        step,
        &model,
        None,
        &constraint_plan,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let values = (0..node_count * 3).map(|value| value as f64).collect();
    let field = RecoveredField::new(
        definition,
        RecoveredFieldSample::Final,
        runmat_analysis_core::AnalysisField::host_f64(
            "structural.displacement",
            vec![node_count, 3],
            values,
        ),
    )
    .unwrap();
    let fields = RecoveredFieldCatalog::new_for_topology(
        plan.topology_digest,
        RecoveredFieldEntityCounts {
            structural_nodes: node_count,
            solid_elements: 0,
            solid_integration_points: 0,
            beam_elements: beam_count,
            shell_elements: shell_count,
            connectors: 0,
            constraint_groups: 0,
        },
        vec![field],
    )
    .unwrap();
    let results = evaluate_structural_observations(&plan, &model, None, &fields).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].entities.len(), node_count);
    assert_eq!(results[0].field.shape, vec![node_count, 3]);
}

fn solver_layers(
    model: &runmat_analysis_core::AnalysisModel,
) -> (
    StudyIdentity,
    SequenceIdentity,
    StepIdentity,
    OperatorIdentity,
) {
    const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);
    let study = StudyIdentity::from_authored_contract(model, None).unwrap();
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let step = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0]).unwrap();
    let model_digest = Digest::sha256(
        encode_contract(
            b"observation-identity-test/v1\0",
            "solver-model",
            &model.solver_identity_contract(),
            LIMITS,
        )
        .unwrap(),
    );
    let constraints =
        ConstraintPlanIdentity::new(step, Digest::sha256("mesh"), Digest::sha256("constraints"))
            .unwrap();
    let problem = ProblemIdentity::new(step, model_digest, None, constraints, Vec::new()).unwrap();
    let discretization = DiscretizationIdentity::new(
        problem,
        Digest::sha256("mesh"),
        Digest::sha256("formulation"),
    )
    .unwrap();
    let operator = OperatorIdentity::new(discretization, Digest::sha256("operator")).unwrap();
    (study, sequence, step, operator)
}

fn result_identity(
    study: StudyIdentity,
    step: StepIdentity,
    operator: OperatorIdentity,
) -> AnalysisResultIdentity {
    let numerical = NumericalPlanIdentity::new(operator, Digest::sha256("solver")).unwrap();
    let execution = ExecutionPlanIdentity::new(numerical, Digest::sha256("execution")).unwrap();
    let workload = AnalysisWorkloadIdentity::new(
        study,
        step,
        execution,
        AnalysisWorkKind::Solve,
        Digest::sha256("work"),
        vec![Digest::sha256("input")],
    )
    .unwrap();
    AnalysisResultIdentity::new(workload, vec![Digest::sha256("output")]).unwrap()
}

#[test]
fn output_and_accuracy_changes_do_not_invalidate_solver_layers() {
    let mut baseline = fixture_model(FixtureId::CantileverLinearStatic);
    let projection = projection(ElementOrder::Tet4);
    let definition = RecoveredFieldDefinition::new(
        "displacement.recovered",
        ObservationQuantity::Displacement,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::StructuralNodes,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut request = spec("u", ObservationTarget::WholeModel);
    request.quantity = ObservationQuantity::Displacement;
    request.value_shape = ObservationValueShape::Vector3;
    baseline.sequence.steps[0].observations = vec![ObservationRequest::Field(request)];
    let baseline_layers = solver_layers(&baseline);
    let baseline_plan = resolve_observation_plan(
        &baseline.sequence.steps[0],
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let baseline_accuracy = AccuracyIdentity::new(
        baseline_layers.2,
        Digest::sha256(
            encode_contract(
                b"observation-identity-test/v1\0",
                "accuracy",
                &baseline.sequence.steps[0].accuracy,
                CanonicalLimits::new(1024 * 1024, 65_536, 65_536, 32),
            )
            .unwrap(),
        ),
    )
    .unwrap();

    let mut outputs_changed = baseline.clone();
    let changed_spec = outputs_changed.sequence.steps[0].observations[0]
        .spec_mut()
        .unwrap();
    changed_spec.storage.retention = RetentionPolicy::LatestSamples { count: 1 };
    changed_spec.purpose = ObservationPurpose::Visualization;
    let output_layers = solver_layers(&outputs_changed);
    assert_ne!(output_layers.0, baseline_layers.0);
    assert_eq!(output_layers.1, baseline_layers.1);
    assert_eq!(output_layers.2, baseline_layers.2);
    assert_eq!(output_layers.3, baseline_layers.3);
    assert_ne!(
        result_identity(output_layers.0, output_layers.2, output_layers.3),
        result_identity(baseline_layers.0, baseline_layers.2, baseline_layers.3)
    );
    let changed_plan = resolve_observation_plan(
        &outputs_changed.sequence.steps[0],
        &projection,
        &[definition],
        ObservationLimits::default(),
    )
    .unwrap();
    assert_ne!(changed_plan.identity, baseline_plan.identity);

    let mut accuracy_changed = baseline.clone();
    accuracy_changed.sequence.steps[0]
        .accuracy
        .resources
        .maximum_samples += 1;
    let accuracy_layers = solver_layers(&accuracy_changed);
    assert_ne!(accuracy_layers.0, baseline_layers.0);
    assert_eq!(accuracy_layers.1, baseline_layers.1);
    assert_eq!(accuracy_layers.2, baseline_layers.2);
    assert_eq!(accuracy_layers.3, baseline_layers.3);
    let changed_accuracy = AccuracyIdentity::new(
        baseline_layers.2,
        Digest::sha256(
            encode_contract(
                b"observation-identity-test/v1\0",
                "accuracy",
                &accuracy_changed.sequence.steps[0].accuracy,
                CanonicalLimits::new(1024 * 1024, 65_536, 65_536, 32),
            )
            .unwrap(),
        ),
    )
    .unwrap();
    assert_ne!(changed_accuracy, baseline_accuracy);

    let operator = baseline_layers.3;
    assert_eq!(
        NumericalPlanIdentity::new(operator, Digest::sha256("solver")).unwrap(),
        NumericalPlanIdentity::new(operator, Digest::sha256("solver")).unwrap()
    );
}

#[test]
fn solver_semantic_changes_invalidate_solver_layers() {
    let baseline = fixture_model(FixtureId::CantileverLinearStatic);
    let baseline_layers = solver_layers(&baseline);

    let mut procedure_changed = baseline.clone();
    procedure_changed.sequence.steps[0].procedure =
        runmat_analysis_core::AnalysisProcedure::ModalExtraction;
    assert_ne!(solver_layers(&procedure_changed), baseline_layers);

    let mut excitation_changed = baseline.clone();
    excitation_changed.excitations[0]
        .excitation_mut()
        .active_step_ids
        .push("different".into());
    assert_ne!(solver_layers(&excitation_changed), baseline_layers);

    let mut state_changed = baseline;
    state_changed.sequence.steps[0].restart =
        runmat_analysis_core::RestartPolicy::ResumeCheckpoint {
            checkpoint_id: "checkpoint".into(),
            required_state_digest: "state".into(),
        };
    assert_ne!(solver_layers(&state_changed), baseline_layers);
}

#[test]
fn direct_deserialization_rejects_semantically_invalid_resolved_plans() {
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let step = thermal_step(vec![ObservationRequest::Field(spec(
        "temperature",
        ObservationTarget::WholeModel,
    ))]);
    let mut plan = resolve_observation_plan(
        &step,
        &projection,
        &[definition],
        ObservationLimits::default(),
    )
    .unwrap();
    let ResolvedObservationTarget::Nodes {
        indices,
        identities,
    } = &mut plan.observations[0].target
    else {
        panic!("whole-model nodal field must resolve to nodes");
    };
    indices[1] = indices[0];
    identities[1] = identities[0];
    plan.identity = plan.derive_identity().unwrap();

    let encoded = serde_json::to_value(&plan).unwrap();
    let error = serde_json::from_value::<ResolvedObservationPlan>(encoded).unwrap_err();
    assert!(error.to_string().contains("resolved node target"));
}

#[test]
fn resolved_path_identity_rejects_interpolation_weight_substitution() {
    let projection = projection(ElementOrder::Tet10);
    let definition = scalar_definition();
    let mut path = spec(
        "path",
        ObservationTarget::Path {
            positions: vec![
                Position::new([0.1, 0.1, 0.1]),
                Position::new([0.2, 0.3, 0.1]),
            ],
        },
    );
    path.output_location = FieldLocation::Point;
    path.recovery = RecoveryPolicy::ShapeFunctionInterpolation;
    let mut plan = resolve_observation_plan(
        &thermal_step(vec![ObservationRequest::Field(path)]),
        &projection,
        &[definition],
        ObservationLimits::default(),
    )
    .unwrap();
    let ResolvedObservationTarget::Points { samples } = &mut plan.observations[0].target else {
        panic!("path must resolve to finite-element point samples")
    };
    samples[0].shape_values.swap(0, 1);
    plan.identity = plan.derive_identity().unwrap();

    let encoded = serde_json::to_value(&plan).unwrap();
    let error = serde_json::from_value::<ResolvedObservationPlan>(encoded).unwrap_err();
    assert!(error.to_string().contains("resolved point samples"));
}

#[test]
fn direct_deserialization_rejects_semantically_invalid_results() {
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let step = thermal_step(vec![ObservationRequest::Field(spec(
        "temperature",
        ObservationTarget::WholeModel,
    ))]);
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let fields = catalog(&projection, &definition, vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let mut result = evaluate_observations(&plan, &projection, &fields)
        .unwrap()
        .remove(0);
    result.entities[1] = result.entities[0].clone();
    result.logical_identity = result.derive_identity().unwrap();

    let encoded = serde_json::to_value(&result).unwrap();
    let error = serde_json::from_value::<ObservationResult>(encoded).unwrap_err();
    assert!(error.to_string().contains("metadata, values, or identity"));
}

#[test]
fn evaluation_rejects_plan_targets_rebound_to_another_topology_entity() {
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let step = thermal_step(vec![ObservationRequest::Field(spec(
        "temperature",
        ObservationTarget::WholeModel,
    ))]);
    let mut plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let ResolvedObservationTarget::Nodes { identities, .. } = &mut plan.observations[0].target
    else {
        panic!("whole-model nodal field must resolve to nodes");
    };
    identities.swap(0, 1);
    plan.identity = plan.derive_identity().unwrap();
    let fields = catalog(&projection, &definition, vec![4], vec![1.0, 2.0, 3.0, 4.0]);

    let error = evaluate_observations(&plan, &projection, &fields).unwrap_err();
    assert!(error.to_string().contains("differs from the topology"));
}
