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
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    },
    topology::AnalysisProjection,
};

use super::*;

#[path = "tests/final_frequency.rs"]
mod final_frequency;

pub(super) fn scalar_definition() -> RecoveredFieldDefinition {
    RecoveredFieldDefinition::new(
        "temperature.recovered",
        ObservationQuantity::Temperature,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap()
}

#[test]
fn final_observation_selects_only_the_authored_terminal_time_without_relabeling() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let mut step = ProcedureStep::thermal_transient("terminal-field");
    step.independent_domain = runmat_analysis_core::IndependentDomain::TimeSeconds {
        origin: 2.0,
        start: 2.0,
        end: 8.0,
        interval_count: 3,
    };
    step.observations = vec![ObservationRequest::Field(spec(
        "terminal",
        ObservationTarget::WholeModel,
    ))];
    let definition = scalar_definition();
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let observation = &plan.observations[0];
    for time in [2.0, 4.0, 6.0, 8.0, 10.0] {
        assert_eq!(
            resolved_observation_accepts_sample(
                &plan,
                observation,
                RecoveredFieldSample::TimeSeconds(time),
                None
            )
            .unwrap(),
            time == 8.0
        );
    }
    assert!(!resolved_observation_accepts_sample(
        &plan,
        observation,
        RecoveredFieldSample::FrequencyHertz(8.0),
        None
    )
    .unwrap());
    assert!(resolved_observation_accepts_sample(
        &plan,
        observation,
        RecoveredFieldSample::Final,
        None
    )
    .unwrap());
    let fields = |time| {
        catalog_at_sample(
            &projection,
            &definition,
            RecoveredFieldSample::TimeSeconds(time),
            vec![projection.nodes.len()],
            vec![293.15; projection.nodes.len()],
        )
    };
    let final_fields = fields(8.0);
    let results = evaluate_observations(&plan, &projection, &final_fields).unwrap();
    assert_eq!(results[0].sample, RecoveredFieldSample::TimeSeconds(8.0));
    assert!(evaluate_observations(&plan, &projection, &fields(6.0)).is_err());
    let ambiguous = RecoveredFieldCatalog::new(
        &projection,
        [
            RecoveredFieldSample::Final,
            RecoveredFieldSample::TimeSeconds(8.0),
        ]
        .into_iter()
        .map(|sample| {
            RecoveredField::new(
                definition.clone(),
                sample,
                runmat_analysis_core::AnalysisField::host_f64(
                    definition.field_id(), definition.layout(),
                    vec![projection.nodes.len()],
                    vec![293.15; projection.nodes.len()],
                ),
            )
            .unwrap()
        })
        .collect(),
    )
    .unwrap();
    assert!(evaluate_observations(&plan, &projection, &ambiguous).is_err());
}

fn complex_pressure_definition() -> RecoveredFieldDefinition {
    RecoveredFieldDefinition::new(
        "acoustic.pressure.recovered",
        ObservationQuantity::AcousticPressure,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
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
        runmat_analysis_core::AnalysisField::host_f64(definition.field_id(), definition.layout(), shape, values);
    RecoveredFieldCatalog::new(
        projection,
        vec![RecoveredField::new(definition.clone(), sample, storage).unwrap()],
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

pub(super) fn thermal_step(requests: Vec<ObservationRequest>) -> ProcedureStep {
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

pub(super) fn projection(order: ElementOrder) -> AnalysisProjection {
    AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(order)).unwrap()
}

#[test]
fn constraint_reaction_resultant_uses_canonical_group_ownership() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for face in &mut mesh.topology.boundary_faces {
        face.provenance.push(PersistentEntityId::new(
            PersistentEntityKind::Face,
            "root".into(),
            vec!["root".into()],
        ));
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
        force_definition.clone(),
        RecoveredFieldSample::Final,
        runmat_analysis_core::AnalysisField::host_f64(
            "structural.constraint_reaction_force", force_definition.layout(),
            vec![1, 3],
            vec![125.0, -250.0, 375.0],
        ),
    )
    .unwrap();
    let moment_field = RecoveredField::new(
        moment_definition.clone(),
        RecoveredFieldSample::Final,
        runmat_analysis_core::AnalysisField::host_f64(
            "structural.constraint_reaction_moment_about_global_origin", moment_definition.layout(),
            vec![1, 3],
            vec![12.5, -25.0, 37.5],
        ),
    )
    .unwrap();
    let catalog = RecoveredFieldCatalog::new_for_topology(
        plan.topology_digest,
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: mesh.topology.nodes.len(),
            volume_elements: mesh.topology.volume_elements.len(),
            volume_integration_points: mesh.topology.volume_elements.len() * 4,
            oriented_edges: 0,
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
    model
        .connectors
        .push(runmat_analysis_core::AnalysisConnector {
            connector_id: "mount".into(),
            kind: runmat_analysis_core::ConnectorKind::IsotropicGroundSpring {
                target: runmat_analysis_core::RegionTarget::point("node:1"),
                stiffness: runmat_analysis_core::Stiffness::new(1_000.0),
            },
        });
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for face in &mut mesh.topology.boundary_faces {
        face.provenance.push(PersistentEntityId::new(
            PersistentEntityKind::Face,
            "root".into(),
            vec!["root".into()],
        ));
        face.provenance.sort();
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let constraint_plan = crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
        &model,
        Some(&mesh),
    )
    .unwrap();
    let connector_plan =
        crate::assembly::explicit_connectors::compile_model_connector_plan(&model, Some(&mesh), 0)
            .unwrap();
    let source_digest = connector_plan.entries[0].source_digest();
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
        definition.clone(),
        RecoveredFieldSample::Final,
        runmat_analysis_core::AnalysisField::host_f64(
            crate::contracts::FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE, definition.layout(),
            vec![1, 3],
            vec![10.0, -20.0, 30.0],
        ),
    )
    .unwrap();
    let catalog = RecoveredFieldCatalog::new_for_topology(
        plan.topology_digest,
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: mesh.topology.nodes.len(),
            volume_elements: mesh.topology.volume_elements.len(),
            volume_integration_points: mesh.topology.volume_elements.len() * 4,
            oriented_edges: 0,
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

    model.sequence.steps[0].observations[0]
        .spec_mut()
        .unwrap()
        .target = ObservationTarget::Connector {
        connector_id: "mount".into(),
    };
    let mut substituted = connector_plan.clone();
    if let crate::connectors::CanonicalConnectorEntry::IsotropicGroundSpring {
        source_digest, ..
    } = &mut substituted.entries[0]
    {
        *source_digest = Digest::sha256("substituted-connector-source");
    } else {
        panic!("fixture must compile a ground spring");
    }
    assert!(resolve_structural_observation_plan_with_connectors(
        &model.sequence.steps[0],
        &model,
        Some(&mesh),
        &constraint_plan,
        &substituted,
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
            position: Position::new(projection.nodes[0].coordinates_m.canonical_value()),
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
        RecoveredFieldEntityDomain::Nodes,
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
            position: Position::new(node.coordinates_m.canonical_value()),
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
        RecoveredFieldEntityDomain::VolumeIntegrationPoints,
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
        RecoveredFieldEntityDomain::VolumeIntegrationPoints,
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
    assert_eq!(*domain, RecoveredFieldEntityDomain::VolumeIntegrationPoints);
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
        RecoveredFieldEntityDomain::VolumeElements,
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
        RecoveredFieldEntityDomain::VolumeElements,
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
                let [x, y, z] = node.coordinates_m.canonical_value();
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
        .map(|node| node.coordinates_m.canonical_value().iter().sum())
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
            runmat_analysis_core::RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&(face_selector)),
            ),
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

    let mut stale_plan = plan.clone().into_contents();
    stale_plan.identity = ResolvedObservationPlanIdentity(Digest::sha256("stale-plan"));
    assert!(ResolvedObservationPlan::try_from(stale_plan.clone()).is_err());
    stale_plan.step_id = runmat_analysis_core::ProcedureStepId("another-step".into());
    stale_plan.identity = stale_plan.derive_identity().unwrap();
    let stale_plan = ResolvedObservationPlan::try_from(stale_plan).unwrap();
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
    result.validate_source(&plan, &history).unwrap();

    let mut substituted = result.clone();
    substituted.evidence.contributions[0].values[0] += 1.0;
    assert!(substituted.validate_for(&plan, &history).is_err());
    assert!(substituted.validate_source(&plan, &history).is_err());
}

#[test]
fn series_peak_of_spatial_maxima_preserves_the_changing_governing_node() {
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let mut request = spec("moving-peak", ObservationTarget::WholeModel);
    request.sampling = SamplingPolicy::TimeSeconds(vec![0.0, 1.0, 2.0]);
    request.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Maximum,
        series: SeriesReduction::Maximum,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    request.storage.retention = RetentionPolicy::Canonical;
    let mut step = ProcedureStep::thermal_transient("moving-peak-step");
    step.independent_domain = runmat_analysis_core::IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 2.0,
        interval_count: 2,
    };
    step.observations = vec![ObservationRequest::History(request)];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let observation_id = ObservationId("moving-peak".into());
    let mut builder = BoundedObservationHistory::new(&plan, &observation_id).unwrap();
    for (time, values) in [
        (0.0, vec![1.0, 0.0, 0.0, 0.0]),
        (1.0, vec![0.0, 3.0, 0.0, 0.0]),
        (2.0, vec![0.0, 0.0, 2.0, 0.0]),
    ] {
        let sample = RecoveredFieldSample::TimeSeconds(time);
        let fields = catalog_at_sample(&projection, &definition, sample, vec![4], values);
        let result =
            evaluate_observation_sample(&plan, &observation_id, &projection, &fields, sample)
                .unwrap();
        builder.push(sample, result).unwrap();
    }
    let history = builder.finish().unwrap();
    assert_ne!(
        history.samples()[0].result.entities,
        history.samples()[1].result.entities
    );
    let result = reduce_observation_history(
        &plan,
        &history,
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(2).unwrap()),
    )
    .unwrap();
    let SeriesReductionPayload::Values { field, extrema } = &result.payload else {
        panic!("expected scalar peak")
    };
    assert_eq!(field.as_host_f64().unwrap(), &[3.0]);
    assert_eq!(extrema.len(), 1);
    assert_eq!(
        extrema[0].source.coordinate,
        RecoveredFieldSample::TimeSeconds(1.0)
    );
    assert_eq!(
        extrema[0].entity.as_ref(),
        history.samples()[1].result.entities.first()
    );
    result.validate_source(&plan, &history).unwrap();
    assert_eq!(
        SeriesReductionResult::from_canonical_bytes(&result.canonical_bytes().unwrap()).unwrap(),
        result
    );
}

#[test]
fn series_source_validation_does_not_recompute_payload() {
    let (plan, history) = completed_scalar_series(
        SeriesReduction::Mean {
            weighting: SeriesWeighting::UniformSamples,
        },
        ExtremaTiePolicy::FirstCanonical,
        RetentionPolicy::Canonical,
        &[0.0, 1.0, 2.0],
        &[1.0, 2.0, 3.0],
    )
    .unwrap();
    let mut result =
        reduce_observation_history(&plan, &history, ObservationExecutionPolicy::serial()).unwrap();
    let SeriesReductionPayload::Values { field, .. } = &mut result.payload else {
        unreachable!()
    };
    *field = runmat_analysis_core::AnalysisField::host_f64(
        field.field_id.clone(), field.layout.clone(),
        field.shape.clone(),
        vec![99.0],
    );
    let bytes = encode_contract(
        b"runmat-fea-observation/v1\0",
        "series-reduction-result-identity",
        &(
            result.schema_version,
            result.source_history,
            &result.observation_id,
            result.operation,
            result.extrema_ties,
            &result.payload,
        ),
        CanonicalLimits::new(64 * 1024 * 1024, 4_000_000, 65_536, 64),
    )
    .unwrap();
    result.identity =
        serde_json::from_value(serde_json::to_value(Digest::sha256(bytes)).unwrap()).unwrap();
    result.validate_source(&plan, &history).unwrap();
    assert!(result.validate_for(&plan, &history).is_err());
    let (_, foreign_history) = completed_scalar_series(
        SeriesReduction::Mean {
            weighting: SeriesWeighting::UniformSamples,
        },
        ExtremaTiePolicy::FirstCanonical,
        RetentionPolicy::Canonical,
        &[0.0, 1.0, 2.0],
        &[1.0, 2.0, 4.0],
    )
    .unwrap();
    assert!(result.validate_source(&plan, &foreign_history).is_err());
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
        RecoveredFieldEntityDomain::Nodes,
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
    let accuracy =
        crate::results::accuracy::evaluate_accuracy(&AccuracyPlan::baseline("event-accuracy"), &[])
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
        RecoveredFieldEntityDomain::Nodes,
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
        crate::execution::FieldTopologyIdentity::Structural { topology },
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
        vec![source_field.clone()],
    )
    .unwrap();
    let selection = ObservationFieldSelection::structural(
        &recovery,
        &artifact.manifest().unwrap(),
        vec![source_field.identity()],
    )
    .unwrap();
    let source =
        crate::execution::ObservationEvaluationRequest::new(selection, plan.clone()).unwrap();
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
        RecoveredFieldEntityDomain::Nodes,
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
        RecoveredFieldEntityDomain::Nodes,
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
        RecoveredFieldEntityDomain::Nodes,
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

// witness: FEA-SHARED-OBSERVATIONS-001/bounded_partitioned_evaluation @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Invalid and unbounded requests fail before field evaluation.
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
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut request = spec("explicit-displacement", ObservationTarget::WholeModel);
    request.quantity = ObservationQuantity::Displacement;
    request.value_shape = ObservationValueShape::Vector3;
    model.sequence.steps[0].observations = vec![ObservationRequest::Field(request)];
    // The direct evaluator uses serial partition ownership; retain that exact policy.
    let workload = crate::execution::StructuralLinearWorkload {
        schema_version: 2,
        step_id: model.sequence.steps[0].step_id.clone(),
        model: model.clone(),
        backend: crate::ComputeBackend::Cpu,
        preconditioner: crate::solve::preconditioner::SpdPreconditionerKind::Jacobi,
        algebra_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        observation_execution: ObservationExecutionPolicy::serial(),
        require_solver_mesh_for_solid: false,
        state_input: None,
    };
    workload.validate().unwrap();
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
        definition.clone(),
        RecoveredFieldSample::Final,
        runmat_analysis_core::AnalysisField::host_f64(
            "structural.displacement", definition.layout(),
            vec![node_count, 3],
            values,
        ),
    )
    .unwrap();
    let fields = RecoveredFieldCatalog::new_for_topology(
        plan.topology_digest,
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: node_count,
            volume_elements: 0,
            volume_integration_points: 0,
            oriented_edges: 0,
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
    // The publication authority rederives the same explicit topology and authored targets.
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, step).unwrap();
    let numerical_source = FieldRecoverySource::SolverResult(
        SolverResultIdentity::from_step_result(
            producer,
            Digest::sha256("structural sample publication"),
        )
        .unwrap(),
    );
    let recovery_request = FieldRecoveryRequest::new(
        producer,
        numerical_source,
        crate::execution::FieldTopologyIdentity::Structural {
            topology: constraint_plan.topology,
        },
        crate::execution::FieldRecoveryPlan::default(),
    )
    .unwrap();
    let recovered = RecoveredFieldArtifact::new(
        recovery_request.identity(),
        numerical_source,
        plan.topology_digest,
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: node_count,
            volume_elements: 0,
            volume_integration_points: 0,
            oriented_edges: 0,
            beam_elements: beam_count,
            shell_elements: shell_count,
            connectors: 0,
            constraint_groups: 0,
        },
        fields.fields().to_vec(),
    )
    .unwrap();
    let recovered_manifest = recovered.manifest().unwrap();
    let selection = ObservationFieldSelection::structural(
        &recovery_request,
        &recovered_manifest,
        vec![fields.fields()[0].identity()],
    )
    .unwrap();
    let source = ObservationFieldCatalog::structural(recovery_request, recovered_manifest).unwrap();
    source
        .validate_structural_plan(&plan, &model, None)
        .unwrap();
    let evaluation =
        crate::execution::ObservationEvaluationRequest::new(selection, plan.clone()).unwrap();
    let artifact = ObservationArtifact::new(ObservationArtifactInput {
        evaluation_request: evaluation.identity(),
        producer,
        source_selection: evaluation.selection.identity(),
        source: evaluation.selection.source(),
        topology_digest: plan.topology_digest,
        plan_identity: plan.identity,
        results,
    })
    .unwrap();
    let sample = ObservationSamplePublication::new(evaluation, &artifact).unwrap();
    let publication = ObservationPublicationCatalog::new(
        producer,
        plan.clone(),
        source.clone(),
        step,
        vec![ObservationPublicationEntry::Sample(Box::new(sample))],
        crate::execution::ObservationHistorySourceContext::Structural {
            source: crate::execution::StructuralObservationSource::Linear(&workload),
            mesh: None,
        },
    )
    .unwrap();
    assert_eq!(
        ObservationPublicationCatalog::from_canonical_bytes(
            &publication.canonical_bytes().unwrap()
        )
        .unwrap(),
        publication
    );
    let mut foreign_policy = workload.clone();
    foreign_policy.observation_execution =
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(2).unwrap());
    foreign_policy.validate().unwrap();
    assert!(publication
        .validate_for(
            step,
            crate::execution::ObservationHistorySourceContext::Structural {
                source: crate::execution::StructuralObservationSource::Linear(&foreign_policy),
                mesh: None,
            },
        )
        .unwrap_err()
        .to_string()
        .contains("execution policy differs"));
    let mut forged = plan.into_contents();
    let ResolvedObservationTarget::Nodes { indices, .. } = &mut forged.observations[0].target
    else {
        unreachable!()
    };
    for index in indices {
        *index += 1;
    }
    forged.identity = forged.derive_identity().unwrap();
    let forged = ResolvedObservationPlan::try_from(forged).unwrap();
    assert!(source
        .validate_structural_plan(&forged, &model, None)
        .is_err());
    let mut foreign_model = model.clone();
    foreign_model.sequence.steps[0]
        .step_id
        .0
        .push_str("-foreign");
    assert!(source
        .validate_structural_plan(publication.plan(), &foreign_model, None)
        .is_err());
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

// witness: FEA-SHARED-IDENTITY-001/distinct_identity_layers @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Output and accuracy changes do not invalidate solver identity layers.
#[test]
fn output_and_accuracy_changes_do_not_invalidate_solver_layers() {
    let mut baseline = fixture_model(FixtureId::CantileverLinearStatic);
    let projection = projection(ElementOrder::Tet4);
    let definition = RecoveredFieldDefinition::new(
        "displacement.recovered",
        ObservationQuantity::Displacement,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
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
    .unwrap()
    .into_contents();
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
    .unwrap()
    .into_contents();
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
    .unwrap()
    .into_contents();
    let ResolvedObservationTarget::Nodes { identities, .. } = &mut plan.observations[0].target
    else {
        panic!("whole-model nodal field must resolve to nodes");
    };
    identities.swap(0, 1);
    plan.identity = plan.derive_identity().unwrap();
    let plan = ResolvedObservationPlan::try_from(plan).unwrap();
    let fields = catalog(&projection, &definition, vec![4], vec![1.0, 2.0, 3.0, 4.0]);

    let error = evaluate_observations(&plan, &projection, &fields).unwrap_err();
    assert!(error.to_string().contains("differs from the topology"));
}

// witness: FEA-SHARED-OBSERVATIONS-001/governed_observation_corpus @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: A point sample of an element field returns the native value of the element that
//   contains the point (not a centroid match), with that element as its entity, on a
//   three-element Tet4 mesh. Tet10 and path samples of element fields are not exercised.
#[test]
fn element_field_point_sample_reads_the_containing_element() {
    let mesh = runmat_meshing_core::fixtures::three_region_tetrahedron_solver_mesh();
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    assert_eq!(projection.elements.len(), 3);
    let definition = RecoveredFieldDefinition::new(
        "stress.recovered",
        ObservationQuantity::Stress,
        FieldLocation::Element,
        RecoveredFieldEntityDomain::VolumeElements,
        ObservationValueShape::SymmetricTensor3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let fields = catalog(
        &projection,
        &definition,
        vec![3, 6],
        vec![
            100.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            200.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            300.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    );
    for (index, element) in projection.elements.iter().enumerate() {
        // A point strictly inside the element, away from its centroid.
        let corner = projection.nodes[element.node_indices[0]].coordinates_m;
        let position = std::array::from_fn(|axis| {
            0.7 * element.centroid_m[axis] + 0.3 * corner.canonical_value()[axis]
        });
        let mut request = spec(
            "sample",
            ObservationTarget::Point {
                position: Position::new(position),
            },
        );
        request.quantity = ObservationQuantity::Stress;
        request.value_shape = ObservationValueShape::SymmetricTensor3;
        request.source_location = FieldLocation::Element;
        request.output_location = FieldLocation::Element;
        request.component = ObservationComponent::VonMises;
        request.cost = runmat_analysis_core::ObservationCostPolicy::Automatic;
        let mut step = ProcedureStep::structural_static("static");
        step.observations = vec![ObservationRequest::Probe(request)];
        let plan = resolve_observation_plan(
            &step,
            &projection,
            std::slice::from_ref(&definition),
            ObservationLimits::default(),
        )
        .unwrap();
        let result = evaluate_observations(&plan, &projection, &fields).unwrap();
        assert_eq!(
            result[0].field.as_host_f64().unwrap(),
            &[100.0 * (index + 1) as f64]
        );
        assert_eq!(
            result[0].entities,
            vec![ObservationEntity::Element(element.stable_identity)]
        );
    }
}

// witness: FEA-SHARED-OBSERVATIONS-001/governed_observation_corpus @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: an element field over a boundary-face region reads each face's owning element;
//   its area-weighted average and its maximum match the owners' values on a three-element Tet4
//   mesh. Tet10 faces and faces shared by two elements (interfaces) are not exercised.
#[test]
fn element_field_over_a_boundary_region_reads_the_face_owners() {
    use runmat_analysis_core::{
        EntityDimension, ExcitationTarget, ExpectedTopology, RegionTarget, TargetCardinality,
    };
    let mesh = runmat_meshing_core::fixtures::three_region_tetrahedron_solver_mesh();
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let definition = RecoveredFieldDefinition::new(
        "stress.recovered",
        ObservationQuantity::Stress,
        FieldLocation::Element,
        RecoveredFieldEntityDomain::VolumeElements,
        ObservationValueShape::SymmetricTensor3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let element_values = [100.0, 200.0, 300.0];
    let fields = catalog(
        &projection,
        &definition,
        vec![3, 6],
        element_values
            .iter()
            .flat_map(|value| [*value, 0.0, 0.0, 0.0, 0.0, 0.0])
            .collect(),
    );
    let entity = projection.boundary_faces[0].provenance[0].clone();
    let mut region = RegionTarget::new(
        "faces",
        EntityDimension::Face,
        ExpectedTopology::BoundaryFace,
        TargetCardinality::OneOrMore,
    );
    region.bind_resolved_entities(vec![entity.clone()]).unwrap();
    // Expected trace of the element field: owner values weighted by face area.
    let (mut weighted, mut area, mut peak) = (0.0, 0.0, f64::NEG_INFINITY);
    let mut owners = std::collections::BTreeSet::new();
    for face in projection
        .boundary_faces
        .iter()
        .filter(|face| face.provenance.contains(&entity))
    {
        let [a, b, c] = [0, 1, 2].map(|k| projection.nodes[face.node_indices[k]].coordinates_m);
        let (u, v) = (
            [
                b.canonical_value()[0] - a.canonical_value()[0],
                b.canonical_value()[1] - a.canonical_value()[1],
                b.canonical_value()[2] - a.canonical_value()[2],
            ],
            [
                c.canonical_value()[0] - a.canonical_value()[0],
                c.canonical_value()[1] - a.canonical_value()[1],
                c.canonical_value()[2] - a.canonical_value()[2],
            ],
        );
        let cross = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let face_area = 0.5 * cross.iter().map(|x| x * x).sum::<f64>().sqrt();
        let owner = face.adjacent_element_indices[0];
        owners.insert(owner);
        weighted += element_values[owner] * face_area;
        area += face_area;
        peak = peak.max(element_values[owner]);
    }
    assert!(owners.len() >= 1 && area > 0.0);
    for (id, operation, output_location, expected) in [
        (
            "mean",
            SpatialReduction::Average,
            FieldLocation::Global,
            weighted / area,
        ),
        (
            "peak",
            SpatialReduction::Maximum,
            FieldLocation::Element,
            peak,
        ),
    ] {
        let mut request = spec(
            id,
            ObservationTarget::Spatial(ExcitationTarget::Region(region.clone())),
        );
        request.quantity = ObservationQuantity::Stress;
        request.value_shape = ObservationValueShape::SymmetricTensor3;
        request.source_location = FieldLocation::Element;
        request.output_location = output_location;
        request.recovery = RecoveryPolicy::Native;
        request.component = ObservationComponent::VonMises;
        request.cost = runmat_analysis_core::ObservationCostPolicy::Automatic;
        request.reduction = Some(ReductionPolicy {
            spatial: operation,
            series: SeriesReduction::None,
            extrema_ties: ExtremaTiePolicy::FirstCanonical,
        });
        let mut step = ProcedureStep::structural_static("static");
        step.observations = vec![ObservationRequest::Reduction(request)];
        let plan = resolve_observation_plan(
            &step,
            &projection,
            std::slice::from_ref(&definition),
            ObservationLimits::default(),
        )
        .unwrap();
        let result = evaluate_observations(&plan, &projection, &fields).unwrap();
        let value = result[0].field.as_host_f64().unwrap()[0];
        assert!(
            (value - expected).abs() <= 1e-12 * expected,
            "{id}: {value} vs {expected}"
        );
    }
}

// witness: FEA-SHARED-OBSERVATIONS-001/governed_observation_corpus @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: a nodal von Mises field, which the solver provides only as the volume-weighted
//   transfer of element values, binds by its node location and is sampled at a point by shape
//   function interpolation (exact for a nodal field linear in x) on a three-element Tet4 mesh.
//   The transfer itself is witnessed by the recovery tests, not here.
#[test]
fn node_located_von_mises_binds_its_declared_transfer_and_interpolates() {
    let mesh = runmat_meshing_core::fixtures::three_region_tetrahedron_solver_mesh();
    let projection = AnalysisProjection::derive(&mesh).unwrap();
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
    let values = projection
        .nodes
        .iter()
        .map(|node| 1.0e6 * node.coordinates_m.canonical_value()[0] + 3.0e6)
        .collect::<Vec<_>>();
    let fields = catalog(&projection, &definition, vec![values.len()], values);
    let element = &projection.elements[1];
    let corner = projection.nodes[element.node_indices[0]].coordinates_m;
    let position = std::array::from_fn(|axis| {
        0.6 * element.centroid_m[axis] + 0.4 * corner.canonical_value()[axis]
    });
    let mut request = spec(
        "surface_sample",
        ObservationTarget::Point {
            position: Position::new(position),
        },
    );
    request.quantity = ObservationQuantity::VonMisesStress;
    request.value_shape = ObservationValueShape::Scalar;
    request.source_location = FieldLocation::Node;
    request.output_location = FieldLocation::Point;
    request.recovery = RecoveryPolicy::ShapeFunctionInterpolation;
    request.cost = runmat_analysis_core::ObservationCostPolicy::Automatic;
    let mut step = ProcedureStep::structural_static("static");
    step.observations = vec![ObservationRequest::Probe(request)];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let result = evaluate_observations(&plan, &projection, &fields).unwrap();
    let value = result[0].field.as_host_f64().unwrap()[0];
    let expected = 1.0e6 * position[0] + 3.0e6;
    assert!(
        (value - expected).abs() <= 1e-9 * expected,
        "{value} vs {expected}"
    );
}

// witness: FEA-SHARED-OBSERVATIONS-001/governed_observation_corpus @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: a reaction-force resultant over a boundary face region is the discrete sum of the
//   nodal reactions at the nodes of that region's faces, each node counted once, on a
//   three-element Tet4 mesh. Constraint-group targets are witnessed separately.
#[test]
fn reaction_resultant_over_a_face_region_sums_its_nodes_once() {
    use runmat_analysis_core::{
        EntityDimension, ExcitationTarget, ExpectedTopology, RegionTarget, TargetCardinality,
    };
    let mesh = runmat_meshing_core::fixtures::three_region_tetrahedron_solver_mesh();
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let definition = RecoveredFieldDefinition::new(
        "structural.reaction_force",
        ObservationQuantity::ReactionForce,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let values = (0..projection.nodes.len())
        .flat_map(|node| [node as f64 + 1.0, 2.0 * node as f64, -1.0])
        .collect::<Vec<_>>();
    let fields = catalog(
        &projection,
        &definition,
        vec![projection.nodes.len(), 3],
        values.clone(),
    );
    let entity = projection.boundary_faces[0].provenance[0].clone();
    let mut region = RegionTarget::new(
        "supports",
        EntityDimension::Face,
        ExpectedTopology::BoundaryFace,
        TargetCardinality::OneOrMore,
    );
    region.bind_resolved_entities(vec![entity.clone()]).unwrap();
    let nodes = projection
        .boundary_faces
        .iter()
        .filter(|face| face.provenance.contains(&entity))
        .flat_map(|face| face.node_indices.iter().copied())
        .collect::<std::collections::BTreeSet<_>>();
    let expected: [f64; 3] =
        std::array::from_fn(|axis| nodes.iter().map(|node| values[node * 3 + axis]).sum());
    let mut request = spec(
        "support_reaction",
        ObservationTarget::Spatial(ExcitationTarget::Region(region)),
    );
    request.quantity = ObservationQuantity::ReactionForce;
    request.value_shape = ObservationValueShape::Vector3;
    request.source_location = FieldLocation::Node;
    request.output_location = FieldLocation::Global;
    request.cost = runmat_analysis_core::ObservationCostPolicy::Automatic;
    request.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Resultant,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    let mut step = ProcedureStep::structural_static("static");
    step.observations = vec![ObservationRequest::Reduction(request)];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let result = evaluate_observations(&plan, &projection, &fields).unwrap();
    let value = result[0].field.as_host_f64().unwrap();
    for axis in 0..3 {
        assert!(
            (value[axis] - expected[axis]).abs() < 1e-12,
            "{value:?} vs {expected:?}"
        );
    }
}
