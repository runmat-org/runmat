use runmat_analysis_core::{
    ComplexRepresentation, CompressionPolicy, CoordinateFrame, ExtremaTiePolicy, FieldLocation,
    ObservationComponent, ObservationCostEstimate, ObservationId, ObservationPurpose,
    ObservationQuantity, ObservationRequest, ObservationSpec, ObservationStoragePolicy,
    ObservationTarget, ObservationValueShape, PartialAccessPolicy, Position, ProcedureStep,
    RecoveryPolicy, ReductionPolicy, RetentionPolicy, SamplingPolicy, SeriesReduction,
    SignConvention, SignalPrecision, SpatialReduction,
};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

use crate::{
    execution::{
        AccuracyIdentity, AnalysisResultIdentity, AnalysisWorkKind, AnalysisWorkloadIdentity,
        ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity,
        NumericalPlanIdentity, OperatorIdentity, ProblemIdentity, SequenceIdentity, StepIdentity,
        StudyIdentity,
    },
    fixtures::{fixture_model, FixtureId},
    results::field::{
        RecoveredField, RecoveredFieldCatalog, RecoveredFieldDefinition,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    },
    topology::AnalysisProjection,
};

use super::*;

fn scalar_definition() -> RecoveredFieldDefinition {
    RecoveredFieldDefinition::new(
        "temperature.recovered",
        ObservationQuantity::Temperature,
        FieldLocation::Node,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap()
}

fn catalog(
    projection: &AnalysisProjection,
    definition: &RecoveredFieldDefinition,
    shape: Vec<usize>,
    values: Vec<f64>,
) -> RecoveredFieldCatalog {
    let storage =
        runmat_analysis_core::AnalysisField::host_f64(definition.field_id(), shape, values);
    RecoveredFieldCatalog::new(
        projection,
        vec![
            RecoveredField::new(definition.clone(), RecoveredFieldSample::Final, storage).unwrap(),
        ],
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

fn thermal_step(requests: Vec<ObservationRequest>) -> ProcedureStep {
    let mut step = ProcedureStep::thermal_transient("thermal");
    step.independent_domain = runmat_analysis_core::IndependentDomain::Static;
    step.observations = requests;
    step
}

fn projection(order: ElementOrder) -> AnalysisProjection {
    AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(order)).unwrap()
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
    .contains("sampled state series"));

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
