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
    planning::resource::{Count, ObservationResourceFacts},
    results::field::{
        RecoveredField, RecoveredFieldCatalog, RecoveredFieldDefinition,
        RecoveredFieldEntityCounts, RecoveredFieldEntityDomain,
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
    spec.cost = ObservationCostEstimate {
        operation_count: 1_000_000,
        peak_resident_bytes: 10_000_000,
        stored_bytes: 10_000_000,
        transfer_bytes: 10_000_000,
        collective_count: 0,
    };
    spec
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
fn bounded_history_retains_latest_samples_with_independent_identities() {
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let mut history = spec("temperature_history", ObservationTarget::WholeModel);
    history.sampling = SamplingPolicy::TimeSeconds(vec![0.0, 1.0, 2.0]);
    history.storage.retention = RetentionPolicy::LatestSamples { count: 2 };
    history.cost.operation_count = 1_000_000;
    history.cost.peak_resident_bytes = 10_000_000;
    history.cost.stored_bytes = 10_000_000;
    history.cost.transfer_bytes = 10_000_000;
    let mut step = ProcedureStep::thermal_transient("thermal-history");
    step.independent_domain = runmat_analysis_core::IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 2.0,
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
    request.cost = ObservationCostEstimate {
        operation_count: 1_000_000,
        peak_resident_bytes: 10_000_000,
        stored_bytes: 10_000_000,
        transfer_bytes: 10_000_000,
        collective_count: 0,
    };
    let mut step = ProcedureStep::structural_modal("parameter-study");
    step.procedure = runmat_analysis_core::AnalysisProcedure::ModalResponseSpectrum;
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
    .contains("cost estimate understates"));

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
    let step = &model.sequence.steps[0];
    let plan = resolve_structural_observation_plan(
        step,
        &model,
        None,
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
            beam_elements: beam_count,
            shell_elements: shell_count,
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
