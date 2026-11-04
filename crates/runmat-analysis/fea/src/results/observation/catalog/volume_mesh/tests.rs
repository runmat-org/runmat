//! Family-neutral volume-mesh authority: a real nodal field of one family and a complex nodal
//! field of another resolve, select and evaluate through the same catalog.
use super::*;
use crate::execution::{
    FieldRecoveryCost, FieldRecoveryPlan, FieldRecoveryRequirement, FieldRecoverySource,
    FieldTopologyIdentity, SequenceIdentity, SolverResultIdentity, StepIdentity,
};
use crate::results::field::{
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldCatalog, RecoveredFieldDefinition,
    RecoveredFieldEntityCounts, RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation,
    RecoveredFieldSample,
};
use crate::results::observation::{
    evaluate_observations, resolve_observation_plan, ObservationLimits,
    ObservationSelectionCoordinate,
};
use runmat_analysis_core::{
    AnalysisField, AnalysisSequence, ComplexRepresentation, ExtremaTiePolicy, FieldLocation,
    IndependentDomain, ObservationQuantity, ObservationRequest, ObservationTarget,
    ObservationValueShape, ProcedureStep, RecoveryPolicy, ReductionPolicy, SeriesReduction,
    SignalPrecision, SpatialReduction,
};
use runmat_meshing_core::fixtures::slab_mesh;

struct Source {
    request: FieldRecoveryRequest,
    artifact: RecoveredFieldArtifact,
    sequence: SequenceIdentity,
}

fn source(
    step: &ProcedureStep,
    definition: &RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    projection: &AnalysisProjection,
    mesh: &SolverMeshArtifact,
    values: Vec<f64>,
    components: usize,
) -> Source {
    let sequence = SequenceIdentity::from_solver_contract(&AnalysisSequence::from_steps(
        "s",
        vec![step.clone()],
    ))
    .unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, step).unwrap();
    let nodes = projection.nodes.len();
    let count = nodes as u64;
    let plan = FieldRecoveryPlan {
        requirements: vec![FieldRecoveryRequirement {
            support: Default::default(),
            quantity: definition.quantity(),
            source_location: FieldLocation::Node,
            output_location: FieldLocation::Node,
            recovery: RecoveryPolicy::Native,
            value_shape: ObservationValueShape::Scalar,
            entity_domain: RecoveredFieldEntityDomain::Nodes,
            cost: FieldRecoveryCost {
                source_entity_count: count,
                output_entity_count: count,
                topology_incidence_count: 0,
                arithmetic_operation_count: count,
            },
        }],
    };
    let result =
        SolverResultIdentity::from_step_result(producer, Digest::sha256("fields")).unwrap();
    let request = FieldRecoveryRequest::new(
        producer,
        FieldRecoverySource::SolverResult(result),
        FieldTopologyIdentity::VolumeMesh {
            mesh_digest: mesh.canonical_digest,
        },
        plan,
    )
    .unwrap();
    let shape = if components == 1 {
        vec![nodes]
    } else {
        vec![nodes, components]
    };
    let field = RecoveredField::new(
        definition.clone(),
        sample,
        AnalysisField::host_f64(definition.field_id(), definition.layout(), shape, values),
    )
    .unwrap();
    let artifact = RecoveredFieldArtifact::new(
        request.identity(),
        request.source,
        request.topology.discretization_digest(),
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes,
            volume_elements: projection.elements.len(),
            volume_integration_points: 0,
            oriented_edges: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        },
        vec![field],
    )
    .unwrap();
    Source {
        request,
        artifact,
        sequence,
    }
}

fn maximum(id: &str, quantity: ObservationQuantity, complex: bool) -> ObservationRequest {
    maximum_as(
        id,
        quantity,
        complex.then_some(ComplexRepresentation::Magnitude),
    )
}

fn maximum_as(
    id: &str,
    quantity: ObservationQuantity,
    representation: Option<ComplexRepresentation>,
) -> ObservationRequest {
    let mut spec = crate::results::observation::tests::spec(id, ObservationTarget::WholeModel);
    spec.quantity = quantity;
    spec.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Maximum,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::FirstCanonical,
    });
    if let Some(representation) = representation {
        spec.complex_representation = representation;
        spec.storage.precision = SignalPrecision::Real64;
    }
    ObservationRequest::Reduction(spec)
}

fn authority_round_trip(
    step: &ProcedureStep,
    source: &Source,
    mesh: &SolverMeshArtifact,
    projection: &AnalysisProjection,
) -> f64 {
    let manifest = source.artifact.manifest().unwrap();
    let definitions = manifest
        .fields
        .iter()
        .map(|entry| entry.definition.clone())
        .collect::<Vec<_>>();
    let plan =
        resolve_observation_plan(step, projection, &definitions, ObservationLimits::default())
            .unwrap();
    let catalog = ObservationFieldCatalog::volume_mesh_recovery(
        source.request.clone(),
        manifest.clone(),
        source.sequence,
        step.clone(),
    )
    .unwrap();
    catalog.validate_volume_mesh_plan(&plan, mesh).unwrap();
    let selection = catalog
        .canonical_selection(&plan, ObservationSelectionCoordinate::VolumeMeshRecovery)
        .unwrap();
    catalog
        .validate_volume_mesh_selection(&selection, mesh)
        .unwrap();
    catalog
        .validate_volume_mesh_selection_metadata(&selection, plan.topology_digest)
        .unwrap();
    assert_eq!(
        selection.source().observation_topology_digest(),
        projection.projection_digest
    );
    // Cold catalog decoding preserves the producing step authority.
    let cold =
        ObservationFieldCatalog::from_canonical_bytes(&catalog.canonical_bytes().unwrap()).unwrap();
    assert_eq!(cold.volume_mesh_step().unwrap(), step);
    // A step that is not the request's producer cannot be bound to the fields.
    let mut foreign = step.clone();
    foreign.step_id = runmat_analysis_core::ProcedureStepId("other".into());
    assert!(ObservationFieldCatalog::volume_mesh_recovery(
        source.request.clone(),
        manifest,
        source.sequence,
        foreign,
    )
    .is_err());
    let fields = RecoveredFieldCatalog::new(projection, source.artifact.fields().to_vec()).unwrap();
    let results = evaluate_observations(&plan, projection, &fields).unwrap();
    results[0].field.as_host_f64().unwrap()[0]
}

#[test]
fn real_nodal_field_of_a_static_family_uses_the_neutral_volume_mesh_authority() {
    let mesh = slab_mesh(2);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let mut step = ProcedureStep::thermal_transient("heat");
    step.independent_domain = IndependentDomain::Static;
    step.observations = vec![maximum("peak", ObservationQuantity::Temperature, false)];
    let definition = RecoveredFieldDefinition::new(
        "temperature",
        ObservationQuantity::Temperature,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    // T = 300 + x: the maximum over the unit slab is 301 at x = 1.
    let values = projection
        .nodes
        .iter()
        .map(|node| 300.0 + node.coordinates_m.canonical_value()[0])
        .collect();
    let source = source(
        &step,
        &definition,
        RecoveredFieldSample::Final,
        &projection,
        &mesh,
        values,
        1,
    );
    let peak = authority_round_trip(&step, &source, &mesh, &projection);
    assert!((peak - 301.0).abs() < 1e-12, "{peak}");
}

#[test]
fn complex_acoustic_pressure_peak_magnitude_uses_the_same_authority_at_its_frequency() {
    let mesh = slab_mesh(2);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let mut step = ProcedureStep::acoustic_harmonic("sound");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![100.0]);
    step.observations = vec![maximum("peak", ObservationQuantity::AcousticPressure, true)];
    let definition = RecoveredFieldDefinition::new(
        "acoustic.pressure",
        ObservationQuantity::AcousticPressure,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::ComplexInterleaved,
    )
    .unwrap();
    // p = (3 + 4i) x: |p| peaks at 5 Pa on x = 1.
    let values = projection
        .nodes
        .iter()
        .flat_map(|node| {
            [
                3.0 * node.coordinates_m.canonical_value()[0],
                4.0 * node.coordinates_m.canonical_value()[0],
            ]
        })
        .collect();
    let source = source(
        &step,
        &definition,
        RecoveredFieldSample::FrequencyHertz(100.0),
        &projection,
        &mesh,
        values,
        2,
    );
    let peak = authority_round_trip(&step, &source, &mesh, &projection);
    assert!((peak - 5.0).abs() < 1e-12, "{peak}");
}

#[test]
fn sound_pressure_level_peak_is_the_rms_level_of_the_largest_pressure_phasor() {
    let mesh = slab_mesh(2);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let mut step = ProcedureStep::acoustic_harmonic("sound");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![100.0]);
    step.observations = vec![maximum_as(
        "peak_level",
        ObservationQuantity::AcousticPressure,
        Some(ComplexRepresentation::SoundPressureLevel),
    )];
    let definition = RecoveredFieldDefinition::new(
        "acoustic.pressure",
        ObservationQuantity::AcousticPressure,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::ComplexInterleaved,
    )
    .unwrap();
    // |p| = 5 (1 + x) Pa: the peak 10 Pa at x = 1 is 20 log10(10 / (sqrt(2) 20 uPa)) dB.
    let values = projection
        .nodes
        .iter()
        .flat_map(|node| {
            let scale = 1.0 + node.coordinates_m.canonical_value()[0];
            [3.0 * scale, 4.0 * scale]
        })
        .collect();
    let source = source(
        &step,
        &definition,
        RecoveredFieldSample::FrequencyHertz(100.0),
        &projection,
        &mesh,
        values,
        2,
    );
    let level = authority_round_trip(&step, &source, &mesh, &projection);
    let expected = 20.0 * (10.0 / (std::f64::consts::SQRT_2 * 20.0e-6)).log10();
    assert!((level - expected).abs() < 1e-10, "{level} vs {expected}");
}

// The reduced acoustic result path publishes nodal levels and the distributed observation path
// evaluates them from the complex pressure; both use the one RMS definition, with no clamp.
#[test]
fn reduced_nodal_and_distributed_sound_pressure_levels_agree() {
    let mesh = slab_mesh(2);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let mut step = ProcedureStep::acoustic_harmonic("sound");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![100.0]);
    step.observations = vec![maximum_as(
        "peak_level",
        ObservationQuantity::AcousticPressure,
        Some(ComplexRepresentation::SoundPressureLevel),
    )];
    let definition = RecoveredFieldDefinition::new(
        "acoustic.pressure",
        ObservationQuantity::AcousticPressure,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::ComplexInterleaved,
    )
    .unwrap();
    // Peaks above and below the 20 uPa reference: the second level is negative.
    for (peak_pa, expected_db) in [
        (10.0, 110.969_100_130_080_56),
        (1.0e-5, -9.030_899_869_919_436),
    ] {
        let phasors = projection
            .nodes
            .iter()
            .map(|node| {
                let scale = 0.1 * peak_pa * (1.0 + node.coordinates_m.canonical_value()[0]);
                [3.0 * scale, 4.0 * scale]
            })
            .collect::<Vec<_>>();
        let magnitudes = phasors
            .iter()
            .map(|[real, imaginary]| real.hypot(*imaginary))
            .collect::<Vec<_>>();
        let reduced = crate::physics::acoustic::nodal_sound_pressure_levels_db(&magnitudes)
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max);
        let source = source(
            &step,
            &definition,
            RecoveredFieldSample::FrequencyHertz(100.0),
            &projection,
            &mesh,
            phasors.into_iter().flatten().collect(),
            2,
        );
        let distributed = authority_round_trip(&step, &source, &mesh, &projection);
        assert!(
            (reduced - distributed).abs() < 1e-10,
            "reduced {reduced} dB vs distributed {distributed} dB at {peak_pa} Pa"
        );
        assert!(
            (distributed - expected_db).abs() < 1e-9,
            "{distributed} vs {expected_db}"
        );
    }
}
