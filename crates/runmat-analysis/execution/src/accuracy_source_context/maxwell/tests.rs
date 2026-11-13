//! Cold source and authored-plan authority, not an independent discretization witness.
use super::*;

use runmat_analysis_fea::{
    execution::ObservationEvaluationRequest,
    results::{
        field::RecoveredFieldCatalog,
        observation::{
            evaluate_sampled_observations, resolve_sampled_observation_plan, ObservationArtifact,
            ObservationArtifactInput, ObservationFieldCatalog, ObservationFieldSelection,
            ObservationLimits, ObservationPublicationEntry, ObservationSamplePublication,
            ObservationSelectionCoordinate,
        },
    },
};
fn observation() -> ObservationRequest {
    ObservationRequest::Field(ObservationSpec {
        observation_id: ObservationId("B".into()),
        quantity: ObservationQuantity::MagneticFluxDensity,
        value_shape: ObservationValueShape::Vector3,
        target: ObservationTarget::WholeModel,
        source_location: FieldLocation::IntegrationPoint,
        output_location: FieldLocation::IntegrationPoint,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Complex,
        sampling: SamplingPolicy::Final,
        reduction: None,
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Complex128,
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
    })
}

fn catalog(
    workload: &MaxwellHarmonicWorkload,
    mesh: &SolverMeshArtifact,
    result: &MaxwellHarmonicSolverResult,
) -> ObservationPublicationCatalog {
    let recovery = result.native_field_request(workload, mesh).unwrap();
    let fields = result
        .recover_native_fields(workload, mesh, &recovery)
        .unwrap();
    let manifest = fields.manifest().unwrap();
    let sampling = result.sampling_topology();
    let step = &workload.model.sequence.steps[0];
    let definitions = fields
        .fields()
        .iter()
        .map(|f| f.definition().clone())
        .collect::<Vec<_>>();
    let plan = resolve_sampled_observation_plan(
        step,
        sampling,
        mesh,
        &definitions,
        ObservationLimits::default(),
    )
    .unwrap();
    let source = ObservationFieldCatalog::sampled_recovery(
        recovery.clone(),
        manifest.clone(),
        sampling.clone(),
    )
    .unwrap();
    let selected = source
        .canonical_selected_fields(&plan, ObservationSelectionCoordinate::SampledRecovery)
        .unwrap();
    let selection =
        ObservationFieldSelection::sampled_recovery(&recovery, &manifest, sampling, selected)
            .unwrap();
    let evaluation = ObservationEvaluationRequest::new(selection, plan).unwrap();
    let values = RecoveredFieldCatalog::new_for_topology(
        sampling.identity().unwrap().discretization_digest(),
        sampling.entity_counts(),
        fields.into_fields(),
    )
    .unwrap();
    let results = evaluate_sampled_observations(&evaluation.plan, sampling, mesh, &values).unwrap();
    let artifact = ObservationArtifact::new(ObservationArtifactInput {
        evaluation_request: evaluation.identity(),
        producer: evaluation.producer,
        source_selection: evaluation.selection.identity(),
        source: evaluation.selection.source(),
        topology_digest: sampling.identity().unwrap().discretization_digest(),
        plan_identity: evaluation.plan.identity,
        results,
    })
    .unwrap();
    let publication = ObservationPublicationCatalog::new(
        result.producer_identity(),
        evaluation.plan.clone(),
        source,
        step,
        vec![ObservationPublicationEntry::Sample(Box::new(
            ObservationSamplePublication::new(evaluation, &artifact).unwrap(),
        ))],
        ObservationHistorySourceContext::MaxwellHarmonic { workload, mesh },
    )
    .unwrap();
    ObservationPublicationCatalog::from_canonical_bytes(&publication.canonical_bytes().unwrap())
        .unwrap()
}

#[test]
fn original_complex_catalog_and_missing_catalog_authenticate_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut w = workload();
        w.model.sequence.steps[0].observations = vec![observation()];
        let r = w.execute(&mesh).unwrap();
        let receipt = MaxwellNumericalReceiptArtifact::new(&w, &r, &mesh).unwrap();
        let publication = catalog(&w, &mesh, &r);
        let inputs = context(&w, &mesh, &r);
        let refs = inputs.iter().collect::<Vec<_>>();
        let request = AccuracyEvaluationRequest::new(
            receipt.receipt().producer,
            AccuracySourceKind::MaxwellHarmonic,
            receipt.source_result(),
            Some(&publication),
            w.model.sequence.steps[0].accuracy.clone(),
        )
        .unwrap();
        validate(&request, &receipt, Some(&publication), &refs).unwrap();
        assert!(validate(&request, &receipt, None, &refs).is_err());
        let no_catalog = AccuracyEvaluationRequest::new(
            receipt.receipt().producer,
            AccuracySourceKind::MaxwellHarmonic,
            receipt.source_result(),
            None,
            w.model.sequence.steps[0].accuracy.clone(),
        )
        .unwrap();
        assert!(validate(&no_catalog, &receipt, None, &refs).is_err());
    }
}
use crate::prepare_analysis_objects;
use runmat_analysis_core::*;
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ObjectNamespace};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn workload() -> MaxwellHarmonicWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.structural = None;
    model.constraints.clear();
    model.boundaries.clear();
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.initial_conditions.clear();
    model.electromagnetic = Some(ElectromagneticDomain::Harmonic {
        reference_frequency: Frequency::new(1e8),
    });
    let mut step = ProcedureStep::electromagnetic_harmonic("field");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![1e8]);
    model.sequence = AnalysisSequence::from_steps("maxwell-source", vec![step]);
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("solid"));
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(3.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(4.0),
        relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
        conductivity_frequency_response: Vec::new(),
    });
    for face in 0..4 {
        model
            .constraints
            .push(AnalysisConstraint::Electrical(ElectricalConstraint {
                constraint_id: format!("pec:{face}"),
                kind: ElectricalConstraintKind::ElectricPerfectConductor {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(
                            &(format!("face:{face}")),
                        ),
                    ),
                },
            }));
    }
    MaxwellHarmonicWorkload {
        model,
        step_id: ProcedureStepId("field".into()),
        relative_residual_target: 1e-10,
        frequency: runmat_analysis_core::Frequency::new(1e8),
    }
}

fn objects(kind: Kind, identity: Digest, bytes: Vec<u8>) -> PreparedAnalysisObjects {
    prepare_analysis_objects(
        kind,
        identity,
        &bytes,
        1024,
        if kind == Kind::SolverResult {
            ObjectNamespace::ResultValue
        } else {
            ObjectNamespace::InputValue
        },
        ObjectInventoryLimits::default(),
    )
    .unwrap()
}
fn context(
    w: &MaxwellHarmonicWorkload,
    m: &SolverMeshArtifact,
    r: &MaxwellHarmonicSolverResult,
) -> [PreparedAnalysisObjects; 3] {
    [
        objects(
            Kind::MaxwellHarmonicWorkload,
            w.identity().unwrap().digest(),
            w.canonical_bytes().unwrap(),
        ),
        objects(
            Kind::Mesh,
            Digest::from_bytes(*m.canonical_digest.bytes()),
            m.canonical_encode().unwrap(),
        ),
        objects(
            Kind::SolverResult,
            r.identity().unwrap().digest(),
            r.canonical_bytes().unwrap(),
        ),
    ]
}
fn request(
    w: &MaxwellHarmonicWorkload,
    r: &MaxwellNumericalReceiptArtifact,
) -> AccuracyEvaluationRequest {
    AccuracyEvaluationRequest::new(
        r.receipt().producer,
        AccuracySourceKind::MaxwellHarmonic,
        r.source_result(),
        None,
        w.model.sequence.steps[0].accuracy.clone(),
    )
    .unwrap()
}

#[test]
fn cold_context_both_orders_authenticates_exact_original_receipt_and_authored_accuracy() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let w = workload();
        let r = w.execute(&mesh).unwrap();
        let receipt = MaxwellNumericalReceiptArtifact::new(&w, &r, &mesh).unwrap();
        let inputs = context(&w, &mesh, &r);
        let refs = inputs.iter().collect::<Vec<_>>();
        let original = request(&w, &receipt);
        validate(&original, &receipt, None, &refs).unwrap();
        assert!(validate(&original, &receipt, None, &refs[..2]).is_err());
        assert!(validate(&original, &receipt, None, &[refs[0], refs[0], refs[2]]).is_err());
        let mut policy = w.clone();
        policy.model.sequence.steps[0].accuracy.accuracy_plan_id =
            AccuracyPlanId("foreign-plan".into());
        let rehashed = request(&policy, &receipt);
        assert!(validate(&rehashed, &receipt, None, &refs).is_err());
        let mut foreign = inputs.clone();
        foreign[2].manifest.payload_digest = Digest::sha256(b"foreign exact retained source");
        assert!(validate(
            &original,
            &receipt,
            None,
            &foreign.iter().collect::<Vec<_>>()
        )
        .is_err());
    }
}

#[test]
fn coherent_foreign_numerical_intent_result_and_receipt_cannot_replace_original_source() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let w = workload();
    let r = w.execute(&mesh).unwrap();
    let receipt = MaxwellNumericalReceiptArtifact::new(&w, &r, &mesh).unwrap();
    let original = request(&w, &receipt);
    let mut foreign = w.clone();
    foreign.relative_residual_target = 1e-9;
    let changed = foreign.execute(&mesh).unwrap();
    let changed_receipt = MaxwellNumericalReceiptArtifact::new(&foreign, &changed, &mesh).unwrap();
    let inputs = context(&foreign, &mesh, &changed);
    assert!(validate(
        &original,
        &receipt,
        None,
        &inputs.iter().collect::<Vec<_>>()
    )
    .is_err());
    assert!(validate(
        &original,
        &changed_receipt,
        None,
        &inputs.iter().collect::<Vec<_>>()
    )
    .is_err());
    let mut mixed = context(&w, &mesh, &r);
    mixed[2] = inputs[2].clone();
    assert!(validate(&original, &receipt, None, &mixed.iter().collect::<Vec<_>>()).is_err());
}
