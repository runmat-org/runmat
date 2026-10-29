//! Retained-result publication only: no magnetic numerical work-kind or scheduler claim.
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    prepare_analysis_objects, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisOutputPolicy, AnalysisWorkloadInput, AnalysisWorkloadRequest, PreparedAnalysisObjects,
};
use runmat_analysis_fea::execution::{
    ExecutionPlanIdentity, MagnetostaticSolverResult, MagnetostaticWorkload, NumericalPlanIdentity,
    StudyIdentity,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::Digest;
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess, ObjectNamespace};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn retained_result() -> MagnetostaticSolverResult {
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
    );
    model.constraints.clear();
    model.boundaries.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.initial_conditions.clear();
    model.structural = None;
    model.thermal = None;
    model.thermo_mechanical = None;
    model.electro_thermal = None;
    model.cfd = None;
    model.electromagnetic = Some(ElectromagneticDomain::Magnetostatic {
        formulation: MagnetostaticFormulation::MagneticVectorPotential,
    });
    model.sequence = AnalysisSequence::from_steps(
        "magnetic-publication",
        vec![ProcedureStep::electromagnetic_static("potential")],
    );
    model.sequence.steps[0].state_outputs = vec![StateOutput {
        state_id: StateId("retained".into()),
        kind: StateKind::ElectromagneticStatic,
        completeness: StateCompleteness::Complete,
        persistence: StatePersistence::Canonical,
    }];
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("solid"));
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(1.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(1.0),
        relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
        conductivity_frequency_response: vec![],
    });
    for face in 0..4 {
        model
            .constraints
            .push(AnalysisConstraint::Magnetic(MagneticConstraint {
                constraint_id: format!("ground:{face}"),
                kind: MagneticConstraintKind::VectorPotentialGround {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(
                            &(format!("face:{face}")),
                        ),
                    ),
                },
            }));
    }
    model
        .constraints
        .push(AnalysisConstraint::Magnetic(MagneticConstraint {
            constraint_id: "gauge".into(),
            kind: MagneticConstraintKind::CoulombGauge {
                target: RegionTarget::volume(runmat_meshing_core::fixtures::fixture_solid("solid")),
            },
        }));
    let result = MagnetostaticWorkload {
        model,
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-10,
    }
    .execute(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet4))
    .unwrap();
    MagnetostaticSolverResult::from_canonical_bytes(&result.canonical_bytes().unwrap()).unwrap()
}
fn object(
    kind: AnalysisNumericalObjectKind,
    identity: Digest,
    bytes: &[u8],
) -> PreparedAnalysisObjects {
    prepare_analysis_objects(
        kind,
        identity,
        bytes,
        4096,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap()
}
fn invocation_inputs(
    result: &MagnetostaticSolverResult,
    request: &SemanticStatePublicationRequest,
) -> Vec<PreparedAnalysisObjects> {
    let bytes = result.canonical_bytes().unwrap();
    let catalog = result
        .state_publication_profiles(Digest::sha256(&bytes))
        .unwrap();
    let mut inputs = vec![
        object(
            AnalysisNumericalObjectKind::StatePublicationRequest,
            Digest::sha256(request.canonical_bytes().unwrap()),
            &request.canonical_bytes().unwrap(),
        ),
        object(
            AnalysisNumericalObjectKind::SolverResult,
            result.source_result_identity().unwrap().digest(),
            &bytes,
        ),
        object(
            AnalysisNumericalObjectKind::StatePublicationProfiles,
            catalog.identity(),
            &catalog.canonical_bytes().unwrap(),
        ),
    ];
    let plan = runmat_analysis_execution::build_prepared_state_publication_resource_plan(
        request, &catalog, &inputs,
    )
    .unwrap();
    inputs.push(object(
        AnalysisNumericalObjectKind::ResourcePlan,
        plan.identity.digest(),
        &plan.canonical_bytes().unwrap(),
    ));
    inputs
}
fn run(
    result: &MagnetostaticSolverResult,
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    // Test-only opaque plan identity; the kernel checks the concrete publication resource plan.
    let plan: NumericalPlanIdentity = serde_json::from_value(
        serde_json::to_value(Digest::sha256(b"publication-test-plan")).unwrap(),
    )
    .unwrap();
    let host = AnalysisHostWorkload::new(
        AnalysisWorkloadRequest::new(
            StudyIdentity::from_authored_digest(Digest::sha256(b"magnetic-test"), None).unwrap(),
            result.producer(),
            ExecutionPlanIdentity::new(plan, Digest::sha256(b"test-execution")).unwrap(),
            AnalysisWorkKind::PublishState,
            inputs
                .iter()
                .map(|o| AnalysisWorkloadInput {
                    kind: o.manifest.kind,
                    logical_identity: o.manifest.logical_identity,
                    payload_digest: o.manifest.payload_digest,
                    manifest_digest: o.root.digest,
                    manifest_encoded_length: o.root.encoded_length,
                    reference_kind: ValueRefKind::DriverObject,
                })
                .collect(),
            vec![AnalysisNumericalObjectKind::SemanticState],
            std::collections::BTreeSet::from([AnalysisCapabilityRequirement::Host {
                abi: runmat_analysis_execution::ANALYSIS_HOST_ABI.into(),
            }]),
        )
        .unwrap(),
        ArtifactAccess {
            authorization_scope: "magnetic-publication-test".into(),
            encryption_context: Digest::sha256(b"scope"),
        },
        runmat_analysis_execution::AnalysisOutputPolicy::new(
            1024,
            ObjectInventoryLimits::default(),
        )
        .unwrap(),
    )
    .unwrap();
    NativeAnalysisKernel.execute(AnalysisKernelInvocation {
        kind: AnalysisWorkKind::PublishState,
        host: &host,
        inputs,
        output_policy: AnalysisOutputPolicy::new(4096, ObjectInventoryLimits::default()).unwrap(),
    })
}
#[test]
fn magnetic_native_publication_consumes_cold_immutable_authority_and_rejects_foreign_inputs() {
    let result = retained_result();
    let request = result.state_publication_requests().unwrap().remove(0);
    let inputs = invocation_inputs(&result, &request);
    let expected =
        CanonicalSemanticState::from_magnetostatic_publication(&request, &result).unwrap();
    let outputs = run(&result, &inputs).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        CanonicalSemanticState::from_canonical_bytes(outputs[0].whole_payload().unwrap()).unwrap(),
        expected
    );
    let mut foreign = inputs.clone();
    foreign[1] = object(
        AnalysisNumericalObjectKind::SolverResult,
        Digest::sha256(b"foreign"),
        &result.canonical_bytes().unwrap(),
    );
    assert!(run(&result, &foreign).is_err());
    let mut substituted = request.clone();
    substituted.declaration.state_id = StateId("invented".into());
    let mut foreign = inputs.clone();
    let bytes = substituted.canonical_bytes().unwrap();
    foreign[0] = object(
        AnalysisNumericalObjectKind::StatePublicationRequest,
        Digest::sha256(&bytes),
        &bytes,
    );
    assert!(run(&result, &foreign).is_err());
}
