use super::*;
use runmat_analysis_core::*;
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_execution::{ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
use std::collections::BTreeSet;

fn workload() -> MaxwellHarmonicWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.structural = None;
    model.electromagnetic = Some(ElectromagneticDomain::Harmonic {
        reference_frequency: Frequency::new(1e8),
    });
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.constraints.clear();
    model.boundaries.clear();
    model.initial_conditions.clear();
    let mut step = ProcedureStep::electromagnetic_harmonic("field");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![1e8]);
    step.state_outputs = vec![StateOutput {
        state_id: StateId("retained".into()),
        kind: StateKind::ElectromagneticComplex,
        completeness: StateCompleteness::Complete,
        persistence: StatePersistence::Canonical,
    }];
    model.sequence = AnalysisSequence::from_steps("maxwell", vec![step]);
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("solid"));
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(3.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(1.0),
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
fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256("graph"),
        Digest::sha256("source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256("runtime"),
            Digest::sha256("abi"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}
#[test]
fn canonical_prepare_and_rederived_resource_authority_fit_both_orders_and_chunk_policies() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let w = workload();
        let m = canonical_tetrahedron_solver_mesh(order);
        let reference = w.execute(&m).unwrap();
        for chunk in [1024, 8192] {
            let policy =
                AnalysisOutputPolicy::new(chunk, ObjectInventoryLimits::default()).unwrap();
            let task =
                crate::prepare_maxwell_harmonic_task(crate::MaxwellHarmonicTaskPreparation {
                    local_execution:
                        runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
                    authored_study_digest: Digest::sha256("authored"),
                    frozen_project_digest: None,
                    program_revision: revision(),
                    artifact_access: ArtifactAccess {
                        authorization_scope: "maxwell-bridge-test".into(),
                        encryption_context: Digest::sha256("scope"),
                    },
                    workload: w.clone(),
                    mesh: m.clone(),
                    output_policy: policy,
                    required_capabilities: BTreeSet::new(),
                })
                .unwrap();
            assert_eq!(
                task.host.workload.kind,
                runmat_analysis_fea::execution::AnalysisWorkKind::SolveMaxwellHarmonic
            );
            assert_eq!(
                task.host.workload.outputs,
                vec![
                    AnalysisNumericalObjectKind::SolverResult,
                    AnalysisNumericalObjectKind::NumericalReceipt,
                    AnalysisNumericalObjectKind::StatePublicationProfiles
                ]
            );
            let plan =
                rederive_maxwell_harmonic_resource_plan(&w, &m, &task.inputs, policy).unwrap();
            assert_eq!(plan, task.resource_plan);
            assert_eq!(reference, w.execute_with_resource_plan(&m, &plan).unwrap());
            let mut foreign = task.inputs.clone();
            foreign[0] = prepare_analysis_objects(
                AnalysisNumericalObjectKind::MaxwellHarmonicWorkload,
                w.identity().unwrap().digest(),
                b"foreign source",
                chunk,
                ObjectNamespace::InputValue,
                policy.inventory_limits(),
            )
            .unwrap();
            assert!(rederive_maxwell_harmonic_resource_plan(&w, &m, &foreign, policy).is_err());
            let other = AnalysisOutputPolicy::new(
                if chunk == 1024 { 8192 } else { 1024 },
                policy.inventory_limits(),
            )
            .unwrap();
            assert!(rederive_maxwell_harmonic_resource_plan(&w, &m, &task.inputs, other).is_err());
            let mut changed = w.clone();
            changed.relative_residual_target *= 2.0;
            assert!(changed.execute_with_resource_plan(&m, &plan).is_err());
        }
    }
}
#[test]
fn tiny_authored_ceiling_and_wrong_input_inventory_are_rejected() {
    let mut w = workload();
    let m = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let policy = AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default()).unwrap();
    assert!(rederive_maxwell_harmonic_resource_plan(&w, &m, &[], policy).is_err());
    w.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    assert!(prepare_maxwell_harmonic_inputs(
        &w,
        &m,
        policy,
        runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial()
    )
    .is_err());
}

#[test]
fn maxwell_plan_carries_admitted_threads_and_worker_rederivation_converges() {
    let w = workload();
    let m = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let policy = AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default()).unwrap();
    let task = crate::prepare_maxwell_harmonic_task(crate::MaxwellHarmonicTaskPreparation {
        local_execution:
            runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(4_000)
                .unwrap(),
        authored_study_digest: Digest::sha256("authored"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: ArtifactAccess {
            authorization_scope: "maxwell-bridge-test".into(),
            encryption_context: Digest::sha256("scope"),
        },
        workload: w.clone(),
        mesh: m.clone(),
        output_policy: policy,
        required_capabilities: BTreeSet::new(),
    })
    .unwrap();
    assert_eq!(task.resource_plan.request.cpu_millicores, 4_000);
    let plan = rederive_maxwell_harmonic_resource_plan(&w, &m, &task.inputs, policy).unwrap();
    assert_eq!(plan, task.resource_plan);
    assert_eq!(
        w.execute_with_resource_plan(&m, &plan).unwrap(),
        w.execute(&m).unwrap()
    );
}
