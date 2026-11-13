use super::*;
use runmat_analysis_core::{
    AnalysisConstraint, AnalysisSequence, ElectricPotential, ElectricalConstraint,
    ElectricalConstraintKind, ElectromagneticDomain, MaterialElectricalModel, PrescribedValue,
    ProcedureStep, RegionTarget, StaticElectricalEquation,
};
use runmat_analysis_fea::{
    fixtures::{fixture_model, FixtureId},
    properties::compile_property_map,
    topology::AnalysisProjection,
};
use runmat_execution::ProgramEnvironment;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, StableDigest,
};

mod import;

pub(crate) fn preparation() -> ElectricalStudyPreparation {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.constraints.clear();
    model.boundaries.clear();
    model.structural = None;
    model.electromagnetic = Some(ElectromagneticDomain::Static {
        equation: StaticElectricalEquation::Electrostatics,
    });
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(3.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(4.0),
        relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
        conductivity_frequency_response: vec![],
    });
    let face = AnalysisProjection::derive(&mesh).unwrap().boundary_faces[0].provenance[0]
        .analysis_selection_id()
        .unwrap();
    model
        .constraints
        .push(AnalysisConstraint::Electrical(ElectricalConstraint {
            constraint_id: "pressure".into(),
            kind: ElectricalConstraintKind::PrescribedPotential {
                target: RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_boundary_selection(&(face)),
                ),
                value: PrescribedValue::Constant(ElectricPotential::new(1.0)),
            },
        }));
    let step = ProcedureStep::electromagnetic_static("potential");
    model.sequence = AnalysisSequence::from_steps("electrical", vec![step]);
    let compiled = CompiledAnalysisStudy {
        module_path: "tests/electrical.fea".into(),
        study_id: "electrical".into(),
        study_inputs: Vec::new(),
        assessments: Vec::new(),
        outputs: Vec::new(),
        properties: compile_property_map(&model, Some(&mesh)).unwrap(),
        model,
        solver_mesh_digest: mesh.canonical_digest,
    };
    let policy = ElectricalStudyPlanningPolicy {
        element_temperature_k: None,
        operating_time_s: None,
    };
    let workload = plan_electrical_study(&compiled, policy.clone()).unwrap();
    ElectricalStudyPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        machine_memory_bytes: 16 << 30,
        product: ElectricalStudyProduct { compiled, workload },
        policy,
        mesh,
        frozen_project_digest: None,
        artifact_access: ArtifactAccess {
            authorization_scope: "electrical-study-tests".into(),
            encryption_context: Digest::sha256("scope"),
        },
        program_revision: ProgramRevision::new(
            Digest::sha256("graph"),
            Digest::sha256("source"),
            ProgramEnvironment::new(
                1,
                1,
                Digest::sha256("runtime"),
                Digest::sha256("catalog"),
                "matlab",
            )
            .unwrap(),
        )
        .unwrap(),
        output_policy: AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default()).unwrap(),
        required_capabilities: BTreeSet::new(),
    }
}

#[test]
fn electrical_standalone_coordinator_rejects_foreign_mesh_policy_and_definition() {
    let mut input = preparation();
    input.product.compiled.solver_mesh_digest = StableDigest::ZERO;
    assert!(ElectricalStudyCoordinator::new(input).is_err());
    let mut input = preparation();
    input.product.compiled.model.geometry_revision += 1;
    assert!(ElectricalStudyCoordinator::new(input).is_err());
    let mut input = preparation();
    input.product.workload.relative_residual_target *= 0.5;
    assert!(ElectricalStudyCoordinator::new(input).is_err());
}

#[test]
fn electrical_standalone_owned_solve_cannot_be_skipped_or_prepared_twice() {
    let mut study = ElectricalStudyCoordinator::new(preparation()).unwrap();
    assert!(study.prepare_accuracy().is_err());
    assert!(study.prepare_next_state().is_err());
    assert!(study.freeze_baseline().is_err());
    assert!(study.prepare_publication().is_err());
    let definition = study.compiled_study().definition_digest().unwrap();
    let task = study.prepare_solve().unwrap();
    assert_eq!(
        task.host.workload.study,
        runmat_analysis_fea::execution::StudyIdentity::from_authored_digest(definition, None)
            .unwrap()
    );
    assert_eq!(study.phase(), ElectricalStudyPhase::AwaitingSolve);
    assert!(study.prepare_solve().is_err());
    assert!(study.result().is_none());
    assert!(!study.is_complete());
    assert!(study.into_completed().is_err());
}

struct EmptyStore;
impl CacheImport for EmptyStore {
    fn read_verified(
        &self,
        _: Digest,
    ) -> Result<Option<Vec<u8>>, runmat_execution_artifact::ArtifactError> {
        Ok(None)
    }
}

#[test]
fn electrical_failed_response_never_advances_or_installs_usage() {
    let mut study = ElectricalStudyCoordinator::new(preparation()).unwrap();
    let task = study.prepare_solve().unwrap();
    let accounting = crate::tests::phase_execution(
        &task.host,
        &task.program,
        &task.resource_plan.request,
        &ProgramExecutionResponse::ExternalizedSuccess {
            outputs: Vec::new(),
            result_objects: Vec::new(),
        },
    );
    assert!(study
        .complete_solve(
            &EmptyStore,
            ProgramExecutionResponse::ExternalizedSuccess {
                outputs: Vec::new(),
                result_objects: Vec::new()
            },
            accounting
        )
        .is_err());
    assert_eq!(study.phase(), ElectricalStudyPhase::AwaitingSolve);
    assert!(study.result().is_none());
    // Failed solve admission must not install the shared accuracy/state/accounting authority.
    assert!(study.tail.is_none());
    assert!(study.baseline_usage().is_none());
    assert!(study.terminal_usage().is_none());
    assert!(study.closures.is_empty());
}

#[test]
fn electrical_completion_replay_rejects_missing_and_substituted_phase_workloads() {
    assert!(import_electrical_study_completion(&EmptyStore, preparation(), Vec::new()).is_err());
    let mut study = ElectricalStudyCoordinator::new(preparation()).unwrap();
    let task = study.prepare_solve().unwrap();
    let accounting = crate::tests::phase_execution(
        &task.host,
        &task.program,
        &task.resource_plan.request,
        &ProgramExecutionResponse::ExternalizedSuccess {
            outputs: Vec::new(),
            result_objects: Vec::new(),
        },
    );
    let response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: Vec::new(),
        result_objects: Vec::new(),
    };
    let closure = ElectricalStudyResultClosure::new(
        ElectricalStudyResultPhase::Accuracy,
        &task.host,
        &response,
        &accounting,
    );
    assert!(import_electrical_study_completion(&EmptyStore, preparation(), vec![closure]).is_err());
    let mut different = preparation();
    different.frozen_project_digest = Some(Digest::sha256("foreign-project"));
    different.program_revision = ProgramRevision::new(
        Digest::sha256("foreign-graph"),
        Digest::sha256("foreign-source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256("runtime"),
            Digest::sha256("catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap();
    let mut foreign = ElectricalStudyCoordinator::new(different).unwrap();
    let foreign_task = foreign.prepare_solve().unwrap();
    assert_ne!(task.host.workload, foreign_task.host.workload);
    let closure = ElectricalStudyResultClosure::new(
        ElectricalStudyResultPhase::Solve,
        &foreign_task.host,
        &response,
        &accounting,
    );
    assert!(import_electrical_study_completion(&EmptyStore, preparation(), vec![closure]).is_err());
}
