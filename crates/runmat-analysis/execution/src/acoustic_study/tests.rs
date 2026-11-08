use super::*;
use runmat_analysis_core::{
    AcousticConstraint, AcousticConstraintKind, AnalysisConstraint, AnalysisSequence,
    IndependentDomain, MaterialAcousticModel, PrescribedValue, Pressure, ProcedureStep,
    RegionTarget,
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

pub(super) fn preparation() -> AcousticStudyPreparation {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.constraints.clear();
    model.boundaries.clear();
    model.materials[0].acoustic = Some(MaterialAcousticModel {
        speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(2.0),
        damping_ratio: runmat_types::physical::DampingRatio::new(0.0),
    });
    model.materials[0].mass_density_kg_per_m3 = Some(runmat_types::physical::MassDensity::new(3.0));
    let face = AnalysisProjection::derive(&mesh).unwrap().boundary_faces[0].provenance[0]
        .analysis_selection_id()
        .unwrap();
    model
        .constraints
        .push(AnalysisConstraint::Acoustic(AcousticConstraint {
            constraint_id: "pressure".into(),
            kind: AcousticConstraintKind::PrescribedPressure {
                target: RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_boundary_selection(&(face)),
                ),
                value: PrescribedValue::Constant(Pressure::new(1.0)),
            },
        }));
    let mut step = ProcedureStep::acoustic_harmonic("sound");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![0.2]);
    model.sequence = AnalysisSequence::from_steps("acoustic", vec![step]);
    let compiled = CompiledAnalysisStudy {
        module_path: "tests/acoustic.fea".into(),
        study_id: "acoustic".into(),
        study_inputs: Vec::new(),
        assessments: Vec::new(),
        outputs: Vec::new(),
        properties: compile_property_map(&model, Some(&mesh)).unwrap(),
        model,
        solver_mesh_digest: mesh.canonical_digest,
    };
    let policy = AcousticStudyPlanningPolicy::standard();
    let plan = plan_acoustic_study(&compiled, policy).unwrap();
    AcousticStudyPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        product: AcousticStudyProduct { compiled, plan },
        policy,
        mesh,
        frozen_project_digest: None,
        artifact_access: ArtifactAccess {
            authorization_scope: "acoustic-study-tests".into(),
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
fn acoustic_standalone_coordinator_rejects_foreign_mesh_policy_and_definition() {
    let mut input = preparation();
    input.product.compiled.solver_mesh_digest = StableDigest::ZERO;
    assert!(AcousticStudyCoordinator::new(input).is_err());
    let mut input = preparation();
    input.policy.contraction_limit *= 0.5;
    assert!(AcousticStudyCoordinator::new(input).is_err());
    let mut input = preparation();
    input.product.compiled.model.geometry_revision += 1;
    assert!(AcousticStudyCoordinator::new(input).is_err());
    let mut input = preparation();
    input.product.compiled.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![0.2, 0.3]);
    assert!(AcousticStudyCoordinator::new(input).is_err());
}

#[test]
fn acoustic_standalone_owned_solve_cannot_be_skipped_or_prepared_twice() {
    let mut coordinator = AcousticStudyCoordinator::new(preparation()).unwrap();
    let study = coordinator.entry_mut(0).unwrap();
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
    assert_eq!(study.phase(), AcousticStudyPhase::AwaitingSolve);
    assert!(study.prepare_solve().is_err());
    assert!(study.result().is_none());
    assert!(!study.is_complete());
    assert!(coordinator.into_completed().is_err());
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
fn acoustic_failed_response_never_advances_or_installs_usage() {
    let mut coordinator = AcousticStudyCoordinator::new(preparation()).unwrap();
    let study = coordinator.entry_mut(0).unwrap();
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
    assert_eq!(study.phase(), AcousticStudyPhase::AwaitingSolve);
    assert!(study.result().is_none());
    assert!(study.baseline_usage().is_none());
}

#[test]
fn acoustic_completion_replay_rejects_missing_and_substituted_phase_workloads() {
    let mut coordinator = AcousticStudyCoordinator::new(preparation()).unwrap();
    let study = coordinator.entry_mut(0).unwrap();
    let entry = study.entry();
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
    let closure = AcousticStudyResultClosure::new(
        AcousticStudyResultPhase::Accuracy,
        &task.host,
        &response,
        &accounting,
    );
    let assembly = super::completion::AcousticStudyAssemblyClosure {
        workload: task.host.workload.clone(),
        response: response.clone(),
        execution: accounting.clone(),
    };
    assert!(import_acoustic_study_completion(
        &EmptyStore,
        preparation(),
        vec![super::completion::AcousticStudyEntryClosure {
            entry,
            closures: vec![closure]
        }],
        Vec::new(),
        assembly,
        1
    )
    .is_err());
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
    let mut foreign = AcousticStudyCoordinator::new(different).unwrap();
    let foreign_task = foreign.entry_mut(0).unwrap().prepare_solve().unwrap();
    assert_ne!(task.host.workload, foreign_task.host.workload);
    let closure = AcousticStudyResultClosure::new(
        AcousticStudyResultPhase::Solve,
        &foreign_task.host,
        &response,
        &accounting,
    );
    let assembly = super::completion::AcousticStudyAssemblyClosure {
        workload: task.host.workload.clone(),
        response: response.clone(),
        execution: accounting.clone(),
    };
    assert!(import_acoustic_study_completion(
        &EmptyStore,
        preparation(),
        vec![super::completion::AcousticStudyEntryClosure {
            entry,
            closures: vec![closure]
        }],
        Vec::new(),
        assembly,
        1
    )
    .is_err());
}

#[test]
fn authored_frequency_sweep_creates_every_exact_entry_without_parent_singleton_api() {
    let mut input = preparation();
    input.product.compiled.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![0.2, 0.3]);
    input.product.plan = plan_acoustic_study(&input.product.compiled, input.policy).unwrap();
    let expected = input
        .product
        .plan
        .frequencies()
        .iter()
        .map(|f| f.entry())
        .collect::<Vec<_>>();
    let mut study = AcousticStudyCoordinator::new(input).unwrap();
    assert_eq!(study.entries().len(), 2);
    assert_eq!(
        study
            .entries()
            .iter()
            .map(|entry| entry.entry())
            .collect::<Vec<_>>(),
        expected
    );
    assert!(study.prepare_assembly().is_err());
}
