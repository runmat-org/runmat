//! Planning, lifecycle and replay failures; actual cold chains are native witnesses.
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_fea::{
    execution::MaxwellHarmonicWorkload,
    fixtures::{fixture_model, FixtureId},
    properties::compile_property_map,
};
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, StableDigest,
};

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

fn preparation(order: ElementOrder) -> MaxwellStudyPreparation {
    let prior = crate::electrical_study::tests::preparation();
    let mesh = canonical_tetrahedron_solver_mesh(order);
    let model = workload().model;
    let compiled = CompiledAnalysisStudy {
        module_path: "tests/maxwell.fea".into(),
        study_id: "maxwell".into(),
        study_inputs: Vec::new(),
        assessments: Vec::new(),
        outputs: Vec::new(),
        properties: compile_property_map(&model, Some(&mesh)).unwrap(),
        model,
        solver_mesh_digest: mesh.canonical_digest,
    };
    let policy = MaxwellStudyPlanningPolicy {};
    let workload = plan_maxwell_study(&compiled, policy).unwrap();
    MaxwellStudyPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        policy,
        product: MaxwellStudyProduct { compiled, workload },
        mesh,
        frozen_project_digest: prior.frozen_project_digest,
        artifact_access: prior.artifact_access,
        program_revision: prior.program_revision,
        output_policy: prior.output_policy,
        required_capabilities: prior.required_capabilities,
    }
}

struct EmptyStore;
impl CacheImport for EmptyStore {
    fn read_verified(
        &self,
        _: Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        Ok(None)
    }
}
fn accounting(task: &PreparedMaxwellHarmonicTask) -> AnalysisPhaseExecutionEvidence {
    crate::tests::phase_execution(
        &task.host,
        &task.program,
        &task.resource_plan.request,
        &empty_response(),
    )
}
fn empty_response() -> ProgramExecutionResponse {
    ProgramExecutionResponse::ExternalizedSuccess {
        outputs: Vec::new(),
        result_objects: Vec::new(),
    }
}

#[test]
fn exact_authored_preparation_and_owned_phases_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut study = MaxwellStudyCoordinator::new(preparation(order)).unwrap();
        assert_eq!(study.phase(), MaxwellStudyPhase::ReadyForSolve);
        assert!(study.prepare_field_recovery().is_err());
        assert!(study.prepare_observation_reduction().is_err());
        assert!(study.prepare_observation_assembly().is_err());
        assert!(study.prepare_accuracy().is_err());
        assert!(study.prepare_next_state().is_err());
        assert!(study.freeze_baseline().is_err());
        assert!(study.prepare_publication().is_err());
        let definition = study.compiled_study().definition_digest().unwrap();
        let task = study.prepare_solve().unwrap();
        assert_eq!(
            task.host.workload.kind,
            runmat_analysis_fea::execution::AnalysisWorkKind::SolveMaxwellHarmonic
        );
        assert_eq!(
            task.host.workload.study,
            runmat_analysis_fea::execution::StudyIdentity::from_authored_digest(definition, None)
                .unwrap()
        );
        assert!(study.prepare_solve().is_err());
        assert!(!study.is_complete());
        assert!(study.into_completed().is_err());
        let mut changed = preparation(order);
        changed.product.compiled.solver_mesh_digest = StableDigest::ZERO;
        assert!(MaxwellStudyCoordinator::new(changed).is_err());
        let mut changed = preparation(order);
        changed.product.workload.relative_residual_target *= 0.5;
        assert!(MaxwellStudyCoordinator::new(changed).is_err());
        let mut changed = preparation(order);
        changed.product.compiled.model.materials[0]
            .electrical
            .as_mut()
            .unwrap()
            .relative_permittivity = runmat_types::physical::Quantity::new(
            changed.product.compiled.model.materials[0]
                .electrical
                .as_mut()
                .unwrap()
                .relative_permittivity
                .canonical_value()
                * (2.0),
        );
        assert!(MaxwellStudyCoordinator::new(changed).is_err());
    }
}

#[test]
fn failed_solve_import_never_installs_result_usage_or_terminal_authority() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut study = MaxwellStudyCoordinator::new(preparation(order)).unwrap();
        let task = study.prepare_solve().unwrap();
        let usage = accounting(task);
        let mut allocation = task.resource_plan.request.clone();
        allocation.memory_bytes += 1;
        let foreign = crate::tests::phase_execution(
            &task.host,
            &task.program,
            &allocation,
            &empty_response(),
        );
        assert!(matches!(
            study.complete_solve(&EmptyStore, empty_response(), foreign),
            Err(AnalysisBridgeError::Identity(_))
        ));
        assert_eq!(study.phase(), MaxwellStudyPhase::AwaitingSolve);
        assert!(study.usage.is_none() && study.closures.is_empty());
        assert!(study
            .complete_solve(&EmptyStore, empty_response(), usage)
            .is_err());
        assert_eq!(study.phase(), MaxwellStudyPhase::AwaitingSolve);
        assert!(study.result().is_none());
        assert!(study.fields().is_none());
        assert!(study.usage.is_none());
        assert!(study.tail.is_none());
        assert!(study.baseline_usage().is_none());
        assert!(study.terminal_usage().is_none());
        assert!(study.closures.is_empty());
    }
}

#[test]
fn replay_rejects_missing_wrong_phase_and_foreign_authored_closures() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        assert!(
            import_maxwell_study_completion(&EmptyStore, preparation(order), Vec::new()).is_err()
        );
        let mut study = MaxwellStudyCoordinator::new(preparation(order)).unwrap();
        let task = study.prepare_solve().unwrap();
        let usage = accounting(task);
        let wrong = MaxwellStudyResultClosure::new(
            MaxwellStudyResultPhase::Accuracy,
            &task.host,
            &empty_response(),
            &usage,
        );
        assert!(
            import_maxwell_study_completion(&EmptyStore, preparation(order), vec![wrong]).is_err()
        );
        let mut changed = preparation(order);
        changed.frozen_project_digest = Some(Digest::sha256("foreign project"));
        let mut foreign = MaxwellStudyCoordinator::new(changed).unwrap();
        let foreign_task = foreign.prepare_solve().unwrap();
        assert_ne!(task.host.workload, foreign_task.host.workload);
        let closure = MaxwellStudyResultClosure::new(
            MaxwellStudyResultPhase::Solve,
            &foreign_task.host,
            &empty_response(),
            &usage,
        );
        assert!(
            import_maxwell_study_completion(&EmptyStore, preparation(order), vec![closure])
                .is_err()
        );
    }
}
