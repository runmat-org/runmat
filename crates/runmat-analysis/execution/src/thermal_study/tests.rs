use super::*;
use runmat_analysis_core::{AnalysisSequence, ProcedureStep, Temperature, ThermalDomain};
use runmat_analysis_fea::{
    fixtures::{fixture_model, FixtureId},
    properties::compile_property_map,
    study::{plan_thermal_study, ThermalStudyPlanningPolicy},
    ComputeBackend,
};
use runmat_execution::ProgramEnvironment;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, StableDigest,
};

pub(crate) fn preparation() -> ThermalStudyPreparation {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    // Explicit insulation retains the natural zero-flux equilibrium boundary condition.
    let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh).unwrap();
    let insulated_face = projection.boundary_faces[0].provenance[0]
        .analysis_selection_id()
        .unwrap();
    model
        .boundaries
        .push(runmat_analysis_core::AnalysisBoundary {
            boundary_id: "insulated-face".into(),
            target: runmat_analysis_core::RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&(insulated_face)),
            ),
            operator: runmat_analysis_core::BoundaryOperator::ThermalHeatFlux {
                value: runmat_analysis_core::PrescribedValue::Constant(
                    runmat_analysis_core::HeatFlux::new(0.0),
                ),
            },
        });
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence = AnalysisSequence::from_steps(
        "thermal-study",
        vec![ProcedureStep::thermal_transient("heat")],
    );
    let compiled = CompiledAnalysisStudy {
        module_path: "tests/thermal.fea".into(),
        study_id: "thermal".into(),
        study_inputs: Vec::new(),
        assessments: Vec::new(),
        outputs: Vec::new(),
        properties: compile_property_map(&model, Some(&mesh)).unwrap(),
        model,
        solver_mesh_digest: mesh.canonical_digest,
    };
    let workload = plan_thermal_study(
        &compiled,
        ThermalStudyPlanningPolicy {
            backend: ComputeBackend::Cpu,
            element_heat_source: None,
        },
    )
    .unwrap();
    ThermalStudyPreparation {
        maximum_resident_bytes: None,
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        field_source: None,
        scope: ThermalStudyScope::Standalone,
        product: ThermalStudyProduct { compiled, workload },
        mesh,
        frozen_project_digest: None,
        artifact_access: ArtifactAccess {
            authorization_scope: "thermal-study-tests".into(),
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
fn rejects_foreign_mesh_and_workload_controls() {
    let mut input = preparation();
    input.product.compiled.solver_mesh_digest = StableDigest::ZERO;
    assert!(ThermalStudyCoordinator::new(input).is_err());
    let mut input = preparation();
    input.product.workload.residual_target *= 0.5;
    assert!(ThermalStudyCoordinator::new(input).is_err());
    let mut input = preparation();
    input
        .product
        .workload
        .model
        .thermal
        .as_mut()
        .unwrap()
        .reference_temperature = Temperature::new(400.0);
    assert!(ThermalStudyCoordinator::new(input).is_err());
}

#[test]
fn rejects_forged_compiled_property_ownership() {
    let mut input = preparation();
    input.product.compiled.model.geometry_revision += 1;
    assert!(ThermalStudyCoordinator::new(input).is_err());
}

#[test]
fn lifecycle_cannot_skip_solve_or_publish_incomplete_evidence() {
    let mut study = ThermalStudyCoordinator::new(preparation()).unwrap();
    assert_eq!(study.phase(), ThermalStudyPhase::ReadyForSolve);
    assert!(study.prepare_accuracy().is_err());
    assert!(study.prepare_next_state().is_err());
    assert!(study.freeze_baseline().is_err());
    assert!(study.prepare_publication().is_err());
    assert!(study.result().is_none());
    assert!(study.publication().is_none());
    assert!(study.terminal_usage().is_none());
    assert!(!study.is_complete());
    assert!(study.into_completed().is_err());
}

#[test]
fn prepared_solve_is_owned_and_cannot_be_prepared_twice() {
    let mut study = ThermalStudyCoordinator::new(preparation()).unwrap();
    let expected = study.compiled_study().definition_digest().unwrap();
    let task = study.prepare_solve().unwrap();
    assert_eq!(
        task.host.workload.study,
        runmat_analysis_fea::execution::StudyIdentity::from_authored_digest(expected, None)
            .unwrap()
    );
    assert_eq!(study.phase(), ThermalStudyPhase::AwaitingSolve);
    assert!(study.prepare_solve().is_err());
    assert!(study.prepare_accuracy().is_err());
    assert!(study.prepare_next_state().is_err());
    assert!(study.freeze_baseline().is_err());
    assert!(study.prepare_publication().is_err());
    assert_eq!(study.phase(), ThermalStudyPhase::AwaitingSolve);
}

#[test]
fn all_declared_states_precede_baseline_in_authored_order() {
    let mut input = preparation();
    let mut second = input.product.compiled.model.sequence.steps[0].state_outputs[0].clone();
    second.state_id = runmat_analysis_core::StateId("second-temperature".into());
    input.product.compiled.model.sequence.steps[0]
        .state_outputs
        .push(second);
    input.product.workload.model = input.product.compiled.model.clone();
    let mut study = ThermalStudyCoordinator::new(input).unwrap();
    // Exercise the deterministic routing independently of numerical execution.
    study.advance_state();
    assert_eq!(
        study.phase(),
        ThermalStudyPhase::ReadyForState { ordinal: 0 }
    );
    assert!(study.freeze_baseline().is_err());
    assert!(study.prepare_publication().is_err());
    assert_eq!(
        study.compiled_study().model.sequence.steps[0]
            .state_outputs
            .len(),
        2
    );
}

pub(super) fn selected_preparation() -> ThermalStudyPreparation {
    let mut input = preparation();
    let selected = input.product.compiled.model.sequence.steps[0].clone();
    input.product.compiled.model.sequence = AnalysisSequence::from_steps(
        "mixed-study",
        vec![
            ProcedureStep::structural_static("independent-mechanical"),
            selected,
        ],
    );
    input.product.compiled.model.sequence.dependencies.clear();
    input.product.compiled.properties =
        compile_property_map(&input.product.compiled.model, Some(&input.mesh)).unwrap();
    input.product.workload.model = input.product.compiled.model.clone();
    input.scope = ThermalStudyScope::SelectedStep;
    input
}

#[test]
fn selected_lifecycle_keeps_full_model_and_routes_nonfirst_state_declarations() {
    let input = selected_preparation();
    let model = input.product.compiled.model.clone();
    let mut study = ThermalStudyCoordinator::new(input).unwrap();
    assert_eq!(study.scope(), ThermalStudyScope::SelectedStep);
    assert_eq!(study.selected_step().step_id.0, "heat");
    assert_eq!(study.compiled_study().model, model);
    let task = study.prepare_solve().unwrap();
    assert_eq!(
        task.host.workload.step,
        runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
            runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(&model.sequence)
                .unwrap(),
            &model.sequence.steps[1],
        )
        .unwrap()
    );
    // The unrelated first step has a different state kind; routing must use selected heat.
    assert_eq!(
        study.preparation.selected_step().state_outputs[0].kind,
        runmat_analysis_core::StateKind::Thermal
    );
    study.advance_state();
    assert_eq!(
        study.phase(),
        ThermalStudyPhase::ReadyForState { ordinal: 0 }
    );
}

#[test]
fn selected_lifecycle_rejects_false_standalone_and_foreign_selection() {
    let mut input = selected_preparation();
    input.scope = ThermalStudyScope::Standalone;
    assert!(ThermalStudyCoordinator::new(input).is_err());
    for selected in ["missing", "independent-mechanical"] {
        let mut input = selected_preparation();
        input.product.workload.step_id.0 = selected.into();
        assert!(ThermalStudyCoordinator::new(input).is_err());
    }
}

#[test]
fn original_temperature_fields_survive_borrowed_authored_intent_validation() {
    use runmat_analysis_fea::{
        execution::ThermalSolverResultManifest, physics::thermal::qualification as f,
        results::field::RecoveredField, study::plan_thermal_study_with_fields,
    };
    let mesh = f::mesh(false);
    let producer = f::initial::linear_profile().execute(&mesh).unwrap();
    let manifest = ThermalSolverResultManifest::from_result(&producer).unwrap();
    let entry = manifest.terminal_temperature().unwrap();
    let field = RecoveredField::new(
        entry.definition.clone(),
        entry.sample,
        producer.temperature_snapshots[0].clone(),
    )
    .unwrap();
    let binding = f::initial::temperature_binding(&manifest, field, &mesh);
    let mut input = f::cooling::manufactured(8, binding);
    input.model.sequence.steps[0].accuracy.criteria[0].tolerance =
        runmat_analysis_core::AccuracyTolerance::Relative(f::TARGET);
    let mut p = preparation();
    p.mesh = mesh;
    p.product.compiled.model = input.model;
    p.product.compiled.properties =
        compile_property_map(&p.product.compiled.model, Some(&p.mesh)).unwrap();
    p.product.compiled.solver_mesh_digest = p.mesh.canonical_digest;
    p.product.workload = plan_thermal_study_with_fields(
        &p.product.compiled,
        ThermalStudyPlanningPolicy {
            backend: ComputeBackend::Cpu,
            element_heat_source: None,
        },
        input.initial_fields,
    )
    .unwrap();
    let copy = || {
        let mut copied = preparation();
        copied.scope = p.scope;
        copied.product = p.product.clone();
        copied.mesh = p.mesh.clone();
        copied
    };
    ThermalStudyCoordinator::new(copy()).unwrap();
    for scope in [
        ThermalStudyScope::Standalone,
        ThermalStudyScope::SelectedStep,
    ] {
        let mut selected = copy();
        selected.scope = scope;
        let mut valid = copy();
        valid.scope = scope;
        ThermalStudyCoordinator::new(valid).unwrap();
        selected.product.workload.initial_fields.clear();
        assert!(ThermalStudyCoordinator::new(selected).is_err());
    }
    let mut foreign = copy();
    foreign.product.workload.residual_target *= 0.5;
    assert!(ThermalStudyCoordinator::new(foreign).is_err());
    let mut bounded = p;
    bounded.product.compiled.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    bounded.product.workload.model = bounded.product.compiled.model.clone();
    bounded.product.compiled.properties =
        compile_property_map(&bounded.product.compiled.model, Some(&bounded.mesh)).unwrap();
    assert!(ThermalStudyCoordinator::new(bounded).is_err());
}
