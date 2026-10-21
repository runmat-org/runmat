use super::*;
use crate::physics::magnetic::test_support::authored_model;
use runmat_analysis_core::{ElectromagneticDomain, RestartPolicy, StaticElectricalEquation};
use runmat_meshing_core::StableDigest;

fn compiled() -> CompiledAnalysisStudy {
    let model = authored_model(1.0);
    let properties = crate::properties::compile_property_map(&model, None).unwrap();
    CompiledAnalysisStudy {
        module_path: "tests/magnetic.fea".into(),
        study_id: "magnetic".into(),
        study_inputs: vec![],
        assessments: Vec::new(),
        outputs: vec![],
        model,
        properties,
        solver_mesh_digest: StableDigest::ZERO,
    }
}

fn policy() -> MagneticStudyPlanningPolicy {
    MagneticStudyPlanningPolicy {}
}

#[test]
fn preserves_authored_magnetic_model_and_tightest_target() {
    let mut study = compiled();
    let mut tighter = study.model.sequence.steps[0]
        .accuracy
        .criteria
        .iter()
        .find(|criterion| criterion.contribution == ErrorContributionKind::Algebraic)
        .unwrap()
        .clone();
    tighter.tolerance = AccuracyTolerance::Relative(3e-12);
    tighter.criterion_id = "tighter-magnetic-algebraic".into();
    study.model.sequence.steps[0]
        .accuracy
        .criteria
        .push(tighter);
    let original = study.clone();
    let workload = plan_magnetic_study(&study, policy()).unwrap();
    assert_eq!(workload.model, study.model);
    assert_eq!(workload.relative_residual_target, 3e-12);
    assert_eq!(study, original);
}

#[test]
fn accuracy_sequence_and_requirement_fail_closed() {
    let study = compiled();
    let mut changed = study.clone();
    changed.model.sequence.steps[0].accuracy.criteria.clear();
    assert!(matches!(
        plan_magnetic_study(&changed, policy()),
        Err(MagneticStudyPlanningError::MissingRelativeAlgebraicTarget)
    ));
    changed = study.clone();
    changed
        .model
        .sequence
        .steps
        .push(changed.model.sequence.steps[0].clone());
    assert!(matches!(
        plan_magnetic_study(&changed, policy()),
        Err(MagneticStudyPlanningError::UnsupportedSequence)
    ));
}

#[test]
fn electrical_equations_and_restart_are_not_magnetic_substitutes() {
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        let mut changed = compiled();
        changed.model.electromagnetic = Some(ElectromagneticDomain::Static { equation });
        assert!(plan_magnetic_study(&changed, policy()).is_err());
    }
    let mut changed = compiled();
    changed.model.sequence.steps[0].restart = RestartPolicy::ResumeCheckpoint {
        checkpoint_id: "unqualified".into(),
        required_state_digest: "unqualified".into(),
    };
    assert!(plan_magnetic_study(&changed, policy()).is_err());
}

#[test]
fn invalid_authored_residual_targets_are_not_replaced_by_defaults() {
    for target in [0.0, -1.0, 1.0, f64::INFINITY, f64::NAN] {
        let mut study = compiled();
        for criterion in &mut study.model.sequence.steps[0].accuracy.criteria {
            if criterion.contribution == ErrorContributionKind::Algebraic {
                criterion.tolerance = AccuracyTolerance::Relative(target);
            }
        }
        assert!(plan_magnetic_study(&study, policy()).is_err());
    }
}
