use super::*;
use crate::physics::electrical::tests::{mesh, model, prescribe};
use runmat_analysis_core::{RegionTarget, RestartPolicy, StaticElectricalEquation};
use runmat_meshing_core::{ElementOrder, StableDigest};

fn compiled(equation: StaticElectricalEquation) -> CompiledAnalysisStudy {
    let mut model = model(equation);
    prescribe(
        &mut model,
        "left".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        0.0,
    );
    prescribe(
        &mut model,
        "right".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        2.0,
    );
    let properties = crate::properties::compile_property_map(&model, None).unwrap();
    CompiledAnalysisStudy {
        module_path: "tests/electrical.fea".into(),
        study_id: "electrical".into(),
        study_inputs: vec![],
        assessments: Vec::new(),
        outputs: vec![],
        model,
        properties,
        solver_mesh_digest: StableDigest::ZERO,
    }
}

fn policy() -> ElectricalStudyPlanningPolicy {
    ElectricalStudyPlanningPolicy {
        element_temperature_k: None,
        operating_time_s: None,
    }
}

#[test]
fn plans_both_equations_orders_without_rewriting_authored_intent() {
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        let mut study = compiled(equation);
        let criterion = study.model.sequence.steps[0]
            .accuracy
            .criteria
            .iter_mut()
            .find(|criterion| criterion.contribution == ErrorContributionKind::Algebraic)
            .unwrap();
        criterion.tolerance = AccuracyTolerance::Relative(3e-9);
        let before = study.clone();
        let workload = plan_electrical_study(&study, policy()).unwrap();
        assert_eq!(workload.model, study.model);
        assert_eq!(workload.equation().unwrap(), equation);
        assert_eq!(workload.relative_residual_target, 3e-9);
        assert_eq!(study, before);
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            workload.formulation(&mesh(order, 2)).unwrap();
        }
    }
}

#[test]
fn unsupported_lifecycle_and_missing_accuracy_are_not_silently_lowered() {
    let study = compiled(StaticElectricalEquation::Electrostatics);
    let mut changed = study.clone();
    changed.model.sequence.steps[0].accuracy.criteria.clear();
    assert!(matches!(
        plan_electrical_study(&changed, policy()),
        Err(ElectricalStudyPlanningError::MissingRelativeAlgebraicTarget)
    ));
    changed = study.clone();
    changed
        .model
        .sequence
        .steps
        .push(changed.model.sequence.steps[0].clone());
    assert!(matches!(
        plan_electrical_study(&changed, policy()),
        Err(ElectricalStudyPlanningError::UnsupportedSequence)
    ));
    changed = study.clone();
    changed.model.sequence.steps[0].restart = RestartPolicy::ResumeCheckpoint {
        checkpoint_id: "not-admitted".into(),
        required_state_digest: "not-admitted".into(),
    };
    assert!(plan_electrical_study(&changed, policy()).is_err());
}

#[test]
fn floating_and_integral_terminal_intent_is_not_replaced_by_insulation() {
    use runmat_analysis_core::{ElectricCurrent, ElectricalConstraint, PrescribedValue};
    for kind in [
        ElectricalConstraintKind::FloatingConductor {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_right",
            )),
        },
        ElectricalConstraintKind::Equipotential {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_right",
            )),
        },
        ElectricalConstraintKind::IntegralCurrent {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_right",
            )),
            value: PrescribedValue::Constant(ElectricCurrent::new(1.0)),
        },
    ] {
        let mut study = compiled(StaticElectricalEquation::SteadyConduction);
        study.model.constraints[1] = AnalysisConstraint::Electrical(ElectricalConstraint {
            constraint_id: "right".into(),
            kind,
        });
        assert!(matches!(
            plan_electrical_study(&study, policy()),
            Err(ElectricalStudyPlanningError::UnsupportedConstraint)
        ));
    }
}
