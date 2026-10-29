use super::*;
use runmat_analysis_core::*;
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

pub(in crate::study) fn model() -> AnalysisModel {
    let mut model = crate::physics::magnetic::test_support::authored_model(1.0);
    let frequency = 1e8;
    model.electromagnetic = Some(ElectromagneticDomain::Harmonic {
        reference_frequency: Frequency::new(frequency),
    });
    let mut step = ProcedureStep::electromagnetic_harmonic("field");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![frequency]);
    model.sequence = AnalysisSequence::from_steps("maxwell-study", vec![step]);
    model.constraints.clear();
    for boundary in ["slab_left", "slab_right", "slab_insulated"] {
        model
            .constraints
            .push(AnalysisConstraint::Electrical(ElectricalConstraint {
                constraint_id: boundary.into(),
                kind: ElectricalConstraintKind::ElectricPerfectConductor {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(boundary)),
                    ),
                },
            }));
    }
    let PhysicalSignal::ElectricCurrentDensity(signal) = &mut model.signals[0] else {
        panic!()
    };
    let vector = |value| SignalValue::Vector3([0.0, value, 0.0].map(ElectricCurrentDensity::new));
    signal.precision = SignalPrecision::Complex64;
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::FrequencyHertz,
        knots: vec![SignalKnot::continuous(
            frequency,
            SignalOrdinate::Complex {
                real: vector(0.6),
                imaginary: vector(0.8),
            },
        )],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::HoldEndpoint,
    });
    for action in &mut model.excitations {
        if let ExcitationAction::MagneticVolumeCurrentDensity(action) = action {
            action.active_step_ids = vec!["field".into()];
        }
    }
    model
}

pub(in crate::study) fn mesh(order: ElementOrder, cells: usize) -> SolverMeshArtifact {
    match order {
        ElementOrder::Tet4 => runmat_meshing_core::fixtures::slab_mesh(cells),
        ElementOrder::Tet10 => runmat_meshing_core::fixtures::quadratic_slab_mesh(cells),
    }
}
pub(in crate::study) fn compiled(
    model: AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> CompiledAnalysisStudy {
    let properties = crate::properties::compile_property_map(&model, Some(mesh)).unwrap();
    CompiledAnalysisStudy {
        module_path: "tests/maxwell.fea".into(),
        study_id: "maxwell".into(),
        study_inputs: vec![],
        assessments: Vec::new(),
        outputs: vec![],
        model,
        properties,
        solver_mesh_digest: mesh.canonical_digest,
    }
}
pub(in crate::study) fn policy() -> MaxwellStudyPlanningPolicy {
    MaxwellStudyPlanningPolicy {}
}

#[test]
fn preserves_original_physics_and_tightest_authored_relative_target() {
    let mesh = mesh(ElementOrder::Tet4, 2);
    let mut study = compiled(model(), &mesh);
    let mut criterion = study.model.sequence.steps[0]
        .accuracy
        .criteria
        .iter()
        .find(|criterion| criterion.contribution == ErrorContributionKind::Algebraic)
        .unwrap()
        .clone();
    criterion.criterion_id = "tighter-maxwell".into();
    criterion.tolerance = AccuracyTolerance::Relative(3e-12);
    study.model.sequence.steps[0]
        .accuracy
        .criteria
        .push(criterion);
    let before = study.clone();
    let planned = plan_maxwell_study(&study, policy()).unwrap();
    assert_eq!(planned.model, before.model);
    assert_eq!(planned.step_id, before.model.sequence.steps[0].step_id);
    assert_eq!(planned.relative_residual_target, 3e-12);
    assert_eq!(study, before);
}

#[test]
fn accuracy_static_sequence_frequency_and_restart_fail_closed() {
    let mesh = mesh(ElementOrder::Tet4, 2);
    let study = compiled(model(), &mesh);
    let mut missing_target = study.clone();
    missing_target.model.sequence.steps[0]
        .accuracy
        .criteria
        .clear();
    assert!(matches!(
        plan_maxwell_study(&missing_target, policy()),
        Err(MaxwellStudyPlanningError::MissingRelativeAlgebraicTarget)
    ));
    for target in [0.0, -1.0, 1.0, f64::INFINITY, f64::NAN] {
        let mut changed = study.clone();
        for criterion in &mut changed.model.sequence.steps[0].accuracy.criteria {
            if criterion.contribution == ErrorContributionKind::Algebraic {
                criterion.tolerance = AccuracyTolerance::Relative(target);
            }
        }
        assert!(plan_maxwell_study(&changed, policy()).is_err());
    }
    let mut static_intent = study.clone();
    static_intent.model.electromagnetic = Some(ElectromagneticDomain::Static {
        equation: StaticElectricalEquation::SteadyConduction,
    });
    let mut sweep = study.clone();
    sweep.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![1e8, 2e8]);
    let mut foreign_frequency = study.clone();
    foreign_frequency.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![2e8]);
    let mut restart = study.clone();
    restart.model.sequence.steps[0].restart = RestartPolicy::ResumeCheckpoint {
        checkpoint_id: "not-qualified".into(),
        required_state_digest: "not-qualified".into(),
    };
    let mut sequence = study.clone();
    sequence
        .model
        .sequence
        .steps
        .push(sequence.model.sequence.steps[0].clone());
    // Several authored frequencies are one case each, not an error.
    assert!(plan_maxwell_study(&sweep, policy()).is_ok());
    // A domain reference frequency other than the analysis frequency only names the domain.
    assert!(plan_maxwell_study(&foreign_frequency, policy()).is_ok());
    for changed in [static_intent, restart, sequence] {
        assert!(plan_maxwell_study(&changed, policy()).is_err());
    }
}
