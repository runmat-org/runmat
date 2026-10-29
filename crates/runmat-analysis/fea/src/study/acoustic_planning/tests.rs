use std::collections::BTreeSet;

use runmat_analysis_core::{
    AcousticConstraint, AcousticConstraintKind, AnalysisBoundary, AnalysisConstraint,
    AnalysisProcedure, AnalysisSequence, BoundaryOperator, IndependentDomain,
    MaterialAcousticModel, PhysicsFamily, PrescribedValue, Pressure, ProcedureStep,
    PropertyRegionTarget, RegionTarget, RestartPolicy, SpecificAcousticImpedance,
};
use runmat_meshing_core::StableDigest;

use crate::{
    fixtures::{fixture_model, replace_fixture_excitations, FixtureId},
    properties::compile_property_map,
    topology::test_meshes::slab_mesh,
};

use super::*;

fn compiled_duct() -> CompiledAnalysisStudy {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    replace_fixture_excitations(&mut model, Vec::new());
    model.constraints.clear();
    model.boundaries.clear();
    model.materials[0].acoustic = Some(MaterialAcousticModel {
        speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(2.0),
        damping_ratio: runmat_types::physical::DampingRatio::new(0.0),
    });
    model.materials[0].mass_density_kg_per_m3 =
        (Some(3.0)).map(runmat_types::physical::MassDensity::new);
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    let mut step = ProcedureStep::acoustic_harmonic("acoustic");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![0.2]);
    step.accuracy.criteria[0].tolerance = AccuracyTolerance::Relative(2.0e-9);
    model.sequence = AnalysisSequence::from_steps("duct", vec![step]);
    model
        .constraints
        .push(AnalysisConstraint::Acoustic(AcousticConstraint {
            constraint_id: "pressure".into(),
            kind: AcousticConstraintKind::PrescribedPressure {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_left",
                )),
                value: PrescribedValue::Constant(Pressure::new(1.0)),
            },
        }));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "matched".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        operator: BoundaryOperator::AcousticImpedance {
            impedance: SpecificAcousticImpedance::new(6.0),
        },
    });
    let properties = compile_property_map(&model, None).unwrap();
    CompiledAnalysisStudy {
        module_path: "tests/acoustic-duct.fea".into(),
        study_id: "duct".into(),
        study_inputs: Vec::new(),
        assessments: Vec::new(),
        outputs: Vec::new(),
        model,
        properties,
        solver_mesh_digest: StableDigest::ZERO,
    }
}

#[test]
fn plans_single_canonical_frequency_in_shared_order() {
    let study = compiled_duct();
    let plan = plan_acoustic_study(&study, AcousticStudyPlanningPolicy::standard()).unwrap();
    assert_eq!(plan.frequency_plan().frequencies_hz(), &[0.2]);
    assert_eq!(plan.frequency_plan().reference_frequency_hz(), 0.2);
    assert_eq!(plan.frequencies().len(), 1);
    for (ordinal, planned) in plan.frequencies().iter().enumerate() {
        assert_eq!(planned.entry().ordinal() as usize, ordinal);
        assert_eq!(
            planned.entry().frequency_hz(),
            planned.workload().frequency.canonical_value()
        );
        assert_eq!(planned.workload().model, study.model);
        assert_eq!(planned.workload().solver.relative_tolerance, 2.0e-9);
        plan.frequency_plan()
            .validate_entry(planned.entry())
            .unwrap();
        planned.workload().execute(&slab_mesh(2)).unwrap();
    }
}

#[test]
fn rejects_lifecycle_family_and_procedure_substitution() {
    let mut study = compiled_duct();
    study.model.sequence.steps[0].restart = RestartPolicy::ResumeCheckpoint {
        checkpoint_id: "checkpoint".into(),
        required_state_digest: "state".into(),
    };
    assert!(matches!(
        plan_acoustic_study(&study, AcousticStudyPlanningPolicy::standard()),
        Err(AcousticStudyPlanningError::UnsupportedSequence)
    ));

    study = compiled_duct();
    study.model.sequence.steps[0].family = PhysicsFamily::Thermal;
    assert!(matches!(
        plan_acoustic_study(&study, AcousticStudyPlanningPolicy::standard()),
        Err(AcousticStudyPlanningError::UnsupportedProcedure)
            | Err(AcousticStudyPlanningError::Compilation(_))
            | Err(AcousticStudyPlanningError::Workload(_))
    ));

    study = compiled_duct();
    study.model.sequence.steps[0].procedure = AnalysisProcedure::Static;
    assert!(matches!(
        plan_acoustic_study(&study, AcousticStudyPlanningPolicy::standard()),
        Err(AcousticStudyPlanningError::UnsupportedProcedure)
            | Err(AcousticStudyPlanningError::Compilation(_))
            | Err(AcousticStudyPlanningError::Workload(_))
    ));
}

#[test]
fn rejects_missing_state() {
    let mut study = compiled_duct();
    study.model.sequence.steps[0].state_outputs.clear();
    assert!(matches!(
        plan_acoustic_study(&study, AcousticStudyPlanningPolicy::standard()),
        Err(AcousticStudyPlanningError::UnsupportedStatePublication)
    ));
}

#[test]
fn plans_sweep_with_original_authored_model_and_entry_scoped_states() {
    let mut study = compiled_duct();
    study.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![0.2, 0.4]);
    let plan = plan_acoustic_study(&study, AcousticStudyPlanningPolicy::standard()).unwrap();
    assert_eq!(plan.frequency_plan().frequencies_hz(), &[0.2, 0.4]);
    let mut sources = BTreeSet::new();
    for planned in plan.frequencies() {
        assert_eq!(planned.workload().model, study.model);
        let result = planned.workload().execute(&slab_mesh(2)).unwrap();
        let requests = result.state_publication_requests().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(sources.insert(result.source_result_identity().unwrap()));
    }
}

#[test]
fn rejects_invalid_policy_and_formulation_mismatch() {
    let study = compiled_duct();
    let mut policy = AcousticStudyPlanningPolicy::standard();
    policy.contraction_limit = 1.0;
    assert!(matches!(
        plan_acoustic_study(&study, policy),
        Err(AcousticStudyPlanningError::InvalidNumericalPolicy)
    ));
}

#[test]
fn rejects_sweep_exceeding_the_unchanged_authored_sample_ceiling() {
    let mut study = compiled_duct();
    study.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![0.1, 0.2, 0.3]);
    study.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_samples = 2;
    assert!(matches!(
        plan_acoustic_study(&study, AcousticStudyPlanningPolicy::standard()),
        Err(AcousticStudyPlanningError::SampleLimitExceeded)
    ));
}
