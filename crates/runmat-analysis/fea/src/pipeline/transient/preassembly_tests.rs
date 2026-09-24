use super::*;
use crate::fixtures::{fixture_model, FixtureId};

#[test]
fn scope_preparation_rejects_accepted_schedule_over_sample_ceiling_before_assembly() {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    model.sequence.steps[0].accuracy.resources.maximum_samples = 2;
    let options = TransientSolveOptions {
        time_step_policy: crate::solve::transient::StructuralTimeStepPolicy::Fixed {
            interval_count: 2,
        },
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        ..TransientSolveOptions::default()
    };
    let error = match prepare_transient_block_basis(&model, &options) {
        Ok(_) => panic!("over-ceiling transient schedule was admitted"),
        Err(error) => error,
    };
    assert!(
        matches!(&error, FeaRunError::InvalidModel(message) if message.contains("requires at least 3 samples") && message.contains("ceiling of 2")),
        "{error}"
    );
}

#[test]
fn motion_breakpoint_samples_count_against_transient_ceiling() {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4SupportMotionReference);
    model.sequence.steps[0].accuracy.resources.maximum_samples = 2;
    let options = TransientSolveOptions {
        time_step_policy: crate::solve::transient::StructuralTimeStepPolicy::Fixed {
            interval_count: 1,
        },
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        ..TransientSolveOptions::default()
    };
    let error = match prepare_transient_block_basis(&model, &options) {
        Ok(_) => panic!("over-ceiling motion schedule was admitted"),
        Err(error) => error,
    };
    assert!(
        matches!(&error, FeaRunError::InvalidModel(message) if message.contains("requires 3 samples") && message.contains("ceiling of 2")),
        "{error}"
    );
}

#[test]
fn direct_host_transient_checks_schedule_before_initial_assembly() {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    model.sequence.steps[0].accuracy.resources.maximum_samples = 2;
    let options = TransientSolveOptions {
        time_step_policy: crate::solve::transient::StructuralTimeStepPolicy::Fixed {
            interval_count: 2,
        },
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        ..TransientSolveOptions::default()
    };
    let error = match run_transient_with_options(&model, ComputeBackend::Cpu, options) {
        Ok(_) => panic!("over-ceiling direct transient schedule was admitted"),
        Err(error) => error,
    };
    assert!(
        matches!(&error, FeaRunError::InvalidModel(message) if message.contains("requires at least 3 samples") && message.contains("ceiling of 2")),
        "{error}"
    );
}

#[test]
fn direct_host_samples_follow_the_shared_preassembly_schedule() {
    let model = fixture_model(FixtureId::TransientLong);
    let options = TransientSolveOptions {
        time_step_policy: crate::solve::transient::StructuralTimeStepPolicy::Fixed {
            interval_count: 4,
        },
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        ..TransientSolveOptions::default()
    };
    let prepared = prepare_transient_preassembly(&model, &options).unwrap();
    let expected = prepared
        .schedule
        .points
        .iter()
        .map(|point| point.time_s)
        .collect::<Vec<_>>();
    let result = run_transient_with_options(&model, ComputeBackend::Cpu, options).unwrap();
    assert_eq!(result.time_points_s, expected);
}
