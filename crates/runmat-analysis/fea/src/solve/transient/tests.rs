use super::*;
use crate::{
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    fixtures::{fixture_model, FixtureId},
};
use std::cell::RefCell;

fn with_load_at(summary: &AssemblySummary, dof: usize, value: f64) -> AssemblySummary {
    let mut changed = summary.clone();
    let mut load = summary.operator.load().to_vec();
    load[dof] = value;
    changed.operator = crate::algebra::AssembledRealSystem::try_new(
        summary.operator.operators().clone(),
        summary.operator.constrained_dofs().to_vec(),
        load,
    )
    .unwrap();
    changed
}

fn with_scaled_stiffness(summary: &AssemblySummary, scale: f64) -> AssemblySummary {
    let mut changed = summary.clone();
    let source = summary.operator.operators();
    let stiffness = crate::algebra::RealSparseMatrix::scaled_sum(
        &[(scale, source.stiffness())],
        crate::algebra::OperatorRole::Stiffness,
        source.stiffness().properties(),
    )
    .unwrap();
    let operators = crate::algebra::CanonicalRealOperators::try_new(
        stiffness,
        source.mass().clone(),
        source.damping().clone(),
    )
    .unwrap();
    changed.operator = crate::algebra::AssembledRealSystem::try_new(
        operators,
        summary.operator.constrained_dofs().to_vec(),
        summary.operator.load().to_vec(),
    )
    .unwrap();
    changed
}

#[test]
fn invalid_fixed_step_and_residual_policies_fail_closed() {
    let empty_grid = TransientSolveOptions {
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 0 },
        ..TransientSolveOptions::default()
    };
    assert!(matches!(
        validate_options::<Infallible>(&empty_grid, 0.0, 1.0),
        Err(TransientSolveError::InvalidOptions(_))
    ));
    let invalid_residual = TransientSolveOptions {
        residual_target: f64::NAN,
        ..TransientSolveOptions::default()
    };
    assert!(matches!(
        validate_options::<Infallible>(&invalid_residual, 0.0, 1.0),
        Err(TransientSolveError::InvalidOptions(_))
    ));
}

#[test]
fn time_varying_structural_operator_is_rejected_before_advancing_state() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let step = &model.sequence.steps[0];
    let excitations =
        realize_step_excitations(&model, step, ExcitationSamplePoint::Static).unwrap();
    let summary = assemble_structural_system(
        &model,
        &excitations,
        None,
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let changed = with_scaled_stiffness(&summary, 1.01);

    let result = solve_transient_system_with_assembly(
        &summary,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
            ..TransientSolveOptions::default()
        },
        ComputeBackend::Cpu,
        TransientTimeDomain {
            start_time_s: 0.0,
            end_time_s: 1.0,
            step_boundaries: &[],
        },
        vec![0.0; summary.dof_count],
        vec![0.0; summary.dof_count],
        |_, _| Ok::<_, Infallible>(changed.clone()),
    );

    assert!(matches!(
        result,
        Err(TransientSolveError::StructuralSystemChanged {
            component: "stiffness operator"
        })
    ));
}

#[test]
fn orchestration_publishes_native_second_order_state_and_physical_energy() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let step = &model.sequence.steps[0];
    let excitations =
        realize_step_excitations(&model, step, ExcitationSamplePoint::Static).unwrap();
    let summary = assemble_structural_system(
        &model,
        &excitations,
        None,
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let result = solve_transient_system(
        &summary,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 20 },
            ..TransientSolveOptions::default()
        },
        ComputeBackend::Cpu,
    )
    .unwrap();

    assert_eq!(result.time_points_s.len(), 21);
    assert_eq!(result.displacement_snapshots.len(), 21);
    assert_eq!(result.velocity_snapshots.len(), 21);
    assert_eq!(result.acceleration_snapshots.len(), 21);
    assert_eq!(result.strain_energy_snapshots.len(), 21);
    assert_eq!(result.kinetic_energy_snapshots.len(), 21);
    assert!(result.residual_norms.iter().all(|value| value.is_finite()));
    assert!(result
        .strain_energy_snapshots
        .iter()
        .chain(&result.kinetic_energy_snapshots)
        .all(|value| value.is_finite() && *value >= -1.0e-12));
    assert!(result
        .velocity_snapshots
        .iter()
        .skip(1)
        .any(|snapshot| snapshot.iter().any(|value| value.abs() > 1.0e-12)));
    assert_eq!(
        result.solver_method,
        SolverMethod::NewmarkAverageAccelerationPcg
    );
}

#[test]
fn discontinuity_refreshes_acceleration_from_the_right_limit_before_continuing() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let step = &model.sequence.steps[0];
    let excitations =
        realize_step_excitations(&model, step, ExcitationSamplePoint::Static).unwrap();
    let summary = assemble_structural_system(
        &model,
        &excitations,
        None,
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let free_dof = summary
        .operator
        .constrained_dofs()
        .iter()
        .position(|constrained| !constrained)
        .unwrap();
    let right_limit = with_load_at(
        &summary,
        free_dof,
        summary.operator.load()[free_dof] + 100.0,
    );
    let boundaries = canonical_transient_step_boundaries(&[0.25], &[], 0.0, 1.0).unwrap();
    let samples = RefCell::new(Vec::new());
    let result = solve_transient_system_with_assembly(
        &summary,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
            ..TransientSolveOptions::default()
        },
        ComputeBackend::Cpu,
        TransientTimeDomain {
            start_time_s: 0.0,
            end_time_s: 1.0,
            step_boundaries: &boundaries,
        },
        vec![0.0; summary.dof_count],
        vec![0.0; summary.dof_count],
        |time_s, approach| {
            samples.borrow_mut().push((time_s, approach));
            Ok::<_, Infallible>(if time_s >= 0.25 && approach == SignalSampleApproach::At {
                right_limit.clone()
            } else {
                summary.clone()
            })
        },
    )
    .unwrap();

    assert_eq!(
        samples.into_inner(),
        vec![
            (0.25, SignalSampleApproach::LeftLimit),
            (0.25, SignalSampleApproach::At),
            (0.75, SignalSampleApproach::At),
            (1.0, SignalSampleApproach::At),
        ]
    );
    let jump_index = result
        .time_points_s
        .iter()
        .position(|time| *time == 0.25)
        .unwrap();
    assert_eq!(
        result.assembly_snapshots[jump_index].operator.load()[free_dof],
        right_limit.operator.load()[free_dof]
    );
    let expected = solve_initial_acceleration(
        &right_limit,
        result.displacement_snapshots[jump_index].clone(),
        result.velocity_snapshots[jump_index].clone(),
        LinearAlgebraBackendKind::CpuReference,
        TransientSolveOptions::default().residual_target,
    )
    .unwrap();
    assert_eq!(
        result.acceleration_snapshots[jump_index],
        expected.state().acceleration()
    );
    assert_eq!(
        result.residual_norms[jump_index],
        expected.projected_residual().relative_norm()
    );
}

#[test]
fn endpoint_discontinuity_publishes_the_right_limit_state() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let step = &model.sequence.steps[0];
    let excitations =
        realize_step_excitations(&model, step, ExcitationSamplePoint::Static).unwrap();
    let summary = assemble_structural_system(
        &model,
        &excitations,
        None,
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let free_dof = summary
        .operator
        .constrained_dofs()
        .iter()
        .position(|constrained| !constrained)
        .unwrap();
    let right_limit = with_load_at(
        &summary,
        free_dof,
        summary.operator.load()[free_dof] + 100.0,
    );
    let boundaries = canonical_transient_step_boundaries(&[1.0], &[], 0.0, 1.0).unwrap();
    let samples = RefCell::new(Vec::new());
    let result = solve_transient_system_with_assembly(
        &summary,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
            ..TransientSolveOptions::default()
        },
        ComputeBackend::Cpu,
        TransientTimeDomain {
            start_time_s: 0.0,
            end_time_s: 1.0,
            step_boundaries: &boundaries,
        },
        vec![0.0; summary.dof_count],
        vec![0.0; summary.dof_count],
        |time_s, approach| {
            samples.borrow_mut().push((time_s, approach));
            Ok::<_, Infallible>(if time_s == 1.0 && approach == SignalSampleApproach::At {
                right_limit.clone()
            } else {
                summary.clone()
            })
        },
    )
    .unwrap();

    assert_eq!(
        samples.into_inner(),
        vec![
            (0.5, SignalSampleApproach::At),
            (1.0, SignalSampleApproach::LeftLimit),
            (1.0, SignalSampleApproach::At),
        ]
    );
    let last = result.time_points_s.len() - 1;
    assert_eq!(result.time_points_s[last], 1.0);
    assert_eq!(
        result.assembly_snapshots[last].operator.load()[free_dof],
        right_limit.operator.load()[free_dof]
    );
    let expected = solve_initial_acceleration(
        &right_limit,
        result.displacement_snapshots[last].clone(),
        result.velocity_snapshots[last].clone(),
        LinearAlgebraBackendKind::CpuReference,
        TransientSolveOptions::default().residual_target,
    )
    .unwrap();
    assert_eq!(
        result.acceleration_snapshots[last],
        expected.state().acceleration()
    );
    assert_eq!(
        result.residual_norms[last],
        expected.projected_residual().relative_norm()
    );
}

#[test]
fn assembled_free_oscillator_converges_quadratically_in_native_state() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model
        .constraints
        .push(runmat_analysis_core::AnalysisConstraint::Structural(
            runmat_analysis_core::StructuralConstraint {
                constraint_id: "second_synthetic_anchor".into(),
                frame: runmat_analysis_core::CoordinateFrame::Global,
                kind: runmat_analysis_core::StructuralConstraintKind::Fixed {
                    target: runmat_analysis_core::RegionTarget::authored("second_anchor"),
                },
            },
        ));
    let step = &model.sequence.steps[0];
    let excitations =
        realize_step_excitations(&model, step, ExcitationSamplePoint::Static).unwrap();
    let summary = assemble_structural_system(
        &model,
        &excitations,
        None,
        StructuralFormulationRequirement::ExperimentalSyntheticOperator,
    )
    .expect("the oscillator reference must exercise an admitted assembled operator");
    assert_eq!(summary.operator.constrained_dofs(), &[true, false, true]);

    let stiffness = summary.operator.operators().stiffness().get(1, 1);
    let mass = summary.operator.operators().mass().get(1, 1);
    let damping = summary.operator.operators().damping().get(1, 1);
    assert!(stiffness > 0.0 && mass > 0.0);
    assert_eq!(damping, 0.0);
    let omega = (stiffness / mass).sqrt();
    let end_time_s = 2.0 * std::f64::consts::PI / omega;
    let initial_displacement = 0.75;
    let initial_velocity = -0.2 * omega;
    let exact = oscillator_state(omega, end_time_s, initial_displacement, initial_velocity);

    let errors = [16, 32, 64].map(|interval_count| {
        let result = solve_transient_system_with_assembly(
            &summary,
            TransientSolveOptions {
                time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count },
                residual_target: 1.0e-10,
                ..TransientSolveOptions::default()
            },
            ComputeBackend::Cpu,
            TransientTimeDomain {
                start_time_s: 0.0,
                end_time_s,
                step_boundaries: &[],
            },
            vec![0.0, initial_displacement, 0.0],
            vec![0.0, initial_velocity, 0.0],
            |_, _| Ok::<_, Infallible>(summary.clone()),
        )
        .expect("the assembled oscillator must solve at every refinement");
        assert_eq!(result.time_points_s.len(), interval_count + 1);
        assert_eq!(result.displacement_snapshots.len(), interval_count + 1);
        assert_eq!(result.velocity_snapshots.len(), interval_count + 1);
        assert_eq!(result.acceleration_snapshots.len(), interval_count + 1);
        let last = interval_count;
        normalized_native_state_error(
            omega,
            [
                result.displacement_snapshots[last][1],
                result.velocity_snapshots[last][1],
                result.acceleration_snapshots[last][1],
            ],
            exact,
        )
    });

    let coarse_order = (errors[0] / errors[1]).log2();
    let fine_order = (errors[1] / errors[2]).log2();
    assert!(
        coarse_order > 1.9,
        "coarse refinement order was {coarse_order}, errors={errors:?}"
    );
    assert!(
        fine_order > 1.9,
        "fine refinement order was {fine_order}, errors={errors:?}"
    );
    assert!(errors[2] < 1.0e-2, "fine native-state error={}", errors[2]);
}

fn oscillator_state(omega: f64, time_s: f64, u0: f64, v0: f64) -> [f64; 3] {
    let phase = omega * time_s;
    let displacement = u0 * phase.cos() + v0 / omega * phase.sin();
    let velocity = -u0 * omega * phase.sin() + v0 * phase.cos();
    [displacement, velocity, -omega * omega * displacement]
}

fn normalized_native_state_error(omega: f64, actual: [f64; 3], exact: [f64; 3]) -> f64 {
    let displacement_error = actual[0] - exact[0];
    let velocity_error = (actual[1] - exact[1]) / omega;
    let acceleration_error = (actual[2] - exact[2]) / (omega * omega);
    displacement_error
        .hypot(velocity_error)
        .hypot(acceleration_error)
}
