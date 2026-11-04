use super::*;
use crate::{
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    execution::StructuralTransientCoordinateSpace,
    fixtures::{fixture_model, FixtureId},
};
use std::cell::RefCell;

use super::effective_solve::solve_initial_acceleration;

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
            start_time_s: runmat_types::physical::Time::new(0.0),
            end_time_s: runmat_types::physical::Time::new(1.0),
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
    assert!(result.integration_intervals.is_empty());
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
            start_time_s: runmat_types::physical::Time::new(0.0),
            end_time_s: runmat_types::physical::Time::new(1.0),
            step_boundaries: &boundaries,
        },
        vec![0.0; summary.dof_count],
        vec![0.0; summary.dof_count],
        |time_s, approach| {
            samples.borrow_mut().push((time_s, approach));
            Ok::<_, Infallible>(
                if time_s >= 0.25 && approach != SignalSampleApproach::LeftLimit {
                    right_limit.clone()
                } else {
                    summary.clone()
                },
            )
        },
    )
    .unwrap();

    assert_eq!(
        samples.into_inner(),
        vec![
            (0.25, SignalSampleApproach::LeftLimit),
            (0.25, SignalSampleApproach::At),
            (0.25, SignalSampleApproach::RightLimit),
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
fn bounded_restart_at_a_jump_matches_uninterrupted_physical_histories() {
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
    let authored_point = with_load_at(&summary, free_dof, summary.operator.load()[free_dof] + 40.0);
    let boundaries = canonical_transient_step_boundaries(&[0.25], &[], 0.0, 1.0).unwrap();
    let options = TransientSolveOptions {
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
        ..TransientSolveOptions::default()
    };
    let time_domain = TransientTimeDomain {
        start_time_s: runmat_types::physical::Time::new(0.0),
        end_time_s: runmat_types::physical::Time::new(1.0),
        step_boundaries: &boundaries,
    };
    let assembly = |time_s: f64, approach: SignalSampleApproach| {
        Ok::<_, Infallible>(if time_s == 0.25 && approach == SignalSampleApproach::At {
            authored_point.clone()
        } else if time_s >= 0.25 && approach != SignalSampleApproach::LeftLimit {
            right_limit.clone()
        } else {
            summary.clone()
        })
    };
    let uninterrupted = solve_transient_system_with_assembly(
        &summary,
        options.clone(),
        ComputeBackend::Cpu,
        time_domain,
        vec![0.0; summary.dof_count],
        vec![0.0; summary.dof_count],
        assembly,
    )
    .unwrap();

    let schedule = canonical_transient_schedule(2, 0.0, 1.0, &boundaries).unwrap();
    let mut identity_physical_state =
        |_: f64,
         _: SignalSampleApproach,
         summary: &AssemblySummary,
         state: &state::SecondOrderState| {
            Ok::<_, TransientSolveError<Infallible>>((summary.clone(), state.clone()))
        };
    let mut strategy = advance::DirectTransientStepStrategy::new(
        &summary,
        LinearAlgebraBackendKind::CpuReference,
        options.residual_target,
    );
    let (state, initial) = advance::initialize::<Infallible>(
        &summary,
        &options,
        &schedule,
        vec![0.0; summary.dof_count],
        vec![0.0; summary.dof_count],
        &mut strategy,
        &mut identity_physical_state,
    )
    .unwrap();
    let mut restarted_assembly = |time_s: f64, approach: SignalSampleApproach| {
        Ok::<_, Infallible>(if time_s == 0.25 && approach == SignalSampleApproach::At {
            authored_point.clone()
        } else if time_s >= 0.25 && approach != SignalSampleApproach::LeftLimit {
            right_limit.clone()
        } else {
            summary.clone()
        })
    };
    let first = advance::advance(
        advance::TransientAdvanceContext {
            initial_summary: &summary,
            options: &options,
            backend_kind: LinearAlgebraBackendKind::CpuReference,
            schedule: &schedule,
            interval_retention: TransientIntervalRetention::DiscardAfterBalance,
        },
        state,
        1,
        &mut restarted_assembly,
        &mut identity_physical_state,
        &mut strategy,
    )
    .unwrap();
    assert_eq!(
        first.terminal.continuation_approach,
        SignalSampleApproach::RightLimit
    );
    let terminal = first.terminal;
    let resumed = advance::resume::<Infallible>(
        &schedule,
        advance::TransientResumeInput {
            schedule_ordinal: terminal.schedule_ordinal,
            numerical_state: terminal.numerical_continuation_state.clone(),
            physical_state: crate::execution::StructuralTransientContinuationState {
                displacement: terminal.physical_continuation_state.displacement().to_vec(),
                velocity: terminal.physical_continuation_state.velocity().to_vec(),
                acceleration: terminal.physical_continuation_state.acceleration().to_vec(),
            },
            continuation_approach:
                crate::execution::StructuralTransientContinuationApproach::RightLimit,
            integration_summary: terminal.integration_continuation_summary,
            physical_summary: terminal.physical_continuation_summary,
            balance: terminal.balance,
            accounting: advance::TransientStepStrategy::resumed_accounting(
                &strategy,
                terminal.accounting.counters,
            ),
        },
        summary.dof_count,
    )
    .unwrap();
    let last_ordinal = u64::try_from(schedule.points.len() - 1).unwrap();
    let second = advance::advance(
        advance::TransientAdvanceContext {
            initial_summary: &summary,
            options: &options,
            backend_kind: LinearAlgebraBackendKind::CpuReference,
            schedule: &schedule,
            interval_retention: TransientIntervalRetention::DiscardAfterBalance,
        },
        resumed,
        last_ordinal,
        &mut restarted_assembly,
        &mut identity_physical_state,
        &mut strategy,
    )
    .unwrap();
    let mut samples = first.samples;
    samples.extend(second.samples);

    assert_eq!(
        initial.state.displacement(),
        uninterrupted.displacement_snapshots[0]
    );
    assert_eq!(samples.len() + 1, uninterrupted.time_points_s.len());
    for (sample, index) in samples.iter().zip(1..) {
        assert_eq!(sample.schedule_ordinal as usize, index);
        assert_eq!(
            sample.time_s.canonical_value(),
            uninterrupted.time_points_s[index]
        );
        assert_eq!(
            sample.state.displacement(),
            uninterrupted.displacement_snapshots[index]
        );
        assert_eq!(
            sample.state.velocity(),
            uninterrupted.velocity_snapshots[index]
        );
        assert_eq!(
            sample.state.acceleration(),
            uninterrupted.acceleration_snapshots[index]
        );
        assert_eq!(
            sample.projected_residual_norm,
            uninterrupted.residual_norms[index]
        );
        assert_eq!(sample.reaction, uninterrupted.reaction_snapshots[index]);
        assert_eq!(
            sample.applied_load_work_j.canonical_value(),
            uninterrupted.external_work_j.canonical_values()[index]
        );
        assert_eq!(
            sample.support_work_j.canonical_value(),
            uninterrupted.support_work_j.canonical_values()[index]
        );
        assert_eq!(
            sample.damping_dissipation_j.canonical_value(),
            uninterrupted.damping_dissipation_j.canonical_values()[index]
        );
        assert_eq!(
            sample.energy_balance_residual_j.canonical_value(),
            uninterrupted.energy_balance_residual_j.canonical_values()[index]
        );
        assert_eq!(
            sample.dynamic_balance,
            uninterrupted.dynamic_balance_snapshots[index]
        );
    }
    assert_eq!(
        second.terminal.accounting.counters.host_sync_count,
        u64::from(uninterrupted.solver_host_sync_count)
    );
    assert_eq!(
        second.terminal.accounting.counters.device_apply_k_count,
        u64::from(uninterrupted.device_apply_k_count)
    );
}

#[test]
fn canonical_block_entrypoint_emits_a_restartable_segment_chain() {
    use crate::execution::{
        AnalysisWorkKind, AnalysisWorkloadIdentity, ConstraintPlanIdentity, DiscretizationIdentity,
        ExecutionPlanIdentity, NumericalPlanIdentity, OperatorIdentity, ProblemIdentity,
        SequenceIdentity, SignalMaterializationIdentity, SignalRealizationIdentity,
        SolverExcitationIdentity, StepIdentity, StructuralDampingOperatorIdentity,
        StructuralDofLayoutIdentity, StructuralExcitationSetIdentity,
        StructuralFormulationIdentity, StructuralKinematicsIdentity,
        StructuralMassOperatorIdentity, StructuralStiffnessOperatorIdentity,
        StructuralTransientCheckpointScope, StructuralTransientIntegrator,
        StructuralTransientPrecision, StructuralTransientSolverOptionsIdentity, StudyIdentity,
    };
    use runmat_execution::Digest;

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
    let options = TransientSolveOptions {
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
        ..TransientSolveOptions::default()
    };
    let schedule = canonical_transient_schedule(2, 0.0, 1.0, &boundaries).unwrap();
    let digest = Digest::sha256;
    let study = StudyIdentity::from_authored_digest(digest("study"), None).unwrap();
    let sequence = SequenceIdentity::new(digest("sequence")).unwrap();
    let step_identity = StepIdentity::new(sequence, digest("step")).unwrap();
    let constraints =
        ConstraintPlanIdentity::new(step_identity, digest("mesh"), digest("constraints")).unwrap();
    let materialization =
        SignalMaterializationIdentity::new(digest("signal"), None, digest("policy")).unwrap();
    let realization = SignalRealizationIdentity::new(materialization, digest("content")).unwrap();
    let excitation = SolverExcitationIdentity::new(
        step_identity,
        realization,
        digest("target"),
        digest("projection"),
    )
    .unwrap();
    let excitation_set = StructuralExcitationSetIdentity::new(vec![excitation]).unwrap();
    let problem = ProblemIdentity::new(
        step_identity,
        digest("model"),
        None,
        constraints,
        vec![excitation],
    )
    .unwrap();
    let discretization =
        DiscretizationIdentity::new(problem, digest("mesh"), digest("formulation")).unwrap();
    let operator = OperatorIdentity::new(discretization, digest("operators")).unwrap();
    let numerical = NumericalPlanIdentity::new(operator, digest("solver")).unwrap();
    let execution = ExecutionPlanIdentity::new(numerical, digest("execution")).unwrap();
    let source_workload = AnalysisWorkloadIdentity::new(
        study,
        step_identity,
        execution,
        AnalysisWorkKind::SolveStructuralTransientBlock,
        digest("work"),
        Vec::new(),
    )
    .unwrap();
    let scope = StructuralTransientCheckpointScope {
        study,
        step: step_identity,
        execution_plan: execution,
        source_workload,
        numerical_plan: numerical,
        discretization,
        constraint_plan: constraints,
        stiffness_operator: StructuralStiffnessOperatorIdentity::new(operator).unwrap(),
        mass_operator: StructuralMassOperatorIdentity::new(operator).unwrap(),
        damping_operator: StructuralDampingOperatorIdentity::new(operator).unwrap(),
        excitations: excitation_set,
        dof_layout: StructuralDofLayoutIdentity::new(discretization, digest("layout")).unwrap(),
        formulation: StructuralFormulationIdentity::new(discretization, digest("recovery"))
            .unwrap(),
        kinematics: StructuralKinematicsIdentity::new(
            constraints,
            excitation_set,
            digest("kinematics"),
        )
        .unwrap(),
        dof_count: summary.dof_count,
        coordinate_space: StructuralTransientCoordinateSpace::Direct,
        integrator: StructuralTransientIntegrator::NewmarkAverageAcceleration,
        precision: StructuralTransientPrecision::Real64,
        solver_options: StructuralTransientSolverOptionsIdentity::new(digest("options")).unwrap(),
        schedule,
    };
    let first_request =
        crate::execution::StructuralTransientBlockRequest::new(scope.clone(), None, 1).unwrap();
    let mut make_assembly = |time_s: f64, approach: SignalSampleApproach| {
        Ok::<_, Infallible>(
            if time_s >= 0.25 && approach != SignalSampleApproach::LeftLimit {
                right_limit.clone()
            } else {
                summary.clone()
            },
        )
    };
    let first = solve_structural_transient_block_with_assembly(
        StructuralTransientBlockSolveInput {
            request: &first_request,
            predecessor: None,
            initial_summary: &summary,
            options: &options,
            backend: ComputeBackend::Cpu,
            initial_conditions: Some(StructuralTransientInitialConditions {
                displacement: vec![0.0; summary.dof_count],
                velocity: vec![0.0; summary.dof_count],
            }),
            observation_cursors: Vec::new(),
        },
        &mut make_assembly,
    )
    .unwrap();
    assert_eq!(first.samples().len(), 1);
    assert_eq!(
        first.manifest().terminal_checkpoint.continuation.approach,
        crate::execution::StructuralTransientContinuationApproach::RightLimit
    );
    assert_eq!(
        first.manifest().solver_evidence.counters,
        first.manifest().terminal_checkpoint.counters
    );

    let predecessor = &first.manifest().terminal_checkpoint;
    let second_request =
        crate::execution::StructuralTransientBlockRequest::new(scope, Some(predecessor), 3)
            .unwrap();
    let second = solve_structural_transient_block_with_assembly(
        StructuralTransientBlockSolveInput {
            request: &second_request,
            predecessor: Some(predecessor),
            initial_summary: &summary,
            options: &options,
            backend: ComputeBackend::Cpu,
            initial_conditions: None,
            observation_cursors: Vec::new(),
        },
        &mut make_assembly,
    )
    .unwrap();
    assert_eq!(second.samples().len(), 2);
    assert_eq!(
        second.manifest().request.start_counters,
        predecessor.counters
    );
    assert_eq!(
        second
            .manifest()
            .terminal_checkpoint
            .continuation
            .schedule_ordinal,
        3
    );

    let uninterrupted = solve_transient_system_with_assembly(
        &summary,
        options,
        ComputeBackend::Cpu,
        TransientTimeDomain {
            start_time_s: runmat_types::physical::Time::new(0.0),
            end_time_s: runmat_types::physical::Time::new(1.0),
            step_boundaries: &boundaries,
        },
        vec![0.0; summary.dof_count],
        vec![0.0; summary.dof_count],
        &mut make_assembly,
    )
    .unwrap();
    let samples = first
        .samples()
        .iter()
        .chain(second.samples())
        .collect::<Vec<_>>();
    for (sample, index) in samples.iter().zip(1..) {
        assert_eq!(
            sample.physical.state.displacement,
            uninterrupted.displacement_snapshots[index]
        );
        assert_eq!(
            sample.physical.state.velocity,
            uninterrupted.velocity_snapshots[index]
        );
        assert_eq!(
            sample.physical.state.acceleration,
            uninterrupted.acceleration_snapshots[index]
        );
        assert_eq!(
            sample.physical.applied_load_work_j.canonical_value(),
            uninterrupted.external_work_j.canonical_values()[index]
        );
        assert_eq!(
            sample.physical.support_work_j.canonical_value(),
            uninterrupted.support_work_j.canonical_values()[index]
        );
        assert_eq!(
            sample.physical.damping_dissipation_j.canonical_value(),
            uninterrupted.damping_dissipation_j.canonical_values()[index]
        );
        assert_eq!(
            sample.physical.dynamic_balance,
            uninterrupted.dynamic_balance_snapshots[index]
        );
    }
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
            start_time_s: runmat_types::physical::Time::new(0.0),
            end_time_s: runmat_types::physical::Time::new(1.0),
            step_boundaries: &boundaries,
        },
        vec![0.0; summary.dof_count],
        vec![0.0; summary.dof_count],
        |time_s, approach| {
            samples.borrow_mut().push((time_s, approach));
            Ok::<_, Infallible>(
                if time_s == 1.0 && approach != SignalSampleApproach::LeftLimit {
                    right_limit.clone()
                } else {
                    summary.clone()
                },
            )
        },
    )
    .unwrap();

    assert_eq!(
        samples.into_inner(),
        vec![
            (0.5, SignalSampleApproach::At),
            (1.0, SignalSampleApproach::LeftLimit),
            (1.0, SignalSampleApproach::At),
            (1.0, SignalSampleApproach::RightLimit),
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
                    target: runmat_analysis_core::RegionTarget::authored(
                        runmat_meshing_core::fixtures::fixture_face("second_anchor"),
                    ),
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
        StructuralFormulationRequirement::SyntheticOperator,
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
                start_time_s: runmat_types::physical::Time::new(0.0),
                end_time_s: runmat_types::physical::Time::new(end_time_s),
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

#[test]
fn equal_sized_reduced_states_cannot_change_their_retained_physical_basis() {
    use crate::algebra::{AssembledRealSystem, CanonicalRealOperators};
    use crate::assembly::physical_fields::StructuralReducedBasis;
    use crate::constraints::AffineConstraintReduction;
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let excitations = realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let full = assemble_structural_system(
        &model,
        &excitations,
        None,
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let mut mask = vec![false; full.dof_count];
    mask[0] = true;
    let original = AffineConstraintReduction::from_constrained_dofs(&mask);
    mask[0] = false;
    mask[full.dof_count - 1] = true;
    let foreign = AffineConstraintReduction::from_constrained_dofs(&mask);
    let operators = full.operator.operators();
    let reduced = CanonicalRealOperators::try_new(
        original
            .reduce_sparse_operator(operators.stiffness())
            .unwrap(),
        original.reduce_sparse_operator(operators.mass()).unwrap(),
        original
            .reduce_sparse_operator(operators.damping())
            .unwrap(),
    )
    .unwrap();
    let mut initial = full.clone();
    initial.dof_count = original.reduced_dof_count;
    initial.constrained_dof_count = 0;
    initial.physical_load = original.transpose_apply(&full.physical_load).unwrap();
    initial.constraint_matrix = None;
    initial.affine_constraint_reduction = None;
    initial.operator = AssembledRealSystem::try_new(
        reduced,
        vec![false; original.reduced_dof_count],
        initial.physical_load.clone(),
    )
    .unwrap();
    initial.structural_reduced_basis =
        Some(StructuralReducedBasis::try_new(&full.structural_dof_layout, original).unwrap());
    let mut substituted = initial.clone();
    substituted.structural_reduced_basis =
        Some(StructuralReducedBasis::try_new(&full.structural_dof_layout, foreign).unwrap());
    assert!(matches!(
        require_stable_system::<Infallible>(&initial, &substituted),
        Err(TransientSolveError::StructuralSystemChanged {
            component: "DOF layout"
        })
    ));
}
