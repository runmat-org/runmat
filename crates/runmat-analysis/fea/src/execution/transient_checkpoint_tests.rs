use crate::execution::{
    OperatorIdentity, SolverExcitationIdentity, StructuralTransientBlockRequest,
    StructuralTransientScheduleArrival, StructuralTransientSchedulePoint,
};

use super::*;

fn schedule() -> StructuralTransientSchedule {
    StructuralTransientSchedule::new(vec![
        StructuralTransientSchedulePoint {
            time_s: runmat_types::physical::Time::new(0.0),
            arrival: StructuralTransientScheduleArrival::Initial,
            continuation: StructuralTransientContinuationApproach::At,
        },
        StructuralTransientSchedulePoint {
            time_s: runmat_types::physical::Time::new(0.5),
            arrival: StructuralTransientScheduleArrival::LeftLimit,
            continuation: StructuralTransientContinuationApproach::RightLimit,
        },
        StructuralTransientSchedulePoint {
            time_s: runmat_types::physical::Time::new(1.0),
            arrival: StructuralTransientScheduleArrival::Continuous,
            continuation: StructuralTransientContinuationApproach::At,
        },
    ])
    .unwrap()
}

pub(in crate::execution) fn scope() -> StructuralTransientCheckpointScope {
    StructuralTransientCheckpointScope {
        study: StudyIdentity::derive("test-study", &1_u8).unwrap(),
        step: StepIdentity::derive("test-step", &2_u8).unwrap(),
        execution_plan: ExecutionPlanIdentity::derive("test-execution", &3_u8).unwrap(),
        source_workload: AnalysisWorkloadIdentity::derive("test-workload", &4_u8).unwrap(),
        numerical_plan: NumericalPlanIdentity::derive("test-numerical", &5_u8).unwrap(),
        discretization: DiscretizationIdentity::derive("test-discretization", &6_u8).unwrap(),
        constraint_plan: ConstraintPlanIdentity::derive("test-constraints", &7_u8).unwrap(),
        stiffness_operator: StructuralStiffnessOperatorIdentity::new(
            OperatorIdentity::derive("test-stiffness", &8_u8).unwrap(),
        )
        .unwrap(),
        mass_operator: StructuralMassOperatorIdentity::new(
            OperatorIdentity::derive("test-mass", &9_u8).unwrap(),
        )
        .unwrap(),
        damping_operator: StructuralDampingOperatorIdentity::new(
            OperatorIdentity::derive("test-damping", &10_u8).unwrap(),
        )
        .unwrap(),
        excitations: StructuralExcitationSetIdentity::new(vec![SolverExcitationIdentity::derive(
            "test-excitation",
            &11_u8,
        )
        .unwrap()])
        .unwrap(),
        dof_layout: StructuralDofLayoutIdentity::new(
            DiscretizationIdentity::derive("test-discretization", &6_u8).unwrap(),
            runmat_execution::Digest::sha256(b"test-dof-layout"),
        )
        .unwrap(),
        formulation: StructuralFormulationIdentity::new(
            DiscretizationIdentity::derive("test-discretization", &6_u8).unwrap(),
            runmat_execution::Digest::sha256(b"test-formulation"),
        )
        .unwrap(),
        kinematics: StructuralKinematicsIdentity::new(
            ConstraintPlanIdentity::derive("test-constraints", &7_u8).unwrap(),
            StructuralExcitationSetIdentity::new(vec![SolverExcitationIdentity::derive(
                "test-excitation",
                &11_u8,
            )
            .unwrap()])
            .unwrap(),
            runmat_execution::Digest::sha256(b"test-kinematics"),
        )
        .unwrap(),
        dof_count: 2,
        coordinate_space: StructuralTransientCoordinateSpace::Direct,
        integrator: StructuralTransientIntegrator::NewmarkAverageAcceleration,
        precision: StructuralTransientPrecision::Real64,
        solver_options: StructuralTransientSolverOptionsIdentity::new(
            runmat_execution::Digest::sha256(b"test-solver-options"),
        )
        .unwrap(),
        schedule: schedule(),
    }
}

pub(in crate::execution) fn checkpoint(
    predecessor: Option<&StructuralTransientCheckpoint>,
    ordinal: u64,
) -> StructuralTransientCheckpoint {
    StructuralTransientCheckpoint::new(
        StructuralTransientCheckpointInput {
            scope: scope(),
            block_ordinal: predecessor.map_or(0, |value| value.block_ordinal + 1),
            continuation: StructuralTransientContinuationPoint {
                schedule_ordinal: ordinal,
                time_s: runmat_types::physical::Time::new(if ordinal == 1 { 0.5 } else { 1.0 }),
                approach: if ordinal == 1 {
                    StructuralTransientContinuationApproach::RightLimit
                } else {
                    StructuralTransientContinuationApproach::At
                },
            },
            state: StructuralTransientContinuationState {
                displacement: vec![ordinal as f64, 0.0],
                velocity: vec![0.5, 0.0],
                acceleration: vec![1.0, 0.0],
            },
            numerical_state: StructuralTransientNumericalState::Direct,
            balance: StructuralTransientBalanceAccumulator {
                initial_mechanical_energy_j: runmat_types::physical::Energy::new(1.0),
                applied_load_work_j: runmat_types::physical::Energy::new(ordinal as f64),
                support_work_j: runmat_types::physical::Energy::new(-0.25 * ordinal as f64),
                damping_dissipation_j: runmat_types::physical::Energy::new(0.1 * ordinal as f64),
            },
            counters: StructuralTransientSolverCounters {
                host_sync_count: ordinal,
                ..StructuralTransientSolverCounters::default()
            },
            observation_cursors: Vec::new(),
        },
        predecessor,
    )
    .unwrap()
}

#[test]
fn schedule_binds_arrival_to_the_exact_continuation_side() {
    let canonical = schedule();
    assert_eq!(
        StructuralTransientSchedule::from_canonical_bytes(&canonical.canonical_bytes().unwrap())
            .unwrap(),
        canonical
    );
    let mut invalid = schedule();
    invalid.points[1].continuation = StructuralTransientContinuationApproach::At;
    assert!(invalid.validate().is_err());
    invalid = schedule();
    invalid.points.swap(1, 2);
    assert!(invalid.validate().is_err());
}

#[test]
fn checkpoint_round_trips_and_preserves_right_limit_continuation() {
    let first = checkpoint(None, 1);
    let decoded =
        StructuralTransientCheckpoint::from_canonical_bytes(&first.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(decoded, first);
    assert_eq!(
        decoded.continuation.approach,
        StructuralTransientContinuationApproach::RightLimit
    );

    let second = checkpoint(Some(&first), 2);
    second.validate_against_predecessor(&first).unwrap();
    assert_eq!(second.predecessor, Some(first.identity));
}

#[test]
fn checkpoint_rejects_state_schedule_scope_and_lineage_substitution() {
    let first = checkpoint(None, 1);
    let mut invalid = first.clone();
    invalid.state.velocity.pop();
    assert!(invalid.validate().is_err());

    let mut invalid = first.clone();
    invalid.continuation.approach = StructuralTransientContinuationApproach::At;
    assert!(invalid.validate().is_err());

    let mut invalid = first.clone();
    invalid.scope.mass_operator =
        StructuralMassOperatorIdentity::new(OperatorIdentity::derive("other-mass", &1_u8).unwrap())
            .unwrap();
    assert!(invalid.validate().is_err());

    let second = checkpoint(Some(&first), 2);
    let mut other_scope = scope();
    other_scope.kinematics = StructuralKinematicsIdentity::new(
        other_scope.constraint_plan,
        other_scope.excitations,
        runmat_execution::Digest::sha256(b"other-kinematics"),
    )
    .unwrap();
    let other = StructuralTransientCheckpoint::new(
        StructuralTransientCheckpointInput {
            scope: other_scope,
            block_ordinal: 0,
            continuation: first.continuation.clone(),
            state: first.state.clone(),
            numerical_state: first.numerical_state.clone(),
            balance: first.balance.clone(),
            counters: first.counters.clone(),
            observation_cursors: first.observation_cursors.clone(),
        },
        None,
    )
    .unwrap();
    assert!(second.validate_against_predecessor(&other).is_err());
}

#[test]
fn block_request_is_exactly_bounded_by_its_predecessor_and_schedule() {
    let first = StructuralTransientBlockRequest::new(scope(), None, 1).unwrap();
    assert_eq!(first.start_schedule_ordinal, 0);
    assert_eq!(first.terminal_schedule_ordinal, 1);
    assert_eq!(
        StructuralTransientBlockRequest::from_canonical_bytes(&first.canonical_bytes().unwrap())
            .unwrap(),
        first
    );

    let checkpoint = checkpoint(None, 1);
    let second = StructuralTransientBlockRequest::new(scope(), Some(&checkpoint), 2).unwrap();
    assert_eq!(second.start_schedule_ordinal, 1);
    assert_eq!(second.block_ordinal, 1);

    assert!(StructuralTransientBlockRequest::new(scope(), Some(&checkpoint), 1).is_err());
    assert!(StructuralTransientBlockRequest::new(scope(), Some(&checkpoint), 3).is_err());
}

#[test]
fn checkpoint_scope_has_an_independent_canonical_object_contract() {
    let original = scope();
    let encoded = original.canonical_bytes().unwrap();
    let decoded = StructuralTransientCheckpointScope::from_canonical_bytes(&encoded).unwrap();
    assert_eq!(decoded, original);
    assert_eq!(decoded.identity().unwrap(), original.identity().unwrap());

    let checkpoint_bytes = checkpoint(None, 1).canonical_bytes().unwrap();
    assert!(StructuralTransientCheckpointScope::from_canonical_bytes(&checkpoint_bytes).is_err());

    let mut invalid = original;
    invalid.dof_count = 0;
    assert!(invalid.canonical_bytes().is_err());
}
