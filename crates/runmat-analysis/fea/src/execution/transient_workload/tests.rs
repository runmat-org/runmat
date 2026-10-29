use runmat_analysis_core::{
    IndependentDomain, ProcedureStep, ProcedureStepId, ReferenceStatePolicy, SequenceDependency,
    SequenceDependencyReason, StateId, StateTransition, StateTransitionMethod,
    TopologyTransitionPolicy, TransitionId,
};

use crate::{
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    fixtures::{fixture_model, FixtureId},
    ComputeBackend, ModalSolveOptions, ModalTruncationPolicy,
};

use super::*;
use crate::execution::{
    CanonicalSemanticStateInput, CanonicalSemanticStatePayload, CanonicalStateTransitionInput,
    SolverResultIdentity, StructuralTransientBlockRequest, StructuralTransientContinuationApproach,
    StructuralTransientScheduleArrival, StructuralTransientSchedulePoint,
};

fn modal_case() -> (
    StructuralTransientWorkload,
    SolverMeshArtifact,
    CanonicalSemanticState,
    AssemblySummary,
    StructuralTransientSchedule,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let modal_step = ProcedureStep::structural_modal("modes");
    let mut transient_step =
        ProcedureStep::structural_modal_transient("response", "modes", "modal_basis");
    transient_step.independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 1.0,
        interval_count: 10,
    };
    model.sequence.steps = vec![modal_step, transient_step];
    model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("modes".into()),
        successor: ProcedureStepId("response".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("modal-basis-to-transient".into()),
        producer_step: ProcedureStepId("modes".into()),
        consumer_step: ProcedureStepId("response".into()),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }];
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let modal = crate::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: ModalTruncationPolicy::requested(3),
            solver_mesh: Some(mesh.clone()),
            ..ModalSolveOptions::default()
        },
    )
    .unwrap();
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0]).unwrap();
    let source_result =
        SolverResultIdentity::from_step_result(producer, modal.basis.identity().digest()).unwrap();
    let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: model.sequence.steps[0].state_outputs[0].clone(),
        topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
        source_result,
        payload: CanonicalSemanticStatePayload::ModalBasis {
            basis: Box::new(modal.basis),
        },
    })
    .unwrap();
    let basis_transition = CanonicalStateTransition::new(CanonicalStateTransitionInput {
        sequence,
        producer,
        consumer: StepIdentity::from_solver_contract(sequence, &model.sequence.steps[1]).unwrap(),
        authored: &model.sequence.transitions[0],
        source: &state,
        target_topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
    })
    .unwrap();
    let workload = StructuralTransientWorkload {
        schema_version: 2,
        initial_fields: Vec::new(),
        model: model.clone(),
        step_id: ProcedureStepId("response".into()),
        backend: ComputeBackend::Cpu,
        solver: StructuralTransientSolverPlan::ModalReduced {
            basis_transition: Box::new(basis_transition),
        },
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
        residual_target: 1.0e-8,
    };
    workload.validate().unwrap();
    let excitations = realize_step_excitations(
        &model,
        &model.sequence.steps[1],
        ExcitationSamplePoint::time(0.0),
    )
    .unwrap();
    let assembly = assemble_structural_system(
        &model,
        &excitations,
        Some(mesh.clone()),
        StructuralFormulationRequirement::SyntheticOperator,
    )
    .unwrap();
    let schedule = StructuralTransientSchedule::new(vec![
        StructuralTransientSchedulePoint {
            time_s: runmat_types::physical::Quantity::<runmat_types::physical::kind::Time, _>::new(
                0.0,
            ),
            arrival: StructuralTransientScheduleArrival::Initial,
            continuation: StructuralTransientContinuationApproach::At,
        },
        StructuralTransientSchedulePoint {
            time_s: runmat_types::physical::Quantity::<runmat_types::physical::kind::Time, _>::new(
                0.5,
            ),
            arrival: StructuralTransientScheduleArrival::Continuous,
            continuation: StructuralTransientContinuationApproach::At,
        },
        StructuralTransientSchedulePoint {
            time_s: runmat_types::physical::Quantity::<runmat_types::physical::kind::Time, _>::new(
                1.0,
            ),
            arrival: StructuralTransientScheduleArrival::Continuous,
            continuation: StructuralTransientContinuationApproach::At,
        },
    ])
    .unwrap();
    (workload, mesh, state, assembly, schedule)
}

#[test]
fn modal_transient_scope_binds_exact_basis_projection_and_coordinate_count() {
    let (workload, mesh, state, assembly, schedule) = modal_case();
    let scope = StructuralTransientCheckpointScope::from_prepared_workload(
        StructuralTransientCheckpointScopeInput {
            workload: &workload,
            solver_mesh: &mesh,
            initial_assembly: &assembly,
            modal_state: Some(&state),
            schedule,
        },
    )
    .unwrap();
    let StructuralTransientCoordinateSpace::ModalReduced {
        basis_state,
        basis,
        projection,
        coordinate_count,
    } = &scope.coordinate_space
    else {
        panic!("modal workload produced direct coordinates");
    };
    assert_eq!(*basis_state, state.identity);
    assert_eq!(*basis, state.payload.modal_basis().unwrap().identity());
    assert_eq!(*coordinate_count, projection.mode_count);
    assert_eq!(
        scope.coordinate_space.coordinate_count(scope.dof_count),
        *coordinate_count
    );
    assert!(scope.identity().is_ok());

    let physical_state = crate::execution::StructuralTransientContinuationState {
        displacement: vec![0.0; scope.dof_count],
        velocity: vec![0.0; scope.dof_count],
        acceleration: vec![0.0; scope.dof_count],
    };
    let numerical_state = crate::execution::StructuralTransientNumericalState::ModalReduced {
        basis_state: state.identity,
        coordinates: crate::execution::ModalTransientCoordinateState {
            displacement: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Displacement,
                _,
            >::new(vec![0.0; *coordinate_count]),
            velocity:
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Velocity, _>::new(
                    vec![0.0; *coordinate_count],
                ),
            acceleration: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Acceleration,
                _,
            >::new(vec![0.0; *coordinate_count]),
        },
    };
    let checkpoint = crate::execution::StructuralTransientCheckpoint::new(
        crate::execution::StructuralTransientCheckpointInput {
            scope: scope.clone(),
            block_ordinal: 0,
            continuation: crate::execution::StructuralTransientContinuationPoint {
                schedule_ordinal: 1,
                time_s: runmat_types::physical::Time::new(0.5),
                approach: crate::execution::StructuralTransientContinuationApproach::At,
            },
            state: physical_state.clone(),
            numerical_state: numerical_state.clone(),
            balance: crate::execution::StructuralTransientBalanceAccumulator {
                initial_mechanical_energy_j: runmat_types::physical::Energy::new(0.0),
                applied_load_work_j: runmat_types::physical::Energy::new(0.0),
                support_work_j: runmat_types::physical::Energy::new(0.0),
                damping_dissipation_j: runmat_types::physical::Energy::new(0.0),
            },
            counters: crate::execution::StructuralTransientSolverCounters {
                dense_factorization_count: 1,
                dense_solve_count: 1,
                ..crate::execution::StructuralTransientSolverCounters::default()
            },
            observation_cursors: Vec::new(),
        },
        None,
    )
    .unwrap();
    assert_eq!(
        crate::execution::StructuralTransientCheckpoint::from_canonical_bytes(
            &checkpoint.canonical_bytes().unwrap()
        )
        .unwrap(),
        checkpoint
    );
    assert!(crate::execution::StructuralTransientCheckpoint::new(
        crate::execution::StructuralTransientCheckpointInput {
            scope,
            block_ordinal: 0,
            continuation: checkpoint.continuation.clone(),
            state: physical_state,
            numerical_state: crate::execution::StructuralTransientNumericalState::Direct,
            balance: checkpoint.balance.clone(),
            counters: checkpoint.counters.clone(),
            observation_cursors: Vec::new(),
        },
        None,
    )
    .is_err());
}

#[test]
fn modal_transient_physical_blocks_restart_from_exact_reduced_coordinates() {
    let (workload, mesh, state, _, _) = modal_case();
    let scope = crate::pipeline::transient::prepare_transient_checkpoint_scope(
        &workload,
        &mesh,
        Some(&state),
    )
    .unwrap();
    let terminal = u64::try_from(scope.schedule.points.len() - 1).unwrap();
    assert!(terminal >= 2);
    let solver_mesh = mesh;
    let run =
        |request: &StructuralTransientBlockRequest,
         predecessor: Option<&crate::execution::StructuralTransientCheckpoint>| {
            crate::pipeline::transient::run_transient_block(
                crate::pipeline::transient::StructuralTransientRunBlockInput {
                    workload: &workload,
                    solver_mesh: &solver_mesh,
                    request,
                    predecessor,
                    modal_state: Some(&state),
                    observation_cursors: Vec::new(),
                },
            )
            .unwrap()
        };

    let complete_request =
        StructuralTransientBlockRequest::new(scope.clone(), None, terminal).unwrap();
    let complete = run(&complete_request, None);
    let first_request = StructuralTransientBlockRequest::new(scope.clone(), None, 1).unwrap();
    let first = run(&first_request, None);
    let second_request = StructuralTransientBlockRequest::new(
        scope,
        Some(&first.manifest().terminal_checkpoint),
        terminal,
    )
    .unwrap();
    let second = run(&second_request, Some(&first.manifest().terminal_checkpoint));

    assert_eq!(
        complete.manifest().terminal_checkpoint.state,
        second.manifest().terminal_checkpoint.state
    );
    assert_eq!(
        complete.manifest().terminal_checkpoint.numerical_state,
        second.manifest().terminal_checkpoint.numerical_state
    );
    assert_eq!(
        complete.manifest().terminal_checkpoint.balance,
        second.manifest().terminal_checkpoint.balance
    );
    assert_eq!(
        complete.manifest().terminal_checkpoint.counters,
        second.manifest().terminal_checkpoint.counters
    );
    assert_eq!(
        complete.manifest().solver_evidence.solver_method,
        crate::contracts::SolverMethod::StructuralModalTransientNewmarkDenseFactorization
    );
    assert_eq!(
        complete
            .manifest()
            .solver_evidence
            .counters
            .dense_factorization_count,
        terminal + 1
    );
    assert_eq!(first.samples()[0].physical, complete.samples()[0].physical);
    assert_eq!(
        second.samples().last().unwrap().physical,
        complete.samples().last().unwrap().physical
    );
    assert!(complete.samples().iter().all(|sample| {
        sample.physical.projected_residual_norm <= workload.residual_target
            && sample
                .physical
                .dynamic_balance
                .equilibrium_residual
                .iter()
                .all(|value| value.is_finite())
    }));
}

#[test]
fn modal_transient_scope_rejects_missing_or_substituted_basis_state() {
    let (workload, mesh, state, assembly, schedule) = modal_case();
    let input = |modal_state| StructuralTransientCheckpointScopeInput {
        workload: &workload,
        solver_mesh: &mesh,
        initial_assembly: &assembly,
        modal_state,
        schedule: schedule.clone(),
    };
    assert!(StructuralTransientCheckpointScope::from_prepared_workload(input(None)).is_err());
    let mut substituted = state;
    substituted.topology_digest = Digest::sha256(b"other-topology");
    assert!(
        StructuralTransientCheckpointScope::from_prepared_workload(input(Some(&substituted)))
            .is_err()
    );
}

#[test]
fn modal_transient_workload_is_canonical_and_rejects_accelerator_substitution() {
    let (workload, ..) = modal_case();
    assert_eq!(
        StructuralTransientWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap())
            .unwrap(),
        workload
    );
    let mut accelerated = workload;
    accelerated.backend = ComputeBackend::Gpu;
    assert!(accelerated.validate().is_err());
}
