use runmat_analysis_core::SignalSampleApproach;

use crate::{
    assembly::AssemblySummary,
    contracts::{SolverMethod, SolverPreconditioner},
    execution::{
        ModalTransientCoordinateState, StateIdentity, StructuralTransientCheckpoint,
        StructuralTransientCoordinateSpace, StructuralTransientNumericalState,
        StructuralTransientSegment,
    },
    pipeline::modal_basis::ModalBasisArtifact,
    solve::{
        backend::kind::LinearAlgebraBackendKind,
        dynamics::{
            advance_modal_transient_state, initialize_modal_transient_state,
            refresh_modal_transient_state, ModalProjectionError, ModalReducedSystem,
            ModalTransientInitialState, ModalTransientStep,
        },
    },
};

use super::{
    advance::{
        TransientInitialVectors, TransientNumericalStepOutcome, TransientSolverAccounting,
        TransientStepFailure, TransientStepStrategy,
    },
    block_execution::{
        solve_structural_transient_block_with_strategy, StructuralTransientBlockSolveInput,
        StructuralTransientInitialConditions,
    },
    constraint_state::physical_load,
    state::SecondOrderState,
    TransientSolveError,
};

const INITIAL_RECONSTRUCTION_TOLERANCE: f64 = 1.0e-10;

pub(super) fn projection_error<E>(error: ModalProjectionError) -> TransientSolveError<E> {
    match error {
        ModalProjectionError::Usage(error) => TransientSolveError::Usage(error),
        other => TransientSolveError::ExecutionContract {
            detail: other.to_string(),
        },
    }
}

fn physical_projection_error<E>(error: ModalProjectionError) -> TransientSolveError<E> {
    match error {
        ModalProjectionError::Usage(error) => TransientSolveError::Usage(error),
        other => TransientSolveError::PhysicalMapping {
            detail: other.to_string(),
        },
    }
}

fn continuation_projection_error<E>(error: ModalProjectionError) -> TransientSolveError<E> {
    match error {
        ModalProjectionError::Usage(error) => TransientSolveError::Usage(error),
        other => TransientSolveError::InvalidContinuationState {
            detail: other.to_string(),
        },
    }
}

pub struct StructuralModalTransientBlockSolveInput<'a> {
    pub block: StructuralTransientBlockSolveInput<'a>,
    pub basis_state: StateIdentity,
    pub basis: &'a ModalBasisArtifact,
}

pub fn solve_structural_modal_transient_block_with_assembly<E>(
    input: StructuralModalTransientBlockSolveInput<'_>,
    mut assembly_at: impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, E>,
) -> Result<StructuralTransientSegment, TransientSolveError<E>> {
    let reduced = ModalReducedSystem::project(input.block.initial_summary, input.basis)
        .map_err(projection_error)?;
    let expansion = reduced.clone();
    let mut map_physical_state =
        |_: f64, _: SignalSampleApproach, summary: &AssemblySummary, state: &SecondOrderState| {
            let displacement = (expansion)
                .expand_state(&runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Displacement,
                    _,
                >::new(state.displacement()))
                .map(|field| field.into_canonical_storage())
                .map_err(physical_projection_error)?;
            let velocity = (expansion)
                .expand_state(&runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Velocity,
                    _,
                >::new(state.velocity()))
                .map(|field| field.into_canonical_storage())
                .map_err(physical_projection_error)?;
            let acceleration = (expansion)
                .expand_state(&runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Acceleration,
                    _,
                >::new(state.acceleration()))
                .map(|field| field.into_canonical_storage())
                .map_err(physical_projection_error)?;
            let physical = SecondOrderState::try_new(displacement, velocity, acceleration)
                .map_err(|error| TransientSolveError::PhysicalMapping {
                    detail: error.to_string(),
                })?;
            Ok::<_, TransientSolveError<E>>((summary.clone(), physical))
        };
    let physical_dof_count = input.block.initial_summary.dof_count;
    solve_structural_modal_transient_block_with_mapping(
        input,
        reduced,
        physical_dof_count,
        &mut assembly_at,
        &mut map_physical_state,
    )
}

pub(crate) fn solve_structural_modal_transient_block_with_mapping<E>(
    input: StructuralModalTransientBlockSolveInput<'_>,
    reduced: ModalReducedSystem,
    physical_dof_count: usize,
    assembly_at: &mut impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, E>,
    map_physical_state: &mut impl FnMut(
        f64,
        SignalSampleApproach,
        &AssemblySummary,
        &SecondOrderState,
    ) -> Result<
        (AssemblySummary, SecondOrderState),
        TransientSolveError<E>,
    >,
) -> Result<StructuralTransientSegment, TransientSolveError<E>> {
    validate_scope(&input, &reduced)?;
    let initial_conditions = input
        .block
        .initial_conditions
        .as_ref()
        .map(|initial| project_initial_conditions(&reduced, initial))
        .transpose()?;
    let mut strategy = ModalTransientStepStrategy { reduced };
    let basis_state = input.basis_state;
    let mut restore_checkpoint_state = |checkpoint: &StructuralTransientCheckpoint| {
        let StructuralTransientNumericalState::ModalReduced {
            basis_state: checkpoint_basis,
            coordinates,
        } = &checkpoint.numerical_state
        else {
            return Err(TransientSolveError::InvalidContinuationState {
                detail: "modal transient checkpoint is missing reduced coordinates".into(),
            });
        };
        if *checkpoint_basis != basis_state {
            return Err(TransientSolveError::InvalidContinuationState {
                detail: "modal transient checkpoint names another basis state".into(),
            });
        }
        coordinate_state(coordinates.clone())
            .map_err(|detail| TransientSolveError::InvalidContinuationState { detail })
    };
    let direct = StructuralTransientBlockSolveInput {
        request: input.block.request,
        predecessor: input.block.predecessor,
        initial_summary: input.block.initial_summary,
        options: input.block.options,
        backend: input.block.backend,
        initial_conditions,
        observation_cursors: input.block.observation_cursors,
    };
    solve_structural_transient_block_with_strategy(
        direct,
        physical_dof_count,
        assembly_at,
        map_physical_state,
        &mut restore_checkpoint_state,
        &mut strategy,
    )
}

fn validate_scope<E>(
    input: &StructuralModalTransientBlockSolveInput<'_>,
    reduced: &ModalReducedSystem,
) -> Result<(), TransientSolveError<E>> {
    match &input.block.request.scope.coordinate_space {
        StructuralTransientCoordinateSpace::ModalReduced {
            basis_state,
            basis,
            projection,
            coordinate_count,
        } if *basis_state == input.basis_state
            && *basis == input.basis.identity()
            && projection == reduced.evidence()
            && *coordinate_count == reduced.mode_count() =>
        {
            Ok(())
        }
        _ => Err(TransientSolveError::ExecutionContract {
            detail: "modal transient basis differs from its canonical coordinate space".into(),
        }),
    }
}

fn project_initial_conditions<E>(
    reduced: &ModalReducedSystem,
    initial: &StructuralTransientInitialConditions,
) -> Result<StructuralTransientInitialConditions, TransientSolveError<E>> {
    let displacement = project_representable_state::<E, runmat_types::physical::kind::Displacement>(
        reduced,
        &initial.displacement,
        "displacement",
    )?;
    let velocity = project_representable_state::<E, runmat_types::physical::kind::Velocity>(
        reduced,
        &initial.velocity,
        "velocity",
    )?;
    Ok(StructuralTransientInitialConditions {
        displacement,
        velocity,
    })
}

fn project_representable_state<E, K: crate::solve::dynamics::ModalCoordinateQuantity>(
    reduced: &ModalReducedSystem,
    physical: &[f64],
    quantity: &'static str,
) -> Result<Vec<f64>, TransientSolveError<E>> {
    let coordinates = reduced
        .state_field::<K>(physical)
        .and_then(|field| reduced.project_state::<K, _>(&field))
        .map_err(continuation_projection_error)?;
    let reconstructed = reduced
        .expand_state(&coordinates)
        .map(|field| field.into_canonical_storage())
        .map_err(continuation_projection_error)?;
    let error = reconstructed
        .iter()
        .zip(physical)
        .map(|(reconstructed, physical)| (reconstructed - physical).abs())
        .fold(0.0_f64, f64::max);
    let scale = physical
        .iter()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    if error > INITIAL_RECONSTRUCTION_TOLERANCE * scale {
        return Err(TransientSolveError::InvalidContinuationState {
            detail: format!(
                "authored initial {quantity} is not representable by the retained modal basis: maximum reconstruction error {error}"
            ),
        });
    }
    Ok(coordinates.into_canonical_values())
}

struct ModalTransientStepStrategy {
    reduced: ModalReducedSystem,
}

impl TransientStepStrategy for ModalTransientStepStrategy {
    fn coordinate_count(&self) -> usize {
        self.reduced.mode_count()
    }

    fn solver_method(&self) -> SolverMethod {
        SolverMethod::StructuralModalTransientNewmarkDenseFactorization
    }

    fn resumed_accounting(
        &self,
        counters: crate::execution::StructuralTransientSolverCounters,
    ) -> TransientSolverAccounting {
        TransientSolverAccounting {
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            counters,
            preconditioner: SolverPreconditioner::None,
            accelerator_evidence: Vec::new(),
        }
    }

    fn initialize(
        &mut self,
        summary: &AssemblySummary,
        initial: TransientInitialVectors,
    ) -> Result<TransientNumericalStepOutcome, TransientStepFailure> {
        let load = (self.reduced)
            .load_field(&physical_load(summary))
            .and_then(|field| (self.reduced).project_load(&field))
            .map_err(TransientStepFailure::projection)?;
        initialize_modal_transient_state(
            &self.reduced,
            load.canonical_values(),
            ModalTransientInitialState {
                displacement: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Displacement,
                    _,
                >::new(initial.displacement),
                velocity: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Velocity,
                    _,
                >::new(initial.velocity),
            },
        )
        .map_err(TransientStepFailure::modal)
        .and_then(modal_outcome)
    }

    fn advance(
        &mut self,
        summary: &AssemblySummary,
        previous: &SecondOrderState,
        time_step_s: f64,
    ) -> Result<TransientNumericalStepOutcome, TransientStepFailure> {
        let load = (self.reduced)
            .load_field(&physical_load(summary))
            .and_then(|field| (self.reduced).project_load(&field))
            .map_err(TransientStepFailure::projection)?;
        advance_modal_transient_state(
            &self.reduced,
            time_step_s,
            load.canonical_values(),
            modal_coordinates(previous),
        )
        .map_err(TransientStepFailure::modal)
        .and_then(modal_outcome)
    }

    fn refresh_acceleration(
        &mut self,
        summary: &AssemblySummary,
        previous: &SecondOrderState,
    ) -> Result<TransientNumericalStepOutcome, TransientStepFailure> {
        let load = (self.reduced)
            .load_field(&physical_load(summary))
            .and_then(|field| (self.reduced).project_load(&field))
            .map_err(TransientStepFailure::projection)?;
        refresh_modal_transient_state(
            &self.reduced,
            load.canonical_values(),
            previous.displacement().to_vec(),
            previous.velocity().to_vec(),
        )
        .map_err(TransientStepFailure::modal)
        .and_then(modal_outcome)
    }
}

fn modal_outcome(
    step: ModalTransientStep,
) -> Result<TransientNumericalStepOutcome, TransientStepFailure> {
    Ok(TransientNumericalStepOutcome {
        state: coordinate_state(step.coordinates).map_err(TransientStepFailure::Detail)?,
        projected_relative_residual: step.relative_residual,
        accounting: TransientSolverAccounting::modal_dense(),
    })
}

fn modal_coordinates(state: &SecondOrderState) -> ModalTransientCoordinateState {
    ModalTransientCoordinateState {
        displacement: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Displacement,
            _,
        >::new(state.displacement().to_vec()),
        velocity:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Velocity, _>::new(
                state.velocity().to_vec(),
            ),
        acceleration: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Acceleration,
            _,
        >::new(state.acceleration().to_vec()),
    }
}

fn coordinate_state(
    coordinates: ModalTransientCoordinateState,
) -> Result<SecondOrderState, String> {
    SecondOrderState::try_new(
        coordinates.displacement.into_canonical_values(),
        coordinates.velocity.into_canonical_values(),
        coordinates.acceleration.into_canonical_values(),
    )
    .map_err(|error| error.to_string())
}
