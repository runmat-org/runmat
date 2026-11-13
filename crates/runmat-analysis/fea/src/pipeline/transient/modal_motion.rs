use std::{cell::RefCell, collections::BTreeMap};

use runmat_analysis_core::SignalSampleApproach;

use crate::{
    assembly::AssemblySummary,
    contracts::FeaRunError,
    execution::{StateIdentity, StructuralTransientSegment},
    pipeline::modal_basis::ModalBasisArtifact,
    solve::{
        dynamics::{ModalProjectionError, ModalReducedSystem},
        transient::{
            solve_structural_modal_transient_block_with_mapping, SecondOrderState,
            StructuralModalTransientBlockSolveInput, StructuralTransientBlockSolveInput,
            StructuralTransientInitialConditions, TransientSolveError,
        },
    },
};

use super::motion::{
    assembly_error, snapshot_key, StructuralMotionBlockSolveInput, StructuralMotionSnapshot,
};

pub(super) fn solve_prescribed_modal_motion_block(
    input: StructuralMotionBlockSolveInput<'_>,
    basis_state: StateIdentity,
    basis: &ModalBasisArtifact,
    mut assembly_at: impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, FeaRunError>,
) -> Result<StructuralTransientSegment, TransientSolveError<FeaRunError>> {
    let initial_time_s = input.request.scope.schedule.points[0]
        .time_s
        .canonical_value();
    let initial = StructuralMotionSnapshot::reduce(
        input.summary.clone(),
        input.motions,
        initial_time_s,
        SignalSampleApproach::At,
    )
    .map_err(assembly_error)?;
    let reduced = ModalReducedSystem::project(input.summary, basis).map_err(projection_error)?;
    let relative_initial_conditions = input
        .initial_conditions
        .as_ref()
        .map(|conditions| {
            Ok(StructuralTransientInitialConditions {
                displacement: initial.relative_displacement(&conditions.displacement)?,
                velocity: initial.relative_velocity(&conditions.velocity)?,
            })
        })
        .transpose()
        .map_err(assembly_error)?;
    let snapshots = RefCell::new(BTreeMap::from([(
        snapshot_key(initial_time_s, SignalSampleApproach::At),
        initial.clone(),
    )]));
    let mut modal_assembly_at = |time_s, approach| {
        let snapshot = StructuralMotionSnapshot::reduce(
            assembly_at(time_s, approach)?,
            input.motions,
            time_s,
            approach,
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        let summary = snapshot.modal_summary.clone();
        snapshots
            .borrow_mut()
            .insert(snapshot_key(time_s, approach), snapshot);
        Ok(summary)
    };
    let expansion = reduced.clone();
    let mut map_physical_state =
        |time_s, approach, _: &AssemblySummary, state: &SecondOrderState| {
            let snapshot = snapshots
                .borrow()
                .get(&snapshot_key(time_s, approach))
                .cloned()
                .ok_or_else(|| TransientSolveError::PhysicalMapping {
                    detail: format!(
                    "modal support-motion snapshot at {time_s} s ({approach:?}) was not retained"
                ),
                })?;
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
            let expanded = snapshot
                .lift_homogeneous_state(&displacement, &velocity, &acceleration)
                .map_err(|error| TransientSolveError::PhysicalMapping {
                    detail: error.to_string(),
                })?;
            let state = SecondOrderState::try_new(
                expanded.displacement,
                expanded.velocity,
                expanded.acceleration,
            )
            .map_err(|error| TransientSolveError::PhysicalMapping {
                detail: error.to_string(),
            })?;
            Ok::<_, TransientSolveError<FeaRunError>>((snapshot.full_summary, state))
        };
    let block = StructuralTransientBlockSolveInput {
        request: input.request,
        predecessor: input.predecessor,
        initial_summary: &initial.modal_summary,
        options: input.options,
        backend: input.backend,
        initial_conditions: relative_initial_conditions,
        observation_cursors: input.observation_cursors,
    };
    solve_structural_modal_transient_block_with_mapping(
        StructuralModalTransientBlockSolveInput {
            block,
            basis_state,
            basis,
        },
        reduced,
        input.summary.dof_count,
        &mut modal_assembly_at,
        &mut map_physical_state,
    )
}

fn projection_error(error: ModalProjectionError) -> TransientSolveError<FeaRunError> {
    match error {
        ModalProjectionError::Usage(error) => TransientSolveError::Usage(error),
        other => TransientSolveError::ExecutionContract {
            detail: other.to_string(),
        },
    }
}

fn physical_projection_error(error: ModalProjectionError) -> TransientSolveError<FeaRunError> {
    match error {
        ModalProjectionError::Usage(error) => TransientSolveError::Usage(error),
        other => TransientSolveError::PhysicalMapping {
            detail: other.to_string(),
        },
    }
}
