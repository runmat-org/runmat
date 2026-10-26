use std::{cell::RefCell, collections::BTreeMap};

use runmat_analysis_core::SignalSampleApproach;

use crate::{
    assembly::AssemblySummary,
    contracts::{ComputeBackend, FeaRunError},
    excitation::StructuralMotionSet,
    execution::{
        StructuralTransientBlockRequest, StructuralTransientCheckpoint,
        StructuralTransientContinuationApproach, StructuralTransientObservationCursor,
        StructuralTransientSegment,
    },
    solve::transient::{
        solve_fully_prescribed_transient_block_with_physical_mapping,
        solve_structural_transient_block_with_physical_mapping, SecondOrderState,
        StructuralTransientBlockSolveInput, StructuralTransientInitialConditions,
        TransientSolveError, TransientSolveOptions,
    },
};

use super::{assembly_error, snapshot_key, StructuralMotionSnapshot};

pub(in crate::pipeline::transient) struct StructuralMotionBlockSolveInput<'a> {
    pub(in crate::pipeline::transient) summary: &'a AssemblySummary,
    pub(in crate::pipeline::transient) motions: &'a StructuralMotionSet,
    pub(in crate::pipeline::transient) options: &'a TransientSolveOptions,
    pub(in crate::pipeline::transient) backend: ComputeBackend,
    pub(in crate::pipeline::transient) request: &'a StructuralTransientBlockRequest,
    pub(in crate::pipeline::transient) predecessor: Option<&'a StructuralTransientCheckpoint>,
    pub(in crate::pipeline::transient) initial_conditions:
        Option<StructuralTransientInitialConditions>,
    pub(in crate::pipeline::transient) observation_cursors:
        Vec<StructuralTransientObservationCursor>,
}

pub(in crate::pipeline::transient) fn solve_prescribed_motion_block(
    input: StructuralMotionBlockSolveInput<'_>,
    mut assembly_at: impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, FeaRunError>,
) -> Result<StructuralTransientSegment, TransientSolveError<FeaRunError>> {
    let schedule = &input.request.scope.schedule;
    let initial_time_s = schedule.points[0].time_s.canonical_value();
    let initial = StructuralMotionSnapshot::reduce(
        input.summary.clone(),
        input.motions,
        initial_time_s,
        SignalSampleApproach::At,
    )
    .map_err(assembly_error)?;
    let snapshots = RefCell::new(BTreeMap::from([(
        snapshot_key(initial_time_s, SignalSampleApproach::At),
        initial.clone(),
    )]));
    let mut reduced_assembly_at = |time_s, approach| {
        let snapshot = StructuralMotionSnapshot::reduce(
            assembly_at(time_s, approach)?,
            input.motions,
            time_s,
            approach,
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        let reduced = snapshot.reduced_summary.clone();
        snapshots
            .borrow_mut()
            .insert(snapshot_key(time_s, approach), snapshot);
        Ok(reduced)
    };
    if initial.reduced_summary.dof_count == 0 {
        let mut map_physical_state = |time_s, approach, _: &AssemblySummary| {
            let snapshot = snapshots
                .borrow()
                .get(&snapshot_key(time_s, approach))
                .cloned()
                .ok_or_else(|| {
                    FeaRunError::Assembly(format!(
                        "fully prescribed block snapshot at {time_s} s ({approach:?}) was not retained"
                    ))
                })?;
            let expanded = snapshot
                .expand_state(&[], &[], &[])
                .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
            let state = SecondOrderState::try_new(
                expanded.displacement,
                expanded.velocity,
                expanded.acceleration,
            )
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
            Ok((snapshot.full_summary, state))
        };
        return solve_fully_prescribed_transient_block_with_physical_mapping(
            StructuralTransientBlockSolveInput {
                request: input.request,
                predecessor: input.predecessor,
                initial_summary: &initial.reduced_summary,
                options: input.options,
                backend: input.backend,
                initial_conditions: input.initial_conditions.clone(),
                observation_cursors: input.observation_cursors,
            },
            input.summary.dof_count,
            &mut reduced_assembly_at,
            &mut map_physical_state,
        );
    }
    let initial_conditions =
        input
            .initial_conditions
            .map(|conditions| StructuralTransientInitialConditions {
                displacement: initial.reduce_displacement(&conditions.displacement),
                velocity: initial.reduce_velocity(&conditions.velocity),
            });
    let mut map_physical_state =
        |time_s, approach, _: &AssemblySummary, state: &SecondOrderState| {
            let snapshot = snapshots
                .borrow()
                .get(&snapshot_key(time_s, approach))
                .cloned()
                .ok_or_else(|| {
                    FeaRunError::Assembly(format!(
                    "support-motion block snapshot at {time_s} s ({approach:?}) was not retained"
                ))
                })?;
            let expanded = snapshot
                .expand_state(state.displacement(), state.velocity(), state.acceleration())
                .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
            let physical_state = SecondOrderState::try_new(
                expanded.displacement,
                expanded.velocity,
                expanded.acceleration,
            )
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
            Ok((snapshot.full_summary, physical_state))
        };
    let mut reduce_checkpoint_state = |checkpoint: &StructuralTransientCheckpoint| {
        let time_s = checkpoint.continuation.time_s.canonical_value();
        let approach = match checkpoint.continuation.approach {
            StructuralTransientContinuationApproach::At => SignalSampleApproach::At,
            StructuralTransientContinuationApproach::RightLimit => SignalSampleApproach::RightLimit,
        };
        let snapshot = snapshots
            .borrow()
            .get(&snapshot_key(time_s, approach))
            .cloned()
            .ok_or_else(|| TransientSolveError::InvalidContinuationState {
                detail: format!(
                    "support-motion continuation snapshot at {time_s} s ({approach:?}) was not retained"
                ),
            })?;
        SecondOrderState::try_new(
            snapshot.reduce_displacement(&checkpoint.state.displacement),
            snapshot.reduce_velocity(&checkpoint.state.velocity),
            snapshot.reduce_acceleration(&checkpoint.state.acceleration),
        )
        .map_err(|error| TransientSolveError::InvalidContinuationState {
            detail: error.to_string(),
        })
    };
    solve_structural_transient_block_with_physical_mapping(
        StructuralTransientBlockSolveInput {
            request: input.request,
            predecessor: input.predecessor,
            initial_summary: &initial.reduced_summary,
            options: input.options,
            backend: input.backend,
            initial_conditions,
            observation_cursors: input.observation_cursors,
        },
        input.summary.dof_count,
        &mut reduced_assembly_at,
        &mut map_physical_state,
        &mut reduce_checkpoint_state,
    )
}
