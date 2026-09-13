use std::{cell::RefCell, collections::BTreeMap};

use runmat_analysis_core::{SignalSampleApproach, StructuralTranslationDof};
use thiserror::Error;

use crate::{
    algebra::{spmv, AssembledRealSystem, CanonicalRealOperators},
    assembly::{dofs::StructuralDofKind, AssemblySummary},
    constraints::{AffineConstraintReduction, ConstraintAlgebraError},
    contracts::{ComputeBackend, FeaRunError},
    excitation::{StructuralMotionError, StructuralMotionSet},
    execution::{
        StructuralTransientBlockRequest, StructuralTransientCheckpoint,
        StructuralTransientObservationCursor, StructuralTransientSegment,
    },
    solve::transient::{
        rebuild_transient_balance, solve_fully_prescribed_transient_block_with_physical_mapping,
        solve_structural_transient_block_with_physical_mapping,
        solve_transient_system_with_assembly, solve_transient_system_with_physical_mapping,
        SecondOrderState, StructuralTransientBlockSolveInput, StructuralTransientInitialConditions,
        TransientPhysicalMappingInput, TransientSolveError, TransientSolveOptions,
        TransientSolveResult, TransientTimeDomain,
    },
};

#[derive(Clone, Debug)]
pub(super) struct StructuralMotionSnapshot {
    pub(super) full_summary: AssemblySummary,
    pub(super) reduced_summary: AssemblySummary,
    pub(super) displacement_lift: Vec<f64>,
    pub(super) velocity_lift: Vec<f64>,
    pub(super) acceleration_lift: Vec<f64>,
    pub(super) reduction: AffineConstraintReduction,
}

pub(super) struct StructuralMotionSolveInput<'a> {
    pub(super) summary: &'a AssemblySummary,
    pub(super) motions: &'a StructuralMotionSet,
    pub(super) options: TransientSolveOptions,
    pub(super) backend: ComputeBackend,
    pub(super) time_domain: TransientTimeDomain<'a>,
    pub(super) initial_displacement: Vec<f64>,
    pub(super) initial_velocity: Vec<f64>,
}

pub(super) struct StructuralMotionBlockSolveInput<'a> {
    pub(super) summary: &'a AssemblySummary,
    pub(super) motions: &'a StructuralMotionSet,
    pub(super) options: &'a TransientSolveOptions,
    pub(super) backend: ComputeBackend,
    pub(super) request: &'a StructuralTransientBlockRequest,
    pub(super) predecessor: Option<&'a StructuralTransientCheckpoint>,
    pub(super) initial_conditions: Option<StructuralTransientInitialConditions>,
    pub(super) observation_cursors: Vec<StructuralTransientObservationCursor>,
}

struct ExpandedStructuralState {
    displacement: Vec<f64>,
    velocity: Vec<f64>,
    acceleration: Vec<f64>,
}

#[derive(Debug, Error)]
pub(super) enum StructuralMotionReductionError {
    #[error("support-motion materialization failed: {0}")]
    Materialization(#[from] StructuralMotionError),
    #[error("support motion requires one resolved affine structural constraint row space")]
    MissingConstraintMatrix,
    #[error("support-motion row `{constraint_id}` is not a canonical global translation row")]
    InvalidMotionRow { constraint_id: String },
    #[error("support-motion constraint algebra failed: {0}")]
    ConstraintAlgebra(#[from] ConstraintAlgebraError),
    #[error("support-motion operator reduction failed: {0}")]
    Operator(String),
}

impl StructuralMotionSnapshot {
    pub(super) fn reduce(
        mut full_summary: AssemblySummary,
        motions: &StructuralMotionSet,
        time_s: f64,
        approach: SignalSampleApproach,
    ) -> Result<Self, StructuralMotionReductionError> {
        let source = full_summary
            .constraint_matrix
            .as_ref()
            .ok_or(StructuralMotionReductionError::MissingConstraintMatrix)?;
        let mut displacement = source.clone();
        let mut velocity = source.clone();
        let mut acceleration = source.clone();
        for row in 0..source.row_count() {
            let constraint_id = &source.constraint_ids[row];
            if !motions.has_constraint(constraint_id) {
                velocity.right_hand_side[row] = 0.0;
                acceleration.right_hand_side[row] = 0.0;
                continue;
            }
            let start = source.row_offsets[row];
            let end = source.row_offsets[row + 1];
            if end != start + 1 || source.coefficients[start] != 1.0 {
                return Err(StructuralMotionReductionError::InvalidMotionRow {
                    constraint_id: constraint_id.clone(),
                });
            }
            let dof = match full_summary
                .structural_dof_layout
                .address(source.column_indices[start])
                .map(|address| address.kind)
            {
                Some(StructuralDofKind::Ux) => StructuralTranslationDof::Ux,
                Some(StructuralDofKind::Uy) => StructuralTranslationDof::Uy,
                Some(StructuralDofKind::Uz) => StructuralTranslationDof::Uz,
                _ => {
                    return Err(StructuralMotionReductionError::InvalidMotionRow {
                        constraint_id: constraint_id.clone(),
                    })
                }
            };
            let sample = motions
                .sample(constraint_id, dof, time_s, approach)?
                .ok_or_else(|| StructuralMotionReductionError::InvalidMotionRow {
                    constraint_id: constraint_id.clone(),
                })?;
            displacement.right_hand_side[row] = sample.displacement_m;
            velocity.right_hand_side[row] = sample.velocity_m_per_s;
            acceleration.right_hand_side[row] = sample.acceleration_m_per_s2;
        }
        let displacement_reduction = displacement.affine_reduction()?;
        let velocity_reduction = velocity.affine_reduction()?;
        let acceleration_reduction = acceleration.affine_reduction()?;
        require_same_transform(&displacement_reduction, &velocity_reduction)?;
        require_same_transform(&displacement_reduction, &acceleration_reduction)?;
        let mut homogeneous = source.clone();
        homogeneous.right_hand_side.fill(0.0);
        let reduction = homogeneous.affine_reduction()?;
        require_same_transform(&displacement_reduction, &reduction)?;

        let operators = full_summary.operator.operators();
        let physical_load = full_summary.physical_load.clone();
        let inertia = spmv(operators.mass(), &acceleration_reduction.offset)
            .map_err(|error| StructuralMotionReductionError::Operator(error.to_string()))?;
        let damping = spmv(operators.damping(), &velocity_reduction.offset)
            .map_err(|error| StructuralMotionReductionError::Operator(error.to_string()))?;
        let elastic = spmv(operators.stiffness(), &displacement_reduction.offset)
            .map_err(|error| StructuralMotionReductionError::Operator(error.to_string()))?;
        let effective_full = physical_load
            .iter()
            .zip(inertia)
            .zip(damping)
            .zip(elastic)
            .map(|(((load, inertia), damping), elastic)| load - inertia - damping - elastic)
            .collect::<Vec<_>>();
        let reduced_load = reduction.transpose_apply(&effective_full)?;
        let reduced_operators = CanonicalRealOperators::try_new(
            reduction.reduce_sparse_operator(operators.stiffness())?,
            reduction.reduce_sparse_operator(operators.mass())?,
            reduction.reduce_sparse_operator(operators.damping())?,
        )
        .map_err(|error| StructuralMotionReductionError::Operator(error.to_string()))?;
        let reduced_operator = AssembledRealSystem::try_new(
            reduced_operators,
            vec![false; reduction.reduced_dof_count],
            reduced_load.clone(),
        )
        .map_err(|error| StructuralMotionReductionError::Operator(error.to_string()))?;
        let mut reduced_summary = full_summary.clone();
        reduced_summary.dof_count = reduction.reduced_dof_count;
        reduced_summary.constrained_dof_count = 0;
        reduced_summary.constraint_matrix = None;
        reduced_summary.affine_constraint_reduction = None;
        reduced_summary.physical_load = reduced_load.clone();
        reduced_summary.operator = reduced_operator;
        full_summary.constraint_matrix = Some(displacement);
        full_summary.affine_constraint_reduction = Some(displacement_reduction.clone());
        Ok(Self {
            full_summary,
            reduced_summary,
            displacement_lift: displacement_reduction.offset,
            velocity_lift: velocity_reduction.offset,
            acceleration_lift: acceleration_reduction.offset,
            reduction,
        })
    }

    pub(super) fn reduce_displacement(&self, full: &[f64]) -> Vec<f64> {
        self.reduction
            .free_dof_indices
            .iter()
            .map(|index| full[*index] - self.displacement_lift[*index])
            .collect()
    }

    pub(super) fn reduce_velocity(&self, full: &[f64]) -> Vec<f64> {
        self.reduction
            .free_dof_indices
            .iter()
            .map(|index| full[*index] - self.velocity_lift[*index])
            .collect()
    }

    pub(super) fn reduce_acceleration(&self, full: &[f64]) -> Vec<f64> {
        self.reduction
            .free_dof_indices
            .iter()
            .map(|index| full[*index] - self.acceleration_lift[*index])
            .collect()
    }

    pub(super) fn expand_displacement(
        &self,
        reduced: &[f64],
    ) -> Result<Vec<f64>, StructuralMotionReductionError> {
        expand_with_lift(&self.reduction, reduced, &self.displacement_lift)
    }

    pub(super) fn expand_velocity(
        &self,
        reduced: &[f64],
    ) -> Result<Vec<f64>, StructuralMotionReductionError> {
        expand_with_lift(&self.reduction, reduced, &self.velocity_lift)
    }

    pub(super) fn expand_acceleration(
        &self,
        reduced: &[f64],
    ) -> Result<Vec<f64>, StructuralMotionReductionError> {
        expand_with_lift(&self.reduction, reduced, &self.acceleration_lift)
    }

    fn expand_state(
        &self,
        displacement: &[f64],
        velocity: &[f64],
        acceleration: &[f64],
    ) -> Result<ExpandedStructuralState, StructuralMotionReductionError> {
        Ok(ExpandedStructuralState {
            displacement: self.expand_displacement(displacement)?,
            velocity: self.expand_velocity(velocity)?,
            acceleration: self.expand_acceleration(acceleration)?,
        })
    }
}

pub(super) fn physical_summary_for_recovery(
    summary: AssemblySummary,
    motions: &StructuralMotionSet,
    time_s: f64,
    approach: SignalSampleApproach,
) -> Result<AssemblySummary, StructuralMotionReductionError> {
    Ok(StructuralMotionSnapshot::reduce(summary, motions, time_s, approach)?.full_summary)
}

pub(super) fn solve_prescribed_motion_block(
    input: StructuralMotionBlockSolveInput<'_>,
    mut assembly_at: impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, FeaRunError>,
) -> Result<StructuralTransientSegment, TransientSolveError<FeaRunError>> {
    let schedule = &input.request.scope.schedule;
    let initial_time_s = schedule.points[0].time_s;
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
    let mut reduce_checkpoint_state =
        |time_s, approach, state: &crate::execution::StructuralTransientContinuationState| {
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
                snapshot.reduce_displacement(&state.displacement),
                snapshot.reduce_velocity(&state.velocity),
                snapshot.reduce_acceleration(&state.acceleration),
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

pub(super) fn solve_prescribed_motion(
    input: StructuralMotionSolveInput<'_>,
    mut assembly_at: impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, FeaRunError>,
) -> Result<TransientSolveResult, TransientSolveError<FeaRunError>> {
    let StructuralMotionSolveInput {
        summary,
        motions,
        options,
        backend,
        time_domain,
        initial_displacement,
        initial_velocity,
    } = input;
    let initial = StructuralMotionSnapshot::reduce(
        summary.clone(),
        motions,
        time_domain.start_time_s,
        SignalSampleApproach::At,
    )
    .map_err(assembly_error)?;
    let reduced_displacement = initial.reduce_displacement(&initial_displacement);
    let reduced_velocity = initial.reduce_velocity(&initial_velocity);
    let snapshots = RefCell::new(BTreeMap::from([(
        snapshot_key(time_domain.start_time_s, SignalSampleApproach::At),
        initial.clone(),
    )]));
    let mut reduced_assembly_at = |time_s, approach| {
        let snapshot = StructuralMotionSnapshot::reduce(
            assembly_at(time_s, approach)?,
            motions,
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
    let mut map_physical_state =
        |time_s, approach, _: &AssemblySummary, state: &SecondOrderState| {
            let snapshot = snapshots
                .borrow()
                .get(&snapshot_key(time_s, approach))
                .cloned()
                .ok_or_else(|| {
                    FeaRunError::Assembly(format!(
                    "support-motion physical snapshot at {time_s} s ({approach:?}) was not retained"
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
    let mut result = if initial.reduced_summary.dof_count == 0 {
        solve_transient_system_with_assembly(
            &initial.reduced_summary,
            options,
            backend,
            time_domain,
            reduced_displacement,
            reduced_velocity,
            &mut reduced_assembly_at,
        )?
    } else {
        solve_transient_system_with_physical_mapping(
            TransientPhysicalMappingInput {
                summary: &initial.reduced_summary,
                options,
                backend,
                time_domain,
                initial_displacement: reduced_displacement,
                initial_velocity: reduced_velocity,
            },
            &mut reduced_assembly_at,
            &mut map_physical_state,
        )?
    };

    if initial.reduced_summary.dof_count != 0 {
        return Ok(result);
    }

    for interval in &mut result.integration_intervals {
        let start = {
            let snapshots = snapshots.borrow();
            retained_snapshot(
                &snapshots,
                interval.start_time_s,
                interval.start_approach,
                "integration",
            )?
            .clone()
        };
        let expanded = start
            .expand_state(
                &interval.start_displacement,
                &interval.start_velocity,
                &interval.start_acceleration,
            )
            .map_err(assembly_error)?;
        interval.start_displacement = expanded.displacement;
        interval.start_velocity = expanded.velocity;
        interval.start_acceleration = expanded.acceleration;
        interval.start_summary = start.full_summary;

        let end = {
            let snapshots = snapshots.borrow();
            retained_snapshot(
                &snapshots,
                interval.end_time_s,
                interval.end_approach,
                "integration",
            )?
            .clone()
        };
        let expanded = end
            .expand_state(
                &interval.end_displacement,
                &interval.end_velocity,
                &interval.end_acceleration,
            )
            .map_err(assembly_error)?;
        interval.end_displacement = expanded.displacement;
        interval.end_velocity = expanded.velocity;
        interval.end_acceleration = expanded.acceleration;
        interval.end_summary = end.full_summary;
    }

    let mut full_summaries = Vec::with_capacity(result.time_points_s.len());
    for index in 0..result.time_points_s.len() {
        let snapshot = {
            let snapshots = snapshots.borrow();
            retained_snapshot(
                &snapshots,
                result.time_points_s[index],
                SignalSampleApproach::At,
                "public",
            )?
            .clone()
        };
        let expanded = snapshot
            .expand_state(
                &result.displacement_snapshots[index],
                &result.velocity_snapshots[index],
                &result.acceleration_snapshots[index],
            )
            .map_err(assembly_error)?;
        result.displacement_snapshots[index] = expanded.displacement;
        result.velocity_snapshots[index] = expanded.velocity;
        result.acceleration_snapshots[index] = expanded.acceleration;
        full_summaries.push(snapshot.full_summary);
    }
    result.assembly_snapshots = full_summaries;
    rebuild_transient_balance(&mut result)
        .map_err(|detail| TransientSolveError::DynamicBalance { detail })?;
    Ok(result)
}

fn retained_snapshot<'a>(
    snapshots: &'a BTreeMap<(u64, u8), StructuralMotionSnapshot>,
    time_s: f64,
    approach: SignalSampleApproach,
    role: &str,
) -> Result<&'a StructuralMotionSnapshot, TransientSolveError<FeaRunError>> {
    snapshots
        .get(&snapshot_key(time_s, approach))
        .ok_or_else(|| {
            TransientSolveError::Assembly(FeaRunError::Assembly(format!(
                "support-motion {role} snapshot at {time_s} s ({approach:?}) was not retained"
            )))
        })
}

fn snapshot_key(time_s: f64, approach: SignalSampleApproach) -> (u64, u8) {
    let approach = match approach {
        SignalSampleApproach::At => 0,
        SignalSampleApproach::LeftLimit => 1,
        SignalSampleApproach::RightLimit => 2,
    };
    (time_s.to_bits(), approach)
}

fn assembly_error(error: StructuralMotionReductionError) -> TransientSolveError<FeaRunError> {
    TransientSolveError::Assembly(FeaRunError::Assembly(error.to_string()))
}

fn expand_with_lift(
    reduction: &AffineConstraintReduction,
    reduced: &[f64],
    lift: &[f64],
) -> Result<Vec<f64>, StructuralMotionReductionError> {
    let mut full = reduction.expand_linear(reduced)?;
    if full.len() != lift.len() {
        return Err(StructuralMotionReductionError::Operator(
            "support-motion lifting dimension differs from its transform".into(),
        ));
    }
    for (value, lift) in full.iter_mut().zip(lift) {
        *value += lift;
    }
    Ok(full)
}

fn require_same_transform(
    left: &AffineConstraintReduction,
    right: &AffineConstraintReduction,
) -> Result<(), StructuralMotionReductionError> {
    if left.full_dof_count != right.full_dof_count
        || left.reduced_dof_count != right.reduced_dof_count
        || left.free_dof_indices != right.free_dof_indices
        || left.pivot_dof_indices != right.pivot_dof_indices
        || left.recovery_row_offsets != right.recovery_row_offsets
        || left.recovery_column_indices != right.recovery_column_indices
        || left.recovery_coefficients != right.recovery_coefficients
    {
        return Err(StructuralMotionReductionError::ConstraintAlgebra(
            ConstraintAlgebraError::InvalidMatrix(
                "support motion changed the immutable homogeneous constraint transform".into(),
            ),
        ));
    }
    Ok(())
}
