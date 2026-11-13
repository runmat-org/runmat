use std::convert::Infallible;

use runmat_analysis_core::SignalSampleApproach;

use crate::{
    assembly::AssemblySummary,
    contracts::SolverMethod,
    progress::{emit_phase, is_cancelled, FeaProgressPhase, FeaProgressStatus},
    solve::backend::kind::LinearAlgebraBackendKind,
    ComputeBackend,
};

mod advance;
mod balance;
mod block_execution;
mod completed_result;
mod constraint_state;
mod contract;
mod diagnostics;
mod effective_solve;
mod empty;
#[cfg(test)]
mod interval_retention_tests;
mod modal_block;
mod newmark;
mod schedule;
mod segment;
mod state;
#[cfg(test)]
mod tests;

use advance::solve_complete;
use balance::{TransientBalanceHistory, TransientBalanceStep};
pub(crate) use block_execution::{
    solve_fully_prescribed_transient_block_with_physical_mapping,
    solve_structural_transient_block_with_physical_mapping,
};
pub use block_execution::{
    solve_structural_transient_block_with_assembly, StructuralTransientBlockSolveInput,
    StructuralTransientInitialConditions,
};
pub(crate) use completed_result::{
    reconstruct_completed_transient_solve, CompletedTransientSolveInput,
};
use contract::TransientIntegrationInterval;
pub use contract::{
    StructuralTimeStepPolicy, TransientSolveError, TransientSolveOptions, TransientSolveResult,
};
pub(crate) use contract::{
    TransientIntervalRetention, TransientPhysicalMappingInput, TransientTimeDomain,
};
pub(crate) use diagnostics::mechanical_energy_history_diagnostic;
use diagnostics::transient_diagnostics;
use effective_solve::EffectiveDynamicsSolveResult;
pub(crate) use modal_block::solve_structural_modal_transient_block_with_mapping;
pub use modal_block::{
    solve_structural_modal_transient_block_with_assembly, StructuralModalTransientBlockSolveInput,
};
pub(crate) use schedule::{
    canonical_transient_schedule, canonical_transient_step_boundaries, TransientStepBoundary,
};
pub(crate) use state::SecondOrderState;

pub(crate) fn rebuild_transient_balance(
    result: &mut TransientSolveResult,
) -> Result<(), TransientSolveError<crate::contracts::FeaRunError>> {
    let snapshot_count = result.time_points_s.len();
    if snapshot_count == 0
        || result.displacement_snapshots.len() != snapshot_count
        || result.velocity_snapshots.len() != snapshot_count
        || result.acceleration_snapshots.len() != snapshot_count
        || result.assembly_snapshots.len() != snapshot_count
        || result.accepted_time_steps_s.len() + 1 != snapshot_count
    {
        return Err(balance::DynamicBalanceError::InvalidHistory(
            "transient balance reconstruction inputs have inconsistent lengths".into(),
        )
        .into_transient_solve());
    }
    let states = (0..snapshot_count)
        .map(|index| {
            state::SecondOrderState::try_new(
                result.displacement_snapshots[index].clone(),
                result.velocity_snapshots[index].clone(),
                result.acceleration_snapshots[index].clone(),
            )
            .map_err(|error| {
                balance::DynamicBalanceError::InvalidHistory(error.to_string())
                    .into_transient_solve()
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut history = TransientBalanceHistory::start(&result.assembly_snapshots[0], &states[0])
        .map_err(balance::DynamicBalanceError::into_transient_solve)?;
    for (index, published_state) in states.iter().enumerate().skip(1) {
        let interval = result.integration_intervals.get(index - 1).ok_or_else(|| {
            balance::DynamicBalanceError::InvalidHistory(
                "transient integration interval history is incomplete".into(),
            )
            .into_transient_solve()
        })?;
        let integration_start_state = interval.start_state().map_err(|error| {
            balance::DynamicBalanceError::InvalidHistory(error).into_transient_solve()
        })?;
        let integration_end_state = interval.end_state().map_err(|error| {
            balance::DynamicBalanceError::InvalidHistory(error).into_transient_solve()
        })?;
        history
            .push_step(TransientBalanceStep {
                integration_start_summary: &interval.start_summary,
                integration_start_state: &integration_start_state,
                integration_end_summary: &interval.end_summary,
                integration_end_state: &integration_end_state,
                published_summary: &result.assembly_snapshots[index],
                published_state,
                time_step_s: result.accepted_time_steps_s[index - 1],
            })
            .map_err(balance::DynamicBalanceError::into_transient_solve)?;
    }
    result.reaction_snapshots = history.reaction_snapshots().to_vec();
    result.strain_energy_snapshots = history.strain_energy_j().to_vec();
    result.kinetic_energy_snapshots = history.kinetic_energy_j().to_vec();
    result.external_work_j =
        runmat_types::physical::Quantity::new(history.external_work_j().to_vec());
    result.support_work_j =
        runmat_types::physical::Quantity::new(history.support_work_j().to_vec());
    result.damping_dissipation_j =
        runmat_types::physical::Quantity::new(history.damping_dissipation_j().to_vec());
    result.energy_balance_residual_j =
        runmat_types::physical::Quantity::new(history.energy_balance_residual_j().to_vec());
    result.relative_energy_balance_error = history.relative_energy_balance_error().to_vec();
    result.dynamic_balance_snapshots = history.balances().to_vec();
    Ok(())
}

pub fn solve_transient_system(
    summary: &AssemblySummary,
    options: TransientSolveOptions,
    backend: ComputeBackend,
) -> Result<TransientSolveResult, TransientSolveError<Infallible>> {
    let initial_displacement = if let Some(reduction) = &summary.affine_constraint_reduction {
        reduction.offset.clone()
    } else if let Some(matrix) = &summary.constraint_matrix {
        matrix
            .affine_reduction()
            .map_err(|error| TransientSolveError::InitialAcceleration {
                detail: format!(
                    "structural constraints cannot establish an initial state: {error}"
                ),
            })?
            .offset
    } else {
        summary
            .operator
            .load()
            .iter()
            .zip(summary.operator.constrained_dofs())
            .map(|(value, constrained)| if *constrained { *value } else { 0.0 })
            .collect()
    };
    solve_transient_system_with_assembly(
        summary,
        options,
        backend,
        TransientTimeDomain {
            start_time_s: runmat_types::physical::Time::new(0.0),
            end_time_s: runmat_types::physical::Time::new(1.0),
            step_boundaries: &[],
        },
        initial_displacement,
        vec![0.0; summary.dof_count],
        |_, _| Ok(summary.clone()),
    )
}

pub(crate) fn solve_transient_system_with_assembly<E: std::fmt::Display>(
    summary: &AssemblySummary,
    options: TransientSolveOptions,
    backend: ComputeBackend,
    time_domain: TransientTimeDomain<'_>,
    initial_displacement: Vec<f64>,
    initial_velocity: Vec<f64>,
    mut assembly_at: impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, E>,
) -> Result<TransientSolveResult, TransientSolveError<E>> {
    if summary.dof_count == 0 {
        let TransientTimeDomain {
            start_time_s,
            end_time_s,
            step_boundaries,
        } = time_domain;
        // The numerical integrator works in canonical seconds after typed admission.
        let (start_time_s, end_time_s) =
            (start_time_s.canonical_value(), end_time_s.canonical_value());
        validate_options(&options, start_time_s, end_time_s)?;
        if !initial_displacement.is_empty() || !initial_velocity.is_empty() {
            return Err(TransientSolveError::InvalidOptions(
                "a zero-DOF transient system requires empty initial state vectors",
            ));
        }
        return empty::solve(
            summary,
            &options,
            start_time_s,
            end_time_s,
            step_boundaries,
            &mut assembly_at,
        );
    }
    let mut identity_physical_state =
        |_: f64, _: SignalSampleApproach, summary: &AssemblySummary, state: &SecondOrderState| {
            Ok::<_, E>((summary.clone(), state.clone()))
        };
    solve_transient_system_with_physical_mapping(
        TransientPhysicalMappingInput {
            summary,
            options,
            backend,
            time_domain,
            initial_displacement,
            initial_velocity,
            interval_retention: TransientIntervalRetention::DiscardAfterBalance,
        },
        assembly_at,
        &mut identity_physical_state,
    )
}

pub(crate) fn solve_transient_system_with_physical_mapping<E: std::fmt::Display>(
    input: TransientPhysicalMappingInput<'_>,
    mut assembly_at: impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, E>,
    mut map_physical_state: impl FnMut(
        f64,
        SignalSampleApproach,
        &AssemblySummary,
        &SecondOrderState,
    ) -> Result<(AssemblySummary, SecondOrderState), E>,
) -> Result<TransientSolveResult, TransientSolveError<E>> {
    let TransientPhysicalMappingInput {
        summary,
        options,
        backend,
        time_domain,
        initial_displacement,
        initial_velocity,
        interval_retention,
    } = input;
    let TransientTimeDomain {
        start_time_s,
        end_time_s,
        step_boundaries,
    } = time_domain;
    // Retained output times re-enter Time; this scope is the numerical integrator.
    let (start_time_s, end_time_s) = (start_time_s.canonical_value(), end_time_s.canonical_value());
    validate_options(&options, start_time_s, end_time_s)?;
    if summary.dof_count == 0 {
        return Err(TransientSolveError::InvalidOptions(
            "a physically mapped transient solve requires at least one numerical DOF",
        ));
    }
    let backend_kind = match backend {
        ComputeBackend::Cpu => LinearAlgebraBackendKind::CpuReference,
        ComputeBackend::Gpu => LinearAlgebraBackendKind::Accelerator,
    };
    let interval_count = options.time_step_policy.interval_count();
    let schedule =
        canonical_transient_schedule(interval_count, start_time_s, end_time_s, step_boundaries)
            .map_err(|detail| TransientSolveError::Schedule { detail })?;
    let mut typed_physical_state =
        |time_s, approach, summary: &AssemblySummary, state: &SecondOrderState| {
            map_physical_state(time_s, approach, summary, state).map_err(|error| {
                TransientSolveError::PhysicalMapping {
                    detail: error.to_string(),
                }
            })
        };
    let (initial, block) = solve_complete(
        advance::TransientAdvanceContext {
            initial_summary: summary,
            options: &options,
            backend_kind,
            schedule: &schedule,
            interval_retention,
        },
        advance::TransientInitialVectors {
            displacement: initial_displacement,
            velocity: initial_velocity,
        },
        &mut assembly_at,
        &mut typed_physical_state,
    )?;
    let total_steps = block.samples.len();
    emit_phase(
        options.progress_operation.as_str(),
        FeaProgressPhase::Solve,
        FeaProgressStatus::Advanced,
        "second-order transient solve loop complete",
        Some(total_steps as u64),
        Some(total_steps as u64),
    );
    let mut time_points_s = vec![initial.time_s.canonical_value()];
    let mut displacement_snapshots = vec![initial.state.displacement().to_vec()];
    let mut velocity_snapshots = vec![initial.state.velocity().to_vec()];
    let mut acceleration_snapshots = vec![initial.state.acceleration().to_vec()];
    let mut residual_norms = vec![initial.projected_residual_norm];
    let mut accepted_time_steps_s = Vec::with_capacity(total_steps);
    let mut assembly_snapshots = vec![initial.summary];
    let mut reaction_snapshots = vec![initial.reaction];
    let mut integration_intervals = Vec::with_capacity(
        if interval_retention == TransientIntervalRetention::RetainForMotion {
            total_steps
        } else {
            0
        },
    );
    let mut strain_energy_snapshots = vec![initial.strain_energy_j.canonical_value()];
    let mut kinetic_energy_snapshots = vec![initial.kinetic_energy_j.canonical_value()];
    let mut external_work_j = vec![0.0];
    let mut support_work_j = vec![0.0];
    let mut damping_dissipation_j = vec![0.0];
    let mut energy_balance_residual_j = vec![initial.energy_balance_residual_j.canonical_value()];
    let mut relative_energy_balance_error = vec![initial.relative_energy_balance_error];
    let mut dynamic_balance_snapshots = vec![initial.dynamic_balance];
    for sample in block.samples {
        debug_assert_eq!(sample.schedule_ordinal as usize, time_points_s.len());
        time_points_s.push(sample.time_s.canonical_value());
        displacement_snapshots.push(sample.state.displacement().to_vec());
        velocity_snapshots.push(sample.state.velocity().to_vec());
        acceleration_snapshots.push(sample.state.acceleration().to_vec());
        residual_norms.push(sample.projected_residual_norm);
        accepted_time_steps_s.push(sample.accepted_time_step_s.canonical_value());
        assembly_snapshots.push(sample.summary);
        reaction_snapshots.push(sample.reaction);
        if let Some(interval) = sample.integration_interval {
            integration_intervals.push(interval);
        }
        strain_energy_snapshots.push(sample.strain_energy_j.canonical_value());
        kinetic_energy_snapshots.push(sample.kinetic_energy_j.canonical_value());
        external_work_j.push(sample.applied_load_work_j.canonical_value());
        support_work_j.push(sample.support_work_j.canonical_value());
        damping_dissipation_j.push(sample.damping_dissipation_j.canonical_value());
        energy_balance_residual_j.push(sample.energy_balance_residual_j.canonical_value());
        relative_energy_balance_error.push(sample.relative_energy_balance_error);
        dynamic_balance_snapshots.push(sample.dynamic_balance);
    }
    let accounting = block.terminal.accounting;
    let solver_host_sync_count = u32::try_from(accounting.counters.host_sync_count)
        .map_err(|_| TransientSolveError::SolverCounterOverflow)?;
    let device_apply_k_count = u32::try_from(accounting.counters.device_apply_k_count)
        .map_err(|_| TransientSolveError::SolverCounterOverflow)?;
    let device_apply_k_attempt_count =
        u32::try_from(accounting.counters.device_apply_k_attempt_count)
            .map_err(|_| TransientSolveError::SolverCounterOverflow)?;
    let diagnostics = transient_diagnostics(
        &options,
        &accepted_time_steps_s,
        &residual_norms,
        &strain_energy_snapshots,
        &kinetic_energy_snapshots,
        &relative_energy_balance_error,
        accounting.solver_backend,
    );
    Ok(TransientSolveResult {
        schedule,
        converged_steps: total_steps,
        total_steps,
        time_points_s,
        displacement_snapshots,
        velocity_snapshots,
        acceleration_snapshots,
        residual_norms,
        accepted_time_steps_s,
        diagnostics,
        solver_method: SolverMethod::NewmarkAverageAccelerationPcg,
        solver_backend: accounting.solver_backend,
        solver_host_sync_count,
        device_apply_k_count,
        device_apply_k_attempt_count,
        preconditioner: accounting.preconditioner,
        accelerator_evidence: accounting.accelerator_evidence,
        assembly_snapshots,
        reaction_snapshots,
        integration_intervals,
        strain_energy_snapshots,
        kinetic_energy_snapshots,
        external_work_j: runmat_types::physical::Quantity::new(external_work_j),
        support_work_j: runmat_types::physical::Quantity::new(support_work_j),
        damping_dissipation_j: runmat_types::physical::Quantity::new(damping_dissipation_j),
        energy_balance_residual_j: runmat_types::physical::Quantity::new(energy_balance_residual_j),
        relative_energy_balance_error,
        dynamic_balance_snapshots,
    })
}

fn require_stable_system<E>(
    initial: &AssemblySummary,
    candidate: &AssemblySummary,
) -> Result<(), TransientSolveError<E>> {
    if candidate.dof_count != initial.dof_count {
        return Err(TransientSolveError::DofCountChanged);
    }
    if candidate.structural_dof_layout != initial.structural_dof_layout
        || candidate.structural_reduced_basis != initial.structural_reduced_basis
    {
        return Err(TransientSolveError::StructuralSystemChanged {
            component: "DOF layout",
        });
    }
    if candidate.operator.constrained_dofs() != initial.operator.constrained_dofs()
        || candidate.constraint_matrix != initial.constraint_matrix
        || candidate.affine_constraint_reduction != initial.affine_constraint_reduction
    {
        return Err(TransientSolveError::StructuralSystemChanged {
            component: "constraint system",
        });
    }
    let initial_operators = initial.operator.operators();
    let candidate_operators = candidate.operator.operators();
    if candidate_operators.stiffness() != initial_operators.stiffness() {
        return Err(TransientSolveError::StructuralSystemChanged {
            component: "stiffness operator",
        });
    }
    if candidate_operators.mass() != initial_operators.mass() {
        return Err(TransientSolveError::StructuralSystemChanged {
            component: "mass operator",
        });
    }
    if candidate_operators.damping() != initial_operators.damping() {
        return Err(TransientSolveError::StructuralSystemChanged {
            component: "damping operator",
        });
    }
    Ok(())
}

fn validate_options<E>(
    options: &TransientSolveOptions,
    start_time_s: f64,
    end_time_s: f64,
) -> Result<(), TransientSolveError<E>> {
    if options.time_step_policy.interval_count() == 0 {
        return Err(TransientSolveError::InvalidOptions(
            "interval_count must be positive",
        ));
    }
    if !options.residual_target.is_finite() || options.residual_target <= 0.0 {
        return Err(TransientSolveError::InvalidOptions(
            "residual_target must be finite and positive",
        ));
    }
    if !start_time_s.is_finite() || !end_time_s.is_finite() || end_time_s <= start_time_s {
        return Err(TransientSolveError::InvalidOptions(
            "the time domain must be finite and ordered",
        ));
    }
    Ok(())
}
