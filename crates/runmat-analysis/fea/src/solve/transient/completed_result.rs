use crate::execution::{
    StructuralTransientNumericalReceipt, StructuralTransientSampleStep,
    StructuralTransientSolverResult,
};

use super::{
    transient_diagnostics, TransientSolveError, TransientSolveOptions, TransientSolveResult,
};

pub(crate) struct CompletedTransientSolveInput<'a> {
    pub(crate) options: &'a TransientSolveOptions,
    pub(crate) result: StructuralTransientSolverResult,
    pub(crate) receipt: &'a StructuralTransientNumericalReceipt,
}

/// Reconstructs the ephemeral solve result consumed by formulation recovery from a completed,
/// canonical temporal solve. This is deliberately not a solver: physical states, balance terms,
/// counters, and accelerator evidence remain authoritative outputs of the accepted block chain.
pub(crate) fn reconstruct_completed_transient_solve(
    input: CompletedTransientSolveInput<'_>,
) -> Result<TransientSolveResult, TransientSolveError<std::convert::Infallible>> {
    input
        .receipt
        .validate_for_result(&input.result)
        .map_err(contract_error)?;
    let samples = input.result.samples();
    let scope = &input.result.manifest().scope;
    if samples.is_empty()
        || samples.iter().any(|sample| {
            sample.state.displacement.len() != scope.dof_count
                || sample.state.velocity.len() != scope.dof_count
                || sample.state.acceleration.len() != scope.dof_count
        })
    {
        return Err(TransientSolveError::DofCountChanged);
    }
    let first_block =
        input.receipt.blocks.first().ok_or_else(|| {
            contract_error("completed transient receipt contains no solver block")
        })?;
    let solver_host_sync_count = u32::try_from(input.receipt.terminal_counters.host_sync_count)
        .map_err(|_| TransientSolveError::SolverCounterOverflow)?;
    let device_apply_k_count = u32::try_from(input.receipt.terminal_counters.device_apply_k_count)
        .map_err(|_| TransientSolveError::SolverCounterOverflow)?;
    let device_apply_k_attempt_count =
        u32::try_from(input.receipt.terminal_counters.device_apply_k_attempt_count)
            .map_err(|_| TransientSolveError::SolverCounterOverflow)?;

    let mut accepted_time_steps_s = Vec::with_capacity(samples.len().saturating_sub(1));
    for sample in samples.iter().skip(1) {
        let StructuralTransientSampleStep::Accepted { time_step_s } = sample.step else {
            return Err(contract_error(
                "completed transient result contains a non-initial sample without an accepted step",
            ));
        };
        accepted_time_steps_s.push(time_step_s);
    }
    let time_points_s = samples
        .iter()
        .map(|sample| sample.time_s.canonical_value())
        .collect::<Vec<_>>();
    let residual_norms = samples
        .iter()
        .map(|sample| sample.projected_residual_norm)
        .collect::<Vec<_>>();
    let strain_energy_snapshots = samples
        .iter()
        .map(|sample| sample.strain_energy_j.canonical_value())
        .collect::<Vec<_>>();
    let kinetic_energy_snapshots = samples
        .iter()
        .map(|sample| sample.kinetic_energy_j.canonical_value())
        .collect::<Vec<_>>();
    let external_work_j = samples
        .iter()
        .map(|sample| sample.applied_load_work_j.canonical_value())
        .collect::<Vec<_>>();
    let support_work_j = samples
        .iter()
        .map(|sample| sample.support_work_j.canonical_value())
        .collect::<Vec<_>>();
    let damping_dissipation_j = samples
        .iter()
        .map(|sample| sample.damping_dissipation_j.canonical_value())
        .collect::<Vec<_>>();
    let energy_balance_residual_j = samples
        .iter()
        .map(|sample| sample.energy_balance_residual_j.canonical_value())
        .collect::<Vec<_>>();
    let relative_energy_balance_error = samples
        .iter()
        .map(|sample| sample.relative_energy_balance_error)
        .collect::<Vec<_>>();
    let total_steps = accepted_time_steps_s.len();
    let diagnostics = transient_diagnostics(
        input.options,
        &accepted_time_steps_s,
        &residual_norms,
        &strain_energy_snapshots,
        &kinetic_energy_snapshots,
        &relative_energy_balance_error,
        first_block.solver_backend,
    );
    let accelerator_evidence = input
        .receipt
        .blocks
        .iter()
        .flat_map(|block| block.accelerator_evidence.iter().cloned())
        .collect();
    let schedule = scope.schedule.clone();
    let mut displacement_snapshots = Vec::with_capacity(samples.len());
    let mut velocity_snapshots = Vec::with_capacity(samples.len());
    let mut acceleration_snapshots = Vec::with_capacity(samples.len());
    let mut reaction_snapshots = Vec::with_capacity(samples.len());
    let mut dynamic_balance_snapshots = Vec::with_capacity(samples.len());
    // The validated result is consumed here so recovery does not retain a second
    // full set of solver-owned physical state and reaction buffers.
    for sample in input.result.samples {
        displacement_snapshots.push(sample.state.displacement);
        velocity_snapshots.push(sample.state.velocity);
        acceleration_snapshots.push(sample.state.acceleration);
        reaction_snapshots.push(sample.reaction);
        dynamic_balance_snapshots.push(sample.dynamic_balance);
    }

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
        solver_method: first_block.solver_method,
        solver_backend: first_block.solver_backend,
        solver_host_sync_count,
        device_apply_k_count,
        device_apply_k_attempt_count,
        preconditioner: first_block.preconditioner,
        accelerator_evidence,
        assembly_snapshots: Vec::new(),
        reaction_snapshots,
        integration_intervals: Vec::new(),
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

fn contract_error(error: impl std::fmt::Display) -> TransientSolveError<std::convert::Infallible> {
    TransientSolveError::ExecutionContract {
        detail: error.to_string(),
    }
}
