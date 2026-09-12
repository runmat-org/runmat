use std::convert::Infallible;

use runmat_analysis_core::{FormulationRequirement, SignalSampleApproach};
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    algebra::spmv,
    assembly::AssemblySummary,
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::FeaDiagnostic,
    progress::{emit_phase, is_cancelled, FeaProgressPhase, FeaProgressStatus},
    solve::backend::kind::LinearAlgebraBackendKind,
    ComputeBackend,
};

mod constraint_state;
mod diagnostics;
mod effective_solve;
mod newmark;
mod schedule;
mod state;
#[cfg(test)]
mod tests;

pub(crate) use diagnostics::mechanical_energy_history_diagnostic;
use diagnostics::transient_diagnostics;
use effective_solve::{
    solve_average_acceleration_step, solve_initial_acceleration, EffectiveDynamicsSolveResult,
};
use schedule::next_step_boundary;
pub(crate) use schedule::{canonical_transient_step_boundaries, TransientStepBoundary};
use state::SecondOrderState;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientSolveOptions {
    pub formulation_requirement: FormulationRequirement,
    pub time_step_policy: StructuralTimeStepPolicy,
    pub residual_target: f64,
    #[serde(default = "default_progress_operation")]
    pub progress_operation: String,
    #[serde(default)]
    pub solver_mesh: Option<SolverMeshArtifact>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructuralTimeStepPolicy {
    Fixed { interval_count: usize },
}

impl StructuralTimeStepPolicy {
    pub fn interval_count(self) -> usize {
        match self {
            Self::Fixed { interval_count } => interval_count,
        }
    }
}

fn default_progress_operation() -> String {
    "fea.run_transient".to_string()
}

impl Default for TransientSolveOptions {
    fn default() -> Self {
        Self {
            formulation_requirement: FormulationRequirement::Production,
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 10 },
            residual_target: 1.0e-6,
            progress_operation: default_progress_operation(),
            solver_mesh: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientSolveResult {
    pub converged_steps: usize,
    pub total_steps: usize,
    pub time_points_s: Vec<f64>,
    pub displacement_snapshots: Vec<Vec<f64>>,
    pub velocity_snapshots: Vec<Vec<f64>>,
    pub acceleration_snapshots: Vec<Vec<f64>>,
    pub residual_norms: Vec<f64>,
    pub accepted_time_steps_s: Vec<f64>,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub solver_method: SolverMethod,
    pub solver_backend: LinearAlgebraBackendKind,
    pub solver_host_sync_count: u32,
    pub device_apply_k_count: u32,
    pub device_apply_k_attempt_count: u32,
    pub preconditioner: SolverPreconditioner,
    pub accelerator_evidence: Vec<crate::solve::accelerator::AcceleratorSolveEvidence>,
    #[serde(skip)]
    pub(crate) assembly_snapshots: Vec<AssemblySummary>,
    #[serde(skip)]
    pub(crate) strain_energy_snapshots: Vec<f64>,
    #[serde(skip)]
    pub(crate) kinetic_energy_snapshots: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum TransientSolveError<E> {
    #[error("transient solve options are invalid: {0}")]
    InvalidOptions(&'static str),
    #[error("transient solve was cancelled before a complete result was available")]
    Cancelled,
    #[error("initial structural acceleration could not be established: {detail}")]
    InitialAcceleration { detail: String },
    #[error(
        "structural acceleration could not be re-established after a discontinuity at {time_s} s: {detail}"
    )]
    AccelerationRefresh { time_s: f64, detail: String },
    #[error("structural transient step ending at {time_s} s failed: {detail}")]
    Step { time_s: f64, detail: String },
    #[error(
        "structural transient state at {time_s} s exceeded its projected residual target: {residual_norm} > {target}"
    )]
    ResidualTargetExceeded {
        time_s: f64,
        residual_norm: f64,
        target: f64,
    },
    #[error("transient reassembly failed: {0}")]
    Assembly(E),
    #[error("transient reassembly changed the structural degree-of-freedom count")]
    DofCountChanged,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TransientTimeDomain<'a> {
    pub(crate) start_time_s: f64,
    pub(crate) end_time_s: f64,
    pub(crate) step_boundaries: &'a [TransientStepBoundary],
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
            start_time_s: 0.0,
            end_time_s: 1.0,
            step_boundaries: &[],
        },
        initial_displacement,
        vec![0.0; summary.dof_count],
        |_, _| Ok(summary.clone()),
    )
}

pub(crate) fn solve_transient_system_with_assembly<E>(
    summary: &AssemblySummary,
    options: TransientSolveOptions,
    backend: ComputeBackend,
    time_domain: TransientTimeDomain<'_>,
    initial_displacement: Vec<f64>,
    initial_velocity: Vec<f64>,
    mut assembly_at: impl FnMut(f64, SignalSampleApproach) -> Result<AssemblySummary, E>,
) -> Result<TransientSolveResult, TransientSolveError<E>> {
    let TransientTimeDomain {
        start_time_s,
        end_time_s,
        step_boundaries,
    } = time_domain;
    validate_options(&options, start_time_s, end_time_s)?;
    if summary.dof_count == 0 {
        return Ok(empty_result(summary, &options, start_time_s));
    }
    let backend_kind = match backend {
        ComputeBackend::Cpu => LinearAlgebraBackendKind::CpuReference,
        ComputeBackend::Gpu => LinearAlgebraBackendKind::Accelerator,
    };
    let initial = solve_initial_acceleration(
        summary,
        initial_displacement,
        initial_velocity,
        backend_kind,
        options.residual_target,
    )
    .map_err(|error| TransientSolveError::InitialAcceleration {
        detail: error.to_string(),
    })?;
    require_residual_target(&initial, start_time_s, options.residual_target)?;
    let mut state = initial.state().clone();
    let mut time_points_s = vec![start_time_s];
    let mut displacement_snapshots = vec![state.displacement().to_vec()];
    let mut velocity_snapshots = vec![state.velocity().to_vec()];
    let mut acceleration_snapshots = vec![state.acceleration().to_vec()];
    let interval_count = options.time_step_policy.interval_count();
    let mut residual_norms = Vec::with_capacity(interval_count + 1);
    residual_norms.push(initial.projected_residual().relative_norm());
    let mut accepted_time_steps_s = Vec::with_capacity(interval_count);
    let mut assembly_snapshots = vec![summary.clone()];
    let mut strain_energy_snapshots = vec![strain_energy(summary, &state)];
    let mut kinetic_energy_snapshots = vec![kinetic_energy(summary, &state)];
    let mut aggregate = SolveAggregate::default();
    aggregate.push(&initial);
    let nominal_time_step_s = (end_time_s - start_time_s) / interval_count as f64;
    let progress_operation = options.progress_operation.as_str();

    while time_points_s.last().is_some_and(|time| *time < end_time_s) {
        if is_cancelled() {
            emit_phase(
                progress_operation,
                FeaProgressPhase::Solve,
                FeaProgressStatus::Cancelled,
                "transient solve cancelled",
                Some(accepted_time_steps_s.len() as u64),
                None,
            );
            return Err(TransientSolveError::Cancelled);
        }
        let current_time_s = time_points_s.last().copied().unwrap_or(start_time_s);
        let proposed_dt_s = nominal_time_step_s.min(end_time_s - current_time_s);
        let (next_time_s, approach) =
            next_step_boundary(current_time_s, proposed_dt_s, end_time_s, step_boundaries);
        let time_step_s = next_time_s - current_time_s;
        emit_phase(
            progress_operation,
            FeaProgressPhase::Solve,
            FeaProgressStatus::Advanced,
            format!("solving transient step ending at {next_time_s} s"),
            Some(accepted_time_steps_s.len() as u64),
            None,
        );
        let step_summary =
            assembly_at(next_time_s, approach).map_err(TransientSolveError::Assembly)?;
        require_stable_dof_count(summary, &step_summary)?;
        let solved = solve_average_acceleration_step(
            &step_summary,
            &state,
            time_step_s,
            backend_kind,
            options.residual_target,
        )
        .map_err(|source| TransientSolveError::Step {
            time_s: next_time_s,
            detail: source.to_string(),
        })?;
        require_residual_target(&solved, next_time_s, options.residual_target)?;

        // A candidate becomes visible only after its linear solve, physical residual, and state
        // invariants have all succeeded.
        state = solved.state().clone();
        aggregate.push(&solved);
        accepted_time_steps_s.push(time_step_s);
        time_points_s.push(next_time_s);
        displacement_snapshots.push(state.displacement().to_vec());
        velocity_snapshots.push(state.velocity().to_vec());
        acceleration_snapshots.push(state.acceleration().to_vec());
        strain_energy_snapshots.push(strain_energy(&step_summary, &state));
        kinetic_energy_snapshots.push(kinetic_energy(&step_summary, &state));
        assembly_snapshots.push(step_summary);
        residual_norms.push(solved.projected_residual().relative_norm());

        if approach == SignalSampleApproach::LeftLimit {
            let right_limit_summary = assembly_at(next_time_s, SignalSampleApproach::At)
                .map_err(TransientSolveError::Assembly)?;
            require_stable_dof_count(summary, &right_limit_summary)?;
            let refreshed = solve_initial_acceleration(
                &right_limit_summary,
                state.displacement().to_vec(),
                state.velocity().to_vec(),
                backend_kind,
                options.residual_target,
            )
            .map_err(|error| TransientSolveError::AccelerationRefresh {
                time_s: next_time_s,
                detail: error.to_string(),
            })?;
            require_residual_target(&refreshed, next_time_s, options.residual_target)?;
            state = refreshed.state().clone();
            aggregate.push(&refreshed);

            // A public time point denotes the signal value `At` that coordinate. The left-limit
            // state is an internal integration boundary only; replace all assembly-dependent
            // quantities with the right-limit equilibrium used to begin the next interval.
            *acceleration_snapshots
                .last_mut()
                .expect("the accepted state was published above") = state.acceleration().to_vec();
            *strain_energy_snapshots
                .last_mut()
                .expect("the accepted state was published above") =
                strain_energy(&right_limit_summary, &state);
            *kinetic_energy_snapshots
                .last_mut()
                .expect("the accepted state was published above") =
                kinetic_energy(&right_limit_summary, &state);
            *assembly_snapshots
                .last_mut()
                .expect("the accepted state was published above") = right_limit_summary;
            *residual_norms
                .last_mut()
                .expect("the accepted state was published above") =
                refreshed.projected_residual().relative_norm();
        }
    }

    let total_steps = accepted_time_steps_s.len();
    emit_phase(
        progress_operation,
        FeaProgressPhase::Solve,
        FeaProgressStatus::Advanced,
        "second-order transient solve loop complete",
        Some(total_steps as u64),
        Some(total_steps as u64),
    );
    let diagnostics = transient_diagnostics(
        &options,
        &accepted_time_steps_s,
        &residual_norms,
        &strain_energy_snapshots,
        &kinetic_energy_snapshots,
        aggregate.solver_backend,
    );
    Ok(TransientSolveResult {
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
        solver_backend: aggregate.solver_backend,
        solver_host_sync_count: aggregate.host_sync_count,
        device_apply_k_count: aggregate.device_apply_k_count,
        device_apply_k_attempt_count: aggregate.device_apply_k_attempt_count,
        preconditioner: aggregate.preconditioner,
        accelerator_evidence: aggregate.accelerator_evidence,
        assembly_snapshots,
        strain_energy_snapshots,
        kinetic_energy_snapshots,
    })
}

fn require_stable_dof_count<E>(
    initial: &AssemblySummary,
    candidate: &AssemblySummary,
) -> Result<(), TransientSolveError<E>> {
    if candidate.dof_count != initial.dof_count {
        return Err(TransientSolveError::DofCountChanged);
    }
    Ok(())
}

fn require_residual_target<E>(
    solved: &EffectiveDynamicsSolveResult,
    time_s: f64,
    target: f64,
) -> Result<(), TransientSolveError<E>> {
    let residual_norm = solved.projected_residual().relative_norm();
    if !residual_norm.is_finite() || residual_norm > target {
        return Err(TransientSolveError::ResidualTargetExceeded {
            time_s,
            residual_norm,
            target,
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
    if !start_time_s.is_finite() || !end_time_s.is_finite() || end_time_s < start_time_s {
        return Err(TransientSolveError::InvalidOptions(
            "the time domain must be finite and ordered",
        ));
    }
    Ok(())
}

fn empty_result(
    summary: &AssemblySummary,
    options: &TransientSolveOptions,
    start_time_s: f64,
) -> TransientSolveResult {
    TransientSolveResult {
        converged_steps: 0,
        total_steps: 0,
        time_points_s: vec![start_time_s],
        displacement_snapshots: vec![Vec::new()],
        velocity_snapshots: vec![Vec::new()],
        acceleration_snapshots: vec![Vec::new()],
        residual_norms: vec![0.0],
        accepted_time_steps_s: Vec::new(),
        diagnostics: transient_diagnostics(
            options,
            &[],
            &[],
            &[0.0],
            &[0.0],
            LinearAlgebraBackendKind::CpuReference,
        ),
        solver_method: SolverMethod::NewmarkAverageAccelerationPcg,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        solver_host_sync_count: 0,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        preconditioner: SolverPreconditioner::None,
        accelerator_evidence: Vec::new(),
        assembly_snapshots: vec![summary.clone()],
        strain_energy_snapshots: vec![0.0],
        kinetic_energy_snapshots: vec![0.0],
    }
}

struct SolveAggregate {
    solver_backend: LinearAlgebraBackendKind,
    host_sync_count: u32,
    device_apply_k_count: u32,
    device_apply_k_attempt_count: u32,
    preconditioner: SolverPreconditioner,
    accelerator_evidence: Vec<crate::solve::accelerator::AcceleratorSolveEvidence>,
}

impl Default for SolveAggregate {
    fn default() -> Self {
        Self {
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            host_sync_count: 0,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            preconditioner: SolverPreconditioner::None,
            accelerator_evidence: Vec::new(),
        }
    }
}

impl SolveAggregate {
    fn push(&mut self, solved: &EffectiveDynamicsSolveResult) {
        for linear in solved.linear_solves() {
            self.solver_backend = linear.solver_backend;
            self.host_sync_count = self.host_sync_count.saturating_add(linear.host_sync_count);
            self.device_apply_k_count = self
                .device_apply_k_count
                .saturating_add(linear.device_apply_k_count);
            self.device_apply_k_attempt_count = self
                .device_apply_k_attempt_count
                .saturating_add(linear.device_apply_k_attempt_count);
            self.preconditioner = linear.preconditioner;
            self.accelerator_evidence
                .extend(linear.accelerator_evidence.iter().cloned());
        }
    }
}

fn strain_energy(summary: &AssemblySummary, state: &SecondOrderState) -> f64 {
    let force = spmv(
        summary.operator.operators().stiffness(),
        state.displacement(),
    )
    .expect("validated structural displacement dimensions");
    0.5 * dot(state.displacement(), &force)
}

fn kinetic_energy(summary: &AssemblySummary, state: &SecondOrderState) -> f64 {
    let momentum = spmv(summary.operator.operators().mass(), state.velocity())
        .expect("validated structural velocity dimensions");
    0.5 * dot(state.velocity(), &momentum)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}
