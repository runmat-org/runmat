//! Genuine small-strain canonical-solid quasistatic equilibrium.
use crate::{
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::FeaDiagnostic,
    solve::backend::kind::LinearAlgebraBackendKind,
};
use serde::{Deserialize, Serialize};

mod driver;
mod events;
mod options;
mod schedule;
mod work;
pub(crate) use work::NonlinearWorkLimits;
pub(crate) mod solid;
#[cfg(test)]
pub(crate) use driver::solve_nonlinear_system;
pub(crate) use driver::solve_nonlinear_system_at_boundaries;
pub(crate) use driver::{require_fixed_structural_nonlinear, validate_original_terminal_result};
pub(crate) use driver::{
    run_nonlinear_load_block, NonlinearLoadBlockRequest, NonlinearLoadBlockStart,
};
pub(crate) use driver::{ContactBlockExecution, NonlinearExecution};
pub use driver::{NonlinearLoadContinuation, NonlinearLoadWork};
pub(crate) use events::{detect_completed_nonlinear_events, nonlinear_solver_content_digest};
pub use options::NonlinearSolveOptions;
pub(crate) use schedule::LoadIncrementSchedule;
pub use solid::{AcceptedContactEnergy, AcceptedSolidState};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearSolveResult {
    /// Complete terminal converged history. Averaged output fields are not restart authority.
    /// This carrier alone does not authorize durable publication or numerical restart.
    pub accepted_state: AcceptedSolidState,
    pub converged_increments: usize,
    pub total_increments: usize,
    pub initial_load_factor: f64,
    pub load_factors: Vec<f64>,
    pub initial_displacement: Vec<f64>,
    pub initial_equivalent_plastic_strain: Vec<f64>,
    pub initial_contact_pressure: Vec<f64>,
    pub initial_contact_gap: Vec<f64>,
    pub displacement_snapshots: Vec<Vec<f64>>,
    /// Accepted physical force, before subtracting the scaled external force.
    pub internal_force_snapshots: Vec<Vec<f64>>,
    /// Volume-weighted accepted integration-point stress: xx, yy, zz, yz, xz, xy.
    pub stress_snapshots: Vec<Vec<f64>>,
    /// Volume-weighted original total engineering strain: xx, yy, zz, yz, xz, xy.
    pub total_strain_snapshots: Vec<Vec<f64>>,
    pub von_mises_snapshots: Vec<Vec<f64>>,
    pub plastic_strain_snapshots: Vec<Vec<f64>>,
    pub equivalent_plastic_strain_snapshots: Vec<Vec<f64>>,
    pub contact_pressure_snapshots: Vec<Vec<f64>>,
    pub contact_gap_snapshots: Vec<Vec<f64>>,
    pub residual_norms: Vec<f64>,
    pub increment_norms: Vec<f64>,
    /// Accepted undamped pending correction divided by offset-free motion scale.
    pub relative_increment_norms: Vec<f64>,
    pub iteration_counts: Vec<usize>,
    /// Rejected trial increments retried by cutback; every returned accepted state converged.
    pub failed_increments: usize,
    pub line_search_backtracks: usize,
    pub max_line_search_backtracks_per_increment: usize,
    pub tangent_rebuild_count: usize,
    pub iteration_spike_count: usize,
    pub convergence_stall_count: usize,
    pub backtrack_burst_count: usize,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub solver_method: SolverMethod,
    pub solver_backend: LinearAlgebraBackendKind,
    pub solver_host_sync_count: u32,
    pub device_apply_k_count: u32,
    pub device_apply_k_attempt_count: u32,
    pub preconditioner: SolverPreconditioner,
    pub accelerator_evidence: Vec<crate::solve::accelerator::AcceleratorSolveEvidence>,
}
