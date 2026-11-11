use serde::{Deserialize, Serialize};

use crate::{
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::FeaDiagnostic,
    solve::backend::kind::LinearAlgebraBackendKind,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalSolveResult {
    pub converged: bool,
    pub frequencies_hz: Vec<f64>,
    pub mode_shapes: Vec<Vec<f64>>,
    /// Relative residual `‖K x − λ M x‖₂ / (‖K x‖₂ + λ ‖M x‖₂)` of each mode on the reduced
    /// pencil.
    pub residual_norms: Vec<f64>,
    /// Normwise backward error `‖K x − λ M x‖_∞ / ((‖K‖_∞ + λ ‖M‖_∞) ‖x‖_∞)` of each mode on the
    /// reduced pencil.
    #[serde(default)]
    pub backward_errors: Vec<f64>,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub solver_method: SolverMethod,
    pub solver_backend: LinearAlgebraBackendKind,
    pub preconditioner: SolverPreconditioner,
    pub solver_host_sync_count: u32,
    pub device_apply_k_count: u32,
    pub device_apply_k_attempt_count: u32,
    pub accelerator_evidence: Vec<crate::solve::accelerator::AcceleratorSolveEvidence>,
    pub available_mode_count: usize,
    pub boundary_probe_count: usize,
    pub boundary_probe_frequency_hz: Option<f64>,
    /// The admitted subspace dimension when the search reached it with the truncation boundary
    /// still unresolved: the boundary cluster needs more vectors than the plan reserved.
    #[serde(default)]
    pub boundary_unresolved_at_capacity: Option<usize>,
}
