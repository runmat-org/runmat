use crate::{
    algebra::{AssembledRealSystem, CompatibilityError, RealSystemError},
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::{backend::kind::LinearAlgebraBackendKind, preconditioner::SpdPreconditionerKind},
};

use super::LinearSolveResult;

pub(super) fn invalid_system_result(
    operator: &AssembledRealSystem,
    backend: LinearAlgebraBackendKind,
    preconditioner: SpdPreconditionerKind,
    error: RealSystemError,
) -> LinearSolveResult {
    rejected_system_result(
        operator,
        backend,
        preconditioner,
        "FEA_INVALID_REAL_SYSTEM",
        format!("real linear-system preparation failed: {error:?}"),
    )
}

pub(super) fn incompatible_system_result(
    operator: &AssembledRealSystem,
    backend: LinearAlgebraBackendKind,
    preconditioner: SpdPreconditionerKind,
    error: CompatibilityError,
) -> LinearSolveResult {
    rejected_system_result(
        operator,
        backend,
        preconditioner,
        "FEA_INCOMPATIBLE_REAL_SOLVER",
        error.to_string(),
    )
}

pub(super) fn rejected_system_result(
    operator: &AssembledRealSystem,
    backend: LinearAlgebraBackendKind,
    preconditioner: SpdPreconditionerKind,
    code: &str,
    message: String,
) -> LinearSolveResult {
    LinearSolveResult {
        iterations: 0,
        residual_norm: operator
            .load()
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt(),
        converged: false,
        host_sync_count: 0,
        solver_backend: backend,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution: vec![0.0; operator.dof_count()],
        solver_method: SolverMethod::MatrixFreePcg,
        preconditioner: SolverPreconditioner::from_spd(preconditioner),
        diagnostics: vec![FeaDiagnostic {
            code: code.to_string(),
            severity: FeaDiagnosticSeverity::Error,
            message,
        }],
    }
}
