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
    if matches!(
        error,
        RealSystemError::Coercivity(
            crate::algebra::CoercivityError::FactorAdmission { .. }
                | crate::algebra::CoercivityError::AdmissionDimension { .. }
                | crate::algebra::CoercivityError::ResourceLimit { .. }
        )
    ) {
        return rejected_system_result(
            operator,
            backend,
            preconditioner,
            "FEA_SOLVER_WORK_EXCEEDS_ADMISSION",
            error.to_string(),
        );
    }
    if matches!(
        error,
        RealSystemError::Coercivity(crate::algebra::CoercivityError::Cancelled)
    ) {
        let mut result = rejected_system_result(
            operator,
            backend,
            preconditioner,
            "FEA_SOLVER_CANCELLED",
            "linear solve cancelled while proving operator coercivity".into(),
        );
        result.diagnostics[0].severity = FeaDiagnosticSeverity::Warning;
        return result;
    }
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
        spectral_estimate: None,
        iterations: 0,
        residual_norm: operator
            .load()
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt(),
        relative_residual: None,
        converged: false,
        host_sync_count: 0,
        solver_backend: backend,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution: vec![0.0; operator.dof_count()],
        solver_method: requested_provenance(preconditioner).0,
        preconditioner: requested_provenance(preconditioner).1,
        diagnostics: vec![FeaDiagnostic {
            code: code.to_string(),
            severity: FeaDiagnosticSeverity::Error,
            message,
        }],
    }
}

/// Provenance of a requested SPD selection before the policy resolves it (rejections, empty
/// systems).
pub(crate) fn requested_provenance(
    kind: SpdPreconditionerKind,
) -> (SolverMethod, SolverPreconditioner) {
    match kind {
        SpdPreconditionerKind::Auto | SpdPreconditionerKind::Direct => (
            SolverMethod::SparseDirectCholesky,
            SolverPreconditioner::None,
        ),
        other => (
            SolverMethod::AssembledPcg,
            SolverPreconditioner::from_spd(other),
        ),
    }
}

/// Constraint matrices that admit no affine reduction cannot be solved.
pub(crate) fn rejected_constraint_system(
    summary: &super::StructuralLinearSystemView<'_>,
    message: String,
) -> LinearSolveResult {
    rejected_system_result(
        summary.operator,
        LinearAlgebraBackendKind::CpuReference,
        SpdPreconditionerKind::Auto,
        "FEA_INVALID_MULTIPLIER_CONSTRAINT_SYSTEM",
        message,
    )
}
