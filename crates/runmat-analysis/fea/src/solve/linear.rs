use serde::{Deserialize, Serialize};

use crate::{
    algebra::{
        validate_solver_compatibility, AssembledRealSystem, CompatibilityError, RealPreconditioner,
        RealSolver, RealSystemError,
    },
    assembly::AssemblySummary,
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::preconditioner::{build_spd_preconditioner, SpdPreconditionerKind},
    solve::{
        affine::solve_affine_constraint_system,
        backend::{kind::LinearAlgebraBackendKind, linear_algebra::LinearAlgebraBackend},
        multiplier::solve_multiplier_constraint_system,
        runtime_tensor_solver::solve_operator_system_runtime_tensor_with_initial_guess,
    },
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearSolveResult {
    pub iterations: u32,
    pub residual_norm: f64,
    pub converged: bool,
    pub host_sync_count: u32,
    pub solver_backend: LinearAlgebraBackendKind,
    pub device_apply_k_count: u32,
    pub device_apply_k_attempt_count: u32,
    pub solution: Vec<f64>,
    pub solver_method: SolverMethod,
    pub preconditioner: SolverPreconditioner,
    pub diagnostics: Vec<FeaDiagnostic>,
}

fn invalid_system_result(
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

fn incompatible_system_result(
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

fn rejected_system_result(
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

pub fn solve_linear_system(
    summary: &AssemblySummary,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
) -> LinearSolveResult {
    if let Some(reduction) = summary.affine_constraint_reduction.as_ref() {
        return solve_affine_constraint_system(
            summary,
            reduction,
            backend_kind,
            algebra_backend,
            initial_guess,
        );
    }
    if let Some(matrix) = summary.constraint_matrix.as_ref() {
        return solve_multiplier_constraint_system(
            summary,
            matrix,
            backend_kind,
            algebra_backend,
            initial_guess,
        );
    }
    solve_operator_system(
        &summary.operator,
        preconditioner_kind,
        backend_kind,
        algebra_backend,
        initial_guess,
    )
}

pub fn solve_operator_system(
    operator: &AssembledRealSystem,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
) -> LinearSolveResult {
    let mut effective_backend_kind = backend_kind;
    let mut runtime_tensor_fallback = false;
    if backend_kind == LinearAlgebraBackendKind::RuntimeTensor {
        if let Some(result) = solve_operator_system_runtime_tensor_with_initial_guess(
            operator,
            preconditioner_kind,
            initial_guess,
        ) {
            return result;
        }
        effective_backend_kind = LinearAlgebraBackendKind::CpuReference;
        runtime_tensor_fallback = true;
    }

    let has_dofs = operator.dof_count() > 0;
    if !has_dofs {
        return LinearSolveResult {
            iterations: 0,
            residual_norm: 0.0,
            converged: false,
            host_sync_count: 0,
            solver_backend: effective_backend_kind,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            solution: Vec::new(),
            solver_method: SolverMethod::MatrixFreePcg,
            preconditioner: SolverPreconditioner::from_spd(preconditioner_kind),
            diagnostics: vec![FeaDiagnostic {
                code: "FEA_EMPTY_SYSTEM".to_string(),
                severity: FeaDiagnosticSeverity::Warning,
                message: "linear solve skipped because assembled system has zero DOFs".to_string(),
            }],
        };
    }

    let tol = 1.0e-8;
    let mut initial_diagnostics = Vec::new();
    if runtime_tensor_fallback {
        initial_diagnostics.push(FeaDiagnostic {
            code: "FEA_RUNTIME_TENSOR_FALLBACK".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: "runtime-tensor backend unavailable for current provider; fell back to cpu_reference solver"
                .to_string(),
        });
    }
    let prepared = match operator.prepare_linear() {
        Ok(prepared) => prepared,
        Err(error) => {
            return invalid_system_result(
                operator,
                effective_backend_kind,
                preconditioner_kind,
                error,
            )
        }
    };
    if let Err(error) = validate_solver_compatibility(
        prepared.matrix(),
        RealSolver::ConjugateGradient,
        match preconditioner_kind {
            SpdPreconditionerKind::Jacobi => RealPreconditioner::Jacobi,
            SpdPreconditionerKind::Ilu0 => RealPreconditioner::Ilu0,
        },
    ) {
        return incompatible_system_result(
            operator,
            effective_backend_kind,
            preconditioner_kind,
            error,
        );
    }
    let max_iters = iteration_limit(operator, 64);
    let preconditioner = match build_spd_preconditioner(&prepared, preconditioner_kind) {
        Ok(preconditioner) => preconditioner,
        Err(error) => {
            return rejected_system_result(
                operator,
                effective_backend_kind,
                preconditioner_kind,
                "FEA_PRECONDITIONER_SETUP_FAILED",
                error.to_string(),
            )
        }
    };
    let b = prepared.rhs();
    let mut x = match initial_guess {
        Some(values)
            if values.len() == operator.dof_count()
                && values.iter().all(|value| value.is_finite()) =>
        {
            values.to_vec()
        }
        Some(_) => {
            return rejected_system_result(
                operator,
                effective_backend_kind,
                preconditioner_kind,
                "FEA_INVALID_INITIAL_GUESS",
                "initial guess must contain one finite value per solver DOF".to_string(),
            )
        }
        None => vec![0.0; operator.dof_count()],
    };

    let mut r = algebra_backend.vec_sub(b, &crate::algebra::apply_prepared(&prepared, &x));
    let mut z = preconditioner.apply(&r);
    let mut p = z.clone();
    let mut rz_old = algebra_backend.dot(&r, &z);
    let b_norm = algebra_backend.dot(b, b).sqrt().max(1.0);

    let mut diagnostics = initial_diagnostics;
    diagnostics.push(FeaDiagnostic {
        code: "FEA_SOLVER_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "solver=pcg preconditioner={} matrix_free=true",
            preconditioner.kind().as_str()
        ),
    });

    if algebra_backend.dot(&r, &r).sqrt() / b_norm <= tol {
        return LinearSolveResult {
            iterations: 0,
            residual_norm: algebra_backend.dot(&r, &r).sqrt(),
            converged: true,
            host_sync_count: 0,
            solver_backend: effective_backend_kind,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            solution: x,
            solver_method: SolverMethod::MatrixFreePcg,
            preconditioner: SolverPreconditioner::from_spd(preconditioner.kind()),
            diagnostics,
        };
    }

    let mut iterations = 0u32;
    let mut converged = false;
    for _ in 0..max_iters {
        if crate::is_fea_cancelled() {
            break;
        }
        let ap = crate::algebra::apply_prepared(&prepared, &p);
        let denom = algebra_backend.dot(&p, &ap);
        if denom.abs() <= 1.0e-18 {
            break;
        }
        let alpha = rz_old / denom;
        algebra_backend.axpy(alpha, &p, &mut x);
        algebra_backend.axpy(-alpha, &ap, &mut r);
        let residual_norm = algebra_backend.dot(&r, &r).sqrt();
        iterations += 1;
        if residual_norm / b_norm <= tol {
            converged = true;
            break;
        }

        z = preconditioner.apply(&r);
        let rz_new = algebra_backend.dot(&r, &z);
        if rz_old.abs() <= 1.0e-18 {
            break;
        }
        let beta = rz_new / rz_old;
        for i in 0..p.len() {
            p[i] = z[i] + beta * p[i];
        }
        rz_old = rz_new;
    }

    let residual_norm = algebra_backend.dot(&r, &r).sqrt();
    if !converged {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_CG_MAX_ITERS".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: format!(
                "matrix-free cg reached max iterations ({max_iters}) with residual_norm={residual_norm}"
            ),
        });
    }

    LinearSolveResult {
        iterations,
        residual_norm,
        converged,
        host_sync_count: 0,
        solver_backend: effective_backend_kind,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        solution: x,
        solver_method: SolverMethod::MatrixFreePcg,
        preconditioner: SolverPreconditioner::from_spd(preconditioner.kind()),
        diagnostics,
    }
}

fn iteration_limit(operator: &AssembledRealSystem, base_max_iters: usize) -> usize {
    base_max_iters
        .max(operator.dof_count().saturating_mul(2))
        .max(256)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algebra::{
            CanonicalRealOperators, MatrixProperties, OperatorRole, RealSparseAssembler,
            RealSparseMatrix,
        },
        solve::backend::cpu_reference::CpuReferenceBackend,
    };

    fn system(stiffness_diagonal: [f64; 2]) -> AssembledRealSystem {
        let mut stiffness = RealSparseAssembler::square(
            2,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        );
        stiffness.add(0, 0, stiffness_diagonal[0]);
        stiffness.add(0, 1, -1.0);
        stiffness.add(1, 0, -1.0);
        stiffness.add(1, 1, stiffness_diagonal[1]);
        let mass = RealSparseMatrix::diagonal(
            vec![1.0; 2],
            OperatorRole::Mass,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let damping = RealSparseMatrix::diagonal(
            vec![0.0; 2],
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .unwrap();
        AssembledRealSystem::try_new(
            CanonicalRealOperators::try_new(stiffness.finish().unwrap(), mass, damping).unwrap(),
            vec![false; 2],
            vec![1.0, 2.0],
        )
        .unwrap()
    }

    #[test]
    fn cpu_reference_solve_converges_with_small_canonical_residual() {
        let operator = system([4.0, 3.0]);
        let result = solve_operator_system(
            &operator,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
        );
        assert!(result.converged);
        assert!(result.residual_norm <= 1.0e-8);
    }

    #[test]
    fn preconditioner_setup_failure_is_attributable() {
        let operator = system([0.0, 3.0]);
        let result = solve_operator_system(
            &operator,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
        );
        assert!(!result.converged);
        assert_eq!(
            result.diagnostics[0].code,
            "FEA_PRECONDITIONER_SETUP_FAILED"
        );
    }
}
