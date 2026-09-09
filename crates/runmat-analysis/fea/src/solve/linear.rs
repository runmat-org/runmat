use serde::{Deserialize, Serialize};

use crate::{
    algebra::{
        validate_prepared_solver_compatibility, AssembledRealSystem, CompatibilityError,
        RealPreconditioner, RealSolver, RealSystemError,
    },
    assembly::AssemblySummary,
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::preconditioner::{build_spd_preconditioner, SpdPreconditionerKind},
    solve::{
        accelerator::{prepare_accelerator_linear_system, solve_prepared_accelerator_system},
        affine::solve_affine_constraint_system,
        backend::{kind::LinearAlgebraBackendKind, linear_algebra::LinearAlgebraBackend},
        multiplier::solve_multiplier_constraint_system,
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
    pub accelerator_evidence: Option<crate::solve::accelerator::AcceleratorSolveEvidence>,
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
    let mut accelerator_fallback = false;
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
            accelerator_evidence: None,
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
    if let Err(error) = validate_prepared_solver_compatibility(
        &prepared,
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
    // Compatibility is a solver-independent admission boundary. A runtime provider must never
    // receive a system that the reference path would reject, and provider fallback must not alter
    // that decision.
    if backend_kind == LinearAlgebraBackendKind::Accelerator {
        let maximum_iterations = u32::try_from(iteration_limit(operator, 64)).unwrap_or(u32::MAX);
        match prepare_accelerator_linear_system(prepared.clone(), maximum_iterations).and_then(
            |accelerator| {
                solve_prepared_accelerator_system(&accelerator, prepared.rhs(), initial_guess, tol)
            },
        ) {
            Ok(output) => {
                return linear_result_from_accelerator_output(output, preconditioner_kind)
            }
            Err(error) if error.permits_cpu_fallback() => {
                effective_backend_kind = LinearAlgebraBackendKind::CpuReference;
                accelerator_fallback = true;
            }
            Err(error) => {
                return linear_result_from_accelerator_error(
                    operator.dof_count(),
                    operator.load(),
                    preconditioner_kind,
                    error,
                )
            }
        }
    }
    let mut initial_diagnostics = Vec::new();
    if accelerator_fallback {
        initial_diagnostics.push(FeaDiagnostic {
            code: "FEA_ACCELERATOR_ADMISSION_FALLBACK".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: "native sparse execution was rejected before provider allocation; using the cpu_reference solver"
                .to_string(),
        });
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
    for (index, constrained) in prepared.constrained_dofs().iter().copied().enumerate() {
        if constrained {
            x[index] = b[index];
        }
    }

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
            accelerator_evidence: None,
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
        if !denom.is_finite() || denom <= 1.0e-18 {
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
        accelerator_evidence: None,
        solution: x,
        solver_method: SolverMethod::MatrixFreePcg,
        preconditioner: SolverPreconditioner::from_spd(preconditioner.kind()),
        diagnostics,
    }
}

pub(crate) fn linear_result_from_accelerator_output(
    output: crate::solve::accelerator::AcceleratorSolveOutput,
    preconditioner_kind: SpdPreconditionerKind,
) -> LinearSolveResult {
    let provider_evidence = output.evidence.provider();
    let host_sync_count = provider_evidence.host_synchronizations.saturating_add(1);
    let device_apply_k_count = u32::try_from(provider_evidence.spmv_dispatches).unwrap_or(u32::MAX);
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_SOLVER_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "solver=pcg preconditioner={} matrix_free=false backend=accelerator_native_sparse",
            preconditioner_kind.as_str()
        ),
    }];
    if !output.converged {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_ACCELERATOR_TOLERANCE_NOT_SATISFIED".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: "native sparse solve did not satisfy the independently recomputed residual tolerance"
                .to_string(),
        });
    }
    LinearSolveResult {
        iterations: output.iterations,
        residual_norm: output
            .evidence
            .independent_residual_norm()
            .expect("completed accelerator evidence has a residual"),
        converged: output.converged,
        host_sync_count,
        solver_backend: LinearAlgebraBackendKind::Accelerator,
        device_apply_k_count,
        device_apply_k_attempt_count: device_apply_k_count,
        accelerator_evidence: Some(output.evidence),
        solution: output.solution,
        solver_method: SolverMethod::AssembledPcg,
        preconditioner: SolverPreconditioner::from_spd(preconditioner_kind),
        diagnostics,
    }
}

pub(crate) fn linear_result_from_accelerator_error(
    dof_count: usize,
    right_hand_side: &[f64],
    preconditioner_kind: SpdPreconditionerKind,
    error: crate::solve::accelerator::AcceleratorAdapterError,
) -> LinearSolveResult {
    let cancelled = error.phase == crate::solve::accelerator::AcceleratorAdapterPhase::Cancellation;
    LinearSolveResult {
        iterations: 0,
        residual_norm: right_hand_side
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt(),
        converged: false,
        host_sync_count: error
            .evidence
            .as_ref()
            .map_or(0, |evidence| evidence.provider().host_synchronizations),
        solver_backend: LinearAlgebraBackendKind::Accelerator,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: error.evidence.map(|evidence| *evidence),
        solution: vec![0.0; dof_count],
        solver_method: SolverMethod::AssembledPcg,
        preconditioner: SolverPreconditioner::from_spd(preconditioner_kind),
        diagnostics: vec![FeaDiagnostic {
            code: if cancelled {
                "FEA_SOLVER_CANCELLED".to_string()
            } else {
                "FEA_ACCELERATOR_SOLVE_FAILED".to_string()
            },
            severity: if cancelled {
                FeaDiagnosticSeverity::Warning
            } else {
                FeaDiagnosticSeverity::Error
            },
            message: error.message,
        }],
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
    fn noncoercive_operator_is_rejected_before_preconditioner_setup() {
        let operator = system([0.0, 3.0]);
        let result = solve_operator_system(
            &operator,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
        );
        assert!(!result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.diagnostics[0].code, "FEA_INVALID_REAL_SYSTEM");
    }

    #[test]
    fn cpu_and_accelerator_reject_unproved_semidefinite_system_before_iteration() {
        let stiffness = RealSparseMatrix::diagonal(
            vec![1.0, 0.0],
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .unwrap();
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
        let operator = AssembledRealSystem::try_new(
            CanonicalRealOperators::try_new(stiffness, mass, damping).unwrap(),
            vec![false; 2],
            vec![1.0, 0.0],
        )
        .unwrap();

        for backend in [
            LinearAlgebraBackendKind::CpuReference,
            LinearAlgebraBackendKind::Accelerator,
        ] {
            let result = solve_operator_system(
                &operator,
                SpdPreconditionerKind::Jacobi,
                backend,
                &CpuReferenceBackend,
                None,
            );
            assert!(!result.converged);
            assert_eq!(result.iterations, 0);
            assert_eq!(result.device_apply_k_attempt_count, 0);
            assert_eq!(result.diagnostics[0].code, "FEA_INVALID_REAL_SYSTEM");
        }
    }
}
