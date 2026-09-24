use serde::{Deserialize, Serialize};

use crate::{
    algebra::{
        validate_prepared_solver_compatibility, AssembledRealSystem, CompatibilityError,
        PreparedRealLinearSystem, RealPreconditioner, RealSolver, RealSystemError, ScopedSpmvError,
    },
    assembly::AssemblySummary,
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    planning::resource::LocalExecutionPlan,
    solve::preconditioner::{build_spd_preconditioner, SpdPreconditionerKind},
    solve::{
        accelerator::{prepare_accelerator_linear_system, solve_prepared_accelerator_system},
        affine::{solve_affine_constraint_system, solve_affine_constraint_system_with_execution},
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

/// Uses the admitted local CPU plan for ordinary and affine-reduced static stiffness application.
/// Multiplier systems retain their serial solver until their GMRES callback can propagate scoped
/// execution failure.
pub fn solve_linear_system_with_execution(
    summary: &AssemblySummary,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    plan: LocalExecutionPlan,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    if let Some(reduction) = summary.affine_constraint_reduction.as_ref() {
        return solve_affine_constraint_system_with_execution(
            summary,
            reduction,
            backend_kind,
            algebra_backend,
            initial_guess,
            plan,
        );
    }
    if summary.constraint_matrix.is_some() {
        return Ok(solve_linear_system(
            summary,
            preconditioner_kind,
            backend_kind,
            algebra_backend,
            initial_guess,
        ));
    }
    solve_operator_system_scoped(
        &summary.operator,
        preconditioner_kind,
        backend_kind,
        algebra_backend,
        initial_guess,
        plan,
    )
}

fn solve_operator_system_scoped(
    operator: &AssembledRealSystem,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    plan: LocalExecutionPlan,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    solve_operator_system_with_apply(
        operator,
        preconditioner_kind,
        backend_kind,
        algebra_backend,
        initial_guess,
        |prepared, x| {
            if plan.is_serial() {
                Ok(crate::algebra::apply_prepared(prepared, x))
            } else {
                crate::algebra::spmv_scoped(prepared.matrix(), x, plan)
            }
        },
    )
}

pub fn solve_operator_system(
    operator: &AssembledRealSystem,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
) -> LinearSolveResult {
    solve_operator_system_with_apply(
        operator,
        preconditioner_kind,
        backend_kind,
        algebra_backend,
        initial_guess,
        |prepared, x| Ok(crate::algebra::apply_prepared(prepared, x)),
    )
    .expect("reference sparse application is infallible for a prepared system")
}

fn solve_operator_system_with_apply(
    operator: &AssembledRealSystem,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    apply: impl Fn(&PreparedRealLinearSystem, &[f64]) -> Result<Vec<f64>, ScopedSpmvError>,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    let mut effective_backend_kind = backend_kind;
    let mut accelerator_fallback = false;
    let has_dofs = operator.dof_count() > 0;
    if !has_dofs {
        return Ok(LinearSolveResult {
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
        });
    }

    let tol = 1.0e-8;
    let prepared = match operator.prepare_linear() {
        Ok(prepared) => prepared,
        Err(error) => {
            return Ok(invalid_system_result(
                operator,
                effective_backend_kind,
                preconditioner_kind,
                error,
            ))
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
        return Ok(incompatible_system_result(
            operator,
            effective_backend_kind,
            preconditioner_kind,
            error,
        ));
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
                return Ok(linear_result_from_accelerator_output(
                    output,
                    preconditioner_kind,
                ))
            }
            Err(error) if error.permits_cpu_fallback() => {
                effective_backend_kind = LinearAlgebraBackendKind::CpuReference;
                accelerator_fallback = true;
            }
            Err(error) => {
                return Ok(linear_result_from_accelerator_error(
                    operator.dof_count(),
                    operator.load(),
                    preconditioner_kind,
                    error,
                ))
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
            return Ok(rejected_system_result(
                operator,
                effective_backend_kind,
                preconditioner_kind,
                "FEA_PRECONDITIONER_SETUP_FAILED",
                error.to_string(),
            ))
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
            return Ok(rejected_system_result(
                operator,
                effective_backend_kind,
                preconditioner_kind,
                "FEA_INVALID_INITIAL_GUESS",
                "initial guess must contain one finite value per solver DOF".to_string(),
            ))
        }
        None => vec![0.0; operator.dof_count()],
    };
    for (index, constrained) in prepared.constrained_dofs().iter().copied().enumerate() {
        if constrained {
            x[index] = b[index];
        }
    }

    let mut r = algebra_backend.vec_sub(b, &apply(&prepared, &x)?);
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
        return Ok(LinearSolveResult {
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
        });
    }

    let mut iterations = 0u32;
    for _ in 0..max_iters {
        if crate::is_fea_cancelled() {
            break;
        }
        let ap = apply(&prepared, &p)?;
        let denom = algebra_backend.dot(&p, &ap);
        // PCG is scale invariant. An absolute cutoff here rejects valid systems when the
        // preconditioned search direction is small (a common case for dynamics operators and a
        // good warm start). For an admitted SPD operator only non-positive or non-finite
        // curvature is a breakdown.
        if !denom.is_finite() || denom <= 0.0 {
            break;
        }
        let alpha = rz_old / denom;
        algebra_backend.axpy(alpha, &p, &mut x);
        algebra_backend.axpy(-alpha, &ap, &mut r);
        let residual_norm = algebra_backend.dot(&r, &r).sqrt();
        iterations += 1;
        if residual_norm / b_norm <= tol {
            break;
        }

        z = preconditioner.apply(&r);
        let rz_new = algebra_backend.dot(&r, &z);
        if !rz_new.is_finite() || !rz_old.is_finite() || rz_old <= 0.0 || rz_new < 0.0 {
            break;
        }
        let beta = rz_new / rz_old;
        for i in 0..p.len() {
            p[i] = z[i] + beta * p[i];
        }
        rz_old = rz_new;
    }

    // Recurrence residuals can drift from b - A*x. Make the public convergence decision from a
    // freshly evaluated equation residual so downstream physics never inherits a false positive
    // or false negative from recurrence roundoff.
    r = algebra_backend.vec_sub(b, &apply(&prepared, &x)?);
    let residual_norm = algebra_backend.dot(&r, &r).sqrt();
    let converged = residual_norm / b_norm <= tol;
    if !converged {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_CG_NOT_CONVERGED".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: format!(
                "matrix-free cg stopped after {iterations}/{max_iters} iterations with residual_norm={residual_norm} relative_residual={} tolerance={tol}",
                residual_norm / b_norm,
            ),
        });
    }

    Ok(LinearSolveResult {
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
    })
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
    let terminal = error.phase;
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
            code: match terminal {
                crate::solve::accelerator::AcceleratorAdapterPhase::Cancellation => {
                    "FEA_SOLVER_CANCELLED"
                }
                crate::solve::accelerator::AcceleratorAdapterPhase::ProviderLoss => {
                    "FEA_ACCELERATOR_PROVIDER_LOST"
                }
                _ => "FEA_ACCELERATOR_SOLVE_FAILED",
            }
            .to_string(),
            severity: match terminal {
                crate::solve::accelerator::AcceleratorAdapterPhase::Cancellation => {
                    FeaDiagnosticSeverity::Warning
                }
                _ => FeaDiagnosticSeverity::Error,
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
#[path = "linear/tests.rs"]
mod tests;
