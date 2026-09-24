use crate::{
    algebra::{apply_k, dot_scoped, spmv_scoped, PreparedRealLinearSystem, ScopedSpmvError},
    assembly::AssemblySummary,
    constraints::AffineConstraintReduction,
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    planning::resource::LocalExecutionPlan,
    solve::{backend::kind::LinearAlgebraBackendKind, linear::LinearSolveResult},
};

use super::backend::linear_algebra::LinearAlgebraBackend;

pub(super) fn solve_affine_constraint_system(
    summary: &AssemblySummary,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
) -> LinearSolveResult {
    solve_affine_constraint_system_with_apply(
        summary,
        reduction,
        requested_backend,
        algebra_backend,
        initial_guess,
        (
            |full: &[f64]| Ok(apply_k(&summary.operator, full)),
            |left: &[f64], right: &[f64]| Ok(algebra_backend.dot(left, right)),
            None,
        ),
    )
    .expect("reference sparse application is infallible for an assembled system")
}

pub(super) fn solve_affine_constraint_system_with_execution(
    summary: &AssemblySummary,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    plan: LocalExecutionPlan,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    solve_affine_constraint_system_with_apply(
        summary,
        reduction,
        requested_backend,
        algebra_backend,
        initial_guess,
        (
            |full: &[f64]| {
                if plan.is_serial() {
                    Ok(apply_k(&summary.operator, full))
                } else {
                    spmv_scoped(summary.operator.operators().stiffness(), full, plan)
                }
            },
            |left: &[f64], right: &[f64]| {
                if plan.is_serial() {
                    Ok(algebra_backend.dot(left, right))
                } else {
                    dot_scoped(left, right, plan)
                }
            },
            Some(plan),
        ),
    )
}

fn solve_affine_constraint_system_with_apply(
    summary: &AssemblySummary,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    operations: (
        impl Fn(&[f64]) -> Result<Vec<f64>, ScopedSpmvError>,
        impl Fn(&[f64], &[f64]) -> Result<f64, ScopedSpmvError>,
        Option<LocalExecutionPlan>,
    ),
) -> Result<LinearSolveResult, ScopedSpmvError> {
    let (apply_full, dot, vector_plan) = operations;
    let expected_reduction = summary
        .constraint_matrix
        .as_ref()
        .and_then(|matrix| matrix.affine_reduction().ok());
    if expected_reduction.as_ref() != Some(reduction) {
        return Ok(rejected_affine_system(
            summary,
            "affine reduction differs from the canonical resolved constraint matrix",
        ));
    }
    let (reduced_operator, reduced_rhs) = match reduction.reduce_sparse_system(
        summary.operator.operators().stiffness(),
        summary.operator.load(),
    ) {
        Ok(reduced) => reduced,
        Err(error) => return Ok(rejected_affine_system(summary, &error.to_string())),
    };
    let mut reduced_solution = initial_guess
        .map(|guess| {
            reduction
                .free_dof_indices
                .iter()
                .map(|dof| guess.get(*dof).copied().unwrap_or(0.0))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![0.0; reduction.reduced_dof_count]);
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_AFFINE_CONSTRAINT_REDUCTION".into(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "full_dofs={} reduced_dofs={} constraint_rank={}",
            reduction.full_dof_count,
            reduction.reduced_dof_count,
            reduction.pivot_dof_indices.len()
        ),
    }];
    if requested_backend == LinearAlgebraBackendKind::Accelerator {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_AFFINE_CPU_REFERENCE".into(),
            severity: FeaDiagnosticSeverity::Warning,
            message: "affine constraint reduction is executing on the CPU reference backend".into(),
        });
    }

    if reduction.reduced_dof_count == 0 {
        return Ok(LinearSolveResult {
            iterations: 0,
            residual_norm: 0.0,
            converged: true,
            host_sync_count: 0,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            accelerator_evidence: None,
            solution: reduction.offset.clone(),
            solver_method: SolverMethod::MatrixFreePcg,
            preconditioner: SolverPreconditioner::None,
            diagnostics,
        });
    }

    if let Err(error) = PreparedRealLinearSystem::try_new(
        reduced_operator,
        reduced_rhs.clone(),
        vec![false; reduction.reduced_dof_count],
    ) {
        return Ok(rejected_affine_system(summary, &error.to_string()));
    }

    let mut residual = algebra_backend.vec_sub(
        &reduced_rhs,
        &apply_reduced_operator(reduction, &reduced_solution, &apply_full)?,
    );
    let mut direction = residual.clone();
    let mut residual_squared = dot(&residual, &residual)?;
    let right_hand_side_norm = dot(&reduced_rhs, &reduced_rhs)?.sqrt().max(1.0);
    let tolerance = 1.0e-8;
    let maximum_iterations = reduction.reduced_dof_count.saturating_mul(4).max(256);
    let mut iterations = 0_u32;
    let mut converged = residual_squared.sqrt() / right_hand_side_norm <= tolerance;

    while !converged && (iterations as usize) < maximum_iterations {
        if crate::is_fea_cancelled() {
            diagnostics.push(FeaDiagnostic {
                code: "FEA_SOLVER_CANCELLED".into(),
                severity: FeaDiagnosticSeverity::Warning,
                message: "affine reduced solve cancelled".into(),
            });
            break;
        }
        let operator_direction = apply_reduced_operator(reduction, &direction, &apply_full)?;
        let denominator = dot(&direction, &operator_direction)?;
        if !denominator.is_finite() || denominator <= 1.0e-18 {
            diagnostics.push(FeaDiagnostic {
                code: "FEA_AFFINE_REDUCED_BREAKDOWN".into(),
                severity: FeaDiagnosticSeverity::Warning,
                message: format!(
                    "reduced conjugate-gradient operator lost positive definiteness at iteration {}",
                    iterations + 1
                ),
            });
            break;
        }
        let alpha = residual_squared / denominator;
        if let Some(plan) = vector_plan.filter(|plan| !plan.is_serial()) {
            crate::algebra::axpy_scoped(alpha, &direction, &mut reduced_solution, plan)?;
            crate::algebra::axpy_scoped(-alpha, &operator_direction, &mut residual, plan)?;
        } else {
            algebra_backend.axpy(alpha, &direction, &mut reduced_solution);
            algebra_backend.axpy(-alpha, &operator_direction, &mut residual);
        }
        let next_residual_squared = dot(&residual, &residual)?;
        iterations += 1;
        converged = next_residual_squared.sqrt() / right_hand_side_norm <= tolerance;
        if converged {
            residual_squared = next_residual_squared;
            break;
        }
        let beta = next_residual_squared / residual_squared;
        if let Some(plan) = vector_plan.filter(|plan| !plan.is_serial()) {
            crate::algebra::direction_update_scoped(&residual, &mut direction, beta, plan)?;
        } else {
            for (direction, residual) in direction.iter_mut().zip(&residual) {
                *direction = *residual + beta * *direction;
            }
        }
        residual_squared = next_residual_squared;
    }

    diagnostics.push(FeaDiagnostic {
        code: "FEA_SOLVER_METHOD".into(),
        severity: FeaDiagnosticSeverity::Info,
        message: "solver=pcg space=affine_reduced preconditioner=none matrix_free=true".into(),
    });
    Ok(LinearSolveResult {
        iterations,
        residual_norm: residual_squared.sqrt(),
        converged,
        host_sync_count: 0,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution: expand(reduction, &reduced_solution),
        solver_method: SolverMethod::MatrixFreePcg,
        preconditioner: SolverPreconditioner::None,
        diagnostics,
    })
}

fn rejected_affine_system(summary: &AssemblySummary, message: &str) -> LinearSolveResult {
    LinearSolveResult {
        iterations: 0,
        residual_norm: summary
            .operator
            .load()
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt(),
        converged: false,
        host_sync_count: 0,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution: vec![0.0; summary.dof_count],
        solver_method: SolverMethod::MatrixFreePcg,
        preconditioner: SolverPreconditioner::None,
        diagnostics: vec![FeaDiagnostic {
            code: "FEA_INVALID_AFFINE_SOLVER_ADMISSION".into(),
            severity: FeaDiagnosticSeverity::Error,
            message: message.into(),
        }],
    }
}

fn apply_reduced_operator(
    reduction: &AffineConstraintReduction,
    reduced: &[f64],
    apply_full: &impl Fn(&[f64]) -> Result<Vec<f64>, ScopedSpmvError>,
) -> Result<Vec<f64>, ScopedSpmvError> {
    let full = expand_linear(reduction, reduced);
    Ok(transpose_apply(reduction, &apply_full(&full)?))
}

fn expand(reduction: &AffineConstraintReduction, reduced: &[f64]) -> Vec<f64> {
    let mut full = expand_linear(reduction, reduced);
    for (value, offset) in full.iter_mut().zip(&reduction.offset) {
        *value += offset;
    }
    full
}

fn expand_linear(reduction: &AffineConstraintReduction, reduced: &[f64]) -> Vec<f64> {
    (0..reduction.full_dof_count)
        .map(|row| {
            let start = reduction.recovery_row_offsets[row];
            let end = reduction.recovery_row_offsets[row + 1];
            (start..end)
                .map(|entry| {
                    reduction.recovery_coefficients[entry]
                        * reduced[reduction.recovery_column_indices[entry]]
                })
                .sum()
        })
        .collect()
}

fn transpose_apply(reduction: &AffineConstraintReduction, full: &[f64]) -> Vec<f64> {
    let mut reduced = vec![0.0; reduction.reduced_dof_count];
    for (row, value) in full.iter().copied().enumerate() {
        let start = reduction.recovery_row_offsets[row];
        let end = reduction.recovery_row_offsets[row + 1];
        for entry in start..end {
            reduced[reduction.recovery_column_indices[entry]] +=
                reduction.recovery_coefficients[entry] * value;
        }
    }
    reduced
}
