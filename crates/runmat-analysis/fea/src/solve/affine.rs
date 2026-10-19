use crate::{
    algebra::{apply_k, dot_scoped, spmv_scoped, PreparedRealLinearSystem, ScopedSpmvError},
    constraints::{AffineConstraintReduction, AffineTransposeIndex},
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    planning::resource::{LocalExecutionPlan, SolverAdmissionResourcePolicy},
    solve::linear::{ScopedLinearSolveSettings, StructuralLinearSystemView},
    solve::{backend::kind::LinearAlgebraBackendKind, linear::LinearSolveResult},
};

use super::backend::linear_algebra::LinearAlgebraBackend;

pub(super) fn solve_affine_constraint_system(
    summary: &StructuralLinearSystemView<'_>,
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
            |full: &[f64]| Ok(apply_k(summary.operator, full)),
            |left: &[f64], right: &[f64]| Ok(algebra_backend.dot(left, right)),
            None,
        ),
        ScopedLinearSolveSettings::with_default_tolerance(LocalExecutionPlan::serial(), None),
    )
    .expect("reference sparse application is infallible for an assembled system")
}

pub(super) fn solve_affine_constraint_system_with_execution(
    summary: &StructuralLinearSystemView<'_>,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    plan: LocalExecutionPlan,
    admission: Option<&SolverAdmissionResourcePolicy>,
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
                    Ok(apply_k(summary.operator, full))
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
        ScopedLinearSolveSettings::with_default_tolerance(plan, admission),
    )
}

fn solve_affine_constraint_system_with_apply(
    summary: &StructuralLinearSystemView<'_>,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    operations: (
        impl Fn(&[f64]) -> Result<Vec<f64>, ScopedSpmvError>,
        impl Fn(&[f64], &[f64]) -> Result<f64, ScopedSpmvError>,
        Option<LocalExecutionPlan>,
    ),
    settings: ScopedLinearSolveSettings<'_>,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    let admission = settings.admission;
    let reuse = settings.preparation;
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
            relative_residual: Some(0.0),
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

    let preparation = match reuse {
        Some(reuse) => PreparedRealLinearSystem::try_new_reusing(
            reduced_operator,
            reduced_rhs.clone(),
            vec![false; reduction.reduced_dof_count],
            reuse,
        ),
        None => PreparedRealLinearSystem::try_new(
            reduced_operator,
            reduced_rhs.clone(),
            vec![false; reduction.reduced_dof_count],
        ),
    };
    let prepared = match preparation {
        Ok(prepared) => prepared,
        Err(error) => return Ok(rejected_affine_system(summary, &error.to_string())),
    };
    if let Some(admission) = admission {
        let accepted = prepared
            .admissibility()
            .is_some_and(|evidence| admission.admits_certificate(evidence.coercivity()));
        if !accepted {
            return Ok(rejected_affine_system(
                summary,
                "exact reduced-solver work exceeds the admitted resource envelope",
            ));
        }
    }
    drop(prepared);
    let transpose_index = match vector_plan
        .filter(|plan| !plan.is_serial())
        .map(|_| AffineTransposeIndex::try_new(reduction))
        .transpose()
    {
        Ok(index) => index,
        Err(error) => return Ok(rejected_affine_system(summary, &error.to_string())),
    };

    let applied_initial = apply_reduced_operator(
        reduction,
        &reduced_solution,
        &apply_full,
        vector_plan,
        transpose_index.as_ref(),
    )?;
    let mut residual = if let Some(plan) = vector_plan.filter(|plan| !plan.is_serial()) {
        crate::algebra::vector_sub_scoped(&reduced_rhs, &applied_initial, plan)?
    } else {
        algebra_backend.vec_sub(&reduced_rhs, &applied_initial)
    };
    drop(applied_initial);
    let mut direction = residual.clone();
    let mut residual_squared = dot(&residual, &residual)?;
    let right_hand_side_norm = dot(&reduced_rhs, &reduced_rhs)?.sqrt();
    let initial_residual_norm = residual_squared.sqrt();
    let reference_norm = if right_hand_side_norm > 0.0 {
        right_hand_side_norm
    } else {
        initial_residual_norm
    };
    let tolerance = 1.0e-8;
    let maximum_iterations = reduction.reduced_dof_count.saturating_mul(4).max(256);
    let mut iterations = 0_u32;
    let mut converged = initial_residual_norm <= tolerance * reference_norm;

    while !converged && (iterations as usize) < maximum_iterations {
        if crate::is_fea_cancelled() {
            diagnostics.push(FeaDiagnostic {
                code: "FEA_SOLVER_CANCELLED".into(),
                severity: FeaDiagnosticSeverity::Warning,
                message: "affine reduced solve cancelled".into(),
            });
            break;
        }
        let operator_direction = apply_reduced_operator(
            reduction,
            &direction,
            &apply_full,
            vector_plan,
            transpose_index.as_ref(),
        )?;
        let denominator = dot(&direction, &operator_direction)?;
        if !denominator.is_finite() || denominator <= 0.0 {
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
        converged = next_residual_squared.sqrt() <= tolerance * reference_norm;
        if converged {
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

    // Publish convergence from the physical reduced equation, not only the
    // recursively updated residual, which can drift after many CG steps.
    let applied_solution = apply_reduced_operator(
        reduction,
        &reduced_solution,
        &apply_full,
        vector_plan,
        transpose_index.as_ref(),
    )?;
    if let Some(plan) = vector_plan.filter(|plan| !plan.is_serial()) {
        crate::algebra::subtract_into_scoped(&reduced_rhs, &applied_solution, &mut residual, plan)?;
    } else {
        for ((residual_value, right_hand_side), applied) in
            residual.iter_mut().zip(&reduced_rhs).zip(&applied_solution)
        {
            *residual_value = right_hand_side - applied;
        }
    }
    drop(applied_solution);
    let residual_norm = dot(&residual, &residual)?.sqrt();
    converged = residual_norm <= tolerance * reference_norm;
    if !converged {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_AFFINE_CG_NOT_CONVERGED".into(),
            severity: FeaDiagnosticSeverity::Warning,
            message: format!(
                "affine reduced cg stopped after {iterations}/{maximum_iterations} iterations with residual_norm={residual_norm} relative_residual={} tolerance={tolerance}",
                residual_norm / reference_norm,
            ),
        });
    }
    diagnostics.push(FeaDiagnostic {
        code: "FEA_SOLVER_METHOD".into(),
        severity: FeaDiagnosticSeverity::Info,
        message: "solver=pcg space=affine_reduced preconditioner=none matrix_free=true".into(),
    });
    Ok(LinearSolveResult {
        iterations,
        residual_norm,
        relative_residual: crate::solve::convergence::relative_residual(
            residual_norm,
            reference_norm,
        ),
        converged,
        host_sync_count: 0,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution: expand(reduction, &reduced_solution, vector_plan)?,
        solver_method: SolverMethod::MatrixFreePcg,
        preconditioner: SolverPreconditioner::None,
        diagnostics,
    })
}

fn rejected_affine_system(
    summary: &StructuralLinearSystemView<'_>,
    message: &str,
) -> LinearSolveResult {
    LinearSolveResult {
        iterations: 0,
        residual_norm: summary
            .operator
            .load()
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt(),
        relative_residual: None,
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
    plan: Option<LocalExecutionPlan>,
    transpose_index: Option<&AffineTransposeIndex>,
) -> Result<Vec<f64>, ScopedSpmvError> {
    let full = expand_linear(reduction, reduced, plan)?;
    let applied = apply_full(&full)?;
    if let (Some(plan), Some(index)) = (plan.filter(|plan| !plan.is_serial()), transpose_index) {
        index.apply(reduction, &applied, plan)
    } else {
        Ok(transpose_apply(reduction, &applied))
    }
}

fn expand(
    reduction: &AffineConstraintReduction,
    reduced: &[f64],
    plan: Option<LocalExecutionPlan>,
) -> Result<Vec<f64>, ScopedSpmvError> {
    let mut full = expand_linear(reduction, reduced, plan)?;
    if let Some(plan) = plan.filter(|plan| !plan.is_serial()) {
        crate::algebra::axpy_scoped(1.0, &reduction.offset, &mut full, plan)?;
    } else {
        for (value, offset) in full.iter_mut().zip(&reduction.offset) {
            *value += offset;
        }
    }
    Ok(full)
}

fn expand_linear(
    reduction: &AffineConstraintReduction,
    reduced: &[f64],
    plan: Option<LocalExecutionPlan>,
) -> Result<Vec<f64>, ScopedSpmvError> {
    let mut full = vec![0.0; reduction.full_dof_count];
    if let Some(plan) = plan.filter(|plan| !plan.is_serial()) {
        let cancellation = crate::progress::cancellation_token();
        crate::planning::resource::execute_scoped_slices_mut(plan, &mut full, |chunk, output| {
            for (offset, value) in output.iter_mut().enumerate() {
                if cancellation.is_cancelled() {
                    return Err(crate::algebra::SpmvWorkError::Cancelled);
                }
                *value = recovery_row(reduction, reduced, chunk.start() as usize + offset);
            }
            Ok(())
        })?;
    } else {
        for (row, value) in full.iter_mut().enumerate() {
            *value = recovery_row(reduction, reduced, row);
        }
    }
    Ok(full)
}

fn recovery_row(reduction: &AffineConstraintReduction, reduced: &[f64], row: usize) -> f64 {
    let start = reduction.recovery_row_offsets[row];
    let end = reduction.recovery_row_offsets[row + 1];
    (start..end)
        .map(|entry| {
            reduction.recovery_coefficients[entry]
                * reduced[reduction.recovery_column_indices[entry]]
        })
        .sum()
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

pub(super) fn solve_affine_constraint_system_reusing(
    summary: &StructuralLinearSystemView<'_>,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    reuse: &mut crate::algebra::LinearPreparationReuse,
) -> LinearSolveResult {
    let mut settings =
        ScopedLinearSolveSettings::with_default_tolerance(LocalExecutionPlan::serial(), None);
    settings.preparation = Some(reuse);
    solve_affine_constraint_system_with_apply(
        summary,
        reduction,
        requested_backend,
        algebra_backend,
        initial_guess,
        (
            |full: &[f64]| Ok(apply_k(summary.operator, full)),
            |left: &[f64], right: &[f64]| Ok(algebra_backend.dot(left, right)),
            None,
        ),
        settings,
    )
    .expect("reference sparse application is infallible for an assembled system")
}

#[cfg(test)]
mod tests;
