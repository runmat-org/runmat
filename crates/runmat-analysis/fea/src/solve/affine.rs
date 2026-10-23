use crate::{
    algebra::{PreparedRealLinearSystem, ScopedSpmvError},
    constraints::AffineConstraintReduction,
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    planning::resource::{LocalExecutionPlan, SolverAdmissionResourcePolicy},
    solve::linear::{ScopedLinearSolveSettings, StructuralLinearSystemView},
    solve::{
        backend::kind::LinearAlgebraBackendKind, linear::LinearSolveResult,
        preconditioner::SpdPreconditionerKind, shared_sparse,
    },
};

use super::backend::linear_algebra::LinearAlgebraBackend;

pub(super) fn solve_affine_constraint_system(
    summary: &StructuralLinearSystemView<'_>,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    _algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
) -> LinearSolveResult {
    solve_affine_constraint_system_with_settings(
        summary,
        reduction,
        requested_backend,
        initial_guess,
        ScopedLinearSolveSettings::with_default_tolerance(LocalExecutionPlan::serial(), None),
    )
}

pub(super) fn solve_affine_constraint_system_with_execution(
    summary: &StructuralLinearSystemView<'_>,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    _algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    plan: LocalExecutionPlan,
    admission: Option<&SolverAdmissionResourcePolicy>,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    Ok(solve_affine_constraint_system_with_settings(
        summary,
        reduction,
        requested_backend,
        initial_guess,
        ScopedLinearSolveSettings::with_default_tolerance(plan, admission),
    ))
}

/// Solves the affine-reduced SPD system `Tᵀ K T u = Tᵀ (f − K c)` through the shared sparse layer
/// (the coercivity-proving Cholesky factor of the reduced operator is the direct solver) and
/// expands `x = T u + c`.
fn solve_affine_constraint_system_with_settings(
    summary: &StructuralLinearSystemView<'_>,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    initial_guess: Option<&[f64]>,
    settings: ScopedLinearSolveSettings<'_>,
) -> LinearSolveResult {
    let admission = settings.admission;
    let reuse = settings.preparation;
    let plan = settings.plan;
    let tolerance = settings.relative_tolerance;
    let expected_reduction = summary
        .constraint_matrix
        .as_ref()
        .and_then(|matrix| matrix.affine_reduction().ok());
    if expected_reduction.as_ref() != Some(reduction) {
        return rejected_affine_system(
            summary,
            "affine reduction differs from the canonical resolved constraint matrix",
        );
    }
    let stiffness = summary.operator.operators().stiffness();
    let derived_support;
    let reduced_support = match summary.planned_reduced_support {
        Some(planned) => planned,
        None => match reduction.reduce_sparse_pattern(stiffness.pattern()) {
            Ok(pattern) => {
                derived_support = pattern;
                &derived_support
            }
            Err(error) => return rejected_affine_system(summary, &error.to_string()),
        },
    };
    let (reduced_operator, reduced_rhs) = match reduction.reduce_sparse_system_on_support(
        stiffness,
        reduced_support,
        summary.operator.load(),
    ) {
        Ok(reduced) => reduced,
        Err(error) => return rejected_affine_system(summary, &error.to_string()),
    };
    let reduced_guess = initial_guess.map(|guess| {
        reduction
            .free_dof_indices
            .iter()
            .map(|dof| guess.get(*dof).copied().unwrap_or(0.0))
            .collect::<Vec<_>>()
    });
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
        return LinearSolveResult {
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
            solver_method: SolverMethod::SparseDirectCholesky,
            preconditioner: SolverPreconditioner::None,
            diagnostics,
        };
    }

    let reduced_dofs = reduction.reduced_dof_count;
    let reduced_pattern_entries = reduced_operator.pattern().column_indices().len();
    let preparation = shared_sparse::with_plan(plan, move || match reuse {
        Some(reuse) => PreparedRealLinearSystem::try_new_reusing(
            reduced_operator,
            reduced_rhs,
            vec![false; reduced_dofs],
            reuse,
        ),
        None => PreparedRealLinearSystem::try_new(
            reduced_operator,
            reduced_rhs,
            vec![false; reduced_dofs],
        ),
    });
    let prepared = match preparation {
        Ok(prepared) => prepared,
        Err(error) => return rejected_affine_system(summary, &error.to_string()),
    };
    if let Some(admission) = admission {
        let accepted = prepared
            .admissibility()
            .is_some_and(|evidence| admission.admits_certificate(evidence.coercivity()));
        if !accepted {
            let exact = prepared
                .admissibility()
                .map(|evidence| evidence.coercivity().work_counts());
            return rejected_affine_system(
                summary,
                &format!(
                    "exact reduced-solver work (dimension, factor nonzeros, updates, bytes) {exact:?} on a reduced pattern of {} entries (planned support: {}) exceeds the admitted resource envelope {admission:?}",
                    reduced_pattern_entries,
                    summary.planned_reduced_support.is_some(),
                ),
            );
        }
    }
    let factor = prepared
        .admissibility()
        .and_then(|evidence| evidence.factor())
        .map(|factor| factor.0.clone());
    let strategy = shared_sparse::resolve_spd_strategy(
        SpdPreconditionerKind::Auto,
        prepared.dof_count(),
        prepared.matrix().values().len(),
        1,
        factor.is_some(),
    );
    let outcome = shared_sparse::solve_spd(shared_sparse::SpdSolveRequest {
        matrix: prepared.matrix(),
        rhs: prepared.rhs(),
        initial_guess: reduced_guess.as_deref(),
        relative_tolerance: tolerance,
        plan,
        strategy,
        block_size: 1,
        factor: factor.as_deref(),
    });
    if outcome.cancelled {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_SOLVER_CANCELLED".into(),
            severity: FeaDiagnosticSeverity::Warning,
            message: "affine reduced solve cancelled".into(),
        });
    } else if !outcome.converged {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_AFFINE_CG_NOT_CONVERGED".into(),
            severity: FeaDiagnosticSeverity::Warning,
            message: format!(
                "affine reduced {} did not reach tolerance {tolerance}: {}",
                strategy.label(),
                outcome.failure.as_deref().unwrap_or("residual above tolerance"),
            ),
        });
    }
    diagnostics.push(FeaDiagnostic {
        code: "FEA_SOLVER_METHOD".into(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!("{} space=affine_reduced", strategy.label()),
    });
    let (solver_method, preconditioner) = strategy.provenance();
    let solution = match expand(reduction, &outcome.solution, Some(plan)) {
        Ok(solution) => solution,
        Err(error) => return rejected_affine_system(summary, &error.to_string()),
    };
    LinearSolveResult {
        iterations: outcome.iterations,
        residual_norm: outcome.residual_norm,
        relative_residual: crate::solve::convergence::relative_residual(
            outcome.residual_norm,
            outcome.reference_norm,
        ),
        converged: outcome.converged && !outcome.cancelled,
        host_sync_count: 0,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution,
        solver_method,
        preconditioner,
        diagnostics,
    }
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
        solver_method: SolverMethod::SparseDirectCholesky,
        preconditioner: SolverPreconditioner::None,
        diagnostics: vec![FeaDiagnostic {
            code: "FEA_INVALID_AFFINE_SOLVER_ADMISSION".into(),
            severity: FeaDiagnosticSeverity::Error,
            message: message.into(),
        }],
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

pub(super) fn solve_affine_constraint_system_reusing(
    summary: &StructuralLinearSystemView<'_>,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    _algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    reuse: &mut crate::algebra::LinearPreparationReuse,
) -> LinearSolveResult {
    let mut settings =
        ScopedLinearSolveSettings::with_default_tolerance(LocalExecutionPlan::serial(), None);
    settings.preparation = Some(reuse);
    solve_affine_constraint_system_with_settings(
        summary,
        reduction,
        requested_backend,
        initial_guess,
        settings,
    )
}

#[cfg(test)]
mod tests;
