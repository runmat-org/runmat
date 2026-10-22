//! Borrowed numerical solve inputs. Physical assembly/recovery metadata stays with its owner.
use super::{
    solve_operator_system, solve_operator_system_scoped, LinearSolveResult,
    ScopedLinearSolveSettings,
};
use crate::{
    algebra::{AssembledRealSystem, ScopedSpmvError},
    assembly::AssemblySummary,
    constraints::{AffineConstraintReduction, ResolvedConstraintMatrix},
    planning::resource::{LocalExecutionPlan, SolverAdmissionResourcePolicy},
    solve::{
        affine::{solve_affine_constraint_system, solve_affine_constraint_system_with_execution},
        backend::{kind::LinearAlgebraBackendKind, linear_algebra::LinearAlgebraBackend},
        preconditioner::SpdPreconditionerKind,
    },
};

#[derive(Clone, Copy)]
pub(crate) struct StructuralLinearSystemView<'a> {
    pub dof_count: usize,
    pub operator: &'a AssembledRealSystem,
    pub constraint_matrix: Option<&'a ResolvedConstraintMatrix>,
    pub affine_constraint_reduction: Option<&'a AffineConstraintReduction>,
    /// The reduced pattern resource planning analyzed. Affine reduction scatters into exactly
    /// this pattern, so the proof factorization repeats the planned ordering and the admitted
    /// envelope bounds the exact work.
    pub planned_reduced_support: Option<&'a crate::algebra::SparsePattern>,
}

impl<'a> From<&'a AssemblySummary> for StructuralLinearSystemView<'a> {
    fn from(summary: &'a AssemblySummary) -> Self {
        Self {
            dof_count: summary.dof_count,
            operator: &summary.operator,
            constraint_matrix: summary.constraint_matrix.as_ref(),
            affine_constraint_reduction: summary.affine_constraint_reduction.as_ref(),
            planned_reduced_support: None,
        }
    }
}

pub(crate) fn solve_structural_system(
    summary: &StructuralLinearSystemView<'_>,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
) -> LinearSolveResult {
    solve_structural_system_with_preparation(
        summary,
        preconditioner_kind,
        backend_kind,
        algebra_backend,
        initial_guess,
        None,
    )
}

pub(super) fn solve_structural_system_with_execution(
    summary: &StructuralLinearSystemView<'_>,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    plan: LocalExecutionPlan,
    admission: Option<&SolverAdmissionResourcePolicy>,
) -> Result<LinearSolveResult, ScopedSpmvError> {
    if crate::planning::resource::local_chunk_active() {
        return Err(ScopedSpmvError::Execution(
            crate::planning::resource::ScopedLocalExecutionError::NestedExecution,
        ));
    }
    if crate::is_fea_cancelled() {
        // A cancelled solve reports the cancelled admitted work, not a numerical outcome.
        crate::planning::resource::execute_scoped_chunks(plan, 1, |_| {
            Err::<(), _>(crate::algebra::SpmvWorkError::Cancelled)
        })?;
    }
    if let Some(reduction) = summary.affine_constraint_reduction.as_ref() {
        return solve_affine_constraint_system_with_execution(
            summary,
            reduction,
            backend_kind,
            algebra_backend,
            initial_guess,
            plan,
            admission,
        );
    }
    if let Some(matrix) = summary.constraint_matrix.as_ref() {
        // Multiplier enforcement and affine elimination determine the same displacement; the
        // reduced SPD system is the canonical (direct-solvable) form of both.
        let reduction = match matrix.affine_reduction() {
            Ok(reduction) => reduction,
            Err(error) => return Ok(super::admission::rejected_constraint_system(summary, error.to_string())),
        };
        return solve_affine_constraint_system_with_execution(
            summary,
            &reduction,
            backend_kind,
            algebra_backend,
            initial_guess,
            plan,
            admission,
        );
    }
    solve_operator_system_scoped(
        summary.operator,
        preconditioner_kind,
        backend_kind,
        algebra_backend,
        initial_guess,
        ScopedLinearSolveSettings::with_default_tolerance(plan, admission),
    )
}

/// Shared structural routing with bounded process-local exact preparation reuse.
pub(crate) fn solve_structural_system_reusing(
    summary: &StructuralLinearSystemView<'_>,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    reuse: &mut crate::algebra::LinearPreparationReuse,
) -> LinearSolveResult {
    solve_structural_system_with_preparation(
        summary,
        preconditioner_kind,
        backend_kind,
        algebra_backend,
        initial_guess,
        Some(reuse),
    )
}

fn solve_structural_system_with_preparation(
    summary: &StructuralLinearSystemView<'_>,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
    reuse: Option<&mut crate::algebra::LinearPreparationReuse>,
) -> LinearSolveResult {
    if let Some(reduction) = summary.affine_constraint_reduction.as_ref() {
        return match reuse {
            Some(reuse) => crate::solve::affine::solve_affine_constraint_system_reusing(
                summary,
                reduction,
                backend_kind,
                algebra_backend,
                initial_guess,
                reuse,
            ),
            None => solve_affine_constraint_system(
                summary,
                reduction,
                backend_kind,
                algebra_backend,
                initial_guess,
            ),
        };
    }
    if let Some(matrix) = summary.constraint_matrix.as_ref() {
        let reduction = match matrix.affine_reduction() {
            Ok(reduction) => reduction,
            Err(error) => return super::admission::rejected_constraint_system(summary, error.to_string()),
        };
        return match reuse {
            Some(reuse) => crate::solve::affine::solve_affine_constraint_system_reusing(
                summary,
                &reduction,
                backend_kind,
                algebra_backend,
                initial_guess,
                reuse,
            ),
            None => solve_affine_constraint_system(
                summary,
                &reduction,
                backend_kind,
                algebra_backend,
                initial_guess,
            ),
        };
    }
    match reuse {
        None => solve_operator_system(
            summary.operator,
            preconditioner_kind,
            backend_kind,
            algebra_backend,
            initial_guess,
        ),
        Some(reuse) => {
            let mut settings = ScopedLinearSolveSettings::with_default_tolerance(
                LocalExecutionPlan::serial(),
                None,
            );
            settings.preparation = Some(reuse);
            solve_operator_system_scoped(
                summary.operator,
                preconditioner_kind,
                backend_kind,
                algebra_backend,
                initial_guess,
                settings,
            )
            .expect("reference sparse application is infallible for a prepared system")
        }
    }
}
