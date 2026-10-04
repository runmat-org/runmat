//! The same admitted tangent solve provides updates and a pending-correction criterion.
use super::{constraint_error, SolidAttemptWork, SolidNewtonError, SolidNewtonProblem};
use crate::{
    algebra::{
        AssembledRealSystem, CanonicalRealOperators, MatrixProperties, OperatorRole,
        RealSparseAssembler, RealSparseMatrix,
    },
    planning::resource::LocalExecutionPlan,
    solve::{
        backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
        linear::{solve_operator_system_scoped, ScopedLinearSolveSettings},
        preconditioner::SpdPreconditionerKind,
    },
};

impl SolidNewtonProblem<'_> {
    pub(super) fn correction(
        &self,
        tangent: &RealSparseMatrix,
        residual: &[f64],
        factor: f64,
        work: &mut SolidAttemptWork,
    ) -> Result<Vec<f64>, SolidNewtonError> {
        let reduced = self
            .reduction
            .reduce_sparse_operator(tangent)
            .map_err(constraint_error)?;
        let count = reduced.rows();
        if count == 0 {
            return self.reduction.expand_linear(&[]).map_err(constraint_error);
        }
        work.linear_solves += 1;
        let mass = RealSparseAssembler::square(
            count,
            OperatorRole::Mass,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .finish()
        .map_err(constraint_error)?;
        let damping = RealSparseAssembler::square(
            count,
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .finish()
        .map_err(constraint_error)?;
        let operators =
            CanonicalRealOperators::try_new(reduced, mass, damping).map_err(constraint_error)?;
        let operator = AssembledRealSystem::try_new(
            operators,
            vec![false; count],
            residual.iter().map(|r| -r).collect(),
        )
        .map_err(constraint_error)?;
        let solved = solve_operator_system_scoped(
            &operator,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
            ScopedLinearSolveSettings {
                plan: LocalExecutionPlan::serial(),
                admission: self.solver_admission,
                relative_tolerance: self.policy.linear_relative_tolerance,
            },
        )
        .map_err(|e| SolidNewtonError::Evaluation(e.to_string()))?;
        if !solved.converged || solved.solution.iter().any(|v| !v.is_finite()) {
            return Err(SolidNewtonError::NotConverged {
                factor,
                cutbacks: 0,
                reason: format!("physical tangent solve rejected: {:?}", solved.diagnostics),
            });
        }
        self.reduction
            .expand_linear(&solved.solution)
            .map_err(constraint_error)
    }
}
