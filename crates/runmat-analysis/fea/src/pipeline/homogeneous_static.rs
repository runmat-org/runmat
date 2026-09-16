//! Canonical homogeneous static solve shared by modal correction procedures.

use crate::{
    algebra::{
        apply_k, AssembledRealSystem, CanonicalRealOperators, MatrixProperties, OperatorRole,
        RealSparseMatrix,
    },
    assembly::AssemblySummary,
    constraints::AffineConstraintReduction,
    contracts::FeaRunError,
    solve::{
        preconditioner::{build_spd_preconditioner, SpdPreconditionerKind},
        spd_linear::{solve_spd_system_cg, SpdCgSolveOptions},
    },
};

const STATIC_TOLERANCE: f64 = 1.0e-10;

pub(crate) fn homogeneous_static_iteration_limit(dimension: usize) -> usize {
    dimension.saturating_mul(20).max(1_000)
}

/// Prepared reduced stiffness and preconditioner for one homogeneous modal constraint space.
pub(crate) struct HomogeneousStaticSolver {
    reduction: AffineConstraintReduction,
    system: AssembledRealSystem,
    preconditioner: Box<dyn crate::solve::preconditioner::SpdPreconditioner>,
}

pub(crate) struct HomogeneousStaticSolution {
    pub displacement: Vec<f64>,
    pub equilibrium_residual_ratio: f64,
}

impl HomogeneousStaticSolver {
    pub(crate) fn new(summary: &AssemblySummary) -> Result<Self, FeaRunError> {
        let reduction = homogeneous_reduction(summary)?;
        let stiffness = reduction
            .reduce_sparse_operator(summary.operator.operators().stiffness())
            .map_err(|error| FeaRunError::Solve(error.to_string()))?;
        let dimension = stiffness.rows();
        let system = AssembledRealSystem::try_new(
            CanonicalRealOperators::try_new(
                stiffness,
                RealSparseMatrix::diagonal(
                    vec![1.0; dimension],
                    OperatorRole::Mass,
                    MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
                )
                .map_err(|error| FeaRunError::Solve(error.to_string()))?,
                RealSparseMatrix::diagonal(
                    vec![0.0; dimension],
                    OperatorRole::Damping,
                    MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
                )
                .map_err(|error| FeaRunError::Solve(error.to_string()))?,
            )
            .map_err(|error| FeaRunError::Solve(error.to_string()))?,
            vec![false; dimension],
            vec![0.0; dimension],
        )
        .map_err(|error| FeaRunError::Solve(error.to_string()))?;
        let prepared = system
            .prepare_linear()
            .map_err(|error| FeaRunError::Solve(error.to_string()))?;
        let preconditioner = build_spd_preconditioner(&prepared, SpdPreconditionerKind::Ilu0)
            .or_else(|_| build_spd_preconditioner(&prepared, SpdPreconditionerKind::Jacobi))
            .map_err(|error| FeaRunError::Solve(error.to_string()))?;
        Ok(Self {
            reduction,
            system,
            preconditioner,
        })
    }

    pub(crate) fn solve(
        &self,
        full_load: &[f64],
    ) -> Result<HomogeneousStaticSolution, FeaRunError> {
        let reduced_load = self
            .reduction
            .transpose_apply(full_load)
            .map_err(|error| FeaRunError::Solve(error.to_string()))?;
        if norm(&reduced_load) <= f64::EPSILON {
            let zero = vec![0.0; self.system.dof_count()];
            return Ok(HomogeneousStaticSolution {
                displacement: self
                    .reduction
                    .expand_linear(&zero)
                    .map_err(|error| FeaRunError::Solve(error.to_string()))?,
                equilibrium_residual_ratio: 0.0,
            });
        }
        let attempt = solve_spd_system_cg(
            &self.system,
            &reduced_load,
            SpdCgSolveOptions {
                max_iters: homogeneous_static_iteration_limit(self.system.dof_count()),
                tol: STATIC_TOLERANCE,
                prepared_accelerator_system: None,
                cpu_preconditioner: Some(self.preconditioner.as_ref()),
                initial_guess: None,
            },
        );
        if !attempt.converged {
            return Err(FeaRunError::Solve(
                "homogeneous static correction did not converge".into(),
            ));
        }
        let equilibrium_residual =
            subtract(&apply_k(&self.system, &attempt.vector), &reduced_load)?;
        let equilibrium_residual_ratio =
            norm(&equilibrium_residual) / norm(&reduced_load).max(f64::MIN_POSITIVE);
        if !equilibrium_residual_ratio.is_finite() || equilibrium_residual_ratio > STATIC_TOLERANCE
        {
            return Err(FeaRunError::Solve(
                "homogeneous static correction failed its equilibrium tolerance".into(),
            ));
        }
        let displacement = self
            .reduction
            .expand_linear(&attempt.vector)
            .map_err(|error| FeaRunError::Solve(error.to_string()))?;
        Ok(HomogeneousStaticSolution {
            displacement,
            equilibrium_residual_ratio,
        })
    }
}

fn homogeneous_reduction(
    summary: &AssemblySummary,
) -> Result<AffineConstraintReduction, FeaRunError> {
    if let Some(matrix) = &summary.constraint_matrix {
        if matrix.right_hand_side.iter().any(|value| *value != 0.0) {
            return Err(FeaRunError::Solve(
                "modal static corrections require homogeneous constraints".into(),
            ));
        }
        matrix
            .affine_reduction()
            .map_err(|error| FeaRunError::Solve(error.to_string()))
    } else {
        Ok(AffineConstraintReduction::from_constrained_dofs(
            summary.operator.constrained_dofs(),
        ))
    }
}

fn subtract(left: &[f64], right: &[f64]) -> Result<Vec<f64>, FeaRunError> {
    if left.len() != right.len() {
        return Err(FeaRunError::Solve(
            "homogeneous static correction dimensions are inconsistent".into(),
        ));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left - right)
        .collect())
}

fn norm(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}
