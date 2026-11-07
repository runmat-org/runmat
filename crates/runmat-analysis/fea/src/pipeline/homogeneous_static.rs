//! Canonical homogeneous static solve shared by modal correction procedures.

use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory};

use crate::{
    algebra::{
        apply_k,
        capacity::{measured_sync, real_spmv_work},
        AssembledRealSystem, CanonicalRealOperators, CoercivityError, MatrixProperties,
        OperatorRole, RealSparseMatrix, RealSystemError, ScopedSpmvError,
    },
    assembly::AssemblySummary,
    constraints::AffineConstraintReduction,
    contracts::FeaRunError,
    solve::{
        affine_expand_linear, affine_recovery_work, affine_scatter_work, affine_transpose_work,
        preconditioner::SpdPreconditionerKind, shared_sparse,
    },
};

const STATIC_TOLERANCE: f64 = 1.0e-10;

/// Kept for resource planning of correction workloads (one direct solve per load).
pub(crate) fn homogeneous_static_iteration_limit(dimension: usize) -> usize {
    dimension.saturating_mul(20).max(1_000)
}

/// Prepared reduced stiffness and its sparse Cholesky factor for one homogeneous modal
/// constraint space; every correction load reuses the factor.
pub(crate) struct HomogeneousStaticSolver {
    reduction: AffineConstraintReduction,
    system: AssembledRealSystem,
    factor: std::sync::Arc<runmat_sparse_solvers::direct::DirectSolver<f64>>,
    /// `||K||_F` of the reduced stiffness, computed once per factor.
    stiffness_norm: f64,
}

pub(crate) struct HomogeneousStaticSolution {
    pub displacement: Vec<f64>,
    pub equilibrium_residual_ratio: f64,
}

impl HomogeneousStaticSolver {
    pub(crate) fn preconditioner_kind(&self) -> SpdPreconditionerKind {
        SpdPreconditionerKind::Direct
    }

    /// Reduces and factors the stiffness. Under an admitted usage context the reduction scatter
    /// is charged as solve work and the Cholesky factorization charges its own reference
    /// operations; a ledger refusal returns [`FeaRunError::Usage`].
    pub(crate) fn new(summary: &AssemblySummary) -> Result<Self, FeaRunError> {
        let reduction = homogeneous_reduction(summary)?;
        let full_stiffness = summary.operator.operators().stiffness();
        let stiffness = metered(
            affine_scatter_work(&reduction, full_stiffness).map_err(scoped_error)?,
            || {
                reduction
                    .reduce_sparse_operator(full_stiffness)
                    .map_err(|error| FeaRunError::Solve(error.to_string()))
            },
        )?;
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
        let prepared = shared_sparse::with_plan(
            crate::planning::resource::LocalExecutionPlan::serial(),
            || system.prepare_linear(),
        )
        .map_err(|error| match error {
            RealSystemError::Coercivity(CoercivityError::Usage(cause)) => FeaRunError::Usage(cause),
            other => FeaRunError::Solve(other.to_string()),
        })?;
        let factor = prepared
            .admissibility()
            .and_then(|evidence| evidence.factor())
            .map(|factor| factor.0.clone())
            .ok_or_else(|| {
                FeaRunError::Solve("homogeneous static stiffness has no Cholesky proof".into())
            })?;
        let values = system.operators().stiffness().values();
        let stiffness_norm = metered(norm_work(values.len())?, || Ok(norm(values)))?;
        Ok(Self {
            reduction,
            system,
            factor,
            stiffness_norm,
        })
    }

    /// One correction load. The direct substitutions, refinement residuals and norms charge
    /// themselves inside the shared direct solver; this owner charges the constraint transpose
    /// and expansion, the equilibrium product `K x - f` and its backward-error norms.
    pub(crate) fn solve(
        &self,
        full_load: &[f64],
    ) -> Result<HomogeneousStaticSolution, FeaRunError> {
        let reduced_load = metered(
            affine_transpose_work(&self.reduction).map_err(scoped_error)?,
            || {
                self.reduction
                    .transpose_apply(full_load)
                    .map_err(|error| FeaRunError::Solve(error.to_string()))
            },
        )?;
        if reduced_load.iter().all(|value| *value == 0.0) {
            let zero = vec![0.0; self.system.dof_count()];
            return Ok(HomogeneousStaticSolution {
                displacement: self.expand(&zero)?,
                equilibrium_residual_ratio: 0.0,
            });
        }
        let arrays = shared_sparse::CsrArrays::of(self.system.operators().stiffness());
        let mut vector = vec![0.0; reduced_load.len()];
        shared_sparse::with_plan(
            crate::planning::resource::LocalExecutionPlan::serial(),
            || {
                self.factor
                    .solve_refined(arrays.view().into(), &reduced_load, &mut vector, 1)
                    .map(|_| ())
            },
        )
        .map_err(|error| match error {
            runmat_sparse_solvers::SolverError::Usage(cause) => FeaRunError::Usage(cause),
            error => FeaRunError::Solve(format!("homogeneous static correction failed: {error}")),
        })?;
        let length = vector.len();
        let applied = metered(real_spmv_work(self.system.operators().stiffness())?, || {
            Ok(apply_k(&self.system, &vector))
        })?;
        let equilibrium_residual = metered(vector_work(length, 1)?, || {
            subtract(&applied, &reduced_load)
        })?;
        // Normwise backward error ||K x - f|| / (||K||_F ||x|| + ||f||). A direct factor meets it
        // near machine precision on any conditioning; ||K x - f|| / ||f|| alone scales with
        // eps * cond(K) and rejects correct solves of stiff models.
        let residual_norm = metered(norm_work(length)?, || Ok(norm(&equilibrium_residual)))?;
        let solution_norm = metered(norm_work(length)?, || Ok(norm(&vector)))?;
        let load_norm = metered(norm_work(length)?, || Ok(norm(&reduced_load)))?;
        // One product, one sum and one quotient.
        let equilibrium_residual_ratio = metered(OperationCount::from_operations(3), || {
            Ok(residual_norm
                / (self.stiffness_norm * solution_norm + load_norm).max(f64::MIN_POSITIVE))
        })?;
        if !equilibrium_residual_ratio.is_finite() || equilibrium_residual_ratio > STATIC_TOLERANCE
        {
            return Err(FeaRunError::Solve(
                "homogeneous static correction failed its equilibrium tolerance".into(),
            ));
        }
        Ok(HomogeneousStaticSolution {
            displacement: self.expand(&vector)?,
            equilibrium_residual_ratio,
        })
    }

    /// `T u` for the homogeneous (offset-free) reduction.
    fn expand(&self, reduced: &[f64]) -> Result<Vec<f64>, FeaRunError> {
        if reduced.len() != self.reduction.reduced_dof_count {
            return Err(FeaRunError::Solve(
                "homogeneous static correction dimensions are inconsistent".into(),
            ));
        }
        metered(
            affine_recovery_work(&self.reduction).map_err(scoped_error)?,
            || affine_expand_linear(&self.reduction, reduced, None).map_err(scoped_error),
        )
    }
}

/// Reserves `work` as solve operations on the active usage ledger (if any), runs `run`, and
/// completes the reservation only when `run` succeeds.
pub(crate) fn metered<R>(
    work: OperationCount,
    run: impl FnOnce() -> Result<R, FeaRunError>,
) -> Result<R, FeaRunError> {
    measured_sync(UsageCategory::Solve, work, run)
}

/// `per_entry` reference operations for each of `length` vector entries.
pub(crate) fn vector_work(length: usize, per_entry: u64) -> Result<OperationCount, FeaRunError> {
    Ok(OperationCount::from_operations(
        u64::try_from(length).map_err(|_| LedgerError::CountOverflow)?,
    )
    .checked_mul(per_entry)
    .map_err(LedgerError::from)?)
}

/// A real dot product of `length` entries.
pub(crate) fn dot_work(length: usize) -> Result<OperationCount, FeaRunError> {
    Ok(OperationCount::direct_real_dot(
        u64::try_from(length).map_err(|_| LedgerError::CountOverflow)?,
    )
    .map_err(LedgerError::from)?)
}

/// A Euclidean norm: the self dot product plus one square root.
pub(crate) fn norm_work(length: usize) -> Result<OperationCount, FeaRunError> {
    Ok(dot_work(length)?
        .checked_add(OperationCount::from_operations(1))
        .map_err(LedgerError::from)?)
}

fn scoped_error(error: ScopedSpmvError) -> FeaRunError {
    match error {
        ScopedSpmvError::Usage(cause) => FeaRunError::Usage(cause),
        other => FeaRunError::Solve(other.to_string()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::RealSparseAssembler;

    #[test]
    fn correction_solves_with_the_proving_cholesky_factor() {
        // Coercive operator on which zero-fill ILU has a negative final pivot; the direct
        // correction is exact regardless.
        let values = [
            [
                2.853656491730362,
                0.0,
                1.7390443217009306,
                1.6154918987303972,
                0.6468259897083044,
            ],
            [0.0, 4.752237345324829, 0.0, 2.2926174104213715, 0.0],
            [
                1.7390443217009306,
                0.0,
                5.047566675581038,
                0.0,
                -2.5766232535243034,
            ],
            [
                1.6154918987303972,
                2.2926174104213715,
                0.0,
                3.101011391496286,
                1.4569321684539318,
            ],
            [
                0.6468259897083044,
                0.0,
                -2.5766232535243034,
                1.4569321684539318,
                3.166770219337195,
            ],
        ];
        let mut assembler = RealSparseAssembler::square(
            5,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        );
        for (row, entries) in values.iter().enumerate() {
            for (column, value) in entries.iter().copied().enumerate() {
                if value != 0.0 {
                    assembler.add(row, column, value);
                }
            }
        }
        let stiffness = assembler.finish().unwrap();
        let system = AssembledRealSystem::try_new(
            CanonicalRealOperators::try_new(
                stiffness,
                RealSparseMatrix::diagonal(
                    vec![1.0; 5],
                    OperatorRole::Mass,
                    MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
                )
                .unwrap(),
                RealSparseMatrix::diagonal(
                    vec![0.0; 5],
                    OperatorRole::Damping,
                    MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
                )
                .unwrap(),
            )
            .unwrap(),
            vec![false; 5],
            vec![0.0; 5],
        )
        .unwrap();
        let prepared = system.prepare_linear().unwrap();
        let factor = prepared
            .admissibility()
            .unwrap()
            .factor()
            .unwrap()
            .0
            .clone();
        let expected = [1.0, -2.0, 0.5, 3.0, -1.0];
        let load = values
            .iter()
            .map(|row| {
                row.iter()
                    .zip(expected)
                    .map(|(entry, value)| entry * value)
                    .sum()
            })
            .collect::<Vec<f64>>();
        let arrays = shared_sparse::CsrArrays::of(system.operators().stiffness());
        let mut solved = vec![0.0; 5];
        factor
            .solve_refined(arrays.view().into(), &load, &mut solved, 1)
            .unwrap();
        for (actual, expected) in solved.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1.0e-12);
        }
        let residual = subtract(&apply_k(&system, &solved), &load).unwrap();
        assert!(norm(&residual) / norm(&load) <= STATIC_TOLERANCE);
    }
}
