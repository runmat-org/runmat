//! Canonical homogeneous static solve shared by modal correction procedures.

use crate::{
    algebra::{
        apply_k, AssembledRealSystem, CanonicalRealOperators, MatrixProperties, OperatorRole,
        RealSparseMatrix,
    },
    assembly::AssemblySummary,
    constraints::AffineConstraintReduction,
    contracts::FeaRunError,
    solve::{preconditioner::SpdPreconditionerKind, shared_sparse},
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
}

pub(crate) struct HomogeneousStaticSolution {
    pub displacement: Vec<f64>,
    pub equilibrium_residual_ratio: f64,
}

impl HomogeneousStaticSolver {
    pub(crate) fn preconditioner_kind(&self) -> SpdPreconditionerKind {
        SpdPreconditionerKind::Direct
    }

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
        let prepared = shared_sparse::with_plan(
            crate::planning::resource::LocalExecutionPlan::serial(),
            || system.prepare_linear(),
        )
        .map_err(|error| FeaRunError::Solve(error.to_string()))?;
        let factor = prepared
            .admissibility()
            .and_then(|evidence| evidence.factor())
            .map(|factor| factor.0.clone())
            .ok_or_else(|| {
                FeaRunError::Solve("homogeneous static stiffness has no Cholesky proof".into())
            })?;
        Ok(Self {
            reduction,
            system,
            factor,
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
        if reduced_load.iter().all(|value| *value == 0.0) {
            let zero = vec![0.0; self.system.dof_count()];
            return Ok(HomogeneousStaticSolution {
                displacement: self
                    .reduction
                    .expand_linear(&zero)
                    .map_err(|error| FeaRunError::Solve(error.to_string()))?,
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
        .map_err(|error| {
            FeaRunError::Solve(format!("homogeneous static correction failed: {error}"))
        })?;
        let equilibrium_residual = subtract(&apply_k(&self.system, &vector), &reduced_load)?;
        // Normwise backward error ||K x - f|| / (||K||_F ||x|| + ||f||). A direct factor meets it
        // near machine precision on any conditioning; ||K x - f|| / ||f|| alone scales with
        // eps * cond(K) and rejects correct solves of stiff models.
        let stiffness_norm = self
            .system
            .operators()
            .stiffness()
            .values()
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let equilibrium_residual_ratio = norm(&equilibrium_residual)
            / (stiffness_norm * norm(&vector) + norm(&reduced_load)).max(f64::MIN_POSITIVE);
        if !equilibrium_residual_ratio.is_finite() || equilibrium_residual_ratio > STATIC_TOLERANCE
        {
            return Err(FeaRunError::Solve(
                "homogeneous static correction failed its equilibrium tolerance".into(),
            ));
        }
        let displacement = self
            .reduction
            .expand_linear(&vector)
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
