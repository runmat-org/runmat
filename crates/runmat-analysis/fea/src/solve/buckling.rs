use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    algebra::{
        AssembledRealSystem, CanonicalRealOperators, MatrixProperties, OperatorRole,
        RealSparseMatrix,
    },
    assembly::AssemblySummary,
    pipeline::structural_linearization::StructuralLinearizationArtifact,
    solve::preconditioner::SpdPreconditionerKind,
};

const POSITIVE_MULTIPLIER_TOLERANCE: f64 = 1.0e-12;
/// Classical Gram–Schmidt passes per block in the shared block Lanczos (CGS2).
pub(crate) const BUCKLING_ORTHOGONALIZATION_PASSES: usize = 2;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BucklingMode {
    pub load_factor: f64,
    /// Relative load-factor error bound `‖K φ + λ K_G φ‖_{K⁻¹} / ‖φ‖_K`.
    pub residual_norm: f64,
    pub dof_shape: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BucklingSolveResult {
    pub converged: bool,
    pub iteration_count: usize,
    pub preconditioner: SpdPreconditionerKind,
    pub modes: Vec<BucklingMode>,
}

#[derive(Debug, Error)]
pub enum BucklingSolveError {
    #[error("buckling base state differs from the current topology or elastic operator")]
    BaseStateMismatch,
    #[error("buckling constraint reduction failed: {0}")]
    ConstraintReduction(String),
    #[error("buckling sparse operator construction failed: {0}")]
    SparseOperatorConstruction(#[source] crate::algebra::RealSparseMatrixError),
    #[error("buckling elastic operator admission failed: {0}")]
    OperatorAdmission(#[source] crate::algebra::RealSystemError),
    #[error("buckling inner elastic solve failed its forcing bound: {0}")]
    InnerSolve(String),
    #[error(
        "buckling eigensolver could not maintain a complete K-orthonormal subspace during {0}"
    )]
    DegenerateSubspace(&'static str),
    #[error("buckling requested mode count and convergence controls must be positive and finite")]
    InvalidControls,
}

pub fn solve_eigenvalue_buckling(
    summary: &AssemblySummary,
    linearization: &StructuralLinearizationArtifact,
    requested_mode_count: usize,
    relative_tolerance: f64,
    maximum_iterations: usize,
) -> Result<BucklingSolveResult, BucklingSolveError> {
    if requested_mode_count == 0
        || maximum_iterations == 0
        || !relative_tolerance.is_finite()
        || relative_tolerance <= 0.0
    {
        return Err(BucklingSolveError::InvalidControls);
    }
    linearization
        .validate()
        .map_err(|_| BucklingSolveError::BaseStateMismatch)?;
    let elastic_digest = summary
        .operator
        .operators()
        .stiffness()
        .numerical_identity()
        .map_err(|_| BucklingSolveError::BaseStateMismatch)?;
    if linearization.elastic_stiffness_digest() != elastic_digest
        || linearization.geometric_stiffness().rows() != summary.dof_count
    {
        return Err(BucklingSolveError::BaseStateMismatch);
    }
    let reduction = summary
        .constraint_matrix
        .as_ref()
        .map(|constraints| constraints.affine_reduction())
        .transpose()
        .map_err(|error| BucklingSolveError::ConstraintReduction(error.to_string()))?
        .unwrap_or_else(|| {
            crate::constraints::AffineConstraintReduction::from_constrained_dofs(
                summary.operator.constrained_dofs(),
            )
        });
    let stiffness = reduction
        .reduce_sparse_operator(summary.operator.operators().stiffness())
        .map_err(|error| BucklingSolveError::ConstraintReduction(error.to_string()))?;
    let geometric = reduction
        .reduce_sparse_operator(linearization.geometric_stiffness())
        .map_err(|error| BucklingSolveError::ConstraintReduction(error.to_string()))?;
    let dimension = stiffness.rows();
    let system = AssembledRealSystem::try_new(
        CanonicalRealOperators::try_new(
            stiffness,
            RealSparseMatrix::diagonal(
                vec![1.0; dimension],
                OperatorRole::Mass,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .map_err(BucklingSolveError::SparseOperatorConstruction)?,
            RealSparseMatrix::diagonal(
                vec![0.0; dimension],
                OperatorRole::Damping,
                MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
            )
            .map_err(BucklingSolveError::SparseOperatorConstruction)?,
        )
        .map_err(BucklingSolveError::OperatorAdmission)?,
        vec![false; dimension],
        vec![0.0; dimension],
    )
    .map_err(BucklingSolveError::OperatorAdmission)?;
    // The coercivity proof of K is a sparse Cholesky; that factor drives the Lanczos operator
    // K⁻¹(−K_G) in the K inner product.
    let prepared = crate::solve::shared_sparse::with_plan(
        crate::planning::resource::LocalExecutionPlan::serial(),
        || system.prepare_linear(),
    )
    .map_err(BucklingSolveError::OperatorAdmission)?;
    let factor = prepared
        .admissibility()
        .and_then(|evidence| evidence.factor())
        .map(|factor| factor.0.clone());
    let stiffness_arrays =
        crate::solve::shared_sparse::CsrArrays::of(system.operators().stiffness());
    let geometric_arrays = crate::solve::shared_sparse::CsrArrays::of(&geometric);
    let options = runmat_sparse_solvers::eigen::EigenOptions {
        count: requested_mode_count.min(dimension),
        tolerance: (0.1 * relative_tolerance).min(1.0e-10),
        max_restarts: maximum_iterations,
        ..Default::default()
    };
    let result = crate::solve::shared_sparse::with_plan(
        crate::planning::resource::LocalExecutionPlan::serial(),
        || {
            let (k, kg) = (stiffness_arrays.view(), geometric_arrays.view());
            match factor.as_deref() {
                Some(solver) => runmat_sparse_solvers::eigen::buckling_with_factor(
                    k,
                    kg,
                    solver,
                    &options,
                    crate::solve::shared_sparse::control(),
                ),
                None => runmat_sparse_solvers::eigen::buckling(
                    k,
                    kg,
                    &options,
                    crate::solve::shared_sparse::control(),
                ),
            }
        },
    )
    .map_err(|error| BucklingSolveError::InnerSolve(error.to_string()))?;
    // Load factors and their relative error bounds come from the shared eigensolver: one
    // definition of the residual for the Lanczos stopping test and for this acceptance test.
    let mut modes = Vec::with_capacity(result.values.len());
    for (index, (&load_factor, &residual_norm)) in
        result.values.iter().zip(&result.residuals).enumerate()
    {
        if !load_factor.is_finite() || 1.0 / load_factor <= POSITIVE_MULTIPLIER_TOLERANCE {
            continue;
        }
        let mut vector = result.vectors[index * dimension..(index + 1) * dimension].to_vec();
        canonicalize_sign(&mut vector);
        modes.push(BucklingMode {
            load_factor,
            residual_norm,
            dof_shape: reduction
                .expand_linear(&vector)
                .map_err(|error| BucklingSolveError::ConstraintReduction(error.to_string()))?,
        });
    }
    modes.sort_by(|left, right| left.load_factor.total_cmp(&right.load_factor));
    modes.truncate(requested_mode_count);
    let converged = result.converged
        && modes.len() == requested_mode_count
        && modes
            .iter()
            .all(|mode| mode.residual_norm <= relative_tolerance);
    Ok(BucklingSolveResult {
        converged,
        iteration_count: result.restarts.max(1),
        preconditioner: SpdPreconditionerKind::Direct,
        modes,
    })
}

fn canonicalize_sign(vector: &mut [f64]) {
    let pivot = vector
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map(|(index, _)| index);
    if pivot.is_some_and(|index| vector[index].is_sign_negative()) {
        for value in vector {
            *value = -*value;
        }
    }
}
