use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    algebra::{
        spmv, AssembledRealSystem, CanonicalRealOperators, MatrixProperties, OperatorRole,
        RealSparseMatrix,
    },
    assembly::AssemblySummary,
    pipeline::structural_linearization::StructuralLinearizationArtifact,
    solve::{
        modal::{
            linear_solve::{solve_k_system_cg, CgSolveOptions},
            subspace::symmetric_eigendecomposition,
        },
        preconditioner::{build_spd_preconditioner, SpdPreconditionerKind},
    },
};

const POSITIVE_MULTIPLIER_TOLERANCE: f64 = 1.0e-12;
const ORTHOGONALIZATION_PASSES: usize = 2;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BucklingMode {
    pub load_factor: f64,
    pub residual_norm: f64,
    pub dof_shape: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BucklingSolveResult {
    pub converged: bool,
    pub iteration_count: usize,
    pub modes: Vec<BucklingMode>,
}

#[derive(Debug, Error)]
pub enum BucklingSolveError {
    #[error("buckling base state differs from the current topology or elastic operator")]
    BaseStateMismatch,
    #[error("buckling constraint reduction failed: {0}")]
    ConstraintReduction(String),
    #[error("buckling elastic operator is not coercive: {0}")]
    NonCoercive(String),
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
        .unwrap_or_else(|| direct_constraint_reduction(summary.operator.constrained_dofs()));
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
            .map_err(|error| BucklingSolveError::NonCoercive(error.to_string()))?,
            RealSparseMatrix::diagonal(
                vec![0.0; dimension],
                OperatorRole::Damping,
                MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
            )
            .map_err(|error| BucklingSolveError::NonCoercive(error.to_string()))?,
        )
        .map_err(|error| BucklingSolveError::NonCoercive(error.to_string()))?,
        vec![false; dimension],
        vec![0.0; dimension],
    )
    .map_err(|error| BucklingSolveError::NonCoercive(error.to_string()))?;
    let prepared = system
        .prepare_linear()
        .map_err(|error| BucklingSolveError::NonCoercive(error.to_string()))?;
    let preconditioner = build_spd_preconditioner(&prepared, SpdPreconditionerKind::Ilu0)
        .or_else(|_| build_spd_preconditioner(&prepared, SpdPreconditionerKind::Jacobi))
        .map_err(|error| BucklingSolveError::NonCoercive(error.to_string()))?;

    let subspace_dimension = dimension.min(requested_mode_count.saturating_add(4));
    let mut basis = deterministic_basis(dimension, subspace_dimension);
    orthonormalize_k(&system, &mut basis);
    if basis.len() != subspace_dimension {
        return Err(BucklingSolveError::DegenerateSubspace("initialization"));
    }
    let mut previous: Vec<f64> = Vec::new();
    let mut modes = Vec::new();
    for iteration in 1..=maximum_iterations {
        let mut filtered = Vec::with_capacity(subspace_dimension);
        for vector in &basis {
            let rhs = spmv(&geometric, vector)
                .map_err(|error| BucklingSolveError::ConstraintReduction(format!("{error:?}")))?
                .into_iter()
                .map(|value| -value)
                .collect::<Vec<_>>();
            let attempt = solve_k_system_cg(
                &system,
                &rhs,
                CgSolveOptions {
                    max_iters: dimension.saturating_mul(20).max(200),
                    tol: relative_tolerance.min(1.0e-10) * 0.1,
                    prepared_accelerator_system: None,
                    cpu_preconditioner: Some(preconditioner.as_ref()),
                    initial_guess: None,
                },
            );
            if !attempt.converged {
                return Err(BucklingSolveError::NonCoercive(
                    "inner elastic solve did not converge".into(),
                ));
            }
            filtered.push(attempt.vector);
        }
        orthonormalize_k(&system, &mut filtered);
        if filtered.len() != subspace_dimension {
            return Err(BucklingSolveError::DegenerateSubspace("inverse iteration"));
        }
        let projected = projected_negative_geometric(&geometric, &filtered)?;
        let (_, rotations) = symmetric_eigendecomposition(projected, subspace_dimension);
        basis = rotate_basis(&filtered, &rotations);
        orthonormalize_k(&system, &mut basis);
        modes = build_modes(&system, &geometric, &reduction, &basis)?;
        modes.truncate(requested_mode_count);
        let current = modes
            .iter()
            .map(|mode| mode.load_factor)
            .collect::<Vec<_>>();
        let stable = previous.len() == current.len()
            && previous.iter().zip(&current).all(|(left, right)| {
                (left - right).abs() <= relative_tolerance * left.abs().max(right.abs()).max(1.0)
            });
        let residuals_converged = modes.len() == requested_mode_count
            && modes
                .iter()
                .all(|mode| mode.residual_norm <= relative_tolerance);
        if stable && residuals_converged {
            return Ok(BucklingSolveResult {
                converged: true,
                iteration_count: iteration,
                modes,
            });
        }
        previous = current;
    }
    Ok(BucklingSolveResult {
        converged: false,
        iteration_count: maximum_iterations,
        modes,
    })
}

fn build_modes(
    system: &AssembledRealSystem,
    geometric: &RealSparseMatrix,
    reduction: &crate::constraints::AffineConstraintReduction,
    basis: &[Vec<f64>],
) -> Result<Vec<BucklingMode>, BucklingSolveError> {
    let mut modes = Vec::new();
    for vector in basis {
        let elastic = spmv(system.operators().stiffness(), vector)
            .map_err(|error| BucklingSolveError::ConstraintReduction(format!("{error:?}")))?;
        let initial_stress = spmv(geometric, vector)
            .map_err(|error| BucklingSolveError::ConstraintReduction(format!("{error:?}")))?;
        let denominator = dot(vector, &elastic);
        let multiplier = -dot(vector, &initial_stress) / denominator.max(1.0e-24);
        if !multiplier.is_finite() || multiplier <= POSITIVE_MULTIPLIER_TOLERANCE {
            continue;
        }
        let load_factor = 1.0 / multiplier;
        let residual = elastic
            .iter()
            .zip(&initial_stress)
            .map(|(k, kg)| (k + load_factor * kg).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = elastic
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            + load_factor.abs()
                * initial_stress
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
        modes.push(BucklingMode {
            load_factor,
            residual_norm: residual / scale.max(1.0e-24),
            dof_shape: reduction
                .expand_linear(vector)
                .map_err(|error| BucklingSolveError::ConstraintReduction(error.to_string()))?,
        });
    }
    modes.sort_by(|left, right| left.load_factor.total_cmp(&right.load_factor));
    Ok(modes)
}

fn projected_negative_geometric(
    geometric: &RealSparseMatrix,
    basis: &[Vec<f64>],
) -> Result<Vec<f64>, BucklingSolveError> {
    let dimension = basis.len();
    let applied = basis
        .iter()
        .map(|vector| spmv(geometric, vector))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| BucklingSolveError::ConstraintReduction(format!("{error:?}")))?;
    let mut projected = vec![0.0; dimension * dimension];
    for row in 0..dimension {
        for column in row..dimension {
            let value = -dot(&basis[row], &applied[column]);
            projected[row * dimension + column] = value;
            projected[column * dimension + row] = value;
        }
    }
    Ok(projected)
}

fn orthonormalize_k(system: &AssembledRealSystem, vectors: &mut Vec<Vec<f64>>) {
    let mut accepted: Vec<Vec<f64>> = Vec::with_capacity(vectors.len());
    for mut vector in vectors.drain(..) {
        for _ in 0..ORTHOGONALIZATION_PASSES {
            for previous in &accepted {
                let projection = dot(
                    previous,
                    &spmv(system.operators().stiffness(), &vector).unwrap(),
                );
                for (value, basis) in vector.iter_mut().zip(previous) {
                    *value -= projection * basis;
                }
            }
        }
        let applied = spmv(system.operators().stiffness(), &vector).unwrap();
        let norm = dot(&vector, &applied).max(0.0).sqrt();
        if norm > 1.0e-12 {
            for value in &mut vector {
                *value /= norm;
            }
            canonicalize_sign(&mut vector);
            accepted.push(vector);
        }
    }
    *vectors = accepted;
}

fn rotate_basis(basis: &[Vec<f64>], rotations: &[f64]) -> Vec<Vec<f64>> {
    let dimension = basis.len();
    (0..dimension)
        .map(|output| {
            let mut vector = vec![0.0; basis[0].len()];
            for input in 0..dimension {
                for (value, basis_value) in vector.iter_mut().zip(&basis[input]) {
                    *value += rotations[input * dimension + output] * basis_value;
                }
            }
            vector
        })
        .collect()
}

fn deterministic_basis(rows: usize, columns: usize) -> Vec<Vec<f64>> {
    (0..columns)
        .map(|column| {
            (0..rows)
                .map(|row| {
                    let phase = ((row + 1) * (column + 1)) as f64;
                    (phase * 0.754_877_666_246_692_7).sin()
                        + (phase * 0.569_840_290_998_053_2).cos()
                })
                .collect()
        })
        .collect()
}

fn direct_constraint_reduction(
    constrained: &[bool],
) -> crate::constraints::AffineConstraintReduction {
    let dimension = constrained.len();
    let free_dof_indices = constrained
        .iter()
        .enumerate()
        .filter_map(|(index, constrained)| (!constrained).then_some(index))
        .collect::<Vec<_>>();
    let free_columns = free_dof_indices
        .iter()
        .enumerate()
        .map(|(column, dof)| (*dof, column))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut row_offsets = Vec::with_capacity(dimension + 1);
    let mut column_indices = Vec::with_capacity(free_dof_indices.len());
    let mut coefficients = Vec::with_capacity(free_dof_indices.len());
    row_offsets.push(0);
    for dof in 0..dimension {
        if let Some(column) = free_columns.get(&dof) {
            column_indices.push(*column);
            coefficients.push(1.0);
        }
        row_offsets.push(column_indices.len());
    }
    crate::constraints::AffineConstraintReduction {
        full_dof_count: dimension,
        reduced_dof_count: free_dof_indices.len(),
        free_dof_indices,
        pivot_dof_indices: constrained
            .iter()
            .enumerate()
            .filter_map(|(index, constrained)| constrained.then_some(index))
            .collect(),
        recovery_row_offsets: row_offsets,
        recovery_column_indices: column_indices,
        recovery_coefficients: coefficients,
        offset: vec![0.0; dimension],
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
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
