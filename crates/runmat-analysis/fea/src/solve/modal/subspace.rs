use crate::algebra::{apply_k, apply_m, AssembledRealSystem};

use super::math::{dot, orthonormalize_mass_block};

const EIGENVALUE_STABILITY_TOLERANCE: f64 = 1.0e-9;
const PROJECTED_SOLVE_TOLERANCE: f64 = 1.0e-13;

#[derive(Debug)]
pub(super) struct RitzCandidate {
    pub(super) eigenvalue: f64,
    pub(super) vector: Vec<f64>,
    pub(super) residual: f64,
}

#[derive(Debug)]
pub(super) struct BlockSubspaceResult {
    pub(super) candidates: Vec<RitzCandidate>,
    pub(super) converged: bool,
    pub(super) iteration_count: usize,
    pub(super) maximum_residual: f64,
}

pub(super) fn solve_lowest_eigenspace<F>(
    system: &AssembledRealSystem,
    subspace_dimension: usize,
    qualified_dimension: usize,
    residual_tolerance: f64,
    maximum_iterations: usize,
    mut solve_stiffness: F,
) -> BlockSubspaceResult
where
    F: FnMut(&[f64], Option<&[f64]>) -> Option<Vec<f64>>,
{
    let dimension = subspace_dimension.min(system.dof_count());
    let qualified_dimension = qualified_dimension.min(dimension);
    let mut basis = deterministic_starting_block(system.dof_count(), dimension);
    orthonormalize_mass_block(system, &mut basis);
    let mut previous_eigenvalues: Vec<f64> = Vec::new();
    let mut candidates = Vec::new();
    let mut maximum_residual = f64::INFINITY;

    for iteration in 1..=maximum_iterations {
        let mut filtered = Vec::with_capacity(dimension);
        for vector in &basis {
            let rhs = apply_m(system, vector);
            let Some(solution) = solve_stiffness(&rhs, None) else {
                return BlockSubspaceResult {
                    candidates,
                    converged: false,
                    iteration_count: iteration,
                    maximum_residual,
                };
            };
            filtered.push(solution);
        }
        orthonormalize_mass_block(system, &mut filtered);
        if filtered.len() != dimension {
            return BlockSubspaceResult {
                candidates,
                converged: false,
                iteration_count: iteration,
                maximum_residual,
            };
        }

        let projected = projected_stiffness(system, &filtered);
        let (eigenvalues, rotations) = symmetric_eigendecomposition(projected, dimension);
        basis = rotate_basis(&filtered, &rotations, dimension);
        orthonormalize_mass_block(system, &mut basis);
        candidates = build_candidates(system, &basis);
        maximum_residual = candidates
            .iter()
            .take(qualified_dimension)
            .map(|candidate| candidate.residual)
            .fold(0.0_f64, f64::max);
        let eigenvalues_stable = previous_eigenvalues.len() == eigenvalues.len()
            && previous_eigenvalues
                .iter()
                .zip(&eigenvalues)
                .take(qualified_dimension)
                .all(|(previous, current)| {
                    let scale = previous.abs().max(current.abs()).max(1.0);
                    (previous - current).abs() / scale <= EIGENVALUE_STABILITY_TOLERANCE
                });
        if maximum_residual <= residual_tolerance && eigenvalues_stable {
            return BlockSubspaceResult {
                candidates,
                converged: true,
                iteration_count: iteration,
                maximum_residual,
            };
        }
        previous_eigenvalues = eigenvalues;
    }

    BlockSubspaceResult {
        candidates,
        converged: false,
        iteration_count: maximum_iterations,
        maximum_residual,
    }
}

fn deterministic_starting_block(row_count: usize, column_count: usize) -> Vec<Vec<f64>> {
    (0..column_count)
        .map(|column| {
            (0..row_count)
                .map(|row| {
                    let phase = (row + 1) as f64 * (column + 1) as f64;
                    (phase * 0.754_877_666_246_692_7).sin()
                        + (phase * 0.569_840_290_998_053_2).cos()
                })
                .collect()
        })
        .collect()
}

fn projected_stiffness(system: &AssembledRealSystem, basis: &[Vec<f64>]) -> Vec<f64> {
    let dimension = basis.len();
    let applied = basis
        .iter()
        .map(|vector| apply_k(system, vector))
        .collect::<Vec<_>>();
    let mut projected = vec![0.0; dimension * dimension];
    for row in 0..dimension {
        for column in row..dimension {
            let value = dot(&basis[row], &applied[column]);
            projected[row * dimension + column] = value;
            projected[column * dimension + row] = value;
        }
    }
    projected
}

fn rotate_basis(basis: &[Vec<f64>], rotations: &[f64], dimension: usize) -> Vec<Vec<f64>> {
    (0..dimension)
        .map(|output_column| {
            let mut vector = vec![0.0; basis[0].len()];
            for input_column in 0..dimension {
                let coefficient = rotations[input_column * dimension + output_column];
                for (value, input) in vector.iter_mut().zip(&basis[input_column]) {
                    *value += coefficient * input;
                }
            }
            canonicalize_sign(&mut vector);
            vector
        })
        .collect()
}

fn build_candidates(system: &AssembledRealSystem, basis: &[Vec<f64>]) -> Vec<RitzCandidate> {
    let mut candidates = basis
        .iter()
        .map(|vector| {
            let k_vector = apply_k(system, vector);
            let m_vector = apply_m(system, vector);
            let eigenvalue =
                (dot(vector, &k_vector) / dot(vector, &m_vector).max(1.0e-24)).max(0.0);
            let residual_norm = k_vector
                .iter()
                .zip(&m_vector)
                .map(|(stiffness, mass)| (stiffness - eigenvalue * mass).powi(2))
                .sum::<f64>()
                .sqrt();
            let scale = k_vector
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
                + eigenvalue.abs()
                    * m_vector
                        .iter()
                        .map(|value| value * value)
                        .sum::<f64>()
                        .sqrt();
            RitzCandidate {
                eigenvalue,
                vector: vector.clone(),
                residual: residual_norm / scale.max(1.0e-24),
            }
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.eigenvalue.total_cmp(&right.eigenvalue));
    candidates
}

fn symmetric_eigendecomposition(mut matrix: Vec<f64>, dimension: usize) -> (Vec<f64>, Vec<f64>) {
    let mut vectors = vec![0.0; dimension * dimension];
    for index in 0..dimension {
        vectors[index * dimension + index] = 1.0;
    }
    let sweep_limit = 12usize
        .saturating_mul(dimension.saturating_mul(dimension))
        .max(1);
    for _ in 0..sweep_limit {
        let Some((row, column, magnitude)) = largest_off_diagonal(&matrix, dimension) else {
            break;
        };
        let scale = matrix[row * dimension + row]
            .abs()
            .max(matrix[column * dimension + column].abs())
            .max(1.0);
        if magnitude <= PROJECTED_SOLVE_TOLERANCE * scale {
            break;
        }
        apply_jacobi_rotation(&mut matrix, &mut vectors, dimension, row, column);
    }
    let mut order = (0..dimension).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        matrix[*left * dimension + *left].total_cmp(&matrix[*right * dimension + *right])
    });
    let eigenvalues = order
        .iter()
        .map(|index| matrix[*index * dimension + *index])
        .collect::<Vec<_>>();
    let mut ordered_vectors = vec![0.0; dimension * dimension];
    for (output, input) in order.into_iter().enumerate() {
        for row in 0..dimension {
            ordered_vectors[row * dimension + output] = vectors[row * dimension + input];
        }
    }
    (eigenvalues, ordered_vectors)
}

fn largest_off_diagonal(matrix: &[f64], dimension: usize) -> Option<(usize, usize, f64)> {
    let mut largest = None;
    for row in 0..dimension {
        for column in row + 1..dimension {
            let magnitude = matrix[row * dimension + column].abs();
            if largest.is_none_or(|(_, _, current)| magnitude > current) {
                largest = Some((row, column, magnitude));
            }
        }
    }
    largest
}

fn apply_jacobi_rotation(
    matrix: &mut [f64],
    vectors: &mut [f64],
    dimension: usize,
    row: usize,
    column: usize,
) {
    let diagonal_delta = matrix[column * dimension + column] - matrix[row * dimension + row];
    let angle = 0.5 * (2.0 * matrix[row * dimension + column]).atan2(diagonal_delta);
    let cosine = angle.cos();
    let sine = angle.sin();
    for index in 0..dimension {
        let left = matrix[index * dimension + row];
        let right = matrix[index * dimension + column];
        matrix[index * dimension + row] = cosine * left - sine * right;
        matrix[index * dimension + column] = sine * left + cosine * right;
    }
    for index in 0..dimension {
        let upper = matrix[row * dimension + index];
        let lower = matrix[column * dimension + index];
        matrix[row * dimension + index] = cosine * upper - sine * lower;
        matrix[column * dimension + index] = sine * upper + cosine * lower;
    }
    matrix[row * dimension + column] = 0.0;
    matrix[column * dimension + row] = 0.0;
    for index in 0..dimension {
        let left = vectors[index * dimension + row];
        let right = vectors[index * dimension + column];
        vectors[index * dimension + row] = cosine * left - sine * right;
        vectors[index * dimension + column] = sine * left + cosine * right;
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{
        CanonicalRealOperators, MatrixProperties, OperatorRole, RealSparseMatrix,
    };

    fn diagonal_system(stiffness: Vec<f64>, mass: Vec<f64>) -> AssembledRealSystem {
        let dimension = stiffness.len();
        let damping = vec![0.0; dimension];
        AssembledRealSystem::try_new(
            CanonicalRealOperators::try_new(
                RealSparseMatrix::diagonal(
                    stiffness,
                    OperatorRole::Stiffness,
                    MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
                )
                .unwrap(),
                RealSparseMatrix::diagonal(
                    mass,
                    OperatorRole::Mass,
                    MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
                )
                .unwrap(),
                RealSparseMatrix::diagonal(
                    damping,
                    OperatorRole::Damping,
                    MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
                )
                .unwrap(),
            )
            .unwrap(),
            vec![false; dimension],
            vec![0.0; dimension],
        )
        .unwrap()
    }

    #[test]
    fn projected_symmetric_solve_orders_repeated_eigenvalues() {
        let (values, vectors) =
            symmetric_eigendecomposition(vec![4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3);
        assert_eq!(values, [1.0, 1.0, 4.0]);
        for column in 0..3 {
            let norm = (0..3)
                .map(|row| vectors[row * 3 + column].powi(2))
                .sum::<f64>();
            assert!((norm - 1.0).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn block_iteration_recovers_a_repeated_generalized_eigenspace() {
        let stiffness = vec![4.0, 6.0, 36.0, 80.0];
        let mass = vec![2.0, 3.0, 4.0, 5.0];
        let system = diagonal_system(stiffness.clone(), mass);
        let result = solve_lowest_eigenspace(&system, 4, 3, 1.0e-12, 8, |rhs, _| {
            Some(
                rhs.iter()
                    .zip(&stiffness)
                    .map(|(value, diagonal)| value / diagonal)
                    .collect(),
            )
        });

        assert!(result.converged);
        assert_eq!(result.candidates.len(), 4);
        for (candidate, expected) in result.candidates.iter().zip([2.0, 2.0, 9.0, 16.0]) {
            assert!((candidate.eigenvalue - expected).abs() <= 1.0e-10);
            assert!(candidate.residual <= 1.0e-12);
        }
        let repeated_inner_product = super::super::math::mass_inner_product(
            &system,
            &result.candidates[0].vector,
            &result.candidates[1].vector,
        );
        assert!(repeated_inner_product.abs() <= 1.0e-12);
    }
}
