const PROJECTED_SOLVE_TOLERANCE: f64 = 1.0e-13;

pub(crate) fn symmetric_eigendecomposition(
    mut matrix: Vec<f64>,
    dimension: usize,
) -> (Vec<f64>, Vec<f64>) {
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

#[cfg(test)]
mod tests {
    use super::symmetric_eigendecomposition;

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
}
