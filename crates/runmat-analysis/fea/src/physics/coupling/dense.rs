pub(crate) fn solve(matrix: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, String> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return Err("dense coupling operator must be square and match the right-hand side".into());
    }
    if matrix.iter().flatten().any(|value| !value.is_finite())
        || rhs.iter().any(|value| !value.is_finite())
    {
        return Err("dense coupling system must contain only finite values".into());
    }

    let mut matrix = matrix.to_vec();
    let mut rhs = rhs.to_vec();
    for pivot in 0..n {
        let mut pivot_row = pivot;
        let mut pivot_abs = matrix[pivot][pivot].abs();
        for (candidate, row) in matrix.iter().enumerate().skip(pivot + 1) {
            let candidate_abs = row[pivot].abs();
            if candidate_abs > pivot_abs {
                pivot_row = candidate;
                pivot_abs = candidate_abs;
            }
        }
        if pivot_abs <= 1.0e-24 {
            return Err(format!(
                "dense coupling operator is singular at pivot {pivot}"
            ));
        }
        if pivot_row != pivot {
            matrix.swap(pivot, pivot_row);
            rhs.swap(pivot, pivot_row);
        }
        let pivot_value = matrix[pivot][pivot];
        for row in pivot + 1..n {
            let factor = matrix[row][pivot] / pivot_value;
            matrix[row][pivot] = 0.0;
            for col in pivot + 1..n {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let mut accum = rhs[row];
        for (col, value) in solution.iter().enumerate().skip(row + 1) {
            accum -= matrix[row][col] * value;
        }
        solution[row] = accum / matrix[row][row];
    }
    if solution.iter().all(|value| value.is_finite()) {
        Ok(solution)
    } else {
        Err("dense coupling solve produced a non-finite solution".into())
    }
}

pub(crate) fn apply(matrix: &[Vec<f64>], x: &[f64]) -> Result<Vec<f64>, String> {
    if matrix.iter().any(|row| row.len() != x.len()) {
        return Err("dense coupling operator width must match the vector".into());
    }
    Ok(matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(x.iter())
                .map(|(coefficient, value)| coefficient * value)
                .sum::<f64>()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{apply, solve};

    #[test]
    fn pivoted_solve_matches_an_independent_two_by_two_answer() {
        let matrix = vec![vec![0.0, 2.0], vec![1.0, 3.0]];
        let rhs = vec![4.0, 7.0];
        let solution = solve(&matrix, &rhs).expect("nonsingular system should solve");
        assert!((solution[0] - 1.0).abs() <= 1.0e-12);
        assert!((solution[1] - 2.0).abs() <= 1.0e-12);
        let reconstructed = apply(&matrix, &solution).expect("shape should remain valid");
        assert!((reconstructed[0] - rhs[0]).abs() <= 1.0e-12);
        assert!((reconstructed[1] - rhs[1]).abs() <= 1.0e-12);
    }

    #[test]
    fn singular_and_malformed_systems_fail_closed() {
        assert!(solve(&[vec![1.0, 2.0]], &[1.0]).is_err());
        assert!(solve(&[vec![1.0, 2.0], vec![2.0, 4.0]], &[1.0, 2.0]).is_err());
        assert!(apply(&[vec![1.0]], &[1.0, 2.0]).is_err());
    }
}
