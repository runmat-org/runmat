use super::domain::AcousticDomainSystem;

pub(super) fn solve_complex_graph_operator(
    system: &AcousticDomainSystem,
    source_real: &[f64],
    source_imag: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = system.diag_real.len().max(1);
    let mut matrix_real = vec![vec![0.0; n]; n];
    let mut matrix_imag = vec![vec![0.0; n]; n];
    for row in 0..n {
        matrix_real[row][row] = system.diag_real[row];
        matrix_imag[row][row] = system.diag_imag[row];
    }
    for edge in &system.edges {
        matrix_real[edge.left][edge.right] -= edge.stiffness;
        matrix_real[edge.right][edge.left] -= edge.stiffness;
    }
    let mut rhs_real = (0..n)
        .map(|index| source_real.get(index).copied().unwrap_or(0.0))
        .collect::<Vec<_>>();
    let mut rhs_imag = (0..n)
        .map(|index| source_imag.get(index).copied().unwrap_or(0.0))
        .collect::<Vec<_>>();
    solve_dense_complex_system(
        &mut matrix_real,
        &mut matrix_imag,
        &mut rhs_real,
        &mut rhs_imag,
    )
}

fn solve_dense_complex_system(
    matrix_real: &mut [Vec<f64>],
    matrix_imag: &mut [Vec<f64>],
    rhs_real: &mut [f64],
    rhs_imag: &mut [f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = rhs_real.len();
    for pivot in 0..n {
        let mut pivot_row = pivot;
        let mut pivot_norm = complex_abs_sq(matrix_real[pivot][pivot], matrix_imag[pivot][pivot]);
        for candidate in pivot + 1..n {
            let candidate_norm =
                complex_abs_sq(matrix_real[candidate][pivot], matrix_imag[candidate][pivot]);
            if candidate_norm > pivot_norm {
                pivot_row = candidate;
                pivot_norm = candidate_norm;
            }
        }
        if pivot_row != pivot {
            matrix_real.swap(pivot, pivot_row);
            matrix_imag.swap(pivot, pivot_row);
            rhs_real.swap(pivot, pivot_row);
            rhs_imag.swap(pivot, pivot_row);
        }
        if pivot_norm <= 1.0e-24 {
            matrix_real[pivot][pivot] += 1.0e-9;
        }
        let (pivot_inv_real, pivot_inv_imag) =
            complex_recip(matrix_real[pivot][pivot], matrix_imag[pivot][pivot]);
        for row in pivot + 1..n {
            let (factor_real, factor_imag) = complex_mul(
                matrix_real[row][pivot],
                matrix_imag[row][pivot],
                pivot_inv_real,
                pivot_inv_imag,
            );
            matrix_real[row][pivot] = 0.0;
            matrix_imag[row][pivot] = 0.0;
            for col in pivot + 1..n {
                let (update_real, update_imag) = complex_mul(
                    factor_real,
                    factor_imag,
                    matrix_real[pivot][col],
                    matrix_imag[pivot][col],
                );
                matrix_real[row][col] -= update_real;
                matrix_imag[row][col] -= update_imag;
            }
            let (rhs_update_real, rhs_update_imag) =
                complex_mul(factor_real, factor_imag, rhs_real[pivot], rhs_imag[pivot]);
            rhs_real[row] -= rhs_update_real;
            rhs_imag[row] -= rhs_update_imag;
        }
    }

    let mut solution_real = vec![0.0; n];
    let mut solution_imag = vec![0.0; n];
    for row in (0..n).rev() {
        let mut accum_real = rhs_real[row];
        let mut accum_imag = rhs_imag[row];
        for col in row + 1..n {
            let (update_real, update_imag) = complex_mul(
                matrix_real[row][col],
                matrix_imag[row][col],
                solution_real[col],
                solution_imag[col],
            );
            accum_real -= update_real;
            accum_imag -= update_imag;
        }
        let (inv_real, inv_imag) = complex_recip(matrix_real[row][row], matrix_imag[row][row]);
        (solution_real[row], solution_imag[row]) =
            complex_mul(accum_real, accum_imag, inv_real, inv_imag);
    }
    (solution_real, solution_imag)
}

fn complex_abs_sq(real: f64, imag: f64) -> f64 {
    real * real + imag * imag
}

fn complex_recip(real: f64, imag: f64) -> (f64, f64) {
    let denom = (real * real + imag * imag).max(1.0e-24);
    (real / denom, -imag / denom)
}

fn complex_mul(a_real: f64, a_imag: f64, b_real: f64, b_imag: f64) -> (f64, f64) {
    (
        a_real * b_real - a_imag * b_imag,
        a_real * b_imag + a_imag * b_real,
    )
}
