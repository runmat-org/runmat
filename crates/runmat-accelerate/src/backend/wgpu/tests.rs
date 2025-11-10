#![cfg(test)]

use super::provider_impl::{host_tensor_from_value, invert_upper_triangular};
use runmat_builtins::{NumericDType, Tensor, Value};

fn make_column_major(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f64) -> Vec<f64> {
    let mut data = vec![0.0; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            data[row + col * rows] = f(row, col);
        }
    }
    data
}

#[test]
fn invert_upper_triangular_produces_identity() {
    let data = vec![
        1.0, 0.0, 0.0, // column 0
        2.0, 4.0, 0.0, // column 1
        3.0, 5.0, 6.0, // column 2
    ];
    let inv = invert_upper_triangular(&data, 3).expect("invert");
    let idx = |row: usize, col: usize| -> usize { row + col * 3 };
    for col in 0..3 {
        for row in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += data[idx(row, k)] * inv[idx(k, col)];
            }
            if row == col {
                assert!((acc - 1.0).abs() < 1e-12);
            } else {
                assert!(acc.abs() < 1e-12);
            }
        }
    }
}

#[test]
fn cholesky_qr_matches_host_qr() {
    let rows = 12;
    let cols = 4;
    let data = make_column_major(rows, cols, |r, c| {
        let base = (r + 1) as f64;
        base.powi((c + 1) as i32) + (c as f64) * 0.25
    });
    // Form Gram matrix G = Y^T Y
    let mut gram = vec![0.0f64; cols * cols];
    for j in 0..cols {
        for i in 0..=j {
            let mut sum = 0.0;
            for r in 0..rows {
                let y_ri = data[r + i * rows];
                let y_rj = data[r + j * rows];
                sum += y_ri * y_rj;
            }
            gram[i + j * cols] = sum;
            if i != j {
                gram[j + i * cols] = sum;
            }
        }
    }

    let tensor_gram = Tensor {
        data: gram.clone(),
        shape: vec![cols, cols],
        rows: cols,
        cols,
        dtype: NumericDType::F64,
    };
    let chol_eval = runmat_runtime::builtins::math::linalg::factor::chol::evaluate(
        Value::Tensor(tensor_gram),
        &[],
    )
    .expect("chol");
    let r_tensor = host_tensor_from_value("qr_chol_r", chol_eval.factor()).expect("chol factor");
    let r_inv = invert_upper_triangular(&r_tensor.data, cols).expect("invert");

    // Construct Q = Y * inv(R)
    let mut q_computed = vec![0.0f64; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            let mut sum = 0.0;
            for k in 0..cols {
                sum += data[row + k * rows] * r_inv[k + col * cols];
            }
            q_computed[row + col * rows] = sum;
        }
    }

    // Check orthogonality Q^T Q ≈ I
    let mut max_diag_err = 0.0f64;
    let mut max_off_diag = 0.0f64;
    for j in 0..cols {
        for i in 0..cols {
            let mut sum = 0.0;
            for row in 0..rows {
                sum += q_computed[row + i * rows] * q_computed[row + j * rows];
            }
            if i == j {
                max_diag_err = max_diag_err.max((sum - 1.0).abs());
            } else {
                max_off_diag = max_off_diag.max(sum.abs());
            }
        }
    }
    assert!(max_diag_err < 1e-6, "max diag error {}", max_diag_err);
    assert!(max_off_diag < 1e-6, "max off-diagonal {}", max_off_diag);

    // Reconstruct the original matrix and compare
    let mut y_reconstructed = vec![0.0f64; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            let mut sum = 0.0;
            for k in 0..cols {
                sum += q_computed[row + k * rows] * r_tensor.data[k + col * cols];
            }
            y_reconstructed[row + col * rows] = sum;
        }
    }
    let mut max_recon_err = 0.0f64;
    for idx in 0..(rows * cols) {
        max_recon_err = max_recon_err.max((y_reconstructed[idx] - data[idx]).abs());
    }
    assert!(
        max_recon_err < 1e-6,
        "max reconstruction error {}",
        max_recon_err
    );

    // Verify R^T R ≈ Gram
    let mut max_gram_err = 0.0f64;
    for j in 0..cols {
        for i in 0..=j {
            let mut sum = 0.0;
            for k in 0..cols {
                sum += r_tensor.data[k + i * cols] * r_tensor.data[k + j * cols];
            }
            max_gram_err = max_gram_err.max((sum - gram[i + j * cols]).abs());
        }
    }
    assert!(
        max_gram_err < 1e-6,
        "max Gram reconstruction error {}",
        max_gram_err
    );

    // Ensure R is upper-triangular
    for j in 0..cols {
        for i in (j + 1)..cols {
            assert!(
                r_tensor.data[i + j * cols].abs() < 1e-8,
                "R lower entry ({}, {}) = {}",
                i,
                j,
                r_tensor.data[i + j * cols]
            );
        }
    }
}
