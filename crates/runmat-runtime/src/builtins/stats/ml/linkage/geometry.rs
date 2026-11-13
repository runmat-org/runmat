//! Euclidean embedding validation for condensed linkage distances.

use super::invalid;
use crate::BuiltinResult;

pub(super) fn validate_euclidean_condensed(
    values: &[f64],
    observations: usize,
) -> BuiltinResult<()> {
    let mut squared = vec![0.0; observations * observations];
    let mut offset = 0usize;
    for col in 0..observations {
        for row in (col + 1)..observations {
            let value = values[offset] * values[offset];
            squared[row * observations + col] = value;
            squared[col * observations + row] = value;
            offset += 1;
        }
    }

    let row_means = (0..observations)
        .map(|row| {
            (0..observations)
                .map(|col| squared[row * observations + col])
                .sum::<f64>()
                / observations as f64
        })
        .collect::<Vec<_>>();
    let col_means = (0..observations)
        .map(|col| {
            (0..observations)
                .map(|row| squared[row * observations + col])
                .sum::<f64>()
                / observations as f64
        })
        .collect::<Vec<_>>();
    let grand_mean = row_means.iter().sum::<f64>() / observations as f64;
    let mut gram = vec![0.0; observations * observations];
    for row in 0..observations {
        for col in 0..observations {
            gram[row * observations + col] = -0.5
                * (squared[row * observations + col] - row_means[row] - col_means[col]
                    + grand_mean);
        }
    }
    ensure_psd(&gram, observations).map_err(|_| {
        invalid("linkage: centroid, median, and ward require Euclidean condensed distances")
    })
}

fn ensure_psd(matrix: &[f64], n: usize) -> Result<(), ()> {
    let max_diag = (0..n)
        .map(|i| matrix[i * n + i].abs())
        .fold(0.0_f64, f64::max);
    let tol = 1.0e-10 * max_diag.max(1.0) * n as f64;
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum < -tol {
                    return Err(());
                }
                l[i * n + j] = sum.max(0.0).sqrt();
            } else if l[j * n + j] > tol {
                l[i * n + j] = sum / l[j * n + j];
            } else if sum.abs() > tol {
                return Err(());
            }
        }
    }
    Ok(())
}
