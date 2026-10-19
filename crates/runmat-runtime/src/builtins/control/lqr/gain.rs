//! One in-place feedback-denominator solve for both allocation policies.

use super::{RiccatiApi, RiccatiIssue};
use crate::BuiltinResult;

/// Solve D*K=C in column-major caller-owned buffers. Partial pivoting avoids
/// constructing an inverse. Singularity is independent of memory policy;
/// the original-equation residual and stability checks certify the result.
pub(super) fn solve_in_place(
    denominator: &mut [f64],
    gain: &mut [f64],
    m: usize,
    n: usize,
    api: RiccatiApi,
    mut checkpoint: impl FnMut() -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    if m == 0 || m.checked_mul(m) != Some(denominator.len()) || m.checked_mul(n) != Some(gain.len())
    {
        return Err(failed(api, "Riccati feedback dimensions are invalid"));
    }
    if denominator.iter().any(|value| !value.is_finite()) {
        return Err(failed(api, "Riccati feedback denominator is nonfinite"));
    }
    for k in 0..m {
        let mut pivot = k;
        for row in (k + 1)..m {
            checkpoint()?;
            if denominator[k * m + row].abs() > denominator[k * m + pivot].abs() {
                pivot = row;
            }
        }
        let diagonal = denominator[k * m + pivot];
        if !diagonal.is_finite() || diagonal == 0.0 {
            return Err(failed(api, "Riccati feedback denominator is singular"));
        }
        if pivot != k {
            for col in 0..m {
                checkpoint()?;
                denominator.swap(col * m + k, col * m + pivot);
            }
            for col in 0..n {
                checkpoint()?;
                gain.swap(col * m + k, col * m + pivot);
            }
        }
        for row in (k + 1)..m {
            let multiplier = denominator[k * m + row] / denominator[k * m + k];
            denominator[k * m + row] = 0.0;
            for col in (k + 1)..m {
                checkpoint()?;
                let prior = denominator[col * m + k];
                denominator[col * m + row] -= multiplier * prior;
            }
            for col in 0..n {
                checkpoint()?;
                let prior = gain[col * m + k];
                gain[col * m + row] -= multiplier * prior;
            }
        }
    }
    for row in (0..m).rev() {
        for col in 0..n {
            let mut value = gain[col * m + row];
            for k in (row + 1)..m {
                checkpoint()?;
                value -= denominator[k * m + row] * gain[col * m + k];
            }
            value /= denominator[row * m + row];
            if !value.is_finite() {
                return Err(failed(api, "Riccati feedback gain is nonfinite"));
            }
            gain[col * m + row] = value;
        }
    }
    checkpoint()
}

fn failed(api: RiccatiApi, message: &'static str) -> crate::RuntimeError {
    api.error(RiccatiIssue::SolverFailed, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn shared_gain_solve_pivots_all_rhs_without_a_memory_policy_threshold() {
        for (mut denominator, mut rhs, expected) in [
            (
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 7.0, 8.0, 15.0],
                [1.0, 2.0, 3.0, 4.0],
            ),
            (
                [1.0e-100, 0.0, 0.0, 1.0e100],
                [2.0e-100, 3.0e100, 4.0e-100, 5.0e100],
                [2.0, 3.0, 4.0, 5.0],
            ),
        ] {
            solve_in_place(
                &mut denominator,
                &mut rhs,
                2,
                2,
                RiccatiApi::Care,
                || Ok(()),
            )
            .unwrap();
            for (actual, expected) in rhs.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 1.0e-12);
            }
        }
    }
}
