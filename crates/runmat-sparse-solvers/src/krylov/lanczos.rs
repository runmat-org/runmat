//! Ritz values of the Lanczos tridiagonal that PCG's coefficients define.
//!
//! PCG on `A` with preconditioner `M` is the Lanczos process on `M⁻¹A`; its coefficients give
//! the tridiagonal `T` with `T_jj = 1/α_j + β_{j−1}/α_{j−1}` and `T_{j,j+1} = √β_j / α_j`.
//! Ritz values lie inside the spectrum of `M⁻¹A`, so the smallest one approaches `λ_min` from
//! above: it is an estimate, never a bound.
use super::pcg::PcgCoefficients;

/// Extreme Ritz values of `M⁻¹A` from one PCG Lanczos segment (an estimate, not a proof).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PcgSpectrumEstimate {
    /// Smallest Ritz value: an over-estimate of `λ_min(M⁻¹A)`.
    pub lambda_min: f64,
    /// Largest Ritz value: an under-estimate of `λ_max(M⁻¹A)`.
    pub lambda_max: f64,
    /// Lanczos steps in the segment (iterations since the start or the last restart).
    pub lanczos_steps: usize,
    /// The coefficients came from flexible CG, whose tridiagonal is only approximately Lanczos.
    pub flexible: bool,
}

impl PcgSpectrumEstimate {
    /// `None` when the segment is empty or a coefficient is not a positive finite number.
    pub fn from_coefficients(coefficients: &PcgCoefficients) -> Option<Self> {
        let (lambda_min, lambda_max) = ritz_extremes(&coefficients.alpha, &coefficients.beta)?;
        Some(Self {
            lambda_min,
            lambda_max,
            lanczos_steps: coefficients.alpha.len(),
            flexible: coefficients.flexible,
        })
    }
}

/// The smallest and largest eigenvalue of the PCG tridiagonal, by Sturm-count bisection.
fn ritz_extremes(alpha: &[f64], beta: &[f64]) -> Option<(f64, f64)> {
    let k = alpha.len();
    if k == 0
        || beta.len() + 1 < k
        || alpha
            .iter()
            .any(|value| !(value.is_finite() && *value > 0.0))
        || beta[..k - 1]
            .iter()
            .any(|value| !(value.is_finite() && *value >= 0.0))
    {
        return None;
    }
    let mut diagonal = Vec::with_capacity(k);
    let mut off_squared = Vec::with_capacity(k - 1);
    for j in 0..k {
        let shift = if j > 0 {
            beta[j - 1] / alpha[j - 1]
        } else {
            0.0
        };
        diagonal.push(1.0 / alpha[j] + shift);
        if j + 1 < k {
            off_squared.push(beta[j] / (alpha[j] * alpha[j]));
        }
    }
    let off = |index: usize| {
        off_squared
            .get(index)
            .map_or(0.0, |value: &f64| value.sqrt())
    };
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    for (j, &value) in diagonal.iter().enumerate() {
        let radius = off(j) + if j > 0 { off(j - 1) } else { 0.0 };
        low = low.min(value - radius);
        high = high.max(value + radius);
    }
    if !(low.is_finite() && high.is_finite()) {
        return None;
    }
    let tiny = f64::EPSILON * low.abs().max(high.abs()).max(f64::MIN_POSITIVE);
    // Eigenvalues of T below x: negative pivots of the LDLᵀ factorization of T − xI.
    let count_below = |x: f64| {
        let mut count = 0;
        let mut pivot = 1.0_f64;
        for j in 0..k {
            let coupling = if j > 0 {
                off_squared[j - 1] / pivot
            } else {
                0.0
            };
            pivot = diagonal[j] - x - coupling;
            if pivot == 0.0 {
                pivot = -tiny;
            }
            if pivot < 0.0 {
                count += 1;
            }
        }
        count
    };
    let eigenvalue = |index: usize| {
        let (mut lower, mut upper) = (low, high);
        for _ in 0..200 {
            let middle = 0.5 * (lower + upper);
            if middle <= lower || middle >= upper {
                break;
            }
            if count_below(middle) > index {
                upper = middle;
            } else {
                lower = middle;
            }
        }
        0.5 * (lower + upper)
    };
    Some((eigenvalue(0), eigenvalue(k - 1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unpreconditioned CG on diag(1, 4, 9) reaches the exact spectrum in three steps.
    #[test]
    fn three_steps_of_cg_on_a_diagonal_recover_its_extreme_eigenvalues() {
        let diagonal = [1.0, 4.0, 9.0];
        let (mut x, mut r, mut p) = ([0.0; 3], [1.0; 3], [1.0; 3]);
        let mut coefficients = PcgCoefficients::default();
        let mut rho: f64 = r.iter().map(|v| v * v).sum();
        for _ in 0..3 {
            let ap: Vec<f64> = p.iter().zip(diagonal).map(|(p, d)| p * d).collect();
            let alpha = rho / p.iter().zip(&ap).map(|(p, a)| p * a).sum::<f64>();
            coefficients.alpha.push(alpha);
            for i in 0..3 {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let next: f64 = r.iter().map(|v| v * v).sum();
            coefficients.beta.push(next / rho);
            for i in 0..3 {
                p[i] = r[i] + next / rho * p[i];
            }
            rho = next;
        }
        let estimate = PcgSpectrumEstimate::from_coefficients(&coefficients).unwrap();
        assert!((estimate.lambda_min - 1.0).abs() < 1e-9, "{estimate:?}");
        assert!((estimate.lambda_max - 9.0).abs() < 1e-9, "{estimate:?}");
        assert_eq!(estimate.lanczos_steps, 3);
        assert!(!estimate.flexible);
    }

    #[test]
    fn an_empty_or_invalid_segment_has_no_estimate() {
        assert!(PcgSpectrumEstimate::from_coefficients(&PcgCoefficients::default()).is_none());
        let negative = PcgCoefficients {
            alpha: vec![-1.0],
            beta: vec![],
            flexible: false,
        };
        assert!(PcgSpectrumEstimate::from_coefficients(&negative).is_none());
    }
}
