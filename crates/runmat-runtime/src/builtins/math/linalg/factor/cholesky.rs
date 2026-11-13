//! Shared dense Cholesky kernel. Callers own value validation and triangle selection.

use num_complex::Complex64;

const EPS: f64 = 1.0e-12;

pub(super) struct CholeskyFactor {
    pub upper: Vec<Complex64>,
    pub info: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum CholeskyError {
    SizeOverflow,
    Allocation,
}

/// Factor a square Hermitian matrix supplied through borrowed indexed access.
///
/// The result is row-major upper triangular. A nonzero `info` identifies the
/// first failing leading minor, using one-based indexing. Empty input succeeds.
pub(super) fn factor_upper(
    order: usize,
    entry: impl Fn(usize, usize) -> Complex64,
) -> Result<CholeskyFactor, CholeskyError> {
    let len = order
        .checked_mul(order)
        .ok_or(CholeskyError::SizeOverflow)?;
    let mut upper = Vec::new();
    upper
        .try_reserve_exact(len)
        .map_err(|_| CholeskyError::Allocation)?;
    upper.resize(len, Complex64::new(0.0, 0.0));
    let mut info = 0;
    'columns: for j in 0..order {
        for i in 0..j {
            let a = entry(i, j);
            let b = entry(j, i);
            let scale = a.norm().max(b.norm()).max(1.0);
            let hermitian = (a - b.conj()).norm() <= EPS * scale;
            if !hermitian {
                info = j + 1;
                break 'columns;
            }
        }
        for i in 0..=j {
            let mut sum = entry(i, j);
            for k in 0..i {
                sum -= upper[k * order + i].conj() * upper[k * order + j];
            }
            if i == j {
                if !sum.re.is_finite()
                    || !sum.im.is_finite()
                    || sum.re <= 0.0
                    || sum.im.abs() > EPS * sum.re.abs().max(1.0)
                {
                    info = j + 1;
                    break 'columns;
                }
                upper[i * order + i] = Complex64::new(sum.re.sqrt(), 0.0);
            } else {
                let denominator = upper[i * order + i];
                // Earlier diagonal steps already proved this real pivot positive.
                // An absolute tolerance here rejects valid small-scale matrices.
                upper[i * order + j] = sum / denominator;
            }
        }
    }
    if info != 0 {
        for row in info.saturating_sub(1).min(order)..order {
            for col in row..order {
                upper[row * order + col] = Complex64::new(0.0, 0.0);
            }
        }
    }
    Ok(CholeskyFactor { upper, info })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_never_reads_input() {
        let factor = factor_upper(0, |_, _| panic!("empty matrix read")).unwrap();
        assert_eq!(factor.info, 0);
        assert!(factor.upper.is_empty());
    }

    #[test]
    fn real_positive_definite_factor() {
        let data = [4.0, 2.0, 2.0, 5.0];
        let factor = factor_upper(2, |r, c| Complex64::new(data[r * 2 + c], 0.0)).unwrap();
        assert_eq!(factor.info, 0);
        assert_eq!(
            factor.upper,
            vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 0.0),
            ]
        );
    }

    #[test]
    fn complex_hermitian_factor() {
        let data = [
            Complex64::new(4.0, 0.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(0.0, -2.0),
            Complex64::new(5.0, 0.0),
        ];
        let factor = factor_upper(2, |r, c| data[r * 2 + c]).unwrap();
        assert_eq!(factor.info, 0);
        assert_eq!(factor.upper[1], Complex64::new(0.0, 1.0));
        assert_eq!(factor.upper[3], Complex64::new(2.0, 0.0));
    }

    #[test]
    fn failed_minor_keeps_existing_partial_factor() {
        let data = [4.0, 2.0, 2.0, 0.0];
        let factor = factor_upper(2, |r, c| Complex64::new(data[r * 2 + c], 0.0)).unwrap();
        assert_eq!(factor.info, 2);
        assert_eq!(factor.upper[0], Complex64::new(2.0, 0.0));
        assert_eq!(factor.upper[3], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn nonfinite_and_nonhermitian_inputs_fail() {
        for data in [[1.0, 0.0, 1.0, 1.0], [1.0, f64::NAN, f64::NAN, 1.0]] {
            assert_ne!(
                factor_upper(2, |r, c| Complex64::new(data[r * 2 + c], 0.0))
                    .unwrap()
                    .info,
                0
            );
        }
    }

    #[test]
    fn overflow_is_rejected_before_access() {
        assert!(matches!(
            factor_upper(usize::MAX, |_, _| panic!("overflow input read")),
            Err(CholeskyError::SizeOverflow)
        ));
    }

    #[test]
    fn positive_diagonal_matrices_are_scale_independent() {
        for scale in [1.0e-26, 1.0, 1.0e26] {
            let factor = factor_upper(2, |r, c| {
                Complex64::new(if r == c { scale } else { 0.0 }, 0.0)
            })
            .unwrap();
            assert_eq!(factor.info, 0);
            assert_eq!(factor.upper[0].re, scale.sqrt());
            assert_eq!(factor.upper[3].re, scale.sqrt());
        }
    }
}
