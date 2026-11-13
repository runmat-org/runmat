//! Normalize generalized outputs through the actual retained B action.
//! Indefinite/non-Hermitian forms retain a valid Euclidean normalization.
use super::{Operator, Scalar, Work};
use crate::BuiltinResult;
use num_complex::Complex64;
impl<T: Scalar> Operator<'_, T> {
    pub async fn normalize_vectors(
        &self,
        vectors: &mut [Complex64],
        native: &mut [T],
        complex: &mut [Complex64],
        work: &mut Work<'_>,
    ) -> BuiltinResult<()> {
        if !self.is_generalized() {
            return Ok(());
        }
        let n = self.b.as_ref().map_or_else(
            || match &self.inverse {
                super::Inverse::Cholesky(f) => f.order(),
                _ => 0,
            },
            |b| b.rows(),
        );
        let (input, tail) = native.split_at_mut(n);
        let (output, scratch) = tail.split_at_mut(n);
        let mass = &mut complex[..n];
        let epsilon = if T::DTYPE == runmat_value::NumericDType::F32 {
            f32::EPSILON as f64
        } else {
            f64::EPSILON
        };
        for vector in vectors.chunks_exact_mut(n) {
            for value in mass.iter_mut() {
                *value = Complex64::new(0.0, 0.0);
                work.tick().await?;
            }
            for imaginary in [false, true] {
                if imaginary && T::COMPLEX {
                    continue;
                }
                let phase = if imaginary {
                    Complex64::new(0.0, 1.0)
                } else {
                    Complex64::new(1.0, 0.0)
                };
                for (destination, &source) in input.iter_mut().zip(vector.iter()) {
                    *destination = T::native(if T::COMPLEX {
                        source
                    } else {
                        Complex64::new(if imaginary { source.im } else { source.re }, 0.0)
                    })
                    .ok_or_else(|| {
                        super::super::invalid("generalized Ritz vector exceeds native range")
                    })?;
                    work.tick().await?;
                }
                self.apply_mass(input, output, scratch, work).await?;
                for (destination, &source) in mass.iter_mut().zip(output.iter()) {
                    *destination += phase * source.complex();
                    work.tick().await?;
                }
            }
            use super::super::basis::metric::{norm_complex, InnerNorm};
            match norm_complex(vector, mass, epsilon, work).await? {
                InnerNorm::Positive(log) => {
                    let norm = log.exp();
                    for value in vector.iter_mut() {
                        *value /= norm;
                        if !value.re.is_finite() || !value.im.is_finite() {
                            return Err(super::super::invalid(
                                "generalized eigenvector normalization exceeds numeric range",
                            ));
                        }
                        work.tick().await?;
                    }
                }
                InnerNorm::Null | InnerNorm::NonPositive if self.uses_metric() => {
                    return Err(super::super::invalid(
                        "generalized eigenvector has no positive B-norm",
                    ))
                }
                InnerNorm::Null | InnerNorm::NonPositive => {}
            }
        }
        work.flush().await
    }
}
