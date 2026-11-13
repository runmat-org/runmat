//! Independent original-pencil residual; callable products are not replayed.
use super::{BuiltinResult, Coefficient, Operator, Scalar, Work};
impl<T: Scalar> Operator<'_, T> {
    /// Verify the original matrix pencil, not merely the inverse action's
    /// Ritz residual. Callable inverse actions cannot manufacture an A view;
    /// their actual cached product transcript supplies the residual instead.
    pub async fn verify_matrix(
        &self,
        values: &[num_complex::Complex64],
        vectors: &[num_complex::Complex64],
        flags: &mut [usize],
        native: &mut [T],
        complex: &mut [num_complex::Complex64],
        tolerance: f64,
        work: &mut Work<'_>,
    ) -> BuiltinResult<()> {
        use num_complex::Complex64;
        let Coefficient::Matrix(a) = &self.coefficient else {
            return Ok(());
        };
        let n = a.rows();
        let (input, tail) = native.split_at_mut(n);
        let (output, scratch) = tail.split_at_mut(n);
        let (av, bv) = complex.split_at_mut(n);
        for (column, &lambda) in values.iter().enumerate() {
            for value in av.iter_mut().chain(bv.iter_mut()) {
                *value = Complex64::new(0.0, 0.0);
                work.tick().await?;
            }
            let vector = &vectors[column * n..(column + 1) * n];
            for imaginary in [false, true] {
                if imaginary && T::COMPLEX {
                    continue;
                }
                let phase = if imaginary {
                    Complex64::new(0.0, 1.0)
                } else {
                    Complex64::new(1.0, 0.0)
                };
                for (destination, &source) in input.iter_mut().zip(vector) {
                    let source = if T::COMPLEX {
                        source
                    } else {
                        Complex64::new(if imaginary { source.im } else { source.re }, 0.0)
                    };
                    *destination = T::native(source)
                        .ok_or_else(|| super::super::invalid("Ritz vector exceeds native range"))?;
                    work.tick().await?;
                }
                a.apply_converted(input, output, T::native, work.runtime)
                    .await
                    .map_err(|e| super::super::operator_error(work.runtime, e))?;
                for (target, &value) in av.iter_mut().zip(output.iter()) {
                    *target += phase * value.complex();
                    work.tick().await?;
                }
                if self.is_generalized() {
                    self.apply_mass(input, output, scratch, work).await?;
                    for (target, &value) in bv.iter_mut().zip(output.iter()) {
                        *target += phase * value.complex();
                        work.tick().await?;
                    }
                }
            }
            let mut anorm = super::super::residual_norm::Norm::new();
            let mut bnorm = anorm;
            let mut residual = anorm;
            for row in 0..n {
                let b = if self.is_generalized() {
                    bv[row]
                } else {
                    vector[row]
                };
                anorm.add(av[row]);
                bnorm.add(b);
                residual.add(av[row] - lambda * b);
                work.tick().await?;
            }
            let converged =
                super::super::residual_norm::converged(residual, anorm, lambda, bnorm, tolerance);
            flags[column] = usize::from(converged);
        }
        work.flush().await
    }
}
