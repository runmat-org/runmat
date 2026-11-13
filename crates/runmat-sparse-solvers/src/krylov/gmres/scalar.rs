//! Native vector arithmetic for the one GMRES recurrence.

use super::{Arithmetic, GmresOperations};
use crate::krylov::usage::{BatchCompletion, ScalarWork, VectorWork};
use num_complex::Complex64;

pub(super) async fn copy<T: Copy, O: GmresOperations<T>>(
    source: &[T],
    target: &mut [T],
    operations: &mut O,
) -> Result<(), O::Error> {
    for (source, target) in source.chunks(1024).zip(target.chunks_mut(1024)) {
        operations.cooperate(source.len() as u64).await?;
        target.copy_from_slice(source);
    }
    Ok(())
}

pub(super) async fn residual<T: Copy + PartialEq, A: Arithmetic<T>, O: GmresOperations<T>>(
    rhs: &[T],
    product: &[T],
    destination: &mut [T],
    operations: &mut O,
) -> Result<bool, O::Error> {
    if rhs.len() != product.len() || rhs.len() != destination.len() {
        return Ok(false);
    }
    let native = operations
        .measured(
            VectorWork::Residual {
                elements: rhs.len(),
                domain: A::DOMAIN,
            },
            |operations| {
                let result = operations.native_residual(rhs, product, destination);
                (
                    result,
                    if result.is_some() {
                        BatchCompletion::Complete
                    } else {
                        BatchCompletion::Skipped
                    },
                )
            },
        )
        .await?;
    if let Some(valid) = native {
        operations.cooperate(rhs.len() as u64).await?;
        return Ok(valid);
    }
    for (start, chunk) in destination.chunks_mut(1024).enumerate() {
        let offset = start * 1024;
        operations.cooperate(chunk.len() as u64).await?;
        let valid = operations
            .measured(
                VectorWork::Residual {
                    elements: chunk.len(),
                    domain: A::DOMAIN,
                },
                |_| {
                    for (index, target) in chunk.iter_mut().enumerate() {
                        let Some(value) = A::from_complex(
                            A::to_complex(rhs[offset + index])
                                - A::to_complex(product[offset + index]),
                        ) else {
                            let per = if A::DOMAIN == crate::krylov::usage::ArithmeticDomain::Real {
                                1
                            } else {
                                2
                            };
                            return (
                                false,
                                BatchCompletion::Prefix(
                                    runmat_simulation_usage::OperationCount::from_operations(
                                        index as u64 * per,
                                    ),
                                ),
                            );
                        };
                        *target = value;
                    }
                    (true, BatchCompletion::Complete)
                },
            )
            .await?;
        if !valid {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) async fn divide<T: Copy + PartialEq, A: Arithmetic<T>, O: GmresOperations<T>>(
    source: &[T],
    target: &mut [T],
    divisor: Norm,
    operations: &mut O,
) -> Result<bool, O::Error> {
    if source.len() != target.len() {
        return Ok(false);
    }
    if let Some(magnitude) = divisor.native_magnitude() {
        let native = operations
            .measured(
                VectorWork::Divide {
                    elements: source.len(),
                    domain: A::DOMAIN,
                },
                |operations| {
                    let applied = operations.native_divide(target, source, magnitude);
                    (
                        applied,
                        if applied {
                            BatchCompletion::Complete
                        } else {
                            BatchCompletion::Skipped
                        },
                    )
                },
            )
            .await?;
        if native {
            operations.cooperate(source.len() as u64).await?;
            return Ok(true);
        }
    }
    for (chunk_index, chunk) in target.chunks_mut(1024).enumerate() {
        let offset = chunk_index * 1024;
        operations.cooperate(chunk.len() as u64).await?;
        let valid = operations
            .measured(
                VectorWork::DivideScaled {
                    elements: chunk.len(),
                    domain: A::DOMAIN,
                },
                |_| {
                    for (index, target) in chunk.iter_mut().enumerate() {
                        let Some(value) = A::from_complex(
                            divisor.normalize(A::to_complex(source[offset + index])),
                        ) else {
                            let per = if A::DOMAIN == crate::krylov::usage::ArithmeticDomain::Real {
                                2
                            } else {
                                4
                            };
                            return (
                                false,
                                BatchCompletion::Prefix(
                                    runmat_simulation_usage::OperationCount::from_operations(
                                        index as u64 * per,
                                    ),
                                ),
                            );
                        };
                        *target = value;
                    }
                    (true, BatchCompletion::Complete)
                },
            )
            .await?;
        if !valid {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) async fn fill<T: Copy, O: GmresOperations<T>>(
    target: &mut [T],
    value: T,
    operations: &mut O,
) -> Result<(), O::Error> {
    for target in target.chunks_mut(1024) {
        operations.cooperate(target.len() as u64).await?;
        target.fill(value);
    }
    Ok(())
}

/// A finite vector's norm need not fit f64. Keep scale and the root of
/// scaled squares separate through ratios, normalization and corrections.
#[derive(Clone, Copy)]
pub(super) struct Norm {
    scale: f64,
    root: f64,
}
impl Norm {
    pub(super) fn one() -> Self {
        Self {
            scale: 1.0,
            root: 1.0,
        }
    }
    pub(super) fn infinity() -> Self {
        Self {
            scale: f64::INFINITY,
            root: 1.0,
        }
    }
    fn from_magnitude(magnitude: f64) -> Self {
        Self {
            scale: magnitude,
            root: 1.0,
        }
    }
    /// The plain magnitude of a norm produced by a native kernel.
    pub(super) fn native_magnitude(self) -> Option<f64> {
        (self.root == 1.0 && self.scale.is_finite() && self.scale > 0.0).then_some(self.scale)
    }
    fn is_one(self) -> bool {
        self.scale == 1.0 && self.root == 1.0
    }
    pub(super) fn is_zero(self) -> bool {
        self.scale == 0.0
    }
    pub(super) fn magnitude(self) -> f64 {
        self.scale * self.root
    }
    pub(super) fn relative(self, denominator: Self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }
        if !self.scale.is_finite() {
            return f64::INFINITY;
        }
        if denominator.is_zero() {
            return self.magnitude();
        }
        let (a, ae) = libm::frexp(self.scale);
        let (b, be) = libm::frexp(denominator.scale);
        let (fraction, shift) = libm::frexp((a / b) * (self.root / denominator.root));
        libm::ldexp(fraction, ae - be + shift)
    }
    pub(super) fn relative_work(self) -> ScalarWork {
        if self.is_zero() || !self.scale.is_finite() {
            ScalarWork::Zero
        } else {
            ScalarWork::NormScale
        }
    }
    pub(super) async fn measured_relative<T, O: GmresOperations<T>>(
        self,
        denominator: Self,
        operations: &mut O,
    ) -> Result<f64, O::Error> {
        operations
            .measured(VectorWork::Scalar(self.relative_work()), |_| {
                (self.relative(denominator), BatchCompletion::Complete)
            })
            .await
    }

    pub(super) async fn measured_magnitude<T, O: GmresOperations<T>>(
        self,
        operations: &mut O,
    ) -> Result<f64, O::Error> {
        operations
            .measured(VectorWork::Scalar(ScalarWork::NormScale), |_| {
                (self.magnitude(), BatchCompletion::Complete)
            })
            .await
    }

    pub(super) async fn measured_less_than<T, O: GmresOperations<T>>(
        self,
        other: Self,
        operations: &mut O,
    ) -> Result<bool, O::Error> {
        if other.is_zero() {
            return Ok(false);
        }
        if other.scale.is_infinite() && self.scale.is_finite() {
            return Ok(true);
        }
        Ok(self.measured_relative(other, operations).await? < 1.0)
    }
    /// `self · factor` for a finite non-negative factor, keeping the scale separate.
    pub(super) fn times(self, factor: f64) -> Self {
        if factor == 0.0 || self.is_zero() {
            return Self {
                scale: 0.0,
                root: 1.0,
            };
        }
        Self {
            scale: self.scale * factor,
            root: self.root,
        }
    }
    pub(super) fn normalize(self, value: Complex64) -> Complex64 {
        (value / self.scale) / self.root
    }
    fn product(self, left: f64, right: f64) -> f64 {
        // Carry the residual scale through multiplication too. In particular
        // tiny beta times a large projected coefficient must not overflow
        // before their exponents cancel.
        if left == 0.0 || right == 0.0 || self.is_zero() {
            return 0.0;
        }
        let (a, ae) = libm::frexp(left);
        let (b, be) = libm::frexp(right);
        let (s, se) = libm::frexp(self.scale);
        let (fraction, shift) = libm::frexp(a * b * s * self.root);
        libm::ldexp(fraction, ae + be + se + shift)
    }
    fn product_complex(self, left: Complex64, right: Complex64) -> Complex64 {
        Complex64::new(
            self.sum_products((left.re, right.re), (-left.im, right.im)),
            self.sum_products((left.re, right.im), (left.im, right.re)),
        )
    }

    fn sum_products(self, left: (f64, f64), right: (f64, f64)) -> f64 {
        if left.0 == 0.0 || left.1 == 0.0 {
            return self.product(right.0, right.1);
        }
        if right.0 == 0.0 || right.1 == 0.0 {
            return self.product(left.0, left.1);
        }
        // Sum the two cross-products before applying the residual scale.
        // Either scaled term can overflow although their difference fits.
        // Independent product exponents also preserve small components when
        // the complex operands themselves span the full native exponent range.
        let (a, ae) = libm::frexp(left.0);
        let (b, be) = libm::frexp(left.1);
        let (c, ce) = libm::frexp(right.0);
        let (d, de) = libm::frexp(right.1);
        let left_exponent = ae + be;
        let right_exponent = ce + de;
        let exponent = left_exponent.max(right_exponent);
        let sum = libm::ldexp(a * b, left_exponent - exponent)
            + libm::ldexp(c * d, right_exponent - exponent);
        let (scale, scale_exponent) = libm::frexp(self.scale);
        let (fraction, shift) = libm::frexp(sum * scale * self.root);
        libm::ldexp(fraction, exponent + scale_exponent + shift)
    }
}

pub(super) async fn norm<T: Copy + PartialEq, A: Arithmetic<T>, O: GmresOperations<T>>(
    values: &[T],
    operations: &mut O,
) -> Result<Option<Norm>, O::Error> {
    let native = operations
        .measured(
            VectorWork::Norm {
                elements: values.len(),
                domain: A::DOMAIN,
            },
            |operations| {
                let result = operations.native_norm(values);
                (
                    result,
                    if result.is_some() {
                        BatchCompletion::Complete
                    } else {
                        BatchCompletion::Skipped
                    },
                )
            },
        )
        .await?;
    if let Some(magnitude) = native {
        operations.cooperate(values.len() as u64).await?;
        return Ok(Some(Norm::from_magnitude(magnitude)));
    }
    if values.is_empty() {
        return Ok(Some(Norm {
            scale: 0.0,
            root: 1.0,
        }));
    }
    // LAPACK-style scaled sum of squares avoids intermediate overflow for
    // otherwise representable norms.
    let mut scale = 0.0_f64;
    let mut squares = 1.0_f64;
    for start in (0..values.len()).step_by(1024) {
        let end = (start + 1024).min(values.len());
        operations.cooperate((end - start) as u64).await?;
        let complete = operations
            .measured(
                if start == 0 {
                    VectorWork::NormSquares {
                        elements: end - start,
                        domain: A::DOMAIN,
                    }
                } else {
                    VectorWork::NormSquaresContinue {
                        elements: end - start,
                        domain: A::DOMAIN,
                    }
                },
                |_| {
                    for value in &values[start..end] {
                        let value = A::to_complex(*value);
                        for component in [value.re.abs(), value.im.abs()] {
                            if !component.is_finite() {
                                return (false, BatchCompletion::Skipped);
                            }
                            if component != 0.0 {
                                if scale < component {
                                    squares = 1.0 + squares * (scale / component).powi(2);
                                    scale = component;
                                } else {
                                    squares += (component / scale).powi(2);
                                }
                            }
                        }
                    }
                    (true, BatchCompletion::Complete)
                },
            )
            .await?;
        if !complete {
            return Ok(None);
        }
    }
    let root = operations
        .measured(VectorWork::NormRoot, |_| {
            (squares.sqrt(), BatchCompletion::Complete)
        })
        .await?;
    Ok(Some(Norm { scale, root }))
}

pub(super) async fn dot<T: Copy + PartialEq, A: Arithmetic<T>, O: GmresOperations<T>>(
    left: &[T],
    right: &[T],
    operations: &mut O,
) -> Result<Option<Complex64>, O::Error> {
    if left.len() != right.len() {
        return Ok(None);
    }
    let native = operations
        .measured(
            VectorWork::Dot {
                elements: left.len(),
                domain: A::DOMAIN,
            },
            |operations| {
                let result = operations.native_dot(left, right);
                (
                    result,
                    if result.is_some() {
                        BatchCompletion::Complete
                    } else {
                        BatchCompletion::Skipped
                    },
                )
            },
        )
        .await?;
    if let Some(product) = native {
        operations.cooperate(left.len() as u64).await?;
        return Ok(Some(product));
    }
    let mut sum = Complex64::new(0.0, 0.0);
    let mut compensation = Complex64::new(0.0, 0.0);
    for start in (0..left.len()).step_by(1024) {
        let end = (start + 1024).min(left.len());
        operations.cooperate((end - start) as u64).await?;
        let complete = operations
            .measured(
                if start == 0 {
                    VectorWork::Dot {
                        elements: end - start,
                        domain: A::DOMAIN,
                    }
                } else {
                    VectorWork::DotContinue {
                        elements: end - start,
                        domain: A::DOMAIN,
                    }
                },
                |_| {
                    for index in start..end {
                        let term = A::to_complex(left[index]).conj() * A::to_complex(right[index]);
                        if !term.re.is_finite() || !term.im.is_finite() {
                            return (false, BatchCompletion::Skipped);
                        }
                        let corrected = term - compensation;
                        let next = sum + corrected;
                        compensation = (next - sum) - corrected;
                        sum = next;
                    }
                    (true, BatchCompletion::Complete)
                },
            )
            .await?;
        if !complete {
            return Ok(None);
        }
    }
    Ok((sum.re.is_finite() && sum.im.is_finite()).then_some(sum))
}

pub(super) async fn axpy<T: Copy + PartialEq, A: Arithmetic<T>, O: GmresOperations<T>>(
    target: &mut [T],
    scale: Complex64,
    source: &[T],
    operations: &mut O,
) -> Result<bool, O::Error> {
    scaled_axpy::<T, A, O>(target, scale, source, Norm::one(), operations).await
}

pub(super) async fn scaled_axpy<T: Copy + PartialEq, A: Arithmetic<T>, O: GmresOperations<T>>(
    target: &mut [T],
    scale: Complex64,
    source: &[T],
    norm: Norm,
    operations: &mut O,
) -> Result<bool, O::Error> {
    if target.len() != source.len() {
        return Ok(false);
    }
    if norm.is_one() {
        let native = operations
            .measured(
                VectorWork::Axpy {
                    elements: target.len(),
                    domain: A::DOMAIN,
                },
                |operations| {
                    let applied = operations.native_axpy(target, scale, source);
                    (
                        applied,
                        if applied {
                            BatchCompletion::Complete
                        } else {
                            BatchCompletion::Skipped
                        },
                    )
                },
            )
            .await?;
        if native {
            operations.cooperate(target.len() as u64).await?;
            return Ok(true);
        }
    }
    for (target, source) in target.chunks_mut(1024).zip(source.chunks(1024)) {
        operations.cooperate(target.len() as u64).await?;
        let complete = operations
            .measured(
                if norm.is_one() {
                    VectorWork::Axpy {
                        elements: target.len(),
                        domain: A::DOMAIN,
                    }
                } else {
                    VectorWork::AxpyScaled {
                        elements: target.len(),
                        domain: A::DOMAIN,
                    }
                },
                |_| {
                    for (index, (target, source)) in target.iter_mut().zip(source).enumerate() {
                        let Some(value) = A::from_complex(
                            A::to_complex(*target)
                                + norm.product_complex(scale, A::to_complex(*source)),
                        ) else {
                            let per_element =
                                if A::DOMAIN == crate::krylov::usage::ArithmeticDomain::Real {
                                    if norm.is_one() {
                                        2
                                    } else {
                                        3
                                    }
                                } else if norm.is_one() {
                                    8
                                } else {
                                    10
                                };
                            return (
                                false,
                                BatchCompletion::Prefix(
                                    runmat_simulation_usage::OperationCount::from_operations(
                                        index as u64 * per_element,
                                    ),
                                ),
                            );
                        };
                        *target = value;
                    }
                    (true, BatchCompletion::Complete)
                },
            )
            .await?;
        if !complete {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_complex_correction_combines_cross_products_before_rounding_range() {
        for (scale, left, right, expected) in [
            (
                2.0,
                Complex64::new(1e308, 3e307),
                Complex64::new(0.9, 0.3),
                Complex64::new(1.62e308, 1.14e308),
            ),
            (
                1e-308,
                Complex64::new(1e308, 1e-308),
                Complex64::new(1e-308, -1e308),
                Complex64::new(2e-308, -1e308),
            ),
        ] {
            let correction = Norm { scale, root: 1.0 }.product_complex(left, right);
            assert!(correction.re.is_finite() && correction.im.is_finite());
            assert!((correction.re / expected.re - 1.0).abs() <= 8.0 * f64::EPSILON);
            assert!((correction.im / expected.im - 1.0).abs() <= 8.0 * f64::EPSILON);
        }
    }
}
