//! Native vector arithmetic for the one GMRES recurrence.

use super::{Arithmetic, GmresOperations};
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
    pub(super) fn less_than(self, other: Self) -> bool {
        if other.is_zero() {
            return false;
        }
        other.scale.is_infinite() && self.scale.is_finite() || self.relative(other) < 1.0
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
    if let Some(magnitude) = operations.native_norm(values) {
        operations.cooperate(values.len() as u64).await?;
        return Ok(Some(Norm::from_magnitude(magnitude)));
    }
    // LAPACK-style scaled sum of squares avoids intermediate overflow for
    // otherwise representable norms.
    let mut scale = 0.0_f64;
    let mut squares = 1.0_f64;
    for (index, value) in values.iter().enumerate() {
        if index % 1024 == 0 {
            operations
                .cooperate((values.len() - index).min(1024) as u64)
                .await?;
        }
        let value = A::to_complex(*value);
        for component in [value.re.abs(), value.im.abs()] {
            if !component.is_finite() {
                return Ok(None);
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
    Ok(Some(Norm {
        scale,
        root: squares.sqrt(),
    }))
}

pub(super) async fn dot<T: Copy + PartialEq, A: Arithmetic<T>, O: GmresOperations<T>>(
    left: &[T],
    right: &[T],
    operations: &mut O,
) -> Result<Option<Complex64>, O::Error> {
    if left.len() != right.len() {
        return Ok(None);
    }
    if let Some(product) = operations.native_dot(left, right) {
        operations.cooperate(left.len() as u64).await?;
        return Ok(Some(product));
    }
    let mut sum = Complex64::new(0.0, 0.0);
    let mut compensation = Complex64::new(0.0, 0.0);
    for (index, (&left_value, &right_value)) in left.iter().zip(right).enumerate() {
        if index % 1024 == 0 {
            operations
                .cooperate((left.len() - index).min(1024) as u64)
                .await?;
        }
        let term = A::to_complex(left_value).conj() * A::to_complex(right_value);
        if !term.re.is_finite() || !term.im.is_finite() {
            return Ok(None);
        }
        let corrected = term - compensation;
        let next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
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
    if norm.is_one() && operations.native_axpy(target, scale, source) {
        operations.cooperate(target.len() as u64).await?;
        return Ok(true);
    }
    let count = target.len();
    for (index, (target, source)) in target.iter_mut().zip(source).enumerate() {
        if index % 1024 == 0 {
            operations
                .cooperate((count - index).min(1024) as u64)
                .await?;
        }
        let Some(value) = A::from_complex(
            A::to_complex(*target) + norm.product_complex(scale, A::to_complex(*source)),
        ) else {
            return Ok(false);
        };
        *target = value;
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
