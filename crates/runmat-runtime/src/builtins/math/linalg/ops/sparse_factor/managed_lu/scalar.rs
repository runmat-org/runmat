use crate::context::host_allocation::HostTypedElement;
use runmat_value::SparseTensor;
use std::ops::{Add, Mul, Sub};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for faer::c32 {}
    impl Sealed for faer::c64 {}
    impl Sealed for runmat_value::ComplexElement<f32> {}
    impl Sealed for runmat_value::ComplexElement<f64> {}
}

pub(in crate::builtins::math::linalg) trait ManagedSparseScalar:
    sealed::Sealed
    + HostTypedElement
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
{
    fn matches(sparse: &SparseTensor) -> bool;
    fn read(sparse: &SparseTensor, index: usize) -> Self;
    fn matches_pencil(sparse: &SparseTensor) -> bool {
        Self::matches(sparse)
    }
    fn read_pencil(sparse: &SparseTensor, index: usize) -> Self {
        Self::read(sparse, index)
    }
    fn one() -> Self;
    fn parts(self) -> (f64, f64);
    fn from_parts(real: f64, imag: f64) -> Option<Self>;
    /// Borrowed coefficient of a complex-multiplied operator. The widened
    /// product is formed before native F32 narrowing, without making a new
    /// factor payload. Finite physical underflow follows native IEEE scalar
    /// multiplication; the selected solve owns its singularity policy.
    fn scaled_complex(self, scale: (f64, f64)) -> Option<Self> {
        if scale == (1.0, 0.0) {
            return self.finite().then_some(self);
        }
        if !scale.0.is_finite() || !scale.1.is_finite() {
            return None;
        }
        let (real, imag) = self.parts();
        let (target_real, target_imag) = finite_complex_product((real, imag), scale);
        Self::from_parts(target_real, target_imag)
    }
    fn divide(self, rhs: Self) -> Self;
    fn conjugated(self) -> Self;
    fn finite(self) -> bool {
        let (real, imag) = self.parts();
        real.is_finite() && imag.is_finite()
    }
    /// Compare magnitudes without overflowing a finite complex modulus.
    fn pivot_greater(self, rhs: Self) -> bool {
        let (ar, ai) = self.parts();
        let (br, bi) = rhs.parts();
        let scale = ar.abs().max(ai.abs()).max(br.abs()).max(bi.abs());
        scale != 0.0 && (ar / scale).hypot(ai / scale) > (br / scale).hypot(bi / scale)
    }
}

/// Products are normalized before either cross term can overflow. This is
/// specifically the borrowed effective-coefficient boundary, not a new
/// factorization or a widened coefficient payload.
fn finite_complex_product((ar, ai): (f64, f64), (br, bi): (f64, f64)) -> (f64, f64) {
    fn product(a: f64, b: f64) -> (f64, i32) {
        if a == 0.0 || b == 0.0 {
            return (a * b, 0);
        }
        let (am, ae) = libm::frexp(a);
        let (bm, be) = libm::frexp(b);
        (am * bm, ae + be)
    }
    fn sum((am, ae): (f64, i32), (bm, be): (f64, i32)) -> f64 {
        if am == 0.0 {
            return libm::scalbn(bm, be);
        }
        if bm == 0.0 {
            return libm::scalbn(am, ae);
        }
        let exponent = ae.max(be);
        let combined = libm::scalbn(am, ae - exponent) + libm::scalbn(bm, be - exponent);
        libm::scalbn(combined, exponent)
    }
    (
        sum(product(ar, br), {
            let (mantissa, exponent) = product(ai, bi);
            (-mantissa, exponent)
        }),
        sum(product(ar, bi), product(ai, br)),
    )
}

/// Split a nonzero finite complex operator scale equally across two
/// triangular stages. Normalizing before the root avoids overflowing the
/// modulus of a finite scale and retains subnormal scales.
fn complex_scale_root(scale: (f64, f64)) -> Option<(f64, f64)> {
    let (real, imag) = scale;
    let largest = real.abs().max(imag.abs());
    if !largest.is_finite() || largest == 0.0 {
        return None;
    }
    let (_, exponent) = libm::frexp(largest);
    let ar = libm::scalbn(real, -exponent);
    let ai = libm::scalbn(imag, -exponent);
    let modulus = ar.hypot(ai);
    let (root_real, root_imag) = if ar >= 0.0 {
        let root_real = ((modulus + ar) * 0.5).sqrt();
        (root_real, ai / (2.0 * root_real))
    } else {
        let sign = if ai.is_sign_negative() { -1.0 } else { 1.0 };
        let root_imag = sign * ((modulus - ar) * 0.5).sqrt();
        (ai / (2.0 * root_imag), root_imag)
    };
    let half = exponent.div_euclid(2);
    let odd = if exponent.rem_euclid(2) == 0 {
        1.0
    } else {
        std::f64::consts::SQRT_2
    };
    let root = (
        libm::scalbn(root_real * odd, half),
        libm::scalbn(root_imag * odd, half),
    );
    (root.0.is_finite() && root.1.is_finite()).then_some(root)
}

/// Cholesky's two borrowed triangular stages multiply to the one requested
/// operator scale. Real execution keeps a negative real multiplier real;
/// complex execution uses the principal root on both stages.
pub(in crate::builtins::math::linalg) fn cholesky_stage_scales(
    scale: (f64, f64),
    real_execution: bool,
) -> Option<((f64, f64), (f64, f64))> {
    if !scale.0.is_finite() || !scale.1.is_finite() || scale == (0.0, 0.0) {
        return None;
    }
    if real_execution && scale.1 == 0.0 && scale.0 < 0.0 {
        let magnitude = (-scale.0).sqrt();
        return Some(((magnitude, 0.0), (-magnitude, 0.0)));
    }
    let root = complex_scale_root(scale)?;
    let reconstructed = finite_complex_product(root, root);
    if (scale.0 != 0.0 && reconstructed.0 == 0.0) || (scale.1 != 0.0 && reconstructed.1 == 0.0) {
        // A rounded square root may erase a much smaller represented phase
        // component. Put the full selected scalar on the first borrowed
        // factor in that case; never reject an otherwise legal IEEE scale.
        return Some((scale, (1.0, 0.0)));
    }
    Some((root, root))
}

macro_rules! real {
    ($ty:ty, $access:ident) => {
        impl ManagedSparseScalar for $ty {
            fn matches(s: &SparseTensor) -> bool {
                s.$access().is_some()
            }
            fn read(s: &SparseTensor, i: usize) -> Self {
                s.$access().expect("validated native class")[i]
            }
            fn matches_pencil(s: &SparseTensor) -> bool {
                s.as_f32_slice().is_some() || s.as_f64_slice().is_some()
            }
            fn read_pencil(s: &SparseTensor, i: usize) -> Self {
                if let Some(values) = s.as_f32_slice() {
                    values[i] as $ty
                } else {
                    s.as_f64_slice().expect("validated real pencil class")[i] as $ty
                }
            }
            fn one() -> Self {
                1.0
            }
            fn parts(self) -> (f64, f64) {
                (self as f64, 0.0)
            }
            fn from_parts(real: f64, imag: f64) -> Option<Self> {
                let value = real as $ty;
                (imag == 0.0 && value.is_finite()).then_some(value)
            }
            fn divide(self, rhs: Self) -> Self {
                self / rhs
            }
            fn conjugated(self) -> Self {
                self
            }
        }
    };
}
real!(f32, as_f32_slice);
real!(f64, as_f64_slice);
macro_rules! complex {
    ($ty:ty, $access:ident, $divide:ident) => {
        impl ManagedSparseScalar for $ty {
            fn matches(s: &SparseTensor) -> bool {
                s.$access().is_some()
            }
            fn read(s: &SparseTensor, i: usize) -> Self {
                let v = s.$access().expect("validated native class")[i];
                Self::new(v.0, v.1)
            }
            fn matches_pencil(s: &SparseTensor) -> bool {
                s.as_complex_f32_slice().is_some()
                    || s.as_complex_f64_slice().is_some()
                    || s.as_f32_slice().is_some()
                    || s.as_f64_slice().is_some()
            }
            fn read_pencil(s: &SparseTensor, i: usize) -> Self {
                if let Some(values) = s.as_complex_f32_slice() {
                    Self::new(values[i].0 as _, values[i].1 as _)
                } else if let Some(values) = s.as_complex_f64_slice() {
                    Self::new(values[i].0 as _, values[i].1 as _)
                } else if let Some(values) = s.as_f32_slice() {
                    Self::new(values[i] as _, 0.0)
                } else {
                    Self::new(
                        s.as_f64_slice().expect("validated floating pencil class")[i] as _,
                        0.0,
                    )
                }
            }
            fn one() -> Self {
                Self::new(1.0, 0.0)
            }
            fn parts(self) -> (f64, f64) {
                (self.re as f64, self.im as f64)
            }
            fn from_parts(real: f64, imag: f64) -> Option<Self> {
                let value = Self::new(real as _, imag as _);
                value.finite().then_some(value)
            }
            fn divide(self, rhs: Self) -> Self {
                let (real, imag) = runmat_value::$divide((self.re, self.im), (rhs.re, rhs.im));
                Self::new(real, imag)
            }
            fn conjugated(self) -> Self {
                Self::new(self.re, -self.im)
            }
        }
    };
}
complex!(faer::c32, as_complex_f32_slice, divide_complex_f32);
complex!(faer::c64, as_complex_f64_slice, divide_complex_f64);

// Retained factor DATA uses the canonical host element from the beginning:
// its owner can move into a HostComplexBuffer without reinterpretation.
macro_rules! canonical_complex {
    ($real:ty, $access:ident, $divide:ident) => {
        impl ManagedSparseScalar for runmat_value::ComplexElement<$real> {
            fn matches(s: &SparseTensor) -> bool {
                s.$access().is_some()
            }
            fn read(s: &SparseTensor, i: usize) -> Self {
                s.$access().expect("validated native class")[i]
            }
            fn one() -> Self {
                Self(1.0, 0.0)
            }
            fn parts(self) -> (f64, f64) {
                (self.0 as f64, self.1 as f64)
            }
            fn from_parts(real: f64, imag: f64) -> Option<Self> {
                let value = Self(real as $real, imag as $real);
                value.finite().then_some(value)
            }
            fn divide(self, rhs: Self) -> Self {
                let (real, imag) = runmat_value::$divide((self.0, self.1), (rhs.0, rhs.1));
                Self(real, imag)
            }
            fn conjugated(self) -> Self {
                Self(self.0, -self.1)
            }
        }
    };
}
canonical_complex!(f32, as_complex_f32_slice, divide_complex_f32);
canonical_complex!(f64, as_complex_f64_slice, divide_complex_f64);

#[cfg(test)]
mod scaled_coefficient_tests {
    use super::{cholesky_stage_scales, complex_scale_root, ManagedSparseScalar};
    use runmat_value::ComplexElement;

    #[test]
    fn complex_product_keeps_finite_cross_term_cancellation() {
        let first = ComplexElement(1.0e308_f64, 5.0e307)
            .scaled_complex((1.8, 0.4))
            .expect("finite coefficient");
        assert!((first.0 / 1.6e308 - 1.0).abs() < 1.0e-15);
        assert!((first.1 / 1.3e308 - 1.0).abs() < 1.0e-15);
        let second = ComplexElement(5.0e307_f64, 1.0e308)
            .scaled_complex((0.4, 1.8))
            .expect("finite coefficient");
        assert!((second.0 / -1.6e308 - 1.0).abs() < 1.0e-15);
        assert!((second.1 / 1.3e308 - 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn native_single_coefficient_multiplies_before_narrowing() {
        let coefficient = (1.0e30_f32)
            .scaled_complex((1.0e-40, 0.0))
            .expect("physical single coefficient");
        assert!((f64::from(coefficient) / 1.0e-10 - 1.0).abs() < 1.0e-6);
        assert_eq!((1.0e-30_f32).scaled_complex((1.0e-30, 0.0)), Some(0.0));
        assert_eq!((1.0e-308_f64).scaled_complex((1.0e-308, 0.0)), Some(0.0));
    }

    #[test]
    fn cholesky_root_splits_extreme_and_complex_scales_without_modulus_overflow() {
        for scale in [
            (1.0e308, 0.0),
            (1.0e-309, 0.0),
            (1.0, 1.0),
            (-1.0, 0.0),
            (1.0e308, 1.0e308),
        ] {
            let root = complex_scale_root(scale).expect("finite nonzero root");
            let square = ComplexElement(root.0, root.1)
                .scaled_complex(root)
                .expect("finite root square");
            let real_error = (square.0 - scale.0).abs() / scale.0.abs().max(1.0e-309);
            let imag_error = (square.1 - scale.1).abs() / scale.1.abs().max(1.0e-309);
            assert!(real_error < 1.0e-14, "{scale:?}: {root:?}: {square:?}");
            assert!(imag_error < 1.0e-14, "{scale:?}: {root:?}: {square:?}");
        }
        assert!(complex_scale_root((0.0, 0.0)).is_none());
        assert!(complex_scale_root((f64::INFINITY, 0.0)).is_none());
        assert_eq!(
            cholesky_stage_scales((-1.0e308, 1.0e-300), false),
            Some(((-1.0e308, 1.0e-300), (1.0, 0.0)))
        );
        assert_eq!(
            cholesky_stage_scales((-1.0, 0.0), true),
            Some(((1.0, 0.0), (-1.0, 0.0)))
        );
    }
}
