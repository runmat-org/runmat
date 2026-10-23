use super::super::managed_lu::ManagedSparseScalar;

/// Only the four existing native sparse numerical classes. Norm reductions
/// use scaled f64 accumulators; numerical factors and outputs retain T.
pub(in crate::builtins::math::linalg) trait SparseQrScalar:
    ManagedSparseScalar
{
    fn conjugate(self) -> Self;
    fn real(value: f64) -> Self;
    fn epsilon() -> f64;
}
macro_rules! real {
    ($ty:ty) => {
        impl SparseQrScalar for $ty {
            fn conjugate(self) -> Self {
                self
            }
            fn real(value: f64) -> Self {
                value as Self
            }
            fn epsilon() -> f64 {
                <$ty>::EPSILON as f64
            }
        }
    };
}
real!(f32);
real!(f64);
macro_rules! complex {
    ($ty:ty,$real:ty) => {
        impl SparseQrScalar for $ty {
            fn conjugate(self) -> Self {
                Self::new(self.re, -self.im)
            }
            fn real(value: f64) -> Self {
                Self::new(value as $real, 0.0)
            }
            fn epsilon() -> f64 {
                <$real>::EPSILON as f64
            }
        }
    };
}
complex!(faer::c32, f32);
complex!(faer::c64, f64);

macro_rules! canonical_complex {
    ($real:ty) => {
        impl SparseQrScalar for runmat_value::ComplexElement<$real> {
            fn conjugate(self) -> Self {
                Self(self.0, -self.1)
            }
            fn real(value: f64) -> Self {
                Self(value as $real, 0.0)
            }
            fn epsilon() -> f64 {
                <$real>::EPSILON as f64
            }
        }
    };
}
canonical_complex!(f32);
canonical_complex!(f64);
