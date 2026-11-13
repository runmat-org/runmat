use runmat_value::ComplexElement;

use super::super::linear_operator::scalar::LinearOperatorScalar;

/// Substitution-only arithmetic on the same four native host scalar classes
/// as the shared dense/CSC product. No Value conversion or independent dtype
/// dispatch is introduced here.
pub(in crate::builtins::math::linalg) trait TriangularScalar:
    LinearOperatorScalar + PartialEq
{
    fn one() -> Self;
    fn is_zero(self) -> bool;
    fn is_finite(self) -> bool;
    fn subtract_product(accumulator: &mut Self, coefficient: Self, input: Self);
    fn divide(self, diagonal: Self) -> Option<Self>;
    fn divide_ieee(self, diagonal: Self) -> Self;
    fn from_ieee_parts(real: f64, imag: f64) -> Self;
}

macro_rules! real_scalar {
    ($scalar:ty) => {
        impl TriangularScalar for $scalar {
            fn one() -> Self {
                1.0
            }

            fn is_zero(self) -> bool {
                self == 0.0
            }

            fn is_finite(self) -> bool {
                <$scalar>::is_finite(self)
            }

            fn subtract_product(accumulator: &mut Self, coefficient: Self, input: Self) {
                *accumulator = (f64::from(*accumulator) - f64::from(coefficient) * f64::from(input))
                    as $scalar;
            }

            fn divide(self, diagonal: Self) -> Option<Self> {
                <Self as LinearOperatorScalar>::divide_finite(self, diagonal)
            }
            fn divide_ieee(self, diagonal: Self) -> Self {
                self / diagonal
            }
            fn from_ieee_parts(real: f64, _imag: f64) -> Self {
                real as $scalar
            }
        }
    };
}

macro_rules! complex_scalar {
    ($scalar:ty) => {
        impl TriangularScalar for ComplexElement<$scalar> {
            fn one() -> Self {
                ComplexElement(1.0, 0.0)
            }

            fn is_zero(self) -> bool {
                self.0 == 0.0 && self.1 == 0.0
            }

            fn is_finite(self) -> bool {
                self.0.is_finite() && self.1.is_finite()
            }

            fn subtract_product(accumulator: &mut Self, coefficient: Self, input: Self) {
                let real = f64::from(accumulator.0)
                    - (f64::from(coefficient.0) * f64::from(input.0)
                        - f64::from(coefficient.1) * f64::from(input.1));
                let imag = f64::from(accumulator.1)
                    - (f64::from(coefficient.0) * f64::from(input.1)
                        + f64::from(coefficient.1) * f64::from(input.0));
                *accumulator = ComplexElement(real as $scalar, imag as $scalar);
            }

            fn divide(self, diagonal: Self) -> Option<Self> {
                <Self as LinearOperatorScalar>::divide_finite(self, diagonal)
            }
            fn divide_ieee(self, diagonal: Self) -> Self {
                let (re, im) = runmat_value::divide_complex_f64(
                    (f64::from(self.0), f64::from(self.1)),
                    (f64::from(diagonal.0), f64::from(diagonal.1)),
                );
                ComplexElement(re as $scalar, im as $scalar)
            }
            fn from_ieee_parts(real: f64, imag: f64) -> Self {
                ComplexElement(real as $scalar, imag as $scalar)
            }
        }
    };
}

real_scalar!(f32);
real_scalar!(f64);
complex_scalar!(f32);
complex_scalar!(f64);
