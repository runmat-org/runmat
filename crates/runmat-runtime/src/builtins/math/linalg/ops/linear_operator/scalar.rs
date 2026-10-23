use runmat_value::ComplexElement;

/// Native scalar arithmetic shared by dense and CSC matrix-vector products.
/// This is intentionally about the four supported host storage classes, not
/// an independent numeric class or Value-conversion authority.
pub(in crate::builtins::math::linalg) trait LinearOperatorScalar:
    Copy
{
    fn zero() -> Self;
    fn conjugate(self) -> Self;
    fn add_product(accumulator: &mut Self, coefficient: Self, input: Self);
    fn divide_finite(self, denominator: Self) -> Option<Self>;
}

macro_rules! real_scalar {
    ($scalar:ty) => {
        impl LinearOperatorScalar for $scalar {
            fn zero() -> Self {
                0.0
            }

            fn conjugate(self) -> Self {
                self
            }

            fn add_product(accumulator: &mut Self, coefficient: Self, input: Self) {
                *accumulator += coefficient * input;
            }

            fn divide_finite(self, denominator: Self) -> Option<Self> {
                let quotient = (f64::from(self) / f64::from(denominator)) as $scalar;
                quotient.is_finite().then_some(quotient)
            }
        }
    };
}

fn robust_complex_divide(a: f64, b: f64, c: f64, d: f64) -> Option<(f64, f64)> {
    super::super::complex_division::divide_finite_components(a, b, c, d)
        .filter(|(real, imag)| real.is_finite() && imag.is_finite())
}
macro_rules! complex_scalar {
    ($scalar:ty) => {
        impl LinearOperatorScalar for ComplexElement<$scalar> {
            fn zero() -> Self {
                ComplexElement(0.0, 0.0)
            }

            fn conjugate(self) -> Self {
                ComplexElement(self.0, -self.1)
            }

            fn add_product(accumulator: &mut Self, coefficient: Self, input: Self) {
                accumulator.0 += coefficient.0 * input.0 - coefficient.1 * input.1;
                accumulator.1 += coefficient.0 * input.1 + coefficient.1 * input.0;
            }

            fn divide_finite(self, denominator: Self) -> Option<Self> {
                let (real, imag) = robust_complex_divide(
                    f64::from(self.0),
                    f64::from(self.1),
                    f64::from(denominator.0),
                    f64::from(denominator.1),
                )?;
                let quotient = ComplexElement(real as $scalar, imag as $scalar);
                (quotient.0.is_finite() && quotient.1.is_finite()).then_some(quotient)
            }
        }
    };
}

real_scalar!(f32);
real_scalar!(f64);
complex_scalar!(f32);
complex_scalar!(f64);
