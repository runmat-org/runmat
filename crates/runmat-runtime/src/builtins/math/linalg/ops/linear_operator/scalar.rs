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

// Independently adapted from the scaling and guarded summation specified by
// LAPACK 3.12.1's DLADIV/DLADIV1/DLADIV2 routines (modified BSD), which credit
// Michael Baudin and Robert L. Smith's "A Robust Complex Division in Scilab".
// Retained copyright and redistribution terms: LAPACK-LICENSE.txt beside
// this module. The numerical algorithm does not change allocation policy.
// Scaling numerator and denominator independently avoids overflowing a finite
// quotient, while the split product preserves a small component that joint
// normalization would lose.
fn robust_complex_divide(
    numerator_real: f64,
    numerator_imag: f64,
    denominator_real: f64,
    denominator_imag: f64,
) -> Option<(f64, f64)> {
    if !numerator_real.is_finite()
        || !numerator_imag.is_finite()
        || !denominator_real.is_finite()
        || !denominator_imag.is_finite()
        || (denominator_real == 0.0 && denominator_imag == 0.0)
    {
        return None;
    }

    let numerator_scale = numerator_real.abs().max(numerator_imag.abs());
    let denominator_scale = denominator_real.abs().max(denominator_imag.abs());
    let overflow_threshold = f64::MAX * 0.5;
    let underflow_threshold = f64::MIN_POSITIVE * 2.0 / f64::EPSILON;
    let underflow_multiplier = 2.0 / (f64::EPSILON * f64::EPSILON);

    let mut a = numerator_real;
    let mut b = numerator_imag;
    let mut c = denominator_real;
    let mut d = denominator_imag;
    let mut result_scale = 1.0;

    if numerator_scale >= overflow_threshold {
        a *= 0.5;
        b *= 0.5;
        result_scale *= 2.0;
    }
    if denominator_scale >= overflow_threshold {
        c *= 0.5;
        d *= 0.5;
        result_scale *= 0.5;
    }
    if numerator_scale <= underflow_threshold {
        a *= underflow_multiplier;
        b *= underflow_multiplier;
        result_scale /= underflow_multiplier;
    }
    if denominator_scale <= underflow_threshold {
        c *= underflow_multiplier;
        d *= underflow_multiplier;
        result_scale *= underflow_multiplier;
    }

    fn divided_component(a: f64, b: f64, c: f64, d: f64, ratio: f64, t: f64) -> f64 {
        if ratio != 0.0 {
            let product = b * ratio;
            if product != 0.0 {
                (a + product) * t
            } else {
                a * t + (b * t) * ratio
            }
        } else {
            (a + d * (b / c)) * t
        }
    }

    let (real, imag) = if d.abs() <= c.abs() {
        let ratio = d / c;
        let reciprocal = 1.0 / (c + d * ratio);
        (
            divided_component(a, b, c, d, ratio, reciprocal),
            divided_component(b, -a, c, d, ratio, reciprocal),
        )
    } else {
        let ratio = c / d;
        let reciprocal = 1.0 / (d + c * ratio);
        (
            divided_component(b, a, d, c, ratio, reciprocal),
            -divided_component(a, -b, d, c, ratio, reciprocal),
        )
    };
    let real = real * result_scale;
    let imag = imag * result_scale;
    (real.is_finite() && imag.is_finite()).then_some((real, imag))
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
