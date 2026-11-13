//! Finite-interval geometry shared by the scalar quadrature traversals.
//!
//! An interval estimate can exceed binary64 even when the final integral is
//! representable after cancellation. Keep estimates scaled until publication.

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScaledEstimate {
    fraction: f64,
    exponent: i32,
}

impl ScaledEstimate {
    pub(crate) fn from_f64(value: f64) -> Self {
        let (fraction, exponent) = libm::frexp(value);
        Self { fraction, exponent }
    }

    fn from_parts(fraction: f64, exponent: i32) -> Self {
        if fraction == 0.0 {
            return Self::from_f64(0.0);
        }
        let (normalized, shift) = libm::frexp(fraction);
        Self {
            fraction: normalized,
            exponent: exponent + shift,
        }
    }

    pub(crate) fn multiply(self, other: Self) -> Self {
        Self::from_parts(
            self.fraction * other.fraction,
            self.exponent + other.exponent,
        )
    }

    pub(crate) fn multiply_f64(self, factor: f64) -> Self {
        self.multiply(Self::from_f64(factor))
    }

    pub(crate) fn divide(self, other: Self) -> Self {
        debug_assert!(other.fraction != 0.0);
        Self::from_parts(
            self.fraction / other.fraction,
            self.exponent - other.exponent,
        )
    }

    pub(crate) fn is_zero(self) -> bool {
        self.fraction == 0.0
    }

    pub(crate) fn add(self, other: Self) -> Self {
        if self.fraction == 0.0 {
            return other;
        }
        if other.fraction == 0.0 {
            return self;
        }
        let exponent = self.exponent.max(other.exponent);
        let left = libm::ldexp(self.fraction, self.exponent - exponent);
        let right = libm::ldexp(other.fraction, other.exponent - exponent);
        Self::from_parts(left + right, exponent)
    }

    pub(crate) fn subtract(self, other: Self) -> Self {
        self.add(Self {
            fraction: -other.fraction,
            exponent: other.exponent,
        })
    }

    pub(crate) fn abs(self) -> Self {
        Self {
            fraction: self.fraction.abs(),
            exponent: self.exponent,
        }
    }

    pub(crate) fn max(self, other: Self) -> Self {
        if self.abs_le(other) {
            other
        } else {
            self
        }
    }

    pub(crate) fn abs_le(self, other: Self) -> bool {
        let left = self.abs();
        let right = other.abs();
        if left.fraction == 0.0 {
            return true;
        }
        if right.fraction == 0.0 {
            return false;
        }
        left.exponent < right.exponent
            || (left.exponent == right.exponent && left.fraction <= right.fraction)
    }

    pub(crate) fn to_f64(self) -> Option<f64> {
        let value = libm::ldexp(self.fraction, self.exponent);
        value.is_finite().then_some(value)
    }

    pub(crate) fn trace_f64(self) -> f64 {
        libm::ldexp(self.fraction, self.exponent)
    }
}

pub(crate) fn midpoint(a: f64, b: f64) -> f64 {
    // Opposite signs make the sum safe; same signs make the difference safe.
    if (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0) {
        (a + b) * 0.5
    } else {
        a + (b - a) * 0.5
    }
}

pub(crate) fn simpson(a: f64, b: f64, fa: f64, fm: f64, fb: f64) -> ScaledEstimate {
    let width = b - a;
    let interval = if width.is_finite() {
        ScaledEstimate::from_f64(width)
    } else {
        // Only opposite-sign finite endpoints can overflow their difference.
        ScaledEstimate::from_f64(b * 0.5 - a * 0.5).multiply_f64(2.0)
    };
    let average = ScaledEstimate::from_f64(fa)
        .multiply_f64(1.0 / 6.0)
        .add(ScaledEstimate::from_f64(fb).multiply_f64(1.0 / 6.0))
        .add(ScaledEstimate::from_f64(fm).multiply_f64(2.0 / 3.0));
    interval.multiply(average)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_extreme_midpoints_do_not_overflow() {
        assert_eq!(midpoint(-f64::MAX, f64::MAX), 0.0);
        assert_eq!(midpoint(0.0, f64::MAX), f64::MAX * 0.5);
        assert_eq!(midpoint(-f64::MAX, 0.0), -f64::MAX * 0.5);
        assert_eq!(midpoint(0.0, f64::from_bits(1)), 0.0);
    }

    #[test]
    fn scaled_estimates_preserve_cancelling_nonrepresentable_intervals() {
        let huge = 1.0e308;
        let constant = simpson(0.0, huge, 1.0, 1.0, 1.0).to_f64().unwrap();
        assert!((constant - huge).abs() / huge < 1.0e-15);
        assert_eq!(
            simpson(-f64::MAX, f64::MAX, 0.0, 0.0, 0.0).to_f64(),
            Some(0.0)
        );
        let left = simpson(-f64::MAX, 0.0, -f64::MAX, -f64::MAX * 0.5, 0.0);
        let right = simpson(0.0, f64::MAX, 0.0, f64::MAX * 0.5, f64::MAX);
        assert!(left.to_f64().is_none());
        assert!(right.to_f64().is_none());
        assert_eq!(left.add(right).to_f64(), Some(0.0));
        assert!(simpson(-f64::MAX, f64::MAX, 1.0, 1.0, 1.0)
            .to_f64()
            .is_none());
        let tiny = f64::from_bits(1);
        assert!(simpson(-f64::MAX, f64::MAX, tiny, tiny, tiny)
            .to_f64()
            .is_some_and(|value| value > 0.0));
    }
}
