//! Positive scalar products with a binary exponent kept outside f64 range.
//! CG needs ratios of squared norms and curvatures even when each dot product
//! itself exceeds f64 (or underflows it).

#[derive(Clone, Copy)]
pub(super) struct ScaledPositive {
    fraction: f64,
    exponent: i32,
}

impl ScaledPositive {
    pub(super) fn from_factors(sum: f64, left: f64, right: f64) -> Option<Self> {
        if !sum.is_finite() || !left.is_finite() || !right.is_finite() || sum < 0.0 {
            return None;
        }
        if sum == 0.0 || left == 0.0 || right == 0.0 {
            return Some(Self {
                fraction: 0.0,
                exponent: 0,
            });
        }
        let (sum_fraction, sum_exponent) = libm::frexp(sum);
        let (left_fraction, left_exponent) = libm::frexp(left);
        let (right_fraction, right_exponent) = libm::frexp(right);
        let (fraction, shift) = libm::frexp(sum_fraction * left_fraction * right_fraction);
        Some(Self {
            fraction,
            exponent: sum_exponent + left_exponent + right_exponent + shift,
        })
    }

    pub(super) fn is_zero(self) -> bool {
        self.fraction == 0.0
    }

    pub(super) fn less_than(self, other: Self) -> bool {
        if self.is_zero() || other.is_zero() {
            return self.is_zero() && !other.is_zero();
        }
        self.exponent < other.exponent
            || (self.exponent == other.exponent && self.fraction < other.fraction)
    }

    pub(super) fn ratio(self, denominator: Self) -> Option<f64> {
        if denominator.is_zero() {
            return None;
        }
        if self.is_zero() {
            return Some(0.0);
        }
        let (fraction, shift) = libm::frexp(self.fraction / denominator.fraction);
        let value = libm::ldexp(fraction, self.exponent - denominator.exponent + shift);
        value.is_finite().then_some(value)
    }

    /// The represented value as a plain number (may overflow to infinity).
    pub(super) fn value(self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }
        libm::ldexp(self.fraction, self.exponent)
    }

    pub(super) fn norm(self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }
        let odd = self.exponent.rem_euclid(2) != 0;
        let fraction = if odd {
            (self.fraction * 2.0).sqrt()
        } else {
            self.fraction.sqrt()
        };
        libm::ldexp(fraction, self.exponent.div_euclid(2))
    }

    pub(super) fn relative_norm(self, denominator: Self) -> f64 {
        if denominator.is_zero() {
            return self.norm();
        }
        if self.is_zero() {
            return 0.0;
        }
        let (fraction, shift) = libm::frexp(self.fraction / denominator.fraction);
        let ratio = Self {
            fraction,
            exponent: self.exponent - denominator.exponent + shift,
        };
        ratio.norm()
    }
}
