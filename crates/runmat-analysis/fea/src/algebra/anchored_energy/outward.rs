//! Enclosures for exact real operations on finite binary64 inputs.
#[derive(Clone, Copy, Debug)]
pub(super) struct Interval {
    pub lo: f64,
    pub hi: f64,
}
impl Interval {
    pub fn exact(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }
    pub fn add(self, other: Self) -> Self {
        Self {
            lo: (self.lo + other.lo).next_down(),
            hi: (self.hi + other.hi).next_up(),
        }
    }
    pub fn sub(self, other: Self) -> Self {
        Self {
            lo: (self.lo - other.hi).next_down(),
            hi: (self.hi - other.lo).next_up(),
        }
    }
    pub fn mul(self, other: Self) -> Self {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        if products.iter().any(|v| v.is_nan()) {
            return Self {
                lo: f64::NEG_INFINITY,
                hi: f64::INFINITY,
            };
        }
        Self {
            lo: products
                .into_iter()
                .fold(f64::INFINITY, f64::min)
                .next_down(),
            hi: products
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max)
                .next_up(),
        }
    }
    pub fn absolute_upper(self) -> f64 {
        if self.lo.is_finite() && self.hi.is_finite() {
            self.lo.abs().max(self.hi.abs())
        } else {
            f64::INFINITY
        }
    }
}
