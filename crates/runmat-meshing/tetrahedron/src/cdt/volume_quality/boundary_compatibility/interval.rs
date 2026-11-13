//! Outward arithmetic for a sufficient incompatibility proof. Overflow or a
//! denominator spanning zero leaves the proof indeterminate.
#[derive(Clone, Copy)]
pub(super) struct Interval {
    pub(super) lo: f64,
    pub(super) hi: f64,
}
impl Interval {
    pub(super) fn exact(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }
    fn bounds(lo: f64, hi: f64) -> Option<Self> {
        (lo.is_finite() && hi.is_finite() && lo <= hi).then_some(Self { lo, hi })
    }
    pub(super) fn add(self, other: Self) -> Option<Self> {
        Self::bounds(
            (self.lo + other.lo).next_down(),
            (self.hi + other.hi).next_up(),
        )
    }
    pub(super) fn sub(self, other: Self) -> Option<Self> {
        Self::bounds(
            (self.lo - other.hi).next_down(),
            (self.hi - other.lo).next_up(),
        )
    }
    pub(super) fn mul(self, other: Self) -> Option<Self> {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        if products.iter().any(|value| !value.is_finite()) {
            return None;
        }
        Self::bounds(
            products
                .into_iter()
                .fold(f64::INFINITY, f64::min)
                .next_down(),
            products
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max)
                .next_up(),
        )
    }
    pub(super) fn div(self, other: Self) -> Option<Self> {
        if other.lo <= 0. && other.hi >= 0. {
            return None;
        }
        self.mul(Self::bounds(
            (1. / other.hi).next_down(),
            (1. / other.lo).next_up(),
        )?)
    }
    pub(super) fn square(self) -> Option<Self> {
        let a = self.lo * self.lo;
        let b = self.hi * self.hi;
        Self::bounds(
            if self.lo <= 0. && self.hi >= 0. {
                0.
            } else {
                a.min(b).next_down().max(0.)
            },
            a.max(b).next_up(),
        )
    }
    pub(super) fn sqrt(self) -> Option<Self> {
        if self.hi < 0. {
            return None;
        }
        Self::bounds(
            self.lo.max(0.).sqrt().next_down().max(0.),
            self.hi.sqrt().next_up(),
        )
    }
    pub(super) fn abs(self) -> Self {
        if self.lo >= 0. {
            self
        } else if self.hi <= 0. {
            Self {
                lo: -self.hi,
                hi: -self.lo,
            }
        } else {
            Self {
                lo: 0.,
                hi: self.hi.max(-self.lo),
            }
        }
    }
}
