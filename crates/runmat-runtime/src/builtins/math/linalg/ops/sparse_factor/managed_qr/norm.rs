//! Scaled sum of squares avoids squaring large/small native values.
use super::SparseQrScalar;
#[derive(Clone, Copy)]
pub(super) struct Norm {
    pub(super) scale: f64,
    sum: f64,
}
impl Norm {
    pub(super) fn zero() -> Self {
        Self {
            scale: 0.0,
            sum: 1.0,
        }
    }
    pub(super) fn add<T: SparseQrScalar>(&mut self, value: T) {
        let (r, i) = value.parts();
        for part in [r.abs(), i.abs()] {
            if part == 0.0 {
                continue;
            }
            if self.scale < part {
                self.sum = 1.0 + self.sum * (self.scale / part).powi(2);
                self.scale = part;
            } else {
                self.sum += (part / self.scale).powi(2);
            }
        }
    }
    pub(super) fn ratio(self, other: Self) -> f64 {
        if self.scale == 0.0 {
            0.0
        } else if other.scale == 0.0 {
            f64::INFINITY
        } else {
            (self.scale / other.scale) * (self.sum / other.sum).sqrt()
        }
    }
    pub(super) fn magnitude(self) -> f64 {
        self.scale * self.sum.sqrt()
    }
    /// Compare an absolute rank threshold without materializing an
    /// overflowing norm or squaring a tiny native component.
    pub(super) fn at_most(self, tolerance: f64) -> bool {
        self.scale == 0.0 || tolerance == f64::INFINITY || self.scale <= tolerance / self.sum.sqrt()
    }
    pub(super) fn unit<T: SparseQrScalar>(self, value: T) -> T {
        // Divide component-wise through the native scalar. The first divisor
        // is representable whenever the original class contains the input.
        value
            .divide(T::real(self.scale))
            .divide(T::real(self.sum.sqrt()))
    }
}
