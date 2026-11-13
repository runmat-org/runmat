//! Scale-invariant residual comparison without overflowing norms/products.
use num_complex::Complex64;
#[derive(Clone, Copy)]
pub(super) struct Norm {
    scale: f64,
    squares: f64,
    finite: bool,
}
impl Norm {
    pub fn new() -> Self {
        Self {
            scale: 0.0,
            squares: 1.0,
            finite: true,
        }
    }
    pub fn add(&mut self, value: Complex64) {
        for value in [value.re.abs(), value.im.abs()] {
            if !value.is_finite() {
                self.finite = false;
                continue;
            }
            if value != 0.0 {
                if value > self.scale {
                    self.squares = 1.0 + self.squares * (self.scale / value).powi(2);
                    self.scale = value;
                } else {
                    self.squares += (value / self.scale).powi(2);
                }
            }
        }
    }
    fn log(self) -> f64 {
        if self.scale == 0.0 {
            f64::NEG_INFINITY
        } else {
            self.scale.ln() + 0.5 * self.squares.ln()
        }
    }
}
pub(super) fn converged(
    residual: Norm,
    action: Norm,
    lambda: Complex64,
    vector: Norm,
    tolerance: f64,
) -> bool {
    let mut eigenvalue = Norm::new();
    eigenvalue.add(lambda);
    if !residual.finite || !action.finite || !eigenvalue.finite || !vector.finite {
        return false;
    }
    if residual.scale == 0.0 {
        return true;
    }
    let first = action.log();
    let second = eigenvalue.log() + vector.log();
    let largest = first.max(second);
    if largest == f64::NEG_INFINITY {
        return false;
    }
    let denominator = largest + ((first - largest).exp() + (second - largest).exp()).ln();
    residual.log() <= tolerance.ln() + denominator
}
