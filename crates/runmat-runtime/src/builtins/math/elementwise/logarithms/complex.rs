use super::operation::LogarithmOperation;

pub(super) fn evaluate_f64(operation: LogarithmOperation, real: f64, imaginary: f64) -> (f64, f64) {
    evaluate(operation, real, imaginary)
}

pub(super) fn evaluate_f32(operation: LogarithmOperation, real: f32, imaginary: f32) -> (f32, f32) {
    evaluate(operation, real, imaginary)
}

fn evaluate<T: LogFloat>(operation: LogarithmOperation, real: T, imaginary: T) -> (T, T) {
    match operation {
        LogarithmOperation::OnePlus => one_plus(real, imaginary),
        LogarithmOperation::Natural | LogarithmOperation::Binary | LogarithmOperation::Common => {
            let (real, imaginary) = natural(real, imaginary);
            let scale = T::scale(operation);
            (real * scale, imaginary * scale)
        }
    }
}

fn one_plus<T: LogFloat>(real: T, imaginary: T) -> (T, T) {
    let shifted_real = real + T::ONE;
    let magnitude = shifted_real.hypot(imaginary);
    if magnitude == T::ZERO {
        return (T::NEG_INFINITY, T::ZERO);
    }
    let real_part = if real.abs() < T::HALF && imaginary.abs() < T::HALF {
        (T::TWO * real + real * real + imaginary * imaginary).ln_1p() * T::HALF
    } else {
        log_magnitude(shifted_real, imaginary)
    };
    (real_part, imaginary.atan2(shifted_real))
}

fn natural<T: LogFloat>(real: T, imaginary: T) -> (T, T) {
    let magnitude = real.hypot(imaginary);
    if magnitude == T::ZERO {
        (T::NEG_INFINITY, T::ZERO)
    } else {
        (log_magnitude(real, imaginary), imaginary.atan2(real))
    }
}

fn log_magnitude<T: LogFloat>(real: T, imaginary: T) -> T {
    let x = real.abs();
    let y = imaginary.abs();
    // Retain hypot's infinity/NaN semantics, without materializing an
    // overflowing finite magnitude before taking its logarithm.
    if !x.is_finite() || !y.is_finite() {
        return x.hypot(y).ln();
    }
    let (major, minor) = if x >= y { (x, y) } else { (y, x) };
    if major == T::ZERO {
        return T::NEG_INFINITY;
    }
    let ratio = minor / major;
    major.ln() + (ratio * ratio).ln_1p() * T::HALF
}

trait LogFloat:
    Copy
    + PartialEq
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const HALF: Self;
    const NEG_INFINITY: Self;
    fn abs(self) -> Self;
    fn is_finite(self) -> bool;
    fn hypot(self, other: Self) -> Self;
    fn ln(self) -> Self;
    fn ln_1p(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn scale(operation: LogarithmOperation) -> Self;
}

macro_rules! impl_log_float {
    ($type:ty, $log2_e:expr, $log10_e:expr) => {
        impl LogFloat for $type {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const TWO: Self = 2.0;
            const HALF: Self = 0.5;
            const NEG_INFINITY: Self = Self::NEG_INFINITY;
            fn abs(self) -> Self {
                self.abs()
            }
            fn is_finite(self) -> bool {
                self.is_finite()
            }
            fn hypot(self, other: Self) -> Self {
                self.hypot(other)
            }
            fn ln(self) -> Self {
                self.ln()
            }
            fn ln_1p(self) -> Self {
                self.ln_1p()
            }
            fn atan2(self, other: Self) -> Self {
                self.atan2(other)
            }
            fn scale(operation: LogarithmOperation) -> Self {
                match operation {
                    LogarithmOperation::Natural => 1.0,
                    LogarithmOperation::Binary => $log2_e,
                    LogarithmOperation::Common => $log10_e,
                    LogarithmOperation::OnePlus => unreachable!(),
                }
            }
        }
    };
}

impl_log_float!(f32, std::f32::consts::LOG2_E, std::f32::consts::LOG10_E);
impl_log_float!(f64, std::f64::consts::LOG2_E, std::f64::consts::LOG10_E);
