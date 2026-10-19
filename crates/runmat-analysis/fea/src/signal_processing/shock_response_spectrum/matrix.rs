use crate::execution::{AnalysisExecutionError, AnalysisExecutionResult};

#[derive(Clone, Copy, Debug)]
pub(super) struct Matrix2 {
    values: [[f64; 2]; 2],
}

impl Matrix2 {
    pub(super) const fn new(a00: f64, a01: f64, a10: f64, a11: f64) -> Self {
        Self {
            values: [[a00, a01], [a10, a11]],
        }
    }

    pub(super) const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 1.0)
    }

    pub(super) fn apply(self, vector: [f64; 2]) -> [f64; 2] {
        [
            self.values[0][0] * vector[0] + self.values[0][1] * vector[1],
            self.values[1][0] * vector[0] + self.values[1][1] * vector[1],
        ]
    }

    pub(super) fn inverse(self) -> AnalysisExecutionResult<Self> {
        let determinant =
            self.values[0][0] * self.values[1][1] - self.values[0][1] * self.values[1][0];
        if !determinant.is_finite() || determinant.abs() <= f64::MIN_POSITIVE {
            return Err(invalid("SRS oscillator transition matrix is singular"));
        }
        Ok(Self::new(
            self.values[1][1] / determinant,
            -self.values[0][1] / determinant,
            -self.values[1][0] / determinant,
            self.values[0][0] / determinant,
        ))
    }

    pub(super) fn exp(self) -> AnalysisExecutionResult<Self> {
        let norm = self
            .values
            .iter()
            .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let squarings = if norm <= 0.5 {
            0
        } else {
            (norm / 0.5).log2().ceil() as u32
        };
        if squarings > 60 {
            return Err(invalid("SRS oscillator interval is numerically unbounded"));
        }
        let scale = 2.0_f64.powi(-(squarings as i32));
        let scaled = self * scale;
        let mut result = Self::identity();
        let mut term = Self::identity();
        for order in 1..=40 {
            term = term * scaled * (1.0 / order as f64);
            result = result + term;
            if term.max_abs() <= 4.0 * f64::EPSILON * result.max_abs().max(1.0) {
                break;
            }
        }
        for _ in 0..squarings {
            result = result * result;
        }
        if result
            .values
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(invalid("SRS oscillator transition is non-finite"));
        }
        Ok(result)
    }

    fn max_abs(self) -> f64 {
        self.values
            .iter()
            .flatten()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
    }
}

impl std::ops::Add for Matrix2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.values[0][0] + rhs.values[0][0],
            self.values[0][1] + rhs.values[0][1],
            self.values[1][0] + rhs.values[1][0],
            self.values[1][1] + rhs.values[1][1],
        )
    }
}

impl std::ops::Sub for Matrix2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.values[0][0] - rhs.values[0][0],
            self.values[0][1] - rhs.values[0][1],
            self.values[1][0] - rhs.values[1][0],
            self.values[1][1] - rhs.values[1][1],
        )
    }
}

impl std::ops::Mul for Matrix2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.values[0][0] * rhs.values[0][0] + self.values[0][1] * rhs.values[1][0],
            self.values[0][0] * rhs.values[0][1] + self.values[0][1] * rhs.values[1][1],
            self.values[1][0] * rhs.values[0][0] + self.values[1][1] * rhs.values[1][0],
            self.values[1][0] * rhs.values[0][1] + self.values[1][1] * rhs.values[1][1],
        )
    }
}

impl std::ops::Mul<f64> for Matrix2 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(
            self.values[0][0] * rhs,
            self.values[0][1] * rhs,
            self.values[1][0] * rhs,
            self.values[1][1] * rhs,
        )
    }
}

fn invalid(detail: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(detail.into())
}
