//! Closed-domain sampling is geometry arithmetic, independent of kernel tolerance.
use super::{GeometryEvaluationError, GeometryEvaluationErrorKind, ParameterRange};

impl ParameterRange {
    /// Interpolate within a finite ordered domain, preserving its exact endpoints.
    /// Invalid fractions are rejected; evaluator domains are never expanded.
    pub fn parameter_at_fraction(self, fraction: f64) -> Result<f64, GeometryEvaluationError> {
        if !self.start.is_finite() || !self.end.is_finite() || self.start > self.end {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::InvalidResult,
                "parameter sampling requires finite ordered domain endpoints",
            ));
        }
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::ParameterOutsideDomain,
                "parameter sampling fraction must be within [0, 1]",
            ));
        }
        if fraction == 0.0 {
            return Ok(self.start);
        }
        if fraction == 1.0 {
            return Ok(self.end);
        }
        // Same-sign subtraction is finite. Opposite-sign weighted endpoints
        // avoid an overflowing span. Bound only the generated interpolant's
        // rounding; this does not clamp user-supplied evaluator parameters.
        let value = if self.start.is_sign_negative() == self.end.is_sign_negative() {
            self.start + (self.end - self.start) * fraction
        } else {
            self.start * (1.0 - fraction) + self.end * fraction
        };
        Ok(value.clamp(self.start, self.end))
    }

    /// Fixed-size uniform samples, including both authored endpoints.
    pub fn uniform_samples<const COUNT: usize>(
        self,
    ) -> Result<[f64; COUNT], GeometryEvaluationError> {
        if COUNT < 2 {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::InvalidResult,
                "closed-domain sampling requires at least two samples",
            ));
        }
        let mut values = [0.0; COUNT];
        for (index, value) in values.iter_mut().enumerate() {
            *value = self.parameter_at_fraction(index as f64 / COUNT.saturating_sub(1) as f64)?;
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests;
