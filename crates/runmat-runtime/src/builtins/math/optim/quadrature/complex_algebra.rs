//! Scaled complex estimates with real certified-error geometry.

use super::adaptive::QuadratureEvaluator;
use super::algebra::QuadratureArithmeticError;
use super::geometry::ScaledEstimate;
use super::open_algebra::{OpenResult, OpenRuleInputs, OpenSample};

#[derive(Clone, Copy, Debug)]
pub(crate) struct ScaledComplex {
    pub(crate) re: ScaledEstimate,
    pub(crate) im: ScaledEstimate,
}

impl ScaledComplex {
    pub(crate) fn new(re: f64, im: f64) -> Self {
        Self {
            re: ScaledEstimate::from_f64(re),
            im: ScaledEstimate::from_f64(im),
        }
    }

    pub(crate) fn add(self, other: Self) -> Self {
        Self {
            re: self.re.add(other.re),
            im: self.im.add(other.im),
        }
    }

    pub(crate) fn subtract(self, other: Self) -> Self {
        Self {
            re: self.re.subtract(other.re),
            im: self.im.subtract(other.im),
        }
    }

    pub(crate) fn multiply_real(self, factor: ScaledEstimate) -> Self {
        Self {
            re: self.re.multiply(factor),
            im: self.im.multiply(factor),
        }
    }

    pub(crate) fn multiply_complex(self, other: Self) -> Self {
        Self {
            re: self
                .re
                .multiply(other.re)
                .subtract(self.im.multiply(other.im)),
            im: self.re.multiply(other.im).add(self.im.multiply(other.re)),
        }
    }

    pub(crate) fn magnitude(self) -> ScaledEstimate {
        let re = self.re.abs();
        let im = self.im.abs();
        let largest = re.max(im);
        if largest.is_zero() {
            return largest;
        }
        let smallest = if re.abs_le(im) { re } else { im };
        let ratio = smallest.divide(largest).trace_f64();
        largest.multiply_f64((1.0 + ratio * ratio).sqrt())
    }

    pub(crate) fn to_pair(self) -> Option<(f64, f64)> {
        Some((self.re.to_f64()?, self.im.to_f64()?))
    }
}

impl OpenResult for ScaledComplex {
    type Error = ScaledEstimate;

    async fn add<E: QuadratureEvaluator>(
        &self,
        other: &Self,
        _evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError> {
        Ok((*self).add(*other))
    }

    async fn subtract<E: QuadratureEvaluator>(
        &self,
        other: &Self,
        _evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError> {
        Ok((*self).subtract(*other))
    }

    async fn tolerance_error<E: QuadratureEvaluator>(
        &self,
        absolute: f64,
        relative: f64,
        _evaluator: &mut E,
    ) -> Result<Self::Error, QuadratureArithmeticError> {
        Ok(ScaledEstimate::from_f64(absolute).max(self.magnitude().multiply_f64(relative)))
    }
}

impl OpenSample for ScaledComplex {
    type Result = ScaledComplex;
    type Error = ScaledEstimate;

    async fn weighted_sum<E: QuadratureEvaluator<Sample = Self>>(
        samples: &[Self],
        weights: &[ScaledEstimate],
        evaluator: &mut E,
    ) -> Result<Self::Result, QuadratureArithmeticError> {
        if samples.len() != weights.len() {
            return Err(QuadratureArithmeticError::InconsistentComponents);
        }
        let mut sum = ScaledComplex::new(0.0, 0.0);
        for (&sample, &weight) in samples.iter().zip(weights) {
            sum = sum.add(sample.multiply_real(weight));
        }
        evaluator
            .charge(samples.len() as u64)
            .await
            .map_err(QuadratureArithmeticError::execution)?;
        Ok(sum)
    }

    async fn embedded_error<E: QuadratureEvaluator<Sample = Self>>(
        inputs: OpenRuleInputs<'_, Self>,
        evaluator: &mut E,
    ) -> Result<Self::Error, QuadratureArithmeticError> {
        if inputs.samples.len() != inputs.base_weights.len()
            || inputs.samples.len() != inputs.jacobians.len()
        {
            return Err(QuadratureArithmeticError::InconsistentComponents);
        }
        let mean = ScaledComplex {
            re: inputs.kronrod.re.divide(inputs.width),
            im: inputs.kronrod.im.divide(inputs.width),
        };
        let mut resabs = ScaledEstimate::from_f64(0.0);
        let mut resasc = ScaledEstimate::from_f64(0.0);
        for ((&sample, &weight), &jacobian) in inputs
            .samples
            .iter()
            .zip(inputs.base_weights)
            .zip(inputs.jacobians)
        {
            let transformed = sample.multiply_real(jacobian);
            resabs = resabs.add(weight.multiply(transformed.magnitude()));
            resasc = resasc.add(weight.multiply(transformed.subtract(mean).magnitude()));
        }
        evaluator
            .charge(inputs.samples.len() as u64)
            .await
            .map_err(QuadratureArithmeticError::execution)?;
        Ok(super::error_estimate::adjusted_error(
            ScaledComplex::subtract(*inputs.kronrod, *inputs.gauss).magnitude(),
            resabs,
            resasc,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_complex_magnitude_and_products_avoid_intermediate_overflow() {
        let huge = ScaledComplex::new(f64::MAX, f64::MAX);
        assert!(huge.magnitude().to_f64().is_none());
        let zero = huge.multiply_complex(ScaledComplex::new(0.0, 0.0));
        assert_eq!(zero.to_pair(), Some((0.0, 0.0)));
        let orthogonal =
            ScaledComplex::new(1.0e200, 0.0).multiply_complex(ScaledComplex::new(0.0, 1.0e-200));
        assert_eq!(orthogonal.to_pair(), Some((0.0, 1.0)));
    }
}
