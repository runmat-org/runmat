//! Componentwise complex open-rule samples and real absolute errors.

use std::sync::Arc;

use super::adaptive::QuadratureEvaluator;
use super::algebra::{ArrayEstimate, QuadratureArithmeticError, COMPONENT_CHUNK};
use super::complex_algebra::ScaledComplex;
use super::geometry::ScaledEstimate;
use super::open_algebra::{OpenResult, OpenRuleInputs, OpenSample};

#[derive(Clone)]
pub(crate) struct ComplexArrayEstimate(pub(crate) Arc<Vec<ScaledComplex>>);

impl ComplexArrayEstimate {
    async fn zip_map<E: QuadratureEvaluator>(
        &self,
        other: &Self,
        evaluator: &mut E,
        map: impl Fn(ScaledComplex, ScaledComplex) -> ScaledComplex,
    ) -> Result<Self, QuadratureArithmeticError> {
        if self.0.len() != other.0.len() {
            return Err(QuadratureArithmeticError::InconsistentComponents);
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(self.0.len())
            .map_err(QuadratureArithmeticError::Resource)?;
        for (index, (&left, &right)) in self.0.iter().zip(other.0.iter()).enumerate() {
            values.push(map(left, right));
            if (index + 1).is_multiple_of(COMPONENT_CHUNK) {
                evaluator
                    .charge(COMPONENT_CHUNK as u64)
                    .await
                    .map_err(QuadratureArithmeticError::execution)?;
            }
        }
        evaluator
            .charge((self.0.len() % COMPONENT_CHUNK) as u64)
            .await
            .map_err(QuadratureArithmeticError::execution)?;
        Ok(Self(Arc::new(values)))
    }
}

impl OpenResult for ComplexArrayEstimate {
    type Error = ArrayEstimate;

    async fn add<E: QuadratureEvaluator>(
        &self,
        other: &Self,
        evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError> {
        self.zip_map(other, evaluator, ScaledComplex::add).await
    }

    async fn subtract<E: QuadratureEvaluator>(
        &self,
        other: &Self,
        evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError> {
        self.zip_map(other, evaluator, ScaledComplex::subtract)
            .await
    }

    async fn tolerance_error<E: QuadratureEvaluator>(
        &self,
        absolute: f64,
        relative: f64,
        evaluator: &mut E,
    ) -> Result<Self::Error, QuadratureArithmeticError> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(self.0.len())
            .map_err(QuadratureArithmeticError::Resource)?;
        let absolute = ScaledEstimate::from_f64(absolute);
        for (index, &value) in self.0.iter().enumerate() {
            values.push(absolute.max(value.magnitude().multiply_f64(relative)));
            if (index + 1).is_multiple_of(COMPONENT_CHUNK) {
                evaluator
                    .charge(COMPONENT_CHUNK as u64)
                    .await
                    .map_err(QuadratureArithmeticError::execution)?;
            }
        }
        evaluator
            .charge((self.0.len() % COMPONENT_CHUNK) as u64)
            .await
            .map_err(QuadratureArithmeticError::execution)?;
        Ok(ArrayEstimate(Arc::new(values)))
    }
}

impl OpenSample for Arc<Vec<ScaledComplex>> {
    type Result = ComplexArrayEstimate;
    type Error = ArrayEstimate;

    async fn weighted_sum<E: QuadratureEvaluator<Sample = Self>>(
        samples: &[Self],
        weights: &[ScaledEstimate],
        evaluator: &mut E,
    ) -> Result<Self::Result, QuadratureArithmeticError> {
        let Some(first) = samples.first() else {
            return Err(QuadratureArithmeticError::InconsistentComponents);
        };
        if samples.len() != weights.len()
            || samples.iter().any(|sample| sample.len() != first.len())
        {
            return Err(QuadratureArithmeticError::InconsistentComponents);
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(first.len())
            .map_err(QuadratureArithmeticError::Resource)?;
        let mut pending = 0_usize;
        for component in 0..first.len() {
            let mut sum = ScaledComplex::new(0.0, 0.0);
            for (sample, &weight) in samples.iter().zip(weights) {
                sum = sum.add(sample[component].multiply_real(weight));
                pending += 1;
                if pending == COMPONENT_CHUNK {
                    evaluator
                        .charge(COMPONENT_CHUNK as u64)
                        .await
                        .map_err(QuadratureArithmeticError::execution)?;
                    pending = 0;
                }
            }
            values.push(sum);
        }
        evaluator
            .charge(pending as u64)
            .await
            .map_err(QuadratureArithmeticError::execution)?;
        Ok(ComplexArrayEstimate(Arc::new(values)))
    }

    async fn embedded_error<E: QuadratureEvaluator<Sample = Self>>(
        inputs: OpenRuleInputs<'_, Self>,
        evaluator: &mut E,
    ) -> Result<Self::Error, QuadratureArithmeticError> {
        let Some(first) = inputs.samples.first() else {
            return Err(QuadratureArithmeticError::InconsistentComponents);
        };
        if inputs.samples.len() != inputs.base_weights.len()
            || inputs.samples.len() != inputs.jacobians.len()
            || inputs.kronrod.0.len() != first.len()
            || inputs.gauss.0.len() != first.len()
            || inputs
                .samples
                .iter()
                .any(|sample| sample.len() != first.len())
        {
            return Err(QuadratureArithmeticError::InconsistentComponents);
        }
        let mut errors = Vec::new();
        errors
            .try_reserve_exact(first.len())
            .map_err(QuadratureArithmeticError::Resource)?;
        let mut pending = 0_usize;
        for component in 0..first.len() {
            let integrated = inputs.kronrod.0[component];
            let mean = ScaledComplex {
                re: integrated.re.divide(inputs.width),
                im: integrated.im.divide(inputs.width),
            };
            let mut resabs = ScaledEstimate::from_f64(0.0);
            let mut resasc = ScaledEstimate::from_f64(0.0);
            for ((sample, &weight), &jacobian) in inputs
                .samples
                .iter()
                .zip(inputs.base_weights)
                .zip(inputs.jacobians)
            {
                let transformed = sample[component].multiply_real(jacobian);
                resabs = resabs.add(weight.multiply(transformed.magnitude()));
                resasc = resasc.add(weight.multiply(transformed.subtract(mean).magnitude()));
                pending += 1;
                if pending == COMPONENT_CHUNK {
                    evaluator
                        .charge(COMPONENT_CHUNK as u64)
                        .await
                        .map_err(QuadratureArithmeticError::execution)?;
                    pending = 0;
                }
            }
            errors.push(super::error_estimate::adjusted_error(
                integrated.subtract(inputs.gauss.0[component]).magnitude(),
                resabs,
                resasc,
            ));
        }
        evaluator
            .charge(pending as u64)
            .await
            .map_err(QuadratureArithmeticError::execution)?;
        Ok(ArrayEstimate(Arc::new(errors)))
    }
}
