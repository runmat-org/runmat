//! Componentwise estimate algebra for one adaptive quadrature traversal.

use std::collections::TryReserveError;
use std::sync::Arc;

use super::adaptive::QuadratureEvaluator;
use super::geometry::{simpson, ScaledEstimate};
use crate::RuntimeError;

pub(super) const COMPONENT_CHUNK: usize = 4096;

#[derive(Debug)]
pub(crate) enum QuadratureArithmeticError {
    Resource(TryReserveError),
    InconsistentComponents,
    Execution(Box<RuntimeError>),
}

impl QuadratureArithmeticError {
    pub(super) fn execution(source: RuntimeError) -> Self {
        Self::Execution(Box::new(source))
    }
}

pub(crate) trait AdaptiveEstimate: Clone {
    async fn magnitude<E: QuadratureEvaluator>(
        &self,
        evaluator: &mut E,
    ) -> Result<ScaledEstimate, QuadratureArithmeticError>;
    async fn add<E: QuadratureEvaluator>(
        &self,
        other: &Self,
        evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError>;
    async fn subtract<E: QuadratureEvaluator>(
        &self,
        other: &Self,
        evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError>;
    async fn multiply_f64<E: QuadratureEvaluator>(
        &self,
        factor: f64,
        evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError>;
    async fn tolerance<E: QuadratureEvaluator>(
        &self,
        absolute: f64,
        relative: f64,
        evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError>;
    async fn abs_le<E: QuadratureEvaluator>(
        &self,
        other: &Self,
        evaluator: &mut E,
    ) -> Result<bool, QuadratureArithmeticError>;
}

pub(crate) struct EmbeddedRuleInputs<'a, S: QuadratureSample> {
    pub(crate) samples: &'a [S],
    pub(crate) base_weights: &'a [ScaledEstimate],
    pub(crate) jacobians: &'a [ScaledEstimate],
    pub(crate) width: ScaledEstimate,
    pub(crate) kronrod: &'a S::Estimate,
    pub(crate) gauss: &'a S::Estimate,
}

pub(crate) trait QuadratureSample: Clone {
    type Estimate: AdaptiveEstimate;

    async fn embedded_error<E: QuadratureEvaluator<Sample = Self>>(
        inputs: EmbeddedRuleInputs<'_, Self>,
        evaluator: &mut E,
    ) -> Result<Self::Estimate, QuadratureArithmeticError>;

    async fn weighted_sum<E: QuadratureEvaluator<Sample = Self>>(
        samples: &[Self],
        weights: &[ScaledEstimate],
        evaluator: &mut E,
    ) -> Result<Self::Estimate, QuadratureArithmeticError>;

    async fn simpson<E: QuadratureEvaluator<Sample = Self>>(
        a: f64,
        b: f64,
        fa: &Self,
        fm: &Self,
        fb: &Self,
        evaluator: &mut E,
    ) -> Result<Self::Estimate, QuadratureArithmeticError>;
}

impl AdaptiveEstimate for ScaledEstimate {
    async fn magnitude<E: QuadratureEvaluator>(
        &self,
        _evaluator: &mut E,
    ) -> Result<ScaledEstimate, QuadratureArithmeticError> {
        Ok(self.abs())
    }

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

    async fn multiply_f64<E: QuadratureEvaluator>(
        &self,
        factor: f64,
        _evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError> {
        Ok((*self).multiply_f64(factor))
    }

    async fn tolerance<E: QuadratureEvaluator>(
        &self,
        absolute: f64,
        relative: f64,
        _evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError> {
        Ok(ScaledEstimate::from_f64(absolute).max(self.abs().multiply_f64(relative)))
    }

    async fn abs_le<E: QuadratureEvaluator>(
        &self,
        other: &Self,
        _evaluator: &mut E,
    ) -> Result<bool, QuadratureArithmeticError> {
        Ok((*self).abs_le(*other))
    }
}

impl QuadratureSample for f64 {
    type Estimate = ScaledEstimate;

    async fn embedded_error<E: QuadratureEvaluator<Sample = Self>>(
        inputs: EmbeddedRuleInputs<'_, Self>,
        evaluator: &mut E,
    ) -> Result<Self::Estimate, QuadratureArithmeticError> {
        super::error_estimate::scalar(inputs, evaluator).await
    }

    async fn weighted_sum<E: QuadratureEvaluator<Sample = Self>>(
        samples: &[Self],
        weights: &[ScaledEstimate],
        evaluator: &mut E,
    ) -> Result<Self::Estimate, QuadratureArithmeticError> {
        if samples.len() != weights.len() {
            return Err(QuadratureArithmeticError::InconsistentComponents);
        }
        let mut result = ScaledEstimate::from_f64(0.0);
        for (&sample, &weight) in samples.iter().zip(weights) {
            result = result.add(weight.multiply_f64(sample));
        }
        evaluator
            .charge(samples.len() as u64)
            .await
            .map_err(QuadratureArithmeticError::execution)?;
        Ok(result)
    }

    async fn simpson<E: QuadratureEvaluator<Sample = Self>>(
        a: f64,
        b: f64,
        fa: &Self,
        fm: &Self,
        fb: &Self,
        _evaluator: &mut E,
    ) -> Result<Self::Estimate, QuadratureArithmeticError> {
        Ok(simpson(a, b, *fa, *fm, *fb))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ArrayEstimate(pub(crate) Arc<Vec<ScaledEstimate>>);

impl ArrayEstimate {
    async fn zip_map<E: QuadratureEvaluator>(
        &self,
        other: &Self,
        evaluator: &mut E,
        map: impl Fn(ScaledEstimate, ScaledEstimate) -> ScaledEstimate,
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

    async fn map<E: QuadratureEvaluator>(
        &self,
        evaluator: &mut E,
        map: impl Fn(ScaledEstimate) -> ScaledEstimate,
    ) -> Result<Self, QuadratureArithmeticError> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(self.0.len())
            .map_err(QuadratureArithmeticError::Resource)?;
        for (index, &value) in self.0.iter().enumerate() {
            values.push(map(value));
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

impl AdaptiveEstimate for ArrayEstimate {
    async fn magnitude<E: QuadratureEvaluator>(
        &self,
        evaluator: &mut E,
    ) -> Result<ScaledEstimate, QuadratureArithmeticError> {
        let mut largest = ScaledEstimate::from_f64(0.0);
        for (index, &component) in self.0.iter().enumerate() {
            largest = largest.max(component.abs());
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
        Ok(largest)
    }

    async fn add<E: QuadratureEvaluator>(
        &self,
        other: &Self,
        evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError> {
        self.zip_map(other, evaluator, ScaledEstimate::add).await
    }

    async fn subtract<E: QuadratureEvaluator>(
        &self,
        other: &Self,
        evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError> {
        self.zip_map(other, evaluator, ScaledEstimate::subtract)
            .await
    }

    async fn multiply_f64<E: QuadratureEvaluator>(
        &self,
        factor: f64,
        evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError> {
        self.map(evaluator, |value| value.multiply_f64(factor))
            .await
    }

    async fn tolerance<E: QuadratureEvaluator>(
        &self,
        absolute: f64,
        relative: f64,
        evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError> {
        self.map(evaluator, |value| {
            ScaledEstimate::from_f64(absolute).max(value.abs().multiply_f64(relative))
        })
        .await
    }

    async fn abs_le<E: QuadratureEvaluator>(
        &self,
        other: &Self,
        evaluator: &mut E,
    ) -> Result<bool, QuadratureArithmeticError> {
        if self.0.len() != other.0.len() {
            return Err(QuadratureArithmeticError::InconsistentComponents);
        }
        for (index, (left, right)) in self.0.iter().zip(other.0.iter()).enumerate() {
            if !(*left).abs_le(*right) {
                return Ok(false);
            }
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
        Ok(true)
    }
}

impl QuadratureSample for Arc<Vec<f64>> {
    type Estimate = ArrayEstimate;

    async fn embedded_error<E: QuadratureEvaluator<Sample = Self>>(
        inputs: EmbeddedRuleInputs<'_, Self>,
        evaluator: &mut E,
    ) -> Result<Self::Estimate, QuadratureArithmeticError> {
        super::error_estimate::array(inputs, evaluator).await
    }

    async fn weighted_sum<E: QuadratureEvaluator<Sample = Self>>(
        samples: &[Self],
        weights: &[ScaledEstimate],
        evaluator: &mut E,
    ) -> Result<Self::Estimate, QuadratureArithmeticError> {
        let Some(first) = samples.first() else {
            return Err(QuadratureArithmeticError::InconsistentComponents);
        };
        if samples.len() != weights.len()
            || samples.iter().any(|sample| sample.len() != first.len())
        {
            return Err(QuadratureArithmeticError::InconsistentComponents);
        }
        let mut components = Vec::new();
        components
            .try_reserve_exact(first.len())
            .map_err(QuadratureArithmeticError::Resource)?;
        let mut pending_work = 0_usize;
        for component in 0..first.len() {
            let mut sum = ScaledEstimate::from_f64(0.0);
            for (sample, &weight) in samples.iter().zip(weights) {
                sum = sum.add(weight.multiply_f64(sample[component]));
                pending_work += 1;
                if pending_work == COMPONENT_CHUNK {
                    evaluator
                        .charge(COMPONENT_CHUNK as u64)
                        .await
                        .map_err(QuadratureArithmeticError::execution)?;
                    pending_work = 0;
                }
            }
            components.push(sum);
        }
        evaluator
            .charge(pending_work as u64)
            .await
            .map_err(QuadratureArithmeticError::execution)?;
        Ok(ArrayEstimate(Arc::new(components)))
    }

    async fn simpson<E: QuadratureEvaluator<Sample = Self>>(
        a: f64,
        b: f64,
        fa: &Self,
        fm: &Self,
        fb: &Self,
        evaluator: &mut E,
    ) -> Result<Self::Estimate, QuadratureArithmeticError> {
        if fa.len() != fm.len() || fa.len() != fb.len() {
            return Err(QuadratureArithmeticError::InconsistentComponents);
        }
        let mut components = Vec::new();
        components
            .try_reserve_exact(fa.len())
            .map_err(QuadratureArithmeticError::Resource)?;
        for (index, ((&left, &middle), &right)) in
            fa.iter().zip(fm.iter()).zip(fb.iter()).enumerate()
        {
            components.push(simpson(a, b, left, middle, right));
            if (index + 1).is_multiple_of(COMPONENT_CHUNK) {
                evaluator
                    .charge(COMPONENT_CHUNK as u64)
                    .await
                    .map_err(QuadratureArithmeticError::execution)?;
            }
        }
        evaluator
            .charge((fa.len() % COMPONENT_CHUNK) as u64)
            .await
            .map_err(QuadratureArithmeticError::execution)?;
        Ok(ArrayEstimate(Arc::new(components)))
    }
}
