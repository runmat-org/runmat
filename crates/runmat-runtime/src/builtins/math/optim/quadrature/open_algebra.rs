//! Numerical result/error separation for the open Gauss–Kronrod rule.
//!
//! A contour integral can have a complex result while its certified absolute
//! error remains real. Legacy Simpson samples retain their existing algebra.

use super::adaptive::QuadratureEvaluator;
use super::algebra::{AdaptiveEstimate, EmbeddedRuleInputs, QuadratureSample};
use super::geometry::ScaledEstimate;
use super::QuadratureArithmeticError;
use std::sync::Arc;

pub(crate) trait OpenResult: Clone {
    type Error: AdaptiveEstimate;

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
    async fn tolerance_error<E: QuadratureEvaluator>(
        &self,
        absolute: f64,
        relative: f64,
        evaluator: &mut E,
    ) -> Result<Self::Error, QuadratureArithmeticError>;
}

impl<T: AdaptiveEstimate> OpenResult for T {
    type Error = Self;

    async fn add<E: QuadratureEvaluator>(
        &self,
        other: &Self,
        evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError> {
        AdaptiveEstimate::add(self, other, evaluator).await
    }

    async fn subtract<E: QuadratureEvaluator>(
        &self,
        other: &Self,
        evaluator: &mut E,
    ) -> Result<Self, QuadratureArithmeticError> {
        AdaptiveEstimate::subtract(self, other, evaluator).await
    }

    async fn tolerance_error<E: QuadratureEvaluator>(
        &self,
        absolute: f64,
        relative: f64,
        evaluator: &mut E,
    ) -> Result<Self::Error, QuadratureArithmeticError> {
        AdaptiveEstimate::tolerance(self, absolute, relative, evaluator).await
    }
}

pub(crate) struct OpenRuleInputs<'a, S: OpenSample> {
    pub(crate) samples: &'a [S],
    pub(crate) base_weights: &'a [ScaledEstimate],
    pub(crate) jacobians: &'a [ScaledEstimate],
    pub(crate) width: ScaledEstimate,
    pub(crate) kronrod: &'a S::Result,
    pub(crate) gauss: &'a S::Result,
}

pub(crate) trait OpenSample: Clone {
    type Result: OpenResult<Error = Self::Error>;
    type Error: AdaptiveEstimate;

    async fn weighted_sum<E: QuadratureEvaluator<Sample = Self>>(
        samples: &[Self],
        weights: &[ScaledEstimate],
        evaluator: &mut E,
    ) -> Result<Self::Result, QuadratureArithmeticError>;

    async fn embedded_error<E: QuadratureEvaluator<Sample = Self>>(
        inputs: OpenRuleInputs<'_, Self>,
        evaluator: &mut E,
    ) -> Result<Self::Error, QuadratureArithmeticError>;
}

impl OpenSample for f64 {
    type Result = ScaledEstimate;
    type Error = ScaledEstimate;

    async fn weighted_sum<E: QuadratureEvaluator<Sample = Self>>(
        samples: &[Self],
        weights: &[ScaledEstimate],
        evaluator: &mut E,
    ) -> Result<Self::Result, QuadratureArithmeticError> {
        <Self as QuadratureSample>::weighted_sum(samples, weights, evaluator).await
    }

    async fn embedded_error<E: QuadratureEvaluator<Sample = Self>>(
        inputs: OpenRuleInputs<'_, Self>,
        evaluator: &mut E,
    ) -> Result<Self::Error, QuadratureArithmeticError> {
        <Self as QuadratureSample>::embedded_error(
            EmbeddedRuleInputs {
                samples: inputs.samples,
                base_weights: inputs.base_weights,
                jacobians: inputs.jacobians,
                width: inputs.width,
                kronrod: inputs.kronrod,
                gauss: inputs.gauss,
            },
            evaluator,
        )
        .await
    }
}

impl OpenSample for Arc<Vec<f64>> {
    type Result = super::algebra::ArrayEstimate;
    type Error = super::algebra::ArrayEstimate;

    async fn weighted_sum<E: QuadratureEvaluator<Sample = Self>>(
        samples: &[Self],
        weights: &[ScaledEstimate],
        evaluator: &mut E,
    ) -> Result<Self::Result, QuadratureArithmeticError> {
        <Self as QuadratureSample>::weighted_sum(samples, weights, evaluator).await
    }

    async fn embedded_error<E: QuadratureEvaluator<Sample = Self>>(
        inputs: OpenRuleInputs<'_, Self>,
        evaluator: &mut E,
    ) -> Result<Self::Error, QuadratureArithmeticError> {
        <Self as QuadratureSample>::embedded_error(
            EmbeddedRuleInputs {
                samples: inputs.samples,
                base_weights: inputs.base_weights,
                jacobians: inputs.jacobians,
                width: inputs.width,
                kronrod: inputs.kronrod,
                gauss: inputs.gauss,
            },
            evaluator,
        )
        .await
    }
}
