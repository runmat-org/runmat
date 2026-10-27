//! Selected numerical controls. Canonical option parsing supplies these
//! typed values; this module does not accept or alias public option names.
use super::{invalid, sampling::Kind, IDENTITY, INVALID_ARGUMENT};
use crate::builtins::math::optim::common::global_time::{Policy, TimeLimit};
use crate::execution::CallbackParallelMode;
use crate::BuiltinResult;
use std::num::NonZeroUsize;

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Selected {
    pub evaluations: Option<usize>,
    pub minimum_surrogate: Option<usize>,
    pub batch: NonZeroUsize,
    pub sample_distance: f64,
    pub nonlinear_tolerance: f64,
    #[serde(with = "super::checkpoint::bits::scalar")]
    pub objective_limit: f64,
    pub time: TimeLimit,
    #[serde(with = "super::checkpoint::parallel")]
    pub parallel: CallbackParallelMode,
    pub vectorized: bool,
}
impl Default for Selected {
    fn default() -> Self {
        Self {
            evaluations: None,
            minimum_surrogate: None,
            batch: NonZeroUsize::MIN,
            sample_distance: 1e-6,
            nonlinear_tolerance: 1e-3,
            objective_limit: f64::NEG_INFINITY,
            time: TimeLimit::Unbounded,
            parallel: CallbackParallelMode::Off,
            vectorized: false,
        }
    }
}
pub(super) struct Parameters {
    pub evaluations: usize,
    pub minimum_surrogate: usize,
    pub batch: NonZeroUsize,
    pub sample_distance: f64,
    pub nonlinear_tolerance: f64,
    pub objective_limit: f64,
    pub time: TimeLimit,
    pub parallel: CallbackParallelMode,
    pub vectorized: bool,
}
impl Selected {
    pub(super) fn resolve(self, variables: usize) -> BuiltinResult<Parameters> {
        if variables == 0
            || !self.sample_distance.is_finite()
            || self.sample_distance < 0.0
            || !self.nonlinear_tolerance.is_finite()
            || self.nonlinear_tolerance < 0.0
            || self.objective_limit.is_nan()
        {
            return Err(invalid("invalid selected surrogate controls"));
        }
        let required = variables
            .checked_add(1)
            .ok_or_else(|| invalid("minimum surrogate extent overflow"))?;
        let minimum = match self.minimum_surrogate {
            Some(value) => value,
            None => variables
                .checked_mul(2)
                .ok_or_else(|| invalid("default surrogate extent overflow"))?
                .max(20),
        };
        if minimum < required {
            return Err(invalid(
                "MinSurrogatePoints must be at least the original variable count plus one",
            ));
        }
        let evaluations = match self.evaluations {
            Some(value) => value,
            None => variables
                .checked_mul(50)
                .ok_or_else(|| invalid("default evaluation extent overflow"))?
                .max(200),
        };
        let time = self.time.validate(Policy {
            identity: IDENTITY,
            error: &INVALID_ARGUMENT,
        })?;
        Ok(Parameters {
            evaluations,
            minimum_surrogate: minimum.max(self.batch.get()),
            batch: self.batch,
            sample_distance: self.sample_distance,
            nonlinear_tolerance: self.nonlinear_tolerance,
            objective_limit: self.objective_limit,
            time,
            parallel: self.parallel,
            // The documented requested parallel mode wins, even if Auto
            // subsequently has no pool and takes its serial fallback.
            vectorized: self.vectorized && self.parallel == CallbackParallelMode::Off,
        })
    }
}
impl Parameters {
    /// RunMat internal pool policy: hundreds/thousands of cheap modeled
    /// proposals, more crossover samples at a smaller radius, and at least
    /// one complete positive-basis ring for coordinate/orthogonal samplers.
    pub(super) fn samples(
        &self,
        kind: Kind,
        dimensions: usize,
        scale: f64,
    ) -> BuiltinResult<usize> {
        if dimensions == 0 || !scale.is_finite() || scale <= 0.0 || scale > 0.8 {
            return Err(invalid("invalid adaptive sample-pool geometry"));
        }
        let base = (dimensions.min(50) * 100).max(500);
        match kind {
            Kind::Random => Ok(base),
            Kind::Crossover => {
                Ok(((base as f64 * (0.8 / scale).sqrt()).ceil().min(5000.0)) as usize)
            }
            Kind::Orthogonal | Kind::Coordinate => dimensions
                .checked_mul(2)
                .and_then(|n| n.checked_add(2))
                .map(|ring| base.max(ring))
                .ok_or_else(|| invalid("sampler ring extent overflow")),
        }
    }
}
