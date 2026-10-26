//! One exact scaled normal. Multiplication by mesh scales is performed before
//! rounding, so a subnormal/overflowing intermediate cannot erase a face.
use super::Policy;
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};

/// Exact diagonal metric, independent of solver option/control types.
pub(in crate::builtins::math::optim) trait Diagonal {
    fn dimension(&self) -> usize;
    fn factor(&self, column: usize) -> BigRational;
}

pub(in crate::builtins::math::optim) struct Normal {
    pub dimension: usize,
    pub values: Vec<(usize, BigRational)>,
    scale: BigRational,
    normalized_length: f64,
}
impl Normal {
    pub async fn new(
        row: &[(usize, f64)],
        scaling: &impl Diagonal,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut values = policy.buffer(row.len())?;
        let mut previous = None;
        let mut scale = BigRational::zero();
        for &(column, coefficient) in row {
            if column >= scaling.dimension() || previous.is_some_and(|prior| prior >= column) {
                return Err(policy.invalid("direction normal dimension/order mismatch"));
            }
            previous = Some(column);
            let coefficient = BigRational::from_float(coefficient)
                .ok_or_else(|| policy.invalid("nonfinite direction normal"))?;
            let diagonal = scaling.factor(column);
            if !diagonal.is_positive() {
                return Err(policy.invalid("direction scale must be strictly positive"));
            }
            let value = coefficient * diagonal;
            if !value.is_zero() {
                scale = scale.max(value.abs());
                values.push((column, value));
            }
            cooperation.completed(1).await?;
        }
        let mut normalized_length = 0.0f64;
        for (_, value) in &values {
            let component = (value / &scale)
                .to_f64()
                .expect("normalized finite coefficient");
            normalized_length = normalized_length.hypot(component);
            cooperation.completed(1).await?;
        }
        Ok(Self {
            dimension: scaling.dimension(),
            values,
            scale,
            normalized_length,
        })
    }
    pub fn distance(&self, slack: f64) -> Option<f64> {
        if !slack.is_finite() {
            return None;
        }
        Some(self.distance_exact(&BigRational::from_float(slack).expect("finite slack")))
    }
    /// A caller with original exact row activity need not overflow a finite
    /// endpoint subtraction before asking for its scaled binding distance.
    pub fn distance_exact(&self, slack: &BigRational) -> f64 {
        if self.scale.is_zero() {
            return if !slack.is_negative() {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
        }
        let ratio = slack / &self.scale;
        let distance = ratio.to_f64().unwrap_or_else(|| {
            if ratio.is_negative() {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        });
        distance / self.normalized_length
    }
}
