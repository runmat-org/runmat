//! One retained original box; proposal windows never replace its endpoints.
use super::super::invalid;
use crate::BuiltinResult;

pub(in crate::builtins::math::optim::patternsearch) struct Bounds {
    pub(in crate::builtins::math::optim::patternsearch::search) lower: Vec<f64>,
    pub(in crate::builtins::math::optim::patternsearch::search) upper: Vec<f64>,
}
impl Bounds {
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> BuiltinResult<Self> {
        if lower.len() != upper.len()
            || lower
                .iter()
                .zip(&upper)
                .any(|(&l, &u)| l.is_nan() || u.is_nan() || l > u)
        {
            return Err(invalid("invalid original search bounds"));
        }
        Ok(Self { lower, upper })
    }
    pub fn original(&self) -> (&[f64], &[f64]) {
        (&self.lower, &self.upper)
    }
}
