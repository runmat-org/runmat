//! Published NUPS transition rules with explicit RunMat numerical choices.
use super::super::invalid;
use crate::BuiltinResult;

/// An internal algorithm coefficient, not an invented public optimizer field.
pub(in crate::builtins::math::optim::patternsearch) const RELATIVE_DECREASE: f64 = 1e-4;
pub(in crate::builtins::math::optim::patternsearch) fn relative_decrease(
    before: f64,
    after: f64,
) -> Option<f64> {
    if !before.is_finite() || !after.is_finite() {
        return None;
    }
    let scale = before.abs().max(1.0);
    Some(before / scale - after / scale)
}
pub(super) struct Transitions {
    coarse_threshold: f64,
    weak_successes: u8,
}
impl Transitions {
    pub fn new(initial_effective_mesh: f64) -> BuiltinResult<Self> {
        if !initial_effective_mesh.is_finite() || initial_effective_mesh <= 0.0 {
            return Err(invalid(
                "NUPS initial effective mesh must be finite positive",
            ));
        }
        Ok(Self {
            coarse_threshold: initial_effective_mesh,
            weak_successes: 0,
        })
    }
    pub fn successful(&mut self, mesh: f64, before: f64, after: f64) -> BuiltinResult<f64> {
        let improvement = relative_decrease(before, after)
            .filter(|gain| *gain > 0.0)
            .ok_or_else(|| invalid("NUPS success has no finite measured improvement"))?;
        if improvement >= RELATIVE_DECREASE {
            self.weak_successes = 0;
            return Ok(2.0);
        }
        self.weak_successes += 1;
        if self.weak_successes == 3 {
            self.weak_successes = 0;
            Ok(1.5)
        } else if mesh > self.coarse_threshold {
            Ok(2.0)
        } else {
            Ok(1.0)
        }
    }
    pub fn unsuccessful(&mut self) -> f64 {
        self.weak_successes = 0;
        0.5
    }
    pub fn interrupted(&mut self) {
        self.weak_successes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_overflow_stable_gains_and_three_success_transition_are_distinct() {
        assert_eq!(relative_decrease(f64::MAX, -f64::MAX), Some(2.0));
        assert_eq!(relative_decrease(0.0, -f64::MAX), Some(f64::MAX));
        assert_eq!(relative_decrease(f64::INFINITY, 0.0), None);
        let mut state = Transitions::new(1.0).unwrap();
        assert_eq!(state.successful(1.0, 1.0, 1.0 - 1e-6).unwrap(), 1.0);
        assert_eq!(state.successful(1.0, 1.0, 1.0 - 1e-6).unwrap(), 1.0);
        assert_eq!(state.successful(1.0, 1.0, 1.0 - 1e-6).unwrap(), 1.5);
        assert_eq!(state.successful(2.0, 1.0, 1.0 - 1e-6).unwrap(), 2.0);
        assert_eq!(state.unsuccessful(), 0.5);
        assert_eq!(state.successful(1.0, 1.0, 0.5).unwrap(), 2.0);
        assert!(state.successful(1.0, 1.0, 1.0).is_err());
    }
}
