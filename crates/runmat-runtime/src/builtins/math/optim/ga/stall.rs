//! Bounded completed-generation relative-change window. AL controllers own
//! one window per fixed-merit subproblem, not one across multiplier changes.
use super::{invalid, ScalarCooperation};
use crate::BuiltinResult;
use runmat_builtins::OptimGeneticStall;
use std::collections::VecDeque;

pub(super) struct Window {
    limit: usize,
    previous: Option<f64>,
    changes: VecDeque<f64>,
    sum: f64,
    weighted: f64,
    unchanged: usize,
}
impl Window {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            previous: None,
            changes: VecDeque::new(),
            sum: 0.,
            weighted: 0.,
            unchanged: 0,
        }
    }
    pub fn reset(&mut self, limit: usize) {
        self.limit = limit;
        self.previous = None;
        self.changes.clear();
        self.sum = 0.;
        self.weighted = 0.;
        self.unchanged = 0;
    }
    pub async fn observe(
        &mut self,
        best: Option<f64>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if best.is_some_and(|value| !value.is_finite()) {
            return Err(invalid(
                "stall comparison requires finite or undefined merit",
            ));
        }
        let previous = self.previous;
        self.previous = best;
        let (Some(previous), Some(best)) = (previous, best) else {
            // Unknown observations do not count as an unchanged objective.
            self.changes.clear();
            self.sum = 0.;
            self.weighted = 0.;
            self.unchanged = 0;
            return Ok(());
        };
        self.unchanged = if previous == best {
            self.unchanged.saturating_add(1)
        } else {
            0
        };
        if self.limit == 0 {
            return Ok(());
        }
        let scale = previous.abs().max(best.abs()).max(1.);
        let change = (previous / scale - best / scale).abs();
        let expired = if self.changes.len() == self.limit {
            self.changes.pop_front()
        } else {
            None
        };
        if expired.is_none() {
            self.changes.try_reserve(1).map_err(|_| {
                crate::context::host_allocation::host_allocation_runtime_error(
                    cooperation.runtime(),
                    "RunMat:ga:StallWindow",
                    "genetic stopping window",
                    crate::context::host_allocation::HostAllocationError::Numeric(
                        runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                    ),
                )
            })?;
        }
        self.sum = self.sum - expired.unwrap_or(0.) + change;
        self.weighted = 0.5 * self.weighted + 0.5 * change;
        if let Some(expired) = expired {
            self.weighted -= expired * half_power(self.limit.saturating_add(1));
        }
        self.changes.push_back(change);
        cooperation.completed(1).await?;
        Ok(())
    }
    pub fn average(&self, kind: OptimGeneticStall) -> Option<f64> {
        if self.previous.is_none() || self.changes.len() < self.limit {
            return None;
        }
        if self.limit == 0 {
            return Some(0.);
        }
        Some(match kind {
            OptimGeneticStall::AverageChange => self.sum.max(0.) / self.limit as f64,
            OptimGeneticStall::GeometricWeighted => {
                self.weighted.max(0.) / (1. - half_power(self.limit))
            }
        })
    }
    pub fn small(&self, kind: OptimGeneticStall, tolerance: f64) -> bool {
        if tolerance == 0.0 {
            return self.unchanged();
        }
        tolerance.is_finite()
            && tolerance >= 0.
            && self
                .average(kind)
                .is_some_and(|average| average <= tolerance)
    }
    pub fn unchanged(&self) -> bool {
        self.previous.is_some() && self.unchanged >= self.limit
    }
}
fn half_power(power: usize) -> f64 {
    if power > 1074 {
        0.
    } else {
        0.5_f64.powi(power as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn completed_windows_do_not_turn_unknown_values_into_convergence() {
        futures::executor::block_on(async {
            let runtime = crate::user_functions::test_support::exact_fixture_runtime();
            let mut cooperation =
                ScalarCooperation::for_builtin(super::super::IDENTITY, Some(&runtime));
            let mut window = Window::new(2);
            for best in [Some(f64::MAX), Some(-f64::MAX), Some(-f64::MAX)] {
                window.observe(best, &mut cooperation).await.unwrap();
            }
            assert_eq!(window.average(OptimGeneticStall::AverageChange), Some(1.));
            let weighted = window
                .average(OptimGeneticStall::GeometricWeighted)
                .unwrap();
            assert!((weighted - 2. / 3.).abs() < 1e-15);
            assert!(!window.small(OptimGeneticStall::AverageChange, 0.));
            window
                .observe(Some(-f64::MAX), &mut cooperation)
                .await
                .unwrap();
            assert!(window.small(OptimGeneticStall::AverageChange, 0.));
            window.observe(None, &mut cooperation).await.unwrap();
            assert!(!window.small(OptimGeneticStall::AverageChange, 1.));
            window.reset(usize::MAX);
            window.observe(Some(0.), &mut cooperation).await.unwrap();
            assert!(
                window.changes.is_empty(),
                "no limit-sized speculative allocation"
            );
        });
    }
}
