//! Observable stopping time is measured only by the original admitted Clock.
use super::ScalarCooperation;
use crate::builtins::math::optim::common::solver_timing::SolverTiming;
pub(super) use crate::builtins::math::optim::common::swarm_options::TimeLimit;
use crate::BuiltinResult;
use std::time::Duration;
pub(super) struct Timing {
    clock: Option<SolverTiming>,
    last_improvement: Duration,
    maximum: TimeLimit,
    stall: TimeLimit,
}
impl Timing {
    pub fn new(
        maximum: TimeLimit,
        stall: TimeLimit,
        cooperation: &ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let owner = crate::builtins::math::optim::common::swarm_options::Owner::Solver;
        let maximum = maximum.validate(owner.time_policy())?;
        let stall = stall.validate(owner.time_policy())?;
        let selected =
            !matches!(maximum, TimeLimit::Unbounded) || !matches!(stall, TimeLimit::Unbounded);
        let clock = selected
            .then(|| SolverTiming::start(cooperation.runtime()))
            .transpose()?;
        Ok(Self {
            clock,
            last_improvement: Duration::ZERO,
            maximum,
            stall,
        })
    }
    pub fn improved(&mut self) {
        if let Some(clock) = &self.clock {
            self.last_improvement = clock.elapsed();
        }
    }
    pub fn stopped(&self) -> Option<super::state::Termination> {
        let clock = self.clock.as_ref()?;
        let elapsed = clock.elapsed();
        if self.maximum.reached(elapsed) {
            Some(super::state::Termination::Time)
        } else if self
            .stall
            .reached(elapsed.saturating_sub(self.last_improvement))
        {
            Some(super::state::Termination::StallTime)
        } else {
            None
        }
    }
}
