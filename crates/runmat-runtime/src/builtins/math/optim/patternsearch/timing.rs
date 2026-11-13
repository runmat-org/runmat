//! Semantic time uses only the original invocation clock. Keeping days as
//! days avoids overflowing a finite duration while converting it to seconds.
use super::invalid;
use crate::builtins::math::optim::common::solver_timing::SolverTiming;
use crate::{context::RuntimeContext, BuiltinResult};
use std::time::Duration;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub(super) enum TimeLimit {
    Unbounded,
    Seconds(f64),
    Days(f64),
}
impl TimeLimit {
    fn reached(self, elapsed: Duration) -> bool {
        let seconds = elapsed.as_secs_f64();
        match self {
            Self::Unbounded => false,
            Self::Seconds(value) => seconds >= value,
            Self::Days(value) => seconds / 86400.0 >= value,
        }
    }
    pub fn validate(self) -> BuiltinResult<Self> {
        let value = match self {
            Self::Unbounded => return Ok(self),
            Self::Seconds(value) | Self::Days(value) => value,
        };
        if value.is_nan() || value < 0.0 {
            return Err(invalid("MaxTime must be nonnegative"));
        }
        Ok(if value == f64::INFINITY {
            Self::Unbounded
        } else {
            self
        })
    }
}

pub(super) struct Deadline {
    clock: Option<SolverTiming>,
    limit: TimeLimit,
}
impl Deadline {
    pub fn new(
        limit: TimeLimit,
        runtime: Option<&RuntimeContext>,
        mutable_time: bool,
    ) -> BuiltinResult<Self> {
        let limit = limit.validate()?;
        let observed = mutable_time || !matches!(limit, TimeLimit::Unbounded);
        Ok(Self {
            clock: if observed {
                Some(SolverTiming::start(runtime)?)
            } else {
                None
            },
            limit,
        })
    }
    pub fn reconfigure(&mut self, limit: TimeLimit) -> BuiltinResult<()> {
        let limit = limit.validate()?;
        if !matches!(limit, TimeLimit::Unbounded) && self.clock.is_none() {
            return Err(SolverTiming::unavailable());
        }
        self.limit = limit;
        Ok(())
    }
    pub fn reached(&self) -> bool {
        self.clock
            .as_ref()
            .is_some_and(|clock| self.limit.reached(clock.elapsed()))
    }
    pub fn auxiliary(&self) -> BuiltinResult<AuxiliaryTiming<'_>> {
        let original = self.clock.as_ref().ok_or_else(SolverTiming::unavailable)?;
        let (offset, local) = original.subsearch_view();
        Ok(AuxiliaryTiming {
            parent: self,
            original,
            offset,
            local,
        })
    }
}

/// One local duration window, subordinate to the immutable parent epoch.
pub(super) struct AuxiliaryTiming<'a> {
    parent: &'a Deadline,
    original: &'a SolverTiming,
    offset: Duration,
    local: SolverTiming,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AuxiliaryStop {
    Parent,
    Local,
}
impl AuxiliaryTiming<'_> {
    pub fn local(&self) -> &SolverTiming {
        &self.local
    }
    pub fn reached(
        &self,
        maximum: crate::builtins::math::optim::common::global_time::TimeLimit,
    ) -> bool {
        self.stop(maximum).is_some()
    }
    pub fn stop(
        &self,
        maximum: crate::builtins::math::optim::common::global_time::TimeLimit,
    ) -> Option<AuxiliaryStop> {
        let elapsed = self.original.elapsed();
        if self.parent.limit.reached(elapsed) {
            Some(AuxiliaryStop::Parent)
        } else if maximum.reached(elapsed.saturating_sub(self.offset)) {
            Some(AuxiliaryStop::Local)
        } else {
            None
        }
    }
}
