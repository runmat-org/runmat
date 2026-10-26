//! Selected active-set controls on the original solver clock.
use super::*;
use crate::builtins::math::optim::common::{
    global_time::{self, TimeLimit},
    ScalarOptionFields,
};
pub(in crate::builtins::math::optim::quadprog) struct Active {
    pub step: f64,
    pub objective_limit: f64,
    time: TimeLimit,
    clock: Option<SolverTiming>,
}
pub(super) fn time(value: &Value) -> BuiltinResult<TimeLimit> {
    global_time::parse(
        global_time::Policy {
            identity: runmat_builtins::QUADPROG_CATALOG_ENTRY.identity,
            error: &runmat_builtins::QUADPROG_ERROR_INVALID_INPUT,
        },
        value,
    )
}
impl Active {
    pub(super) async fn parse(
        fields: &ScalarOptionFields<'_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let step = fields.real(NAME, Field::TolX, 1e-8)?;
        if step < 0. || !step.is_finite() {
            return Err(invalid("StepTolerance must be nonnegative finite"));
        }
        let objective_limit = fields
            .exact(Field::ObjectiveLimit)
            .map(|value| {
                crate::builtins::math::optim::common::quasi_newton_controls::objective_limit(
                    NAME, value,
                )
            })
            .transpose()?
            .unwrap_or(-1e20);
        let time = match fields.exact(Field::MaxTime) {
            None => TimeLimit::Unbounded,
            Some(value @ Value::GpuTensor(_)) => {
                let value = crate::builtins::math::optim::common::gather_exact_floating_value_with_cooperation(
                    NAME, value.clone(), "MaxTime", c,
                ).await?;
                time(&value)?
            }
            Some(value) => time(value)?,
        };
        Ok(Self {
            step,
            objective_limit,
            time,
            clock: None,
        })
    }
    pub(super) fn bind(
        &mut self,
        clock: Option<&SolverTiming>,
        c: &ScalarCooperation,
    ) -> BuiltinResult<()> {
        self.clock = if let Some(clock) = clock {
            Some(clock.clone())
        } else if !matches!(self.time, TimeLimit::Unbounded) {
            Some(SolverTiming::start(c.runtime())?)
        } else {
            None
        };
        Ok(())
    }
    pub(super) fn reached(&self) -> bool {
        if matches!(self.time, TimeLimit::Unbounded) {
            return false;
        }
        self.clock
            .as_ref()
            .is_some_and(|clock| self.time.reached(clock.elapsed()))
    }
    pub(super) fn require_clock(&self) -> BuiltinResult<()> {
        if !matches!(self.time, TimeLimit::Unbounded) && self.clock.is_none() {
            return Err(SolverTiming::unavailable());
        }
        Ok(())
    }
}
