//! Actual objective invocation and same-point result ownership.
//! Private numerical merit probes do not enter this expensive budget owner.
use super::{
    history::Point,
    invalid,
    numeric::DoubleArray,
    observation::{self, Form, Observation},
};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::call::{descriptor::RepeatableCallable, BuiltinArguments};
use crate::BuiltinResult;
mod input;
mod seed;
pub(super) use input::Input;
pub(super) use seed::Seed;
use std::num::NonZeroUsize;

pub(super) struct Completed {
    pub point: Point,
    pub observation: Observation,
}
pub(super) enum Evaluation {
    Completed(Completed),
    Limit,
}
pub(super) enum BatchEvaluation {
    Completed(Vec<Completed>),
    Limit,
}
pub(super) struct Evaluator<'a> {
    source: &'a RepeatableCallable,
    variables: usize,
    limit: usize,
    evaluated: usize,
}
impl<'a> Evaluator<'a> {
    pub(super) fn new(
        source: &'a RepeatableCallable,
        variables: usize,
        limit: usize,
    ) -> BuiltinResult<Self> {
        if variables == 0 {
            return Err(invalid("objective requires a nonempty original point"));
        }
        Ok(Self {
            source,
            variables,
            limit,
            evaluated: 0,
        })
    }
    pub(super) fn evaluated(&self) -> usize {
        self.evaluated
    }
    pub(super) fn restore_evaluated(&mut self, evaluated: usize) -> BuiltinResult<()> {
        if self.evaluated != 0 {
            return Err(invalid(
                "objective count restoration requires an unused dispatcher",
            ));
        }
        // A resumed total limit can be below the already completed count.
        // remaining() then yields zero, never a fresh allowance.
        self.evaluated = evaluated;
        Ok(())
    }
    pub(super) fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.evaluated)
    }

    pub(super) async fn evaluate(
        &mut self,
        coordinates: Vec<f64>,
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Evaluation> {
        match self
            .invoke(
                coordinates,
                NonZeroUsize::MIN,
                Form::Point,
                tolerance,
                cooperation,
            )
            .await?
        {
            BatchEvaluation::Limit => Ok(Evaluation::Limit),
            BatchEvaluation::Completed(mut rows) => {
                let completed = rows
                    .pop()
                    .ok_or_else(|| invalid("scalar objective lost its actual row"))?;
                if !rows.is_empty() {
                    return Err(invalid("scalar objective returned extra actual rows"));
                }
                Ok(Evaluation::Completed(completed))
            }
        }
    }

    /// Coordinates are already in MATLAB column-major m-by-n row layout.
    /// A batch is indivisible: the caller sizes it to remaining(), and no
    /// prefix is submitted when the full actual row count exceeds the grant.
    pub(super) async fn evaluate_batch(
        &mut self,
        coordinates: Vec<f64>,
        rows: NonZeroUsize,
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<BatchEvaluation> {
        self.invoke(
            coordinates,
            rows,
            Form::Vectorized(rows),
            tolerance,
            cooperation,
        )
        .await
    }

    async fn invoke(
        &mut self,
        coordinates: Vec<f64>,
        rows: NonZeroUsize,
        form: Form,
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<BatchEvaluation> {
        cooperation.completed(0).await?;
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(invalid("invalid objective constraint tolerance"));
        }
        if rows.get() > self.remaining() {
            return Ok(BatchEvaluation::Limit);
        }
        let input = Input::new(coordinates, self.variables, form, cooperation).await?;
        let arguments = BuiltinArguments::from_borrowed(
            std::slice::from_ref(input.argument().value()),
            cooperation.runtime(),
        )?;
        // bounded by remaining(): this is actual attempted point count, not
        // invocation count; argument-admission failure never spends a call.
        self.evaluated += rows.get();
        let returned = self
            .source
            .invoke_retained(cooperation.runtime(), arguments, 1)
            .await?;
        let completed = input.finish(returned, tolerance, cooperation).await?;
        Ok(BatchEvaluation::Completed(completed))
    }
}
