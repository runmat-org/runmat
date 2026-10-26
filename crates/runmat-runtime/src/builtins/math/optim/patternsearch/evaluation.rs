//! One original callable evaluation per mesh point. Only undefined numerical
//! values are unsuccessful trials; callback exceptions/control errors escape.
mod choice;
mod history;
use super::{buffer, invalid};
use crate::builtins::math::optim::common::{
    self, constrained_numeric as numeric, ScalarCooperation,
};
use crate::{BuiltinResult, RuntimeError};
pub(super) use choice::improving;
use numeric::Policy;
mod results;
mod point_axis;
pub(super) use point_axis::PointAxis;
pub(super) mod batch;
use runmat_value::Value;

pub(super) struct Measurement {
    pub f: f64,
    pub inequality: runmat_value::ExactArrayStorage<f64>,
    pub equality: runmat_value::ExactArrayStorage<f64>,
}

#[derive(Clone, Copy)]
pub(super) enum CallbackRole {
    Objective,
    NonlinearConstraints,
}

impl CallbackRole {
    fn requested_outputs(self) -> usize {
        match self {
            Self::Objective => 1,
            Self::NonlinearConstraints => 2,
        }
    }
}

pub(super) struct Point {
    pub x: Vec<f64>,
    pub f: f64,
    pub inequality: runmat_value::ExactArrayStorage<f64>,
    pub equality: runmat_value::ExactArrayStorage<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stop {
    Undefined,
    Budget,
    Cached,
}
pub(super) type Evaluation = Result<Point, Stop>;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ObservationDemand {
    MaySkipVisited,
    Fresh,
}

/// Borrowed immutable invocation facts. Copying this view copies no payload.
#[derive(Clone, Copy)]
pub(super) struct Context<'a> {
    pub objective: &'a Value,
    pub constraints: Option<&'a Value>,
    pub shape: &'a [usize],
    pub scalar: bool,
    pub policy: Policy,
    pub execution: Option<&'a batch::Execution>,
    pub demand: ObservationDemand,
}

/// The one mutable observation/budget owner, independent of a temporary view.
pub(super) struct State {
    pub limit: usize,
    pub callbacks: CallbackCounts,
    pub objective_calls: usize,
    pub constraint_calls: usize,
    pub extents: Option<(usize, usize)>,
    pub cache: Option<super::cache::Cache>,
    pub history: Option<super::search::rbf::History>,
    pub original_bounds: Option<super::search::Bounds>,
    pub execution: batch::Execution,
}

/// Objective point budget and semantic invocation counts are different for a
/// vectorized wave. These counters record attempted calls, not repaired points.
#[derive(Default, Clone, Copy)]
pub(super) struct CallbackCounts {
    pub objective: usize,
    pub nonlinear: usize,
}
impl CallbackCounts {
    pub fn charge(&mut self, role: CallbackRole, count: usize) -> BuiltinResult<()> {
        let target = match role {
            CallbackRole::Objective => &mut self.objective,
            CallbackRole::NonlinearConstraints => &mut self.nonlinear,
        };
        *target = target
            .checked_add(count)
            .ok_or_else(|| invalid("semantic callback count overflow"))?;
        Ok(())
    }
}

/// Numerical calls borrow the same state and original cooperation together.
pub(super) struct Evaluator<'a> {
    pub context: Context<'a>,
    pub state: &'a mut State,
    pub cooperation: &'a mut ScalarCooperation,
}

pub(super) fn select(
    sequence: crate::execution::RootedValueSequence,
    width: usize,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let _entry = runtime.map(crate::context::RuntimeContext::enter);
    sequence
        .resolve_sequence(
            runmat_types::SequenceUse::from_requested_outputs(
                runmat_types::RequestedOutputCount::Exactly(width),
            ),
            crate::sequence::SequenceResolutionContext::default(),
        )
        .map_err(consumer_refusal)
}
fn consumer_refusal(mut error: RuntimeError) -> RuntimeError {
    // This is our own invalid result cardinality, not an original callback
    // exception. Never authorize the dispatcher to replay that callback.
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

impl Evaluator<'_> {
    pub(super) fn execution(&self) -> &batch::Execution {
        self.context.execution.unwrap_or(&self.state.execution)
    }
    pub fn point_axis(&self) -> PointAxis {
        PointAxis::from_initial_shape(self.context.shape)
    }

    pub fn direct(&self) -> bool {
        self.execution().direct()
    }
    pub fn remaining(&self) -> usize {
        self.state.limit.saturating_sub(self.state.objective_calls)
    }

    async fn argument(&mut self, x: &[f64]) -> BuiltinResult<Value> {
        let mut copy = buffer(x.len())?;
        for chunk in x.chunks(1024) {
            copy.extend_from_slice(chunk);
            self.cooperation.completed(chunk.len() as u64).await?;
        }
        if self.execution().vectorized {
            common::vector_to_value(
                "patternsearch",
                copy,
                &self.point_axis().argument_shape(x.len(), 1),
                false,
            )
        } else {
            common::vector_to_value(
                "patternsearch",
                copy,
                self.context.shape,
                self.context.scalar,
            )
        }
    }

    pub async fn evaluate(&mut self, x: Vec<f64>) -> BuiltinResult<Evaluation> {
        if self.execution().vectorized {
            let mut points = buffer(1)?;
            points.push(x);
            return self
                .evaluate_wave(points)
                .await?
                .pop()
                .ok_or_else(|| invalid("single point evaluation result missing"));
        }
        self.evaluate_scalar(x).await
    }

    async fn evaluate_scalar(&mut self, x: Vec<f64>) -> BuiltinResult<Evaluation> {
        Ok(self.measure_point(&x).await?.map(|value| Point {
            x,
            f: value.f,
            inequality: value.inequality,
            equality: value.equality,
        }))
    }
    pub async fn measure_point(&mut self, x: &[f64]) -> BuiltinResult<Result<Measurement, Stop>> {
        self.cooperation.completed(0).await?;
        if self.remaining() == 0 {
            return Ok(Err(Stop::Budget));
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Ok(Err(Stop::Undefined));
        }
        if let Some(cache) = self
            .state
            .cache
            .as_ref()
            .filter(|_| self.context.demand == ObservationDemand::MaySkipVisited)
        {
            if cache.contains(x, self.cooperation).await? {
                return Ok(Err(Stop::Cached));
            }
        }
        let argument = self.argument(x).await?;
        if let Some(cache) = &mut self.state.cache {
            cache.record(&argument)?;
        }
        // The count records an actual attempted callback, including an original
        // failure. MaxFunctionEvaluations is an objective-call limit, not the
        // sum of objective and nonlinear calls at the same point.
        self.state.objective_calls += 1;
        if let Some(history) = &mut self.state.history {
            history.attempted(self.state.objective_calls)?;
        }
        let returned = self
            .invoke_one(CallbackRole::Objective, argument.clone())
            .await?;
        let Some(f) = self.objective_result(returned).await? else {
            return Ok(Err(Stop::Undefined));
        };
        if self.context.constraints.is_none() {
            return self
                .complete_measurement(
                    &argument,
                    self.state.objective_calls,
                    x.len(),
                    f,
                    Vec::new(),
                    Vec::new(),
                )
                .await
                .map(Ok);
        }
        self.state.constraint_calls = self
            .state
            .constraint_calls
            .checked_add(1)
            .ok_or_else(|| invalid("constraint callback count overflow"))?;
        let returned = self
            .invoke_one(CallbackRole::NonlinearConstraints, argument.clone())
            .await?;
        let Some((inequality, equality)) = self.constraint_result(returned).await? else {
            return Ok(Err(Stop::Undefined));
        };
        self.complete_measurement(
            &argument,
            self.state.objective_calls,
            x.len(),
            f,
            inequality,
            equality,
        )
        .await
        .map(Ok)
    }
}
