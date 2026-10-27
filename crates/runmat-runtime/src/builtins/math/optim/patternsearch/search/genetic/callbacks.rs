//! Every user event of the auxiliary lifecycle drives the same scoped merit
//! service. The registry owns only its port; the evaluator remains borrowed.
use super::super::super::{
    evaluation::{Context, Evaluator, Point},
    invalid,
};
use super::oracle::Oracle;
use super::Stop;
use crate::builtins::math::optim::{
    common::{population::projection::Arguments, ScalarCooperation},
    ga::{
        callbacks::{Event, Invocation},
        evaluation::source::{Request, Routes, Source, Wave},
    },
};
use crate::{
    call::{descriptor::RepeatableCallable, BuiltinArguments},
    execution::{scoped_callable::Scope, CallbackParallelMode, RootedValueSequence},
    BuiltinResult,
};
use runmat_value::Value;
use std::ops::ControlFlow;

// Scope is the first field so its capability retires before remaining domain
// state is destroyed. No handler or evaluator reference is stored in Value.
pub(super) struct Callbacks<'a> {
    pub scope: Scope,
    pub oracle: Oracle<'a>,
}
impl Source for Callbacks<'_> {
    fn retune(&mut self, maximum: crate::builtins::math::optim::common::global_time::TimeLimit) {
        self.oracle.maximum = maximum;
    }
    fn refresh_counts(
        &self,
        objective: &mut crate::builtins::math::optim::common::population::Counts,
        nonlinear: &mut crate::builtins::math::optim::common::population::Counts,
    ) -> BuiltinResult<()> {
        self.oracle.refresh_counts(objective, nonlinear)
    }
    async fn seed_initial<P: Arguments>(
        &mut self,
        population: &P,
        supplied: &mut [Option<crate::builtins::math::optim::ga::evaluation::fitness::Supplied>],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        self.oracle
            .seed_initial(population, supplied, cooperation)
            .await
    }
    fn select_routes(
        &mut self,
        objective: &Value,
        nonlinear: Option<&Value>,
        vectorized: bool,
        parallel: CallbackParallelMode,
        cooperation: &ScalarCooperation,
    ) -> BuiltinResult<Routes> {
        self.oracle
            .select_routes(objective, nonlinear, vectorized, parallel, cooperation)
    }
    async fn next<'a, P: Arguments + 'a>(
        &'a mut self,
        request: Request<'a, P>,
        cooperation: &'a mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<Stop, Wave>> {
        cooperation.completed(0).await?;
        let end = match self.oracle.end(&request)? {
            ControlFlow::Continue(end) => end,
            ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
        };
        let projected = if request.population.numeric_rows().is_none() {
            let prepared = request
                .population
                .prepare_projection(&request.selected[request.start..end], cooperation)
                .await?;
            match self.projection(prepared, cooperation).await? {
                ControlFlow::Continue(argument) => Some(argument),
                ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
            }
        } else {
            None
        };
        self.oracle.next(request, end, projected, cooperation).await
    }
}
impl Invocation for Callbacks<'_> {
    type Stop = Stop;
    fn stop(&self, event: Event) -> Option<Stop> {
        if matches!(
            event,
            Event::Observer(runmat_builtins::OptimGeneticObserverState::Done)
        ) {
            return None;
        }
        self.oracle
            .timing
            .stop(self.oracle.maximum)
            .map(Stop::from)
            .or_else(|| {
                (self.oracle.state.objective_calls >= self.oracle.state.limit)
                    .then_some(Stop::Budget)
            })
    }
    fn protocol<'a, T: 'a, F: std::future::Future<Output = BuiltinResult<T>> + 'a>(
        &'a mut self,
        make: impl FnOnce(Option<crate::context::RuntimeContext>) -> F + 'a,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<ControlFlow<Stop, T>>> + 'a {
        self.drive(make, Event::Work, cooperation)
    }
    fn call<'a>(
        &'a mut self,
        target: &'a RepeatableCallable,
        arguments: BuiltinArguments,
        outputs: usize,
        event: Event,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<ControlFlow<Stop, RootedValueSequence>>> + 'a
    {
        self.drive(
            move |original| async move {
                target
                    .invoke_retained(original.as_ref(), arguments, outputs)
                    .await
            },
            event,
            cooperation,
        )
    }
}
impl Callbacks<'_> {
    async fn drive<'a, T: 'a, F: std::future::Future<Output = BuiltinResult<T>> + 'a>(
        &'a mut self,
        make: impl FnOnce(Option<crate::context::RuntimeContext>) -> F + 'a,
        event: Event,
        cooperation: &'a mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<Stop, T>> {
        cooperation.completed(0).await?;
        if let Some(stop) = self.stop(event) {
            return Ok(ControlFlow::Break(stop));
        }
        let Oracle {
            context,
            state,
            execution,
            base,
            initial,
            geometry,
            tolerance,
            augmented,
            timing,
            maximum,
            best,
            started: _,
        } = &mut self.oracle;
        let execution = execution
            .as_ref()
            .ok_or_else(|| invalid("genetic merit route was not selected"))?;
        let mut evaluator = Evaluator {
            context: Context {
                execution: Some(execution),
                demand: super::super::super::evaluation::ObservationDemand::Fresh,
                ..*context
            },
            state,
            cooperation,
        };
        let augmented = *augmented;
        let timing = *timing;
        let maximum = *maximum;
        let mut merit = async |point: &Point, cooperation: &mut ScalarCooperation| match augmented {
            Some(augmented) => augmented.merit(point, cooperation).await,
            None => Ok(Some(point.f)),
        };
        self.scope
            .run(
                move |_, original| Ok(make(original)),
                async |request| {
                    super::super::custom::measurement::measure(
                        request,
                        super::super::super::evaluation::PointAxis::Rows,
                        base,
                        *initial,
                        super::super::merit::Reuse::RequestedMeasurement,
                        geometry,
                        *tolerance,
                        || timing.stop(maximum).map(Stop::from),
                        Stop::Budget,
                        best,
                        &mut merit,
                        &mut evaluator,
                    )
                    .await
                },
            )
            .await
    }
}
