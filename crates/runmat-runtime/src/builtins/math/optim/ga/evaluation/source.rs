//! The numerical fitness loop has one wave source. A borrowed search may stop
//! normally before a wave; the ordinary solver cannot produce such a stop.
use super::super::{common, ScalarCooperation};
use crate::{execution::CallbackBatch, BuiltinResult};
use common::{
    callback_source::Callable,
    population::{projection::Arguments, Counts, Fitness, Invocation, Policy},
};
use futures::FutureExt;
use runmat_value::Value;
use std::{convert::Infallible, ops::ControlFlow};

pub(in crate::builtins::math::optim) struct Routes {
    pub objective: Option<CallbackBatch>,
    pub nonlinear: Option<CallbackBatch>,
}

pub(in crate::builtins::math::optim) enum Wave {
    Dispatched(common::population::Wave),
    /// Already interpreted measurements from the enclosing original oracle.
    /// This is not a fabricated callback return or a supplied-score claim.
    Measured {
        start: usize,
        end: usize,
        values: Fitness,
    },
}
impl Wave {
    pub fn range(&self) -> (usize, usize) {
        match self {
            Self::Dispatched(
                common::population::Wave::Rows { start, end, .. }
                | common::population::Wave::Vectorized { start, end, .. },
            )
            | Self::Measured { start, end, .. } => (*start, *end),
        }
    }
}

pub(in crate::builtins::math::optim) struct Request<'a, P: Arguments> {
    pub policy: Policy,
    pub function: Callable<'a>,
    pub vectorized: bool,
    pub parallel: Option<&'a CallbackBatch>,
    pub counts: &'a mut Counts,
    pub population: &'a P,
    pub selected: &'a [usize],
    pub start: usize,
    pub outputs: usize,
}
pub(in crate::builtins::math::optim) trait Source:
    super::super::callbacks::Invocation
{
    fn retune(&mut self, _maximum: common::global_time::TimeLimit) {}
    fn refresh_counts(
        &self,
        _objective: &mut Counts,
        _nonlinear: &mut Counts,
    ) -> BuiltinResult<()> {
        Ok(())
    }
    /// Only the initial population may inherit an already observed seed.
    /// Later missing or forced rows always go through `next`, even when their
    /// coordinates equal that seed. Explicit supplied scores retain priority.
    async fn seed_initial<P: Arguments>(
        &mut self,
        _population: &P,
        _supplied: &mut [Option<super::fitness::Supplied>],
        _cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        Ok(())
    }
    fn select_routes(
        &mut self,
        objective: &Value,
        nonlinear: Option<&Value>,
        _vectorized: bool,
        parallel: crate::execution::CallbackParallelMode,
        cooperation: &ScalarCooperation,
    ) -> BuiltinResult<Routes> {
        Ok(Routes {
            objective: CallbackBatch::select(cooperation.runtime(), objective, parallel)?,
            nonlinear: nonlinear
                .map(|target| CallbackBatch::select(cooperation.runtime(), target, parallel))
                .transpose()?
                .flatten(),
        })
    }
    fn next<'a, P: Arguments + 'a>(
        &'a mut self,
        request: Request<'a, P>,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<ControlFlow<Self::Stop, Wave>>> + 'a;
}
pub(in crate::builtins::math::optim) use super::super::callbacks::Direct;
impl<S: Source> Source for &mut S {
    fn retune(&mut self, maximum: common::global_time::TimeLimit) {
        (**self).retune(maximum);
    }
    fn refresh_counts(&self, objective: &mut Counts, nonlinear: &mut Counts) -> BuiltinResult<()> {
        (**self).refresh_counts(objective, nonlinear)
    }
    async fn seed_initial<P: Arguments>(
        &mut self,
        population: &P,
        supplied: &mut [Option<super::fitness::Supplied>],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        (**self)
            .seed_initial(population, supplied, cooperation)
            .await
    }
    fn select_routes(
        &mut self,
        objective: &Value,
        nonlinear: Option<&Value>,
        vectorized: bool,
        parallel: crate::execution::CallbackParallelMode,
        cooperation: &ScalarCooperation,
    ) -> BuiltinResult<Routes> {
        (**self).select_routes(objective, nonlinear, vectorized, parallel, cooperation)
    }
    fn next<'a, P: Arguments + 'a>(
        &'a mut self,
        request: Request<'a, P>,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<ControlFlow<S::Stop, Wave>>> + 'a {
        (**self).next(request, cooperation)
    }
}
impl Source for Direct {
    fn next<'a, P: Arguments + 'a>(
        &'a mut self,
        request: Request<'a, P>,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<ControlFlow<Infallible, Wave>>> + 'a {
        Invocation {
            policy: request.policy,
            function: request.function,
            vectorized: request.vectorized,
            parallel: request.parallel,
            counts: request.counts,
            cooperation,
        }
        .next(
            request.population,
            request.selected,
            request.start,
            request.outputs,
        )
        .map(|returned| returned.map(|wave| ControlFlow::Continue(Wave::Dispatched(wave))))
    }
}
