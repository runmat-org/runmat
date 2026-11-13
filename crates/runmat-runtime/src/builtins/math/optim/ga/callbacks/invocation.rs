//! One canonical invocation adapter, specialized statically by its owner.
//! Direct calls cannot produce an auxiliary-domain stop.
use super::super::ScalarCooperation;
use crate::{
    call::{descriptor::RepeatableCallable, BuiltinArguments},
    execution::RootedValueSequence,
    BuiltinResult,
};
use futures::FutureExt;
use std::{convert::Infallible, ops::ControlFlow};

#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) enum Event {
    Work,
    Observer(runmat_builtins::OptimGeneticObserverState),
}

pub(in crate::builtins::math::optim) trait Invocation {
    type Stop: 'static;
    fn stop(&self, _event: Event) -> Option<Self::Stop> {
        None
    }
    fn protocol<'a, T: 'a, F: std::future::Future<Output = BuiltinResult<T>> + 'a>(
        &'a mut self,
        make: impl FnOnce(Option<crate::context::RuntimeContext>) -> F + 'a,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<ControlFlow<Self::Stop, T>>> + 'a;
    async fn projection<'a>(
        &'a mut self,
        prepared: super::super::common::population::projection::Prepared<'a>,
        cooperation: &'a mut ScalarCooperation,
    ) -> BuiltinResult<
        ControlFlow<Self::Stop, super::super::common::population::projection::Argument>,
    > {
        use super::super::common::population::projection::Prepared;
        match prepared {
            Prepared::Ready(argument) => Ok(ControlFlow::Continue(argument)),
            prepared @ Prepared::Subscript { .. } => {
                self.protocol(move |runtime| prepared.execute(runtime), cooperation)
                    .await
            }
        }
    }
    fn call<'a>(
        &'a mut self,
        target: &'a RepeatableCallable,
        arguments: BuiltinArguments,
        outputs: usize,
        event: Event,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<ControlFlow<Self::Stop, RootedValueSequence>>>
           + 'a;
}
pub(in crate::builtins::math::optim) struct Direct;
impl<I: Invocation> Invocation for &mut I {
    type Stop = I::Stop;
    fn stop(&self, event: Event) -> Option<Self::Stop> {
        (**self).stop(event)
    }
    fn protocol<'a, T: 'a, F: std::future::Future<Output = BuiltinResult<T>> + 'a>(
        &'a mut self,
        make: impl FnOnce(Option<crate::context::RuntimeContext>) -> F + 'a,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<ControlFlow<I::Stop, T>>> + 'a {
        (**self).protocol(make, cooperation)
    }
    fn call<'a>(
        &'a mut self,
        target: &'a RepeatableCallable,
        arguments: BuiltinArguments,
        outputs: usize,
        event: Event,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<ControlFlow<I::Stop, RootedValueSequence>>> + 'a
    {
        (**self).call(target, arguments, outputs, event, cooperation)
    }
}
impl Invocation for Direct {
    type Stop = Infallible;
    fn protocol<'a, T: 'a, F: std::future::Future<Output = BuiltinResult<T>> + 'a>(
        &'a mut self,
        make: impl FnOnce(Option<crate::context::RuntimeContext>) -> F + 'a,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<ControlFlow<Infallible, T>>> + 'a {
        make(cooperation.runtime().cloned()).map(|returned| returned.map(ControlFlow::Continue))
    }
    fn call<'a>(
        &'a mut self,
        target: &'a RepeatableCallable,
        arguments: BuiltinArguments,
        outputs: usize,
        _event: Event,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<ControlFlow<Infallible, RootedValueSequence>>>
           + 'a {
        target
            .invoke_retained(cooperation.runtime(), arguments, outputs)
            .map(|returned| returned.map(ControlFlow::Continue))
    }
}
pub(in super::super) fn take_direct<T>(value: ControlFlow<Infallible, T>) -> T {
    match value {
        ControlFlow::Continue(value) => value,
        ControlFlow::Break(never) => match never {},
    }
}
