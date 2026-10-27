//! Active GA callbacks bind through the actual selected program producer.
//! No worker eligibility, new registry or synthetic source identity is used.
use super::{inputs::Problem, ScalarCooperation};
use crate::{call::descriptor::RepeatableCallable, execution::RootedValue, BuiltinResult};
use runmat_value::Value;
mod invocation;
pub(super) use invocation::take_direct;
pub(in crate::builtins::math::optim) use invocation::{Direct, Event, Invocation};
pub(super) struct Sources {
    pub objective: RepeatableCallable,
    pub nonlinear: Option<RepeatableCallable>,
}
pub(super) fn unavailable(message: &str) -> crate::RuntimeError {
    let mut error = crate::runtime_error::semantic_error(
        "RunMat:Execution:SelectedCallableUnavailable",
        message,
    );
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
pub(super) fn select(
    value: &Value,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<RepeatableCallable> {
    let runtime = cooperation
        .runtime()
        .ok_or_else(|| unavailable("ga callbacks require their selected execution context"))?;
    let program = runtime.selected_graph_program().ok_or_else(|| {
        unavailable("ga callbacks require original selected-program source admission")
    })?;
    let callsite = {
        let _scope = runtime.enter();
        crate::callsite::current_function_input_callsite()
            .or_else(crate::callsite::current_callsite)
    };
    program
        .select_repeatable_callable(runtime, RootedValue::in_context(value.clone(), runtime)?)
        .map(|selected| selected.with_callsite(callsite))
}
impl Sources {
    pub fn select(problem: &Problem, cooperation: &ScalarCooperation) -> BuiltinResult<Self> {
        Ok(Self {
            objective: select(&problem.objective, cooperation)?,
            nonlinear: problem
                .nonlinear
                .as_ref()
                .map(|value| select(value, cooperation))
                .transpose()?,
        })
    }
}
#[cfg(test)]
pub(super) mod fixture {
    use super::*;
    pub fn target(
        runtime: &crate::context::RuntimeContext,
        fixtures: &crate::user_functions::test_support::ExactNamedFixtures,
        name: &str,
    ) -> RepeatableCallable {
        struct NoTextLookup;
        impl crate::call::descriptor::FunctionNameResolver for NoTextLookup {
            fn resolve_function(&self, _: &str) -> Option<runmat_types::FunctionId> {
                None
            }
        }
        let value =
            crate::user_functions::test_support::exact_local_value(name, fixtures.binding(name));
        RepeatableCallable::select_rooted(
            Some(runtime),
            RootedValue::in_context(value, runtime).unwrap(),
            &NoTextLookup,
        )
        .unwrap()
    }
    pub fn sources(
        runtime: &crate::context::RuntimeContext,
        fixtures: &crate::user_functions::test_support::ExactNamedFixtures,
        objective: &str,
        nonlinear: Option<&str>,
    ) -> Sources {
        Sources {
            objective: target(runtime, fixtures, objective),
            nonlinear: nonlinear.map(|name| target(runtime, fixtures, name)),
        }
    }
}
