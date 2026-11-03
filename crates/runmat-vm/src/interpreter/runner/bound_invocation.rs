//! Select the concrete bound-call future inside its existing admitted owner.
//! Pure registry selection uses the exact supplied definition authority;
//! effectful prior-invoker admission stays in its original polling context.
use super::{FunctionRegistry, ProcessDriverHandle, VmResult};
use futures::future::Either;
use runmat_runtime::{
    call::RootedFunctionArguments, context::RuntimeContext, execution::RootedValueSequence,
    user_functions::BoundFunctionInvoker,
};
use runmat_value::{BoundFunctionAuthority, BoundFunctionBinding, LexicalEnvironment};
use std::{future::Future, rc::Rc};

pub(crate) struct Invocation {
    pub runtime: RuntimeContext,
    pub binding: BoundFunctionBinding,
    pub args: RootedFunctionArguments,
    pub requested_outputs: usize,
    pub driver: Option<ProcessDriverHandle>,
    pub lexical_environment: Option<LexicalEnvironment>,
}

// Construct at the admitted destination rather than adding a future-sized
// by-value return frame beneath an active numerical callback.
#[inline(always)]
pub(super) fn prepare(
    invocation: Invocation,
    registry: Rc<FunctionRegistry>,
    previous: Option<Rc<BoundFunctionInvoker>>,
) -> impl Future<Output = VmResult<RootedValueSequence>> {
    let function = runmat_hir::FunctionId(invocation.binding.function);
    let current_portable = match &invocation.binding.authority {
        BoundFunctionAuthority::Portable { origin } => registry
            .get_current(function)
            .is_some_and(|body| body.binding_origin == Some(*origin)),
        BoundFunctionAuthority::Local { .. } | BoundFunctionAuthority::SelectedNative { .. } => {
            false
        }
    };
    let selected = if current_portable {
        Some(registry.as_ref().clone())
    } else {
        registry.owned_view(function, &invocation.binding.authority)
    };
    if let Some(selected) = selected {
        Either::Left(super::super_constructor::invoke_selected_bound(
            invocation.runtime,
            invocation.binding,
            invocation.args,
            invocation.requested_outputs,
            selected,
            invocation.driver,
            invocation.lexical_environment,
        ))
    } else {
        Either::Right(await_previous(invocation, previous))
    }
}

async fn await_previous(
    invocation: Invocation,
    previous: Option<Rc<BoundFunctionInvoker>>,
) -> VmResult<RootedValueSequence> {
    let Some(previous) = previous else {
        return Err(super::mex(
            "UnavailableBoundFunction",
            "bound function definition is unavailable in this execution owner",
        ));
    };
    let Invocation {
        runtime,
        binding,
        args,
        requested_outputs,
        driver,
        lexical_environment,
    } = invocation;
    let selected_runtime = runtime.clone();
    let environment_roots = if driver.is_some() {
        lexical_environment
            .as_ref()
            .map(runmat_runtime::call::lexical::root_environment)
            .transpose()?
    } else {
        None
    };
    super::await_prior_bound_invoker(driver, selected_runtime, environment_roots, move || {
        previous(
            runtime,
            binding,
            args,
            requested_outputs,
            lexical_environment,
        )
    })
    .await
}
