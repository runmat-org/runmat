//! Construct one callback route before polling it. Sibling routes do not
//! enlarge the active callback frame or acquire another continuation owner.
use super::{CallableCallError, MappedCallable};
use crate::{call::BuiltinArguments, context::RuntimeContext, execution::RootedValueSequence};
use futures::{future::Either, FutureExt};
use runmat_types::{BuiltinId, CallableFallbackPolicy, CallableIdentity, SymbolName};
use runmat_value::Value;
use std::future::Future;

type SequenceResult = std::result::Result<RootedValueSequence, CallableCallError>;

#[inline(never)]
pub(super) fn execute<'a>(
    callable: &'a MappedCallable,
    context: Option<&'a RuntimeContext>,
    arguments: BuiltinArguments,
    closure_owner: Option<crate::call::RetainedCallableTarget>,
    requested_outputs: usize,
) -> impl Future<Output = std::result::Result<crate::execution::RootedValue, CallableCallError>> + 'a
{
    let future = match callable {
        MappedCallable::Scoped(handle) => Either::Left(
            crate::call::feval::invoke_retained(
                context,
                Value::ScopedFunctionHandle(handle.clone()),
                arguments,
                requested_outputs,
                None,
            )
            .map(|result| result.map_err(CallableCallError::Runtime)),
        ),
        MappedCallable::Closure(closure) => Either::Right(Either::Left(closure_call(
            context,
            closure,
            closure_owner.expect("selected closure owner"),
            arguments,
            requested_outputs,
        ))),
        MappedCallable::ExternalName { name } => Either::Right(Either::Right(Either::Left(
            external(context, name, arguments, requested_outputs),
        ))),
        MappedCallable::Builtin { identity } => Either::Right(Either::Right(Either::Right(named(
            context,
            CallableIdentity::Builtin(BuiltinId(identity.name.into())),
            arguments,
            requested_outputs,
        )))),
        MappedCallable::DynamicName { name } => Either::Right(Either::Right(Either::Right(named(
            context,
            CallableIdentity::DynamicName(SymbolName(name.clone())),
            arguments,
            requested_outputs,
        )))),
    };
    future.map(|result| {
        result.and_then(|sequence| {
            crate::call::arguments::project_rooted_legacy_builtin_value_abi(sequence)
                .map_err(CallableCallError::Runtime)
        })
    })
}

async fn closure_call(
    context: Option<&RuntimeContext>,
    closure: &crate::call::descriptor::RetainedClosureValue,
    owner: crate::call::RetainedCallableTarget,
    arguments: BuiltinArguments,
    requested_outputs: usize,
) -> SequenceResult {
    crate::call::descriptor::invoke_selected_retained_closure(
        context,
        closure,
        owner,
        arguments,
        requested_outputs,
    )
    .await
    .map_err(CallableCallError::Runtime)?
    .ok_or_else(|| CallableCallError::SemanticUnavailable {
        function_name: closure.as_closure().function_name.clone(),
        function: closure
            .as_closure()
            .binding
            .as_ref()
            .expect("only a bound closure can decline")
            .function
            .to_string(),
    })
}

async fn external(
    context: Option<&RuntimeContext>,
    name: &str,
    arguments: BuiltinArguments,
    requested_outputs: usize,
) -> SequenceResult {
    let identity = crate::external_callable_identity_for_name(name);
    let request = crate::user_functions::CallableRequest::rooted_resolved(
        identity.clone(),
        CallableFallbackPolicy::ExternalBoundary,
        arguments
            .into_supplied_function_arguments(context)
            .map_err(CallableCallError::Runtime)?,
        requested_outputs,
    )
    .with_selected_runtime(context);
    match crate::user_functions::try_call_semantic_descriptor_owned(request).await {
        std::ops::ControlFlow::Break(result) => result.map_err(CallableCallError::Runtime),
        std::ops::ControlFlow::Continue(_) => {
            Err(CallableCallError::UndefinedExternal { identity })
        }
    }
}

fn named(
    context: Option<&RuntimeContext>,
    identity: CallableIdentity,
    arguments: BuiltinArguments,
    requested_outputs: usize,
) -> impl Future<Output = SequenceResult> + '_ {
    crate::dispatch_retained_callable_with_policy_in_runtime(
        context,
        identity,
        CallableFallbackPolicy::RuntimeNameResolution,
        arguments,
        requested_outputs,
        None,
    )
    .map(|result| result.map_err(CallableCallError::Runtime))
}
