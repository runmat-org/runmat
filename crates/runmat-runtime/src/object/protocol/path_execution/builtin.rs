//! Default indexing after registered-target selection. Bypass the original
//! receiver's protocol once; nested receivers keep ordinary dispatch.
use super::*;

pub(crate) fn read_builtin_subscript_path<'a>(
    runtime: Option<&'a RuntimeContext>,
    base: Value,
    path: ObjectSubscriptPath,
    request: SubscriptReadRequest,
) -> crate::execution::RuntimeContinuationFuture<'a, RootedValueSequence> {
    super::super::context::admit_future_with(runtime, move || async move {
        let steps = path.into_steps_in_context(runtime)?.into_iter();
        read_default_steps(
            runtime,
            base,
            steps,
            request,
            None,
            ProtocolDispatch::Default,
        )
        .await
    })
}

pub(crate) fn assign_builtin_subscript_path<'a>(
    runtime: Option<&'a RuntimeContext>,
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
    access: super::super::ObjectAccessContext,
) -> crate::execution::RuntimeContinuationFuture<'a, RootedValue> {
    super::super::context::admit_future_with(runtime, move || async move {
        assign_default_steps(
            runtime,
            base,
            path.into_steps_in_context(runtime)?.into_iter(),
            values,
            None,
            &access,
        )
        .await
    })
}
