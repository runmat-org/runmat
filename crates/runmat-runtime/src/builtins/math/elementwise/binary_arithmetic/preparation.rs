//! Prepare metadata once, then retire it before polling the selected body.
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_builtins::BuiltinCatalogIdentity;
use runmat_value::Value;
use std::future::Future;
pub(crate) mod managed;
mod selected_execution;
mod selection;

#[inline(always)]
pub(super) fn execute<'a, F, Fut>(
    context: Option<&'a RuntimeContext>,
    identity: BuiltinCatalogIdentity,
    operator: runmat_types::OperatorKind,
    input_count: usize,
    left: Value,
    right: Value,
    body: F,
) -> impl Future<Output = BuiltinResult<RootedValue>> + 'a
where
    F: FnOnce(Value, Value) -> Fut + 'a,
    Fut: Future<Output = BuiltinResult<RootedValue>> + 'a,
{
    execute_in_context(context, identity, operator, input_count, left, right, body)
}

/// The exact instruction-scoped context is borrowed by selection and execution;
/// no preparation context or allocation authority is reconstructed.
#[inline(always)]
pub(super) fn execute_in_context<'a, F, Fut>(
    context: Option<&'a RuntimeContext>,
    identity: BuiltinCatalogIdentity,
    operator: runmat_types::OperatorKind,
    input_count: usize,
    left: Value,
    right: Value,
    body: F,
) -> impl Future<Output = BuiltinResult<RootedValue>> + 'a
where
    F: FnOnce(Value, Value) -> Fut + 'a,
    Fut: Future<Output = BuiltinResult<RootedValue>> + 'a,
{
    crate::execution::after_result(
        futures::future::lazy(move |_| {
            selection::prepare(context, identity, operator, input_count, left, right)
        }),
        move |selected| selected_execution::execute(context, identity, operator, selected, body),
    )
}
