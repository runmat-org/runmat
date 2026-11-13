use crate::context::preparation::{execute_invocation, BuiltinPreparationAdmission};
use crate::BuiltinResult;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogEntry, BuiltinPreparationFormId};
use runmat_value::Value;
use std::future::Future;

#[cfg(test)]
mod tests;

pub(super) async fn binary<F, Fut>(
    context: Option<&crate::context::RuntimeContext>,
    entry: &'static BuiltinCatalogEntry,
    lhs: Value,
    rhs: Value,
    body: F,
) -> BuiltinResult<crate::execution::RootedValue>
where
    F: FnOnce(Value, Value) -> Fut,
    Fut: Future<Output = BuiltinResult<crate::execution::RootedValue>>,
{
    execute_invocation(
        context,
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count: 2,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![lhs, rhs],
        |mut operands| {
            let rhs = operands
                .pop()
                .expect("preparation preserves two admitted operands");
            let lhs = operands
                .pop()
                .expect("preparation preserves two admitted operands");
            body(lhs, rhs)
        },
    )
    .await
}
