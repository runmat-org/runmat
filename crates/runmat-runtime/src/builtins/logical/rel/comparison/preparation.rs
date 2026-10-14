use crate::context::preparation::{execute_invocation, BuiltinPreparationAdmission};
use crate::BuiltinResult;
use runmat_builtins::*;
use runmat_value::Value;
use std::future::Future;

#[cfg(test)]
mod tests;

pub(super) async fn execute<F, Fut>(
    context: Option<&crate::context::RuntimeContext>,
    lhs: Value,
    rhs: Value,
    operator: RelationalOperator,
    body: F,
) -> BuiltinResult<crate::execution::RootedValue>
where
    F: FnOnce(Value, Value) -> Fut,
    Fut: Future<Output = BuiltinResult<crate::execution::RootedValue>>,
{
    let entry = match operator {
        RelationalOperator::Equal => &EQ_CATALOG_ENTRY,
        RelationalOperator::NotEqual => &NE_CATALOG_ENTRY,
        RelationalOperator::LessThan => &LT_CATALOG_ENTRY,
        RelationalOperator::LessThanOrEqual => &LE_CATALOG_ENTRY,
        RelationalOperator::GreaterThan => &GT_CATALOG_ENTRY,
        RelationalOperator::GreaterThanOrEqual => &GE_CATALOG_ENTRY,
    };
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
