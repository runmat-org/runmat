use crate::builtins::common::integer_conversion::{admits_preparation, cast_value, CastError};
use crate::context::preparation::{execute_current_invocation, BuiltinPreparationAdmission};
use crate::{BuiltinResult, RuntimeError};
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinPreparationFormId};
use runmat_types::IntegerClass;
use runmat_value::Value;

pub(super) async fn execute(
    identity: BuiltinCatalogIdentity,
    value: Value,
    target: IntegerClass,
    error: fn(CastError) -> RuntimeError,
) -> BuiltinResult<Value> {
    if !admits_preparation(&value).map_err(error)? {
        return cast_value(value, target).await.map_err(error);
    }
    execute_current_invocation(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count: 1,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![value],
        |mut operands| async move {
            // The shared carrier checks the count. Re-admit inside cast_value:
            // an injected service may intentionally transform its operand.
            cast_value(operands.pop().expect("one admitted operand"), target)
                .await
                .map_err(error)
        },
    )
    .await
}
