//! Automatic preparation after the owning rounding form has admitted its inputs.
use crate::context::preparation::{execute_current_invocation, BuiltinPreparationAdmission};
use crate::BuiltinResult;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogEntry, BuiltinPreparationFormId};
use runmat_value::{NumericDType, Value};
use std::future::Future;

pub(super) async fn unary<F, Fut>(
    entry: &'static BuiltinCatalogEntry,
    value: Value,
    body: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<Value>>,
{
    // These are the real dense forms served by the unary provider hooks.
    // Integer identity, complex, tabular and resident forms keep their own body.
    if !matches!(&value, Value::Tensor(tensor) if matches!(tensor.numeric_dtype(), NumericDType::F32 | NumericDType::F64))
    {
        return body(value).await;
    }
    execute_current_invocation(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count: 1,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![value],
        |mut values| body(values.remove(0)),
    )
    .await
}
