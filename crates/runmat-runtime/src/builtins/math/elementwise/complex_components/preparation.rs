//! Preparation after the component domain admits a dense real floating array.
use crate::context::preparation::{execute_invocation, BuiltinPreparationAdmission};
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinPreparationFormId};
use runmat_value::{NumericDType, Value};
use std::future::Future;

pub(super) async fn execute<F, Fut>(
    runtime: Option<&RuntimeContext>,
    identity: BuiltinCatalogIdentity,
    value: Value,
    body: F,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
{
    if !matches!(&value, Value::Tensor(t) if matches!(t.numeric_dtype(), NumericDType::F32 | NumericDType::F64))
    {
        return body(value).await;
    }
    execute_invocation(
        runtime,
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
        |mut values| body(values.remove(0)),
    )
    .await
}
