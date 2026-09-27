//! Preparation of an admitted dense floating pair. Controls stay with the caller.
use crate::builtins::common::broadcast::BroadcastPlan;
use crate::context::preparation::{execute_current_invocation, BuiltinPreparationAdmission};
use crate::BuiltinResult;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinPreparationFormId};
use runmat_value::{NumericDType, Value};
use std::future::Future;

pub(super) async fn execute<F, Fut>(
    identity: BuiltinCatalogIdentity,
    input_count: usize,
    left: Value,
    right: Value,
    body: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Value, Value) -> Fut,
    Fut: Future<Output = BuiltinResult<Value>>,
{
    let admitted = input_count == 2
        && match (&left, &right) {
            (Value::Tensor(a), Value::Tensor(b)) => {
                matches!(a.numeric_dtype(), NumericDType::F32 | NumericDType::F64)
                    && matches!(b.numeric_dtype(), NumericDType::F32 | NumericDType::F64)
                    && BroadcastPlan::new(&a.shape, &b.shape).is_ok()
            }
            _ => false,
        };
    if !admitted {
        // The owning implementation supplies its established error and handles
        // the other representations. In particular, invalid shapes never upload.
        return body(left, right).await;
    }
    execute_current_invocation(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![left, right],
        |mut values| {
            let right = values.remove(1);
            let left = values.remove(0);
            body(left, right)
        },
    )
    .await
}
