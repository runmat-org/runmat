//! Preparation of an admitted dense floating pair. Controls stay with the caller.
use crate::builtins::common::broadcast::BroadcastPlan;
use crate::context::preparation::{execute_invocation, BuiltinPreparationAdmission};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinPreparationFormId};
use runmat_value::{NumericDType, Value};
use std::future::Future;

pub(super) async fn execute<F, Fut>(
    context: Option<&RuntimeContext>,
    identity: BuiltinCatalogIdentity,
    input_count: usize,
    left: Value,
    right: Value,
    body: F,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value, Value) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
{
    execute_in_context(context, identity, input_count, left, right, body).await
}

/// Explicit context-bearing version for canonical bindings. The preparation
/// service receives the same instruction-scoped context as the selected body.
pub(super) async fn execute_in_context<F, Fut>(
    context: Option<&RuntimeContext>,
    identity: BuiltinCatalogIdentity,
    input_count: usize,
    left: Value,
    right: Value,
    body: F,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value, Value) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
{
    let _inputs = RootedValueLease::for_values([&left, &right].into_iter())
        .map_err(|cause| cause.into_runtime_error())?;
    let admitted = input_count == 2
        && match (&left, &right) {
            (Value::Tensor(a), Value::Tensor(b)) => {
                matches!(a.numeric_dtype(), NumericDType::F32 | NumericDType::F64)
                    && matches!(b.numeric_dtype(), NumericDType::F32 | NumericDType::F64)
                    && BroadcastPlan::new(a.shape(), b.shape()).is_ok()
            }
            _ => false,
        };
    if !admitted {
        // The owning implementation supplies its established error and handles
        // the other representations. In particular, invalid shapes never upload.
        return body(left, right).await;
    }
    let admission = BuiltinPreparationAdmission {
        binding: BuiltinBindingIdentity {
            builtin: identity,
            variant: "default",
        },
        form: BuiltinPreparationFormId(0),
        signature: 0,
        input_count,
        requested_outputs: crate::output_count::current_output_count(),
    };
    let execute = |mut values: Vec<Value>| {
        let right = values.remove(1);
        let left = values.remove(0);
        body(left, right)
    };
    execute_invocation(context, admission, vec![left, right], execute).await
}
