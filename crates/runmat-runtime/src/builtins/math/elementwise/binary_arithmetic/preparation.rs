//! Preparation of an admitted dense floating pair. Controls stay with the caller.
use crate::builtins::common::broadcast::BroadcastPlan;
use crate::context::preparation::{
    execute_admitted, execute_current_invocation, AdmittedBuiltinPreparation,
    BuiltinPreparationAdmission,
};
use crate::context::RuntimeContext;
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
    execute_in_context(None, identity, input_count, left, right, body).await
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
    if let Some(context) = context.filter(|context| context.vm_fusion_host_route().is_some()) {
        let request = AdmittedBuiltinPreparation::new(admission, vec![left, right])?;
        execute_admitted(context, request, execute).await
    } else {
        execute_current_invocation(admission, vec![left, right], execute).await
    }
}
