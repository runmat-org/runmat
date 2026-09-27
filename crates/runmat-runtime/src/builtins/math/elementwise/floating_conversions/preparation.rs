//! Unary data preparation after the owning conversion admits its output template.
use crate::builtins::common::gpu_helpers;
use crate::context::preparation::{execute_current_invocation, BuiltinPreparationAdmission};
use crate::BuiltinResult;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinPreparationFormId};
use runmat_value::Value;
use std::future::Future;

#[cfg(test)]
mod tests;

pub(super) async fn execute<F, Fut>(
    identity: BuiltinCatalogIdentity,
    value: Value,
    signature: usize,
    input_count: usize,
    body: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<Value>>,
{
    // Text, sparse, complex and symbolic conversions retain their specialized
    // runtime-owned representation routes. Unsupported values reach the same
    // domain error without calling a preparation service.
    let admitted = match &value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Tensor(_)
        | Value::LogicalArray(_) => true,
        Value::GpuTensor(handle) => {
            gpu_helpers::exact_provider_for_handle(handle).is_some()
                && gpu_helpers::expected_handle_numeric_element_type(handle).is_ok()
                && handle.descriptor.storage == Some(runmat_accelerate_api::GpuTensorStorage::Real)
        }
        _ => false,
    };
    if !admitted {
        return body(value).await;
    }
    execute_current_invocation(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature,
            input_count,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![value],
        |mut operands| body(operands.pop().expect("one admitted conversion operand")),
    )
    .await
}
