//! Domain admission for shape-preserving floating trigonometric data operands.
//! Controls and output prototypes remain in the identity's parsed state.
use crate::builtins::common::gpu_helpers;
use crate::context::preparation::{execute_current_invocation, BuiltinPreparationAdmission};
use crate::{build_runtime_error, BuiltinResult};
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinPreparationFormId};
use runmat_value::{NumericDType, Value};
use std::future::Future;

#[cfg(test)]
mod tests;

fn admits(value: &Value, identity: BuiltinCatalogIdentity) -> BuiltinResult<bool> {
    Ok(match value {
        Value::Num(_) => true,
        Value::Tensor(tensor) => matches!(
            tensor.numeric_dtype(),
            NumericDType::F32 | NumericDType::F64
        ),
        Value::GpuTensor(handle) => {
            if gpu_helpers::exact_provider_for_handle(handle).is_none()
                || !gpu_helpers::expected_handle_numeric_element_type(handle).is_ok_and(|element| {
                    gpu_helpers::numeric_descriptor_matches(
                        handle,
                        element,
                        runmat_accelerate_api::handle_storage(handle),
                    )
                })
            {
                return Err(build_runtime_error(format!(
                    "{}: malformed or unowned GPU input",
                    identity.name
                ))
                .with_builtin(identity.name)
                .with_identifier(format!("RunMat:{}:Internal", identity.name))
                .build());
            }
            runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::Real
        }
        _ => false,
    })
}

pub(crate) async fn unary<F, Fut>(
    identity: BuiltinCatalogIdentity,
    value: Value,
    execute: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<Value>>,
{
    unary_form(identity, value, 0, 1, execute).await
}

pub(crate) async fn unary_form<F, Fut>(
    identity: BuiltinCatalogIdentity,
    value: Value,
    signature: usize,
    input_count: usize,
    execute: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<Value>>,
{
    if !admits(&value, identity)? {
        // These representations use the identity's exact host/complex or
        // owner-restoring path; they are not inputs to a real floating hook.
        return execute(value).await;
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
        |mut operands| async move {
            let value = operands.pop().expect("one admitted trigonometric operand");
            if !admits(&value, identity)? {
                return Err(build_runtime_error(format!(
                    "{}: preparation changed admitted floating input representation",
                    identity.name
                ))
                .with_builtin(identity.name)
                .with_identifier(format!("RunMat:{}:Internal", identity.name))
                .build());
            }
            execute(value).await
        },
    )
    .await
}
