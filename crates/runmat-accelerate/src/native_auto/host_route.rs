//! VM host instructions and selected canonical builtins retain separate
//! execution owners. Generic automatic promotion has neither authority.

use super::{promote_binary, promote_unary, BinaryOp, UnaryOp};
use runmat_accelerate_api::ProviderExecutionResult;
use runmat_runtime::context::RuntimeContext;
use runmat_value::{NumericDType, Value};

pub async fn promote_binary_in_context(
    runtime: &RuntimeContext,
    op: BinaryOp,
    a: &Value,
    b: &Value,
) -> ProviderExecutionResult<(Value, Value)> {
    promote_binary_with_owner(runtime, op, a, b, BinaryExecutionOwner::Generic).await
}

/// A VM arithmetic instruction has already selected its canonical async
/// builtin. Once an operand is resident, that builtin owns the exact provider
/// hook and any pre-effect host continuation; generic auto-offload has no
/// operation-specific readback or output contract to substitute for it.
pub async fn promote_builtin_binary_in_context(
    runtime: &RuntimeContext,
    op: BinaryOp,
    a: &Value,
    b: &Value,
) -> ProviderExecutionResult<(Value, Value)> {
    promote_binary_with_owner(runtime, op, a, b, BinaryExecutionOwner::CanonicalBuiltin).await
}

#[derive(Clone, Copy)]
enum BinaryExecutionOwner {
    Generic,
    CanonicalBuiltin,
}

/// The canonical unary builtin owns resident execution and any legal host
/// continuation. Generic promotion cannot replace that operation's contract.
pub async fn promote_builtin_unary_in_context(
    runtime: &RuntimeContext,
    op: UnaryOp,
    value: &Value,
) -> ProviderExecutionResult<Value> {
    if let Some(route) = runtime.vm_fusion_host_route() {
        if route.allows_host_promotion(runtime) {
            if host_floating(value) {
                return Ok(value.clone());
            }
            route.invalidate();
        }
    }
    if matches!(value, Value::GpuTensor(_)) {
        return Ok(value.clone());
    }
    promote_unary(op, value).await
}

fn host_floating(value: &Value) -> bool {
    matches!(value, Value::Num(_))
        || matches!(value, Value::Tensor(tensor) if matches!(tensor.numeric_dtype(), NumericDType::F32 | NumericDType::F64))
}

async fn promote_binary_with_owner(
    runtime: &RuntimeContext,
    op: BinaryOp,
    a: &Value,
    b: &Value,
    owner: BinaryExecutionOwner,
) -> ProviderExecutionResult<(Value, Value)> {
    if let Some(route) = runtime.vm_fusion_host_route() {
        if route.allows_host_promotion(runtime) {
            let scalar = |value: &Value| {
                matches!(value, Value::Num(_))
                    || matches!(value, Value::Tensor(tensor) if tensor.len() == 1)
            };
            if host_floating(a)
                && host_floating(b)
                && (matches!(op, BinaryOp::Elementwise)
                    || (matches!(op, BinaryOp::MatMul) && (scalar(a) || scalar(b))))
            {
                return Ok((a.clone(), b.clone()));
            }
            route.invalidate();
        }
    }
    if matches!(owner, BinaryExecutionOwner::CanonicalBuiltin)
        && (matches!(a, Value::GpuTensor(_)) || matches!(b, Value::GpuTensor(_)))
    {
        return Ok((a.clone(), b.clone()));
    }
    promote_binary(op, a, b).await
}
