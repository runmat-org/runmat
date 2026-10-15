//! A selected VM host instruction may bypass automatic promotion only for its
//! exact admitted F32 operands. Ordinary calls continue through native auto.

use super::{promote_binary, BinaryOp};
use runmat_accelerate_api::ProviderExecutionResult;
use runmat_runtime::context::RuntimeContext;
use runmat_value::{NumericDType, Value};

pub async fn promote_binary_in_context(
    runtime: &RuntimeContext,
    op: BinaryOp,
    a: &Value,
    b: &Value,
) -> ProviderExecutionResult<(Value, Value)> {
    if let Some(route) = runtime.vm_fusion_host_route() {
        if route.allows_host_promotion(runtime) {
            let host = |value: &Value| {
                matches!(value, Value::Num(_))
                    || matches!(value, Value::Tensor(tensor) if matches!(tensor.numeric_dtype(), NumericDType::F32 | NumericDType::F64))
            };
            let scalar = |value: &Value| {
                matches!(value, Value::Num(_))
                    || matches!(value, Value::Tensor(tensor) if tensor.len() == 1)
            };
            if host(a)
                && host(b)
                && (matches!(op, BinaryOp::Elementwise)
                    || (matches!(op, BinaryOp::MatMul) && (scalar(a) || scalar(b))))
            {
                return Ok((a.clone(), b.clone()));
            }
            route.invalidate();
        }
    }
    promote_binary(op, a, b).await
}
