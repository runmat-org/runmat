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
        if route.allows_host_promotion(runtime) && matches!(op, BinaryOp::Elementwise) {
            if matches!((a, b),
                (Value::Tensor(left), Value::Tensor(right))
                    if left.numeric_dtype() == NumericDType::F32
                        && right.numeric_dtype() == NumericDType::F32)
            {
                return Ok((a.clone(), b.clone()));
            }
            route.invalidate();
        }
    }
    promote_binary(op, a, b).await
}
