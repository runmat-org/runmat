mod host;

use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::BuiltinResult;

use super::{provider, BUILTIN_NAME};

pub(super) async fn evaluate(significand: Value, exponent: Value) -> BuiltinResult<Value> {
    match (significand, exponent) {
        (Value::GpuTensor(left), Value::GpuTensor(right)) => {
            if let Some(output) = provider::try_binary_direct(&left, &right)? {
                return Ok(output);
            }
            let left = gather(left).await?;
            let right = gather(right).await?;
            host::evaluate(left, right)
        }
        (Value::GpuTensor(left), right) => host::evaluate(gather(left).await?, right),
        (left, Value::GpuTensor(right)) => host::evaluate(left, gather(right).await?),
        (left, right) => host::evaluate(left, right),
    }
}

async fn gather(handle: runmat_accelerate_api::GpuTensorHandle) -> BuiltinResult<Value> {
    gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))
}
