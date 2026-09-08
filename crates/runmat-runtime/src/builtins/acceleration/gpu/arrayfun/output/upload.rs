use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;
use runmat_accelerate_api::set_handle_logical;
use runmat_value::{Tensor, Value};

use super::super::error::arrayfun_flow;

pub(in crate::builtins::acceleration::gpu::arrayfun) fn maybe_upload_uniform(
    value: Value,
    has_gpu_input: bool,
) -> BuiltinResult<Value> {
    if !has_gpu_input {
        return Ok(value);
    }
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(value),
    };

    match value {
        Value::Tensor(tensor) => {
            let handle = gpu_helpers::upload_tensor(provider, &tensor)
                .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::LogicalArray(logical) => {
            let data: Vec<f64> = logical
                .data
                .iter()
                .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
                .collect();
            let tensor = Tensor::new(data, logical.shape.clone())
                .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
            let handle = gpu_helpers::upload_tensor(provider, &tensor)
                .map_err(|e| arrayfun_flow(format!("arrayfun: {e}")))?;
            set_handle_logical(&handle, true);
            Ok(Value::GpuTensor(handle))
        }
        other => Ok(other),
    }
}
