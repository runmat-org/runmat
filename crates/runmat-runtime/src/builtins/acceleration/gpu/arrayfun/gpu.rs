use crate::BuiltinResult;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    arrayfun_gpu_callback, ArrayfunGpuBinary, ArrayfunGpuCallback, ArrayfunGpuUnary,
};
use runmat_value::Value;

use super::callback::Callable;

pub(super) async fn try_fast_path(
    callable: &Callable,
    inputs: &[Value],
    error_handler: Option<&Callable>,
) -> BuiltinResult<Option<Value>> {
    if inputs.is_empty() || error_handler.is_some() {
        return Ok(None);
    }
    if !inputs
        .iter()
        .all(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Ok(None);
    }

    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    let Some(identity) = callable.builtin_identity() else {
        return Ok(None);
    };
    let Some(operation) = arrayfun_gpu_callback(identity) else {
        return Ok(None);
    };

    let mut handles: Vec<GpuTensorHandle> = Vec::with_capacity(inputs.len());
    for value in inputs {
        if let Value::GpuTensor(handle) = value {
            handles.push(handle.clone());
        }
    }

    if handles.len() >= 2 {
        let base_shape = handles[0].shape.clone();
        if handles
            .iter()
            .skip(1)
            .any(|handle| handle.shape != base_shape)
        {
            return Ok(None);
        }
    }

    let result = match operation {
        ArrayfunGpuCallback::Unary(operation) if handles.len() == 1 => match operation {
            ArrayfunGpuUnary::Sin => provider.unary_sin(&handles[0]).await,
            ArrayfunGpuUnary::Cos => provider.unary_cos(&handles[0]).await,
            ArrayfunGpuUnary::Abs => provider.unary_abs(&handles[0]).await,
            ArrayfunGpuUnary::Exp => provider.unary_exp(&handles[0]).await,
            ArrayfunGpuUnary::Log => provider.unary_log(&handles[0]).await,
            ArrayfunGpuUnary::Sqrt => provider.unary_sqrt(&handles[0]).await,
        },
        ArrayfunGpuCallback::Binary(operation) if handles.len() == 2 => match operation {
            ArrayfunGpuBinary::Add => provider.elem_add(&handles[0], &handles[1]).await,
            ArrayfunGpuBinary::Subtract => provider.elem_sub(&handles[0], &handles[1]).await,
            ArrayfunGpuBinary::Multiply => provider.elem_mul(&handles[0], &handles[1]).await,
            ArrayfunGpuBinary::RightDivide => provider.elem_div(&handles[0], &handles[1]).await,
            ArrayfunGpuBinary::LeftDivide => provider.elem_div(&handles[1], &handles[0]).await,
        },
        ArrayfunGpuCallback::Unary(_) | ArrayfunGpuCallback::Binary(_) => return Ok(None),
    };

    match result {
        Ok(handle) => Ok(Some(Value::GpuTensor(handle))),
        Err(_) => Ok(None),
    }
}
