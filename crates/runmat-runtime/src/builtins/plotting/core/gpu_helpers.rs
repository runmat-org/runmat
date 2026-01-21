//! Shared helpers for GPU-aware plotting builtins.

use futures::executor::block_on;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::{call_builtin_async, gather_if_needed_async, value_contains_gpu, BuiltinResult};

use super::plotting_error;

#[cfg(feature = "plot-core")]
use glam::Vec3;
#[cfg(feature = "plot-core")]
use runmat_plot::core::BoundingBox;

/// Compute the min/max bounds for a GPU tensor by delegating to the runtime
/// `min`/`max` builtins. Results are returned as `f32` so they can flow directly
/// into plotting bounding boxes.
pub async fn axis_bounds_async(
    handle: &GpuTensorHandle,
    context: &'static str,
) -> BuiltinResult<(f32, f32)> {
    let min_value = call_builtin_async("min", &[Value::GpuTensor(handle.clone())])
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, context))?;
    let max_value = call_builtin_async("max", &[Value::GpuTensor(handle.clone())])
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, context))?;
    let min_scalar = value_to_scalar_async(min_value, context).await?;
    let max_scalar = value_to_scalar_async(max_value, context).await?;
    Ok((min_scalar as f32, max_scalar as f32))
}

pub fn axis_bounds(handle: &GpuTensorHandle, context: &'static str) -> BuiltinResult<(f32, f32)> {
    block_on(axis_bounds_async(handle, context))
}

/// Gather a GPU tensor handle into host memory.
pub async fn gather_tensor_from_gpu_async(
    handle: GpuTensorHandle,
    name: &'static str,
) -> BuiltinResult<Tensor> {
    let value = Value::GpuTensor(handle);
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, name))?;
    Tensor::try_from(&gathered).map_err(|e| plotting_error(name, format!("{name}: {e}")))
}

pub fn gather_tensor_from_gpu(handle: GpuTensorHandle, name: &'static str) -> BuiltinResult<Tensor> {
    block_on(gather_tensor_from_gpu_async(handle, name))
}

/// Convert a runtime value (potentially GPU-resident) into a concrete scalar.
pub async fn value_to_scalar_async(mut value: Value, context: &'static str) -> BuiltinResult<f64> {
    if value_contains_gpu(&value) {
        value = gather_if_needed_async(&value)
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, context))?;
    }
    match value {
        Value::Num(n) => Ok(n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(t) => {
            t.data.first().copied().ok_or_else(|| {
                plotting_error(context, format!("{context}: expected scalar result"))
            })
        }
        _ => Err(plotting_error(
            context,
            format!("{context}: expected numeric scalar result"),
        )),
    }
}

/// Build a bounding box from GPU-resident X/Y vectors.
#[cfg(feature = "plot-core")]
pub async fn gpu_xy_bounds_async(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    context: &'static str,
) -> BuiltinResult<BoundingBox> {
    let (min_x, max_x) = axis_bounds_async(x, context).await?;
    let (min_y, max_y) = axis_bounds_async(y, context).await?;
    Ok(BoundingBox::new(
        Vec3::new(min_x, min_y, 0.0),
        Vec3::new(max_x, max_y, 0.0),
    ))
}

#[cfg(feature = "plot-core")]
pub fn gpu_xy_bounds(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    context: &'static str,
) -> BuiltinResult<BoundingBox> {
    block_on(gpu_xy_bounds_async(x, y, context))
}
