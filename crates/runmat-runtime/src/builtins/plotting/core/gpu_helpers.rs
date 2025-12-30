//! Shared helpers for GPU-aware plotting builtins.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};

use crate::{call_builtin, gather_if_needed, value_contains_gpu};

#[cfg(feature = "plot-core")]
use glam::Vec3;
#[cfg(feature = "plot-core")]
use runmat_plot::core::BoundingBox;

/// Compute the min/max bounds for a GPU tensor by delegating to the runtime
/// `min`/`max` builtins. Results are returned as `f32` so they can flow directly
/// into plotting bounding boxes.
pub fn axis_bounds(handle: &GpuTensorHandle, context: &str) -> Result<(f32, f32), String> {
    let min_value = call_builtin("min", &[Value::GpuTensor(handle.clone())])?;
    let max_value = call_builtin("max", &[Value::GpuTensor(handle.clone())])?;
    let min_scalar = value_to_scalar(min_value, context)?;
    let max_scalar = value_to_scalar(max_value, context)?;
    Ok((min_scalar as f32, max_scalar as f32))
}

/// Gather a GPU tensor handle into host memory.
pub fn gather_tensor_from_gpu(handle: GpuTensorHandle, name: &str) -> Result<Tensor, String> {
    let value = Value::GpuTensor(handle);
    let gathered = gather_if_needed(&value)?;
    Tensor::try_from(&gathered).map_err(|e| format!("{name}: {e}"))
}

/// Convert a runtime value (potentially GPU-resident) into a concrete scalar.
pub fn value_to_scalar(mut value: Value, context: &str) -> Result<f64, String> {
    if value_contains_gpu(&value) {
        value = gather_if_needed(&value)?;
    }
    match value {
        Value::Num(n) => Ok(n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(t) => t
            .data
            .first()
            .copied()
            .ok_or_else(|| format!("{context}: expected scalar result")),
        _ => Err(format!("{context}: expected numeric scalar result")),
    }
}

/// Build a bounding box from GPU-resident X/Y vectors.
#[cfg(feature = "plot-core")]
pub fn gpu_xy_bounds(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    context: &str,
) -> Result<BoundingBox, String> {
    let (min_x, max_x) = axis_bounds(x, context)?;
    let (min_y, max_y) = axis_bounds(y, context)?;
    Ok(BoundingBox::new(
        Vec3::new(min_x, min_y, 0.0),
        Vec3::new(max_x, max_y, 0.0),
    ))
}
