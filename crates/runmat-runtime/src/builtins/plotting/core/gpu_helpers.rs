//! Shared helpers for GPU-aware plotting builtins.

use futures::executor::block_on;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::{gather_if_needed_async, value_contains_gpu, BuiltinResult};

use super::plotting_error;

#[cfg(feature = "plot-core")]
use glam::Vec3;
#[cfg(feature = "plot-core")]
use runmat_plot::core::BoundingBox;
#[cfg(feature = "plot-core")]
use runmat_plot::SharedWgpuContext;

/// Ensure a shared WGPU plotting context is installed and return it.
///
/// On web, this is critical for the "zero-copy" plotting path: plotting builtins
/// may execute before any renderer surface has been installed, so we proactively
/// seed the shared context from the active acceleration provider.
#[cfg(feature = "plot-core")]
pub fn ensure_shared_wgpu_context(name: &'static str) -> BuiltinResult<SharedWgpuContext> {
    super::context::ensure_context_from_provider()
        .map_err(|err| plotting_error(name, format!("{name}: {}", err.message())))?;
    runmat_plot::shared_wgpu_context()
        .ok_or_else(|| plotting_error(name, format!("{name}: plotting GPU context unavailable")))
}

/// Compute the min/max bounds for a GPU tensor by delegating to the runtime
/// `min`/`max` builtins. Results are returned as `f32` so they can flow directly
/// into plotting bounding boxes.
pub async fn axis_bounds_async(
    handle: &GpuTensorHandle,
    context: &'static str,
) -> BuiltinResult<(f32, f32)> {
    // IMPORTANT: `min(A)`/`max(A)` have MATLAB semantics (reduce along a dimension), which for
    // matrices returns a vector (e.g. 1xN) and would trigger a small-but-unwanted download.
    // For plotting bounds we always want the global extrema, so call the provider global
    // reduction hooks directly when available.
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(handle) {
        let min_handle = provider.reduce_min(handle).await.map_err(|err| {
            plotting_error(context, format!("{context}: reduce_min failed: {err}"))
        })?;
        let max_handle = provider.reduce_max(handle).await.map_err(|err| {
            plotting_error(context, format!("{context}: reduce_max failed: {err}"))
        })?;

        let min_scalar =
            value_to_scalar_async(Value::GpuTensor(min_handle.clone()), context).await?;
        let max_scalar =
            value_to_scalar_async(Value::GpuTensor(max_handle.clone()), context).await?;

        // These temporary scalar handles are purely intermediate; free them eagerly.
        let _ = provider.free(&min_handle);
        let _ = provider.free(&max_handle);

        return Ok((min_scalar as f32, max_scalar as f32));
    }

    // Fallback: gather and compute on the host (should be rare; provider should exist for GPU tensors).
    let tensor = gather_tensor_from_gpu_async(handle.clone(), context).await?;
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &v in &tensor.data {
        if v.is_finite() {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
    }
    if !min_val.is_finite() || !max_val.is_finite() {
        min_val = 0.0;
        max_val = 0.0;
    }
    Ok((min_val as f32, max_val as f32))
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

pub fn gather_tensor_from_gpu(
    handle: GpuTensorHandle,
    name: &'static str,
) -> BuiltinResult<Tensor> {
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
