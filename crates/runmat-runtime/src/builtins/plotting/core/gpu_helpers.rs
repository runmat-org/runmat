//! Shared helpers for GPU-aware plotting builtins.

use futures::executor::block_on;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::{NumericScalar, Tensor, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::{gather_if_needed_async, value_contains_gpu, BuiltinResult};

use super::plotting_error;

/// Preserve the original runtime/provider cause after a completed preparation
/// stage. This is not an availability classifier and cannot authorize replay.
pub(crate) fn preparation_after_effect(
    error: crate::RuntimeError,
    name: &'static str,
) -> crate::RuntimeError {
    preparation_failure(error, name, true)
}

/// A required rendering stage cannot authorize host replay. Join known
/// completed effects with the original typed provider disposition.
pub(crate) fn preparation_failure(
    error: crate::RuntimeError,
    name: &'static str,
    mut after_effect: bool,
) -> crate::RuntimeError {
    let mut cursor: &(dyn std::error::Error + 'static) = &error;
    loop {
        after_effect |= cursor
            .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
            .is_some_and(runmat_accelerate_api::ProviderExecutionError::is_after_effect);
        let Some(source) = cursor.source() else {
            break;
        };
        cursor = source;
    }
    let source = runmat_accelerate_api::ProviderExecutionError::from_source(error);
    crate::runtime_error::with_provider_execution_source(
        plotting_error(name, format!("{name}: GPU preparation failed: {source}")),
        if after_effect {
            source.after_effect()
        } else {
            source.with_unproven_effects()
        },
    )
}

#[cfg(feature = "plot-core")]
#[path = "gpu_exports.rs"]
mod exports;
#[cfg(feature = "plot-core")]
pub(crate) use exports::GpuBufferExports;

#[path = "gpu_bounds.rs"]
mod bounds;
pub(crate) use bounds::NumericBounds;

pub(crate) async fn numeric_bounds_protecting(
    handle: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
    context: &'static str,
) -> BuiltinResult<NumericBounds> {
    bounds::numeric_bounds_protecting(handle, protected, context).await
}

pub(crate) async fn axis_bounds_protecting(
    handle: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
    context: &'static str,
) -> BuiltinResult<(f32, f32)> {
    bounds::axis_bounds_protecting(handle, protected, context).await
}

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
    super::context::ensure_context_from_provider().map_err(|err| {
        super::plotting_error_with_source(name, format!("{name}: {}", err.message()), err)
    })?;
    runmat_plot::shared_wgpu_context()
        .ok_or_else(|| plotting_error(name, format!("{name}: plotting GPU context unavailable")))
}

/// Optional rendering admission borrows the inputs' actual device authority,
/// never a context cached from a different ambient provider.
#[cfg(feature = "plot-core")]
pub(crate) fn prepare_shared_wgpu_context(
    inputs: &[&GpuTensorHandle],
    name: &'static str,
) -> BuiltinResult<Option<SharedWgpuContext>> {
    let mut owner = None;
    for input in inputs {
        input.allocation_lease().validate_live().map_err(|source| {
            crate::runtime_error::with_provider_execution_source(
                plotting_error(name, format!("{name}: plotting input has retired")),
                runmat_accelerate_api::ProviderExecutionError::from_source(source),
            )
        })?;
        let current = runmat_accelerate_api::provider_for_handle(input).ok_or_else(|| {
            preparation_failure(
                plotting_error(name, format!("{name}: no provider owns plotting input")),
                name,
                false,
            )
        })?;
        if let Some(previous) = owner {
            if !runmat_accelerate_api::same_provider_instance(previous, current) {
                return Ok(None);
            }
        } else {
            owner = Some(current);
        }
    }
    let Some(owner) = owner else {
        return Ok(None);
    };
    let Some(runmat_accelerate_api::AccelContextHandle::Wgpu(context)) =
        owner.export_context(runmat_accelerate_api::AccelContextKind::Plotting)
    else {
        return Ok(None);
    };
    super::context::install_wgpu_context(&context);
    Ok(Some(SharedWgpuContext {
        instance: context.instance,
        access: context.access,
        adapter: context.adapter,
        adapter_info: context.adapter_info,
        limits: context.limits,
        features: context.features,
    }))
}

/// Compute global bounds through the exact owner's reduction hooks. Temporary
/// results are guarded across errors and suspension; coordinates stay borrowed.
/// Results are returned as `f32` for plotting bounding boxes.
pub async fn axis_bounds_async(
    handle: &GpuTensorHandle,
    context: &'static str,
) -> BuiltinResult<(f32, f32)> {
    bounds::axis_bounds_protecting(handle, &[handle], context).await
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
    crate::builtins::common::tensor::value_into_tensor_for(name, gathered)
        .map_err(|e| plotting_error(name, format!("{name}: {e}")))
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
        Value::Tensor(t) => t
            .numeric_value_at(0)
            .map(numeric_scalar_to_f64)
            .ok_or_else(|| plotting_error(context, format!("{context}: expected scalar result"))),
        _ => Err(plotting_error(
            context,
            format!("{context}: expected numeric scalar result"),
        )),
    }
}

fn numeric_scalar_to_f64(value: NumericScalar) -> f64 {
    match value {
        NumericScalar::F64(value) => value,
        NumericScalar::F32(value) => f64::from(value),
        value => value
            .into_int_value()
            .expect("non-floating numeric scalar is integer")
            .to_f64(),
    }
}

/// Build a bounding box from GPU-resident X/Y vectors.
#[cfg(feature = "plot-core")]
pub async fn gpu_xy_bounds_async(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    context: &'static str,
) -> BuiltinResult<BoundingBox> {
    let protected = [x, y];
    let (min_x, max_x) = bounds::axis_bounds_protecting(x, &protected, context).await?;
    let (min_y, max_y) = bounds::axis_bounds_protecting(y, &protected, context)
        .await
        .map_err(|error| preparation_after_effect(error, context))?;
    Ok(BoundingBox::new(
        Vec3::new(min_x, min_y, 0.0),
        Vec3::new(max_x, max_y, 0.0),
    ))
}

#[cfg(feature = "plot-core")]
pub async fn gpu_xyz_bounds_async(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    z: &GpuTensorHandle,
    context: &'static str,
) -> BuiltinResult<BoundingBox> {
    let protected = [x, y, z];
    let (min_x, max_x) = bounds::axis_bounds_protecting(x, &protected, context).await?;
    let (min_y, max_y) = bounds::axis_bounds_protecting(y, &protected, context)
        .await
        .map_err(|error| preparation_after_effect(error, context))?;
    let (min_z, max_z) = bounds::axis_bounds_protecting(z, &protected, context)
        .await
        .map_err(|error| preparation_after_effect(error, context))?;
    Ok(BoundingBox::new(
        Vec3::new(min_x, min_y, min_z),
        Vec3::new(max_x, max_y, max_z),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntegerStorage, NumericStorage};

    #[test]
    fn value_to_scalar_reads_authoritative_single_and_integer_storage() {
        let single =
            Tensor::from_numeric_storage(NumericStorage::F32(vec![1.25]), vec![1, 1]).unwrap();
        assert_eq!(
            block_on(value_to_scalar_async(Value::Tensor(single), "test")).unwrap(),
            1.25
        );

        let integer =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .unwrap();
        assert_eq!(
            block_on(value_to_scalar_async(Value::Tensor(integer), "test")).unwrap(),
            9_007_199_254_740_993_u64 as f64
        );
    }
}
