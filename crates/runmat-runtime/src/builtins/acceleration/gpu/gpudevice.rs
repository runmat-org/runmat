//! Session-owned GPUDevice class and real provider lifecycle control.

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericScalar, Value};

use crate::{context::RuntimeContext, execution::RootedValue};
use runmat_accelerate_api::AccelProvider;
use runmat_builtins::GpuDeviceProperty;
mod object;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "gpuDevice";

use runmat_builtins::{
    gpu_device_index_from_float, gpu_device_index_from_integer, GPU_DEVICE_ACTIVE_INDEX,
    GPU_DEVICE_ERROR_DEVICE_NOT_AVAILABLE, GPU_DEVICE_ERROR_INVALID_INDEX,
    GPU_DEVICE_ERROR_LIFECYCLE, GPU_DEVICE_ERROR_NO_PROVIDER, GPU_DEVICE_ERROR_TOO_MANY_INPUTS,
    GPU_DEVICE_ERROR_UNSUPPORTED_ARGUMENT, GPU_DEVICE_LOGICAL_SELECTOR_EXTENSION,
};

fn gpu_device_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    gpu_device_error_with_message(error.message, error)
}

fn gpu_device_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::acceleration::gpu::gpudevice")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gpuDevice",
    op_kind: GpuOpKind::Custom("device-info"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host GPUDevice handle with session-scoped selection and physical reset; no computational GPU kernel is launched.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::acceleration::gpu::gpudevice"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gpuDevice",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; returns a host-resident GPUDevice handle or an empty deselection result.",
};

/// Query/select an actual device owner, never substitute gpuInfo metadata.
#[runtime_builtin(
    name = "gpuDevice",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::acceleration::gpu::gpudevice"
)]
async fn gpu_device_builtin(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<RootedValue> {
    if args
        .first()
        .is_some_and(|value| matches!(value, Value::Bool(_) | Value::LogicalArray(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GPU_DEVICE_LOGICAL_SELECTOR_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args.len() > 1 {
        return Err(gpu_device_error(&GPU_DEVICE_ERROR_TOO_MANY_INPUTS));
    }
    let runtime =
        runtime.ok_or_else(|| lifecycle("gpuDevice requires an active RuntimeContext"))?;
    let empty = args.first().is_some_and(|value| {
        matches!(value,
        Value::Tensor(t) if tensor::tensor_element_len(t) == 0)
            || matches!(value, Value::LogicalArray(array) if array.data.is_empty())
    });
    if empty {
        if matches!(
            runtime.gpu_device_selection(),
            crate::context::gpu_device::GpuDeviceSelection::Deselected
        ) {
            return empty_result(&runtime);
        }
        let (provider, _) = current_device(&runtime)?;
        runtime.reset_gpu_device(provider, true).await?;
        return empty_result(&runtime);
    }
    let selected = args
        .first()
        .map(|value| {
            parse_device_index(value).and_then(|index| {
                index.ok_or_else(|| gpu_device_error(&GPU_DEVICE_ERROR_UNSUPPORTED_ARGUMENT))
            })
        })
        .transpose()?;
    let (provider, index) = match selected {
        Some(index) => (device_at_index(index)?, index),
        None => current_device(&runtime)?,
    };
    if selected.is_some() {
        runtime.reset_gpu_device(provider, false).await?;
    } else if !runtime.select_gpu_device_without_reset(provider) {
        return Err(lifecycle(
            "GPU session cannot select a device during a pending or failed reset",
        ));
    }
    object::create(&runtime, provider, index)
}

fn lifecycle(message: impl Into<String>) -> RuntimeError {
    gpu_device_error_with_message(message, &GPU_DEVICE_ERROR_LIFECYCLE)
}
fn device_at_index(index: u32) -> BuiltinResult<&'static dyn AccelProvider> {
    let mut after = None;
    let mut current = 0u32;
    while let Some((id, provider)) = runmat_accelerate_api::next_registered_provider(after) {
        after = Some(id);
        if !runmat_accelerate_api::device_control_provider_is_authorized(provider)
            || provider.gpu_device_control().is_none()
        {
            continue;
        }
        current = current
            .checked_add(1)
            .ok_or_else(|| lifecycle("GPU device index exhausted"))?;
        if index == current {
            return Ok(provider);
        }
    }
    Err(gpu_device_error_with_message(
        format!("gpuDevice: GPU device with index {index} not available"),
        &GPU_DEVICE_ERROR_DEVICE_NOT_AVAILABLE,
    ))
}
fn current_device(runtime: &RuntimeContext) -> BuiltinResult<(&'static dyn AccelProvider, u32)> {
    let preferred = match runtime.gpu_device_selection() {
        crate::context::gpu_device::GpuDeviceSelection::Selected(provider) => Some(provider),
        crate::context::gpu_device::GpuDeviceSelection::Resetting(_)
        | crate::context::gpu_device::GpuDeviceSelection::ResetFailed(_) => {
            return Err(lifecycle(
                "GPU session is fenced by a pending or failed reset",
            ))
        }
        _ => runmat_accelerate_api::provider_without_registry_allocation(),
    };
    let mut after = None;
    let mut index = 0u32;
    let mut first = None;
    while let Some((id, provider)) = runmat_accelerate_api::next_registered_provider(after) {
        after = Some(id);
        if !runmat_accelerate_api::device_control_provider_is_authorized(provider)
            || provider.gpu_device_control().is_none()
        {
            continue;
        }
        index = index
            .checked_add(1)
            .ok_or_else(|| lifecycle("GPU device index exhausted"))?;
        first.get_or_insert((provider, index));
        if preferred.is_some_and(|selected| {
            runmat_accelerate_api::same_provider_instance(provider, selected)
        }) {
            return Ok((provider, index));
        }
    }
    first.ok_or_else(|| gpu_device_error(&GPU_DEVICE_ERROR_NO_PROVIDER))
}

fn parse_device_index(value: &Value) -> BuiltinResult<Option<u32>> {
    match value {
        Value::Int(i) => integer_to_index(i),
        Value::Num(n) => float_to_index(*n),
        Value::Bool(b) => {
            if *b {
                Ok(Some(GPU_DEVICE_ACTIVE_INDEX))
            } else {
                Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into())
            }
        }
        Value::Tensor(t) => match tensor::tensor_element_len(t) {
            0 => Ok(None),
            1 => match t
                .numeric_value_at(0)
                .expect("tensor storage length matches shape")
            {
                NumericScalar::F64(value) => float_to_index(value),
                NumericScalar::F32(value) => float_to_index(f64::from(value)),
                value => integer_to_index(
                    &value
                        .into_int_value()
                        .expect("non-floating numeric scalar is integer"),
                ),
            },
            _ => Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into()),
        },
        Value::LogicalArray(la) => match la.data.len() {
            0 => Ok(None),
            1 => {
                if la.data[0] != 0 {
                    Ok(Some(GPU_DEVICE_ACTIVE_INDEX))
                } else {
                    Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into())
                }
            }
            _ => Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into()),
        },
        _ => Ok(None),
    }
}

fn integer_to_index(value: &IntValue) -> BuiltinResult<Option<u32>> {
    value
        .try_to_u64()
        .and_then(|raw| gpu_device_index_from_integer(i128::from(raw)).ok())
        .map(Some)
        .ok_or_else(|| gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into())
}

fn float_to_index(raw: f64) -> BuiltinResult<Option<u32>> {
    gpu_device_index_from_float(raw)
        .map(Some)
        .map_err(|_| gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX))
}

fn empty_result(runtime: &RuntimeContext) -> BuiltinResult<RootedValue> {
    use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
    let allocation = |cause| {
        crate::context::host_allocation::host_allocation_runtime_error(
            Some(runtime),
            "gpuDevice:HostAllocation",
            "GPUDevice empty output admission failed",
            cause,
        )
    };
    let bytes = runmat_value::ExactHostTensorAllocation::requested_bytes(
        runmat_value::NumericDType::F64,
        &[0, 0],
    )
    .map_err(|cause| allocation(cause.into()))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        bytes as u64,
        runmat_value::ExactHostTensorAllocation::retained_claim_count(),
    )
    .map_err(allocation)?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained).map_err(allocation)?,
    )
    .map_err(allocation)?;
    let empty = if let Some(receipt) = &receipt {
        receipt
            .with_output_allocation(runtime, |permit| {
                runmat_value::ExactHostTensorAllocation::new(
                    runmat_value::NumericDType::F64,
                    &[0, 0],
                    permit,
                )?
                .into_tensor()
            })
            .map_err(allocation)?
    } else {
        runmat_value::Tensor::new(Vec::new(), vec![0, 0]).map_err(lifecycle)?
    };
    RootedValue::in_optional_context(Value::Tensor(empty), Some(runtime))
}

macro_rules! property_binding {
    ($function:ident, $binding:literal, $property:ident) => {
        #[runtime_builtin(name = $binding, binding_variant = "default", context = true,
                                    builtin_path = "crate::builtins::acceleration::gpu::gpudevice")]
        async fn $function(runtime: Option<RuntimeContext>, device: Value) -> BuiltinResult<Value> {
            let runtime = runtime
                .ok_or_else(|| lifecycle("GPUDevice property requires its session context"))?;
            object::read(&runtime, &device, GpuDeviceProperty::$property)
        }
    };
}
property_binding!(device_name, "__runmat_gpudevice_name", Name);
property_binding!(device_index, "__runmat_gpudevice_index", Index);
property_binding!(
    device_total_memory,
    "__runmat_gpudevice_total_memory",
    TotalMemory
);
property_binding!(
    device_available_memory,
    "__runmat_gpudevice_available_memory",
    AvailableMemory
);
property_binding!(
    device_supports_double,
    "__runmat_gpudevice_supports_double",
    SupportsDouble
);
property_binding!(
    device_available,
    "__runmat_gpudevice_available",
    DeviceAvailable
);
property_binding!(
    device_selected,
    "__runmat_gpudevice_selected",
    DeviceSelected
);

#[runtime_builtin(
    name = "__runmat_gpudevice_reset",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::acceleration::gpu::gpudevice"
)]
async fn reset_device(runtime: Option<RuntimeContext>, device: Value) -> BuiltinResult<Value> {
    let runtime =
        runtime.ok_or_else(|| lifecycle("GPUDevice reset requires its session context"))?;
    let (provider, _) = object::identity(&runtime, &device)?;
    runtime.reset_gpu_device(provider, false).await?;
    Ok(Value::Num(0.0))
}
#[runtime_builtin(
    name = "__runmat_gpudevice_wait",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::acceleration::gpu::gpudevice"
)]
async fn wait_device(runtime: Option<RuntimeContext>, device: Value) -> BuiltinResult<Value> {
    use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
    let runtime =
        runtime.ok_or_else(|| lifecycle("GPUDevice wait requires its session context"))?;
    let mut cooperation = runtime.cpu_cooperation();
    cooperation
        .checkpoint(0)
        .await
        .map_err(|cause| cause.into_runtime_error_in_context(Some(&runtime)))?;
    let (provider, _) = object::identity(&runtime, &device)?;
    let control = provider
        .gpu_device_control()
        .ok_or_else(|| lifecycle("GPUDevice wait is unsupported"))?;
    let bytes = control
        .control_host_bytes()
        .ok_or_else(|| lifecycle("GPUDevice wait cannot be admitted"))?;
    let allocation = |cause| {
        crate::context::host_allocation::host_allocation_runtime_error(
            Some(&runtime),
            "gpuDevice:HostAllocation",
            "GPUDevice wait admission failed",
            cause,
        )
    };
    let retained = HostAllocationPlan::retained_metadata_bytes(bytes).map_err(allocation)?;
    let receipt = HostAllocationReceipt::admit(
        Some(&runtime),
        HostAllocationPlan::new(0, retained).map_err(allocation)?,
    )
    .map_err(allocation)?;
    let claim = receipt
        .as_ref()
        .map(|receipt| receipt.claim_retained_metadata(&runtime, bytes))
        .transpose()
        .map_err(allocation)?;
    // Defer even the provider future's construction until cancellation has
    // been polled. Cancelling the waiter does not revoke the backend callback's
    // original completion clone or its retained physical allocation claim.
    cooperation
        .await_cancellable(async {
            control
                .wait_for_device(claim)
                .ok_or_else(|| lifecycle("GPUDevice wait is unsupported by this backend"))?
                .await
                .map_err(|cause| {
                    crate::runtime_error::with_provider_execution_source(
                        lifecycle("GPUDevice wait failed"),
                        runmat_accelerate_api::ProviderExecutionError::from(cause).after_effect(),
                    )
                })
        })
        .await
        .map_err(|cause| cause.into_runtime_error_in_context(Some(&runtime)))??;
    Ok(Value::Num(0.0))
}

#[cfg(test)]
mod tests;
