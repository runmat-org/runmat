//! Existing class/GC owners hold the device handle. No provider registry,
//! raw device pointer, mutable public metadata or new Value carrier is added.
use super::*;
use crate::{
    class_registry::{ClassRegistration, RuntimeClass, RuntimeMethod, RuntimeProperty},
    context::RuntimeContext,
    execution::RootedValue,
};
use runmat_accelerate_api::AccelProvider;
use runmat_builtins::{GpuDeviceProperty, GPU_DEVICE_CLASS};
use runmat_types::{MemberAccess, MemberName, MethodName};
use std::collections::HashMap;

const PROVIDER: &str = "__runmat_gpudevice_provider";
const SESSION: &str = "__runmat_gpudevice_session";
const INDEX: &str = "__runmat_gpudevice_index";
static REGISTERED: ClassRegistration = ClassRegistration::new(GPU_DEVICE_CLASS);

fn ensure_class() {
    REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        let mut methods = HashMap::new();
        for property in GpuDeviceProperty::ALL {
            let name: MemberName = property.name().into();
            properties.insert(
                name.clone(),
                RuntimeProperty {
                    name: name.clone(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: true,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Private,
                    default_value: None,
                },
            );
            let getter = MethodName::property_getter(&name);
            methods.insert(getter.clone(), method(getter, property.binding()));
        }
        for name in [PROVIDER, SESSION, INDEX] {
            properties.insert(
                name.into(),
                RuntimeProperty {
                    name: name.into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: MemberAccess::Private,
                    set_access: MemberAccess::Private,
                    default_value: None,
                },
            );
        }
        for (name, binding) in [
            ("reset", "__runmat_gpudevice_reset"),
            ("wait", "__runmat_gpudevice_wait"),
        ] {
            let name: MethodName = name.into();
            methods.insert(name.clone(), method(name, binding));
        }
        crate::class_registry::register_builtin_class(
            GPU_DEVICE_CLASS,
            RuntimeClass {
                name: GPU_DEVICE_CLASS.into(),
                parent: Some(runmat_types::standard::HANDLE.into()),
                methods,
                properties,
                events: Default::default(),
            },
        );
    });
}
fn method(name: MethodName, binding: &'static str) -> RuntimeMethod {
    RuntimeMethod {
        name,
        is_static: false,
        is_abstract: false,
        is_sealed: true,
        access: MemberAccess::Public,
        function_name: binding.into(),
        implicit_class_argument: None,
        role: runmat_types::ClassMethodRole::Ordinary,
    }
}
pub(super) fn create(
    runtime: &RuntimeContext,
    provider: &'static dyn AccelProvider,
    index: u32,
) -> BuiltinResult<RootedValue> {
    ensure_class();
    let fields = [
        (PROVIDER, Value::Int(IntValue::U32(provider.device_id()))),
        (
            SESSION,
            Value::Int(IntValue::U64(runtime.gpu_device_session_key())),
        ),
        (INDEX, Value::Int(IntValue::U32(index))),
        (crate::HANDLE_VALID_FLAG_PROPERTY, Value::Bool(true)),
    ];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        Some(runtime),
        GPU_DEVICE_CLASS.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    object.with_owned(|object| {
        let target_root = runmat_gc::gc_allocate_rooted(object)
            .map_err(|cause| lifecycle(format!("GPUDevice allocation failed: {cause}")))?;
        let output = RootedValue::in_context(
            Value::HandleObject(runmat_value::HandleRef {
                class_name: GPU_DEVICE_CLASS.into(),
                target: target_root.handle(),
                valid: true,
            }),
            runtime,
        )?;
        target_root
            .unroot()
            .map_err(|cause| lifecycle(format!("GPUDevice publication failed: {cause}")))?;
        Ok(output)
    })
}

pub(super) fn identity(
    runtime: &RuntimeContext,
    value: &Value,
) -> BuiltinResult<(&'static dyn AccelProvider, u32)> {
    let Value::HandleObject(handle) = value else {
        return Err(lifecycle("Expected a GPUDevice handle"));
    };
    if !handle.class_name.is(GPU_DEVICE_CLASS)
        || !handle.valid
        || !crate::is_handle_valid(handle)
        || !crate::class_registry::is_registered_builtin_class(&GPU_DEVICE_CLASS.owned())
    {
        return Err(lifecycle(
            "GPUDevice handle is invalid or its class authority was replaced",
        ));
    }
    let (device, index) = runmat_gc::gc_with_value(&handle.target, |object| {
        let Value::Object(object) = object else { return Err(lifecycle("Invalid GPUDevice target")); };
        if !object.is_class(GPU_DEVICE_CLASS) || !matches!(object.property(SESSION), Some(Value::Int(IntValue::U64(session))) if *session == runtime.gpu_device_session_key()) {
            return Err(lifecycle("GPUDevice belongs to another session"));
        }
        match (object.property(PROVIDER), object.property(INDEX)) {
            (Some(Value::Int(IntValue::U32(device))), Some(Value::Int(IntValue::U32(index)))) => Ok((*device, *index)),
            _ => Err(lifecycle("GPUDevice has no issued provider identity")),
        }
    }).map_err(|cause| lifecycle(format!("GPUDevice target is unavailable: {cause}")))??;
    let mut after = None;
    while let Some((id, provider)) = runmat_accelerate_api::next_registered_provider(after) {
        if id == device {
            if !runmat_accelerate_api::device_control_provider_is_authorized(provider) {
                return Err(lifecycle("GPUDevice lacks the execution device lease"));
            }
            return Ok((provider, index));
        }
        after = Some(id);
    }
    Err(lifecycle("GPUDevice physical owner is unavailable"))
}

pub(super) fn read(
    runtime: &RuntimeContext,
    value: &Value,
    property: GpuDeviceProperty,
) -> BuiltinResult<Value> {
    let (provider, index) = identity(runtime, value)?;
    let control = provider
        .gpu_device_control()
        .ok_or_else(|| lifecycle("GPUDevice facts are unsupported by this backend"))?;
    let facts = control.gpu_device_facts();
    Ok(match property {
        GpuDeviceProperty::Name => name_value(runtime, facts.name)?,
        GpuDeviceProperty::Index => Value::Num(f64::from(index)),
        GpuDeviceProperty::TotalMemory => Value::Num(facts.total_memory_bytes.ok_or_else(|| lifecycle("TotalMemory is not available from this backend; buffer allocation limits are not VRAM"))? as f64),
        GpuDeviceProperty::AvailableMemory => Value::Num(facts.available_memory_bytes.ok_or_else(|| lifecycle("AvailableMemory is not measured by this backend; total memory is not free memory"))? as f64),
        GpuDeviceProperty::SupportsDouble => Value::Bool(facts.supports_double.ok_or_else(|| lifecycle("Double precision capability is not reported by this backend"))?),
        GpuDeviceProperty::DeviceAvailable => Value::Bool(facts.available),
        GpuDeviceProperty::DeviceSelected => Value::Bool(matches!(runtime.gpu_device_selection(), crate::context::gpu_device::GpuDeviceSelection::Selected(selected) if runmat_accelerate_api::same_provider_instance(provider, selected))),
    })
}
fn name_value(runtime: &RuntimeContext, name: &str) -> BuiltinResult<Value> {
    use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};
    let allocation = |cause| {
        crate::context::host_allocation::host_allocation_runtime_error(
            Some(runtime),
            "gpuDevice:HostAllocation",
            "GPUDevice Name output admission failed",
            cause,
        )
    };
    if HostAllocationReceipt::policy(Some(runtime)).map_err(allocation)?
        == HostAllocationPolicy::Unbounded
    {
        return Ok(Value::CharArray(runmat_value::CharArray::new_row(name)));
    }
    let count = name.encode_utf16().count();
    let value = crate::builtins::common::character_output::build(
        runtime,
        2,
        count,
        0,
        |axis| Ok(if axis == 0 { 1 } else { count }),
        |output, _, _| {
            for (slot, unit) in output.iter_mut().zip(name.encode_utf16()) {
                *slot = runmat_value::Utf16CodeUnit::from_raw(unit);
            }
            Ok(())
        },
        crate::builtins::common::character_output::CharacterBuildErrors {
            admission: |runtime, cause| {
                crate::context::host_allocation::host_allocation_runtime_error(
                    Some(runtime),
                    "gpuDevice:HostAllocation",
                    "GPUDevice Name output admission failed",
                    cause,
                )
            },
            invalid: lifecycle,
        },
    )?;
    Ok(Value::CharArray(value))
}
