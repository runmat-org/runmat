use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::Value;

use crate::BuiltinResult;

pub(super) struct ResidentInput {
    provider: &'static dyn AccelProvider,
    prototype: GpuTensorHandle,
}

pub(super) fn resident_input(value: &Value) -> Option<ResidentInput> {
    let Value::GpuTensor(prototype) = value else {
        return None;
    };
    runmat_accelerate_api::provider_for_handle(prototype).map(|provider| ResidentInput {
        provider,
        prototype: prototype.clone(),
    })
}

pub(super) fn restore(
    resident: Option<&ResidentInput>,
    g: Value,
    gl: Value,
) -> BuiltinResult<(Value, Value)> {
    let Some(resident) = resident else {
        return Ok((g, gl));
    };
    let host = [g, gl];
    let mut restored = Vec::with_capacity(host.len());
    for value in host.iter().cloned() {
        let protected = std::iter::once(resident.prototype.clone())
            .chain(restored.iter().filter_map(handle))
            .collect::<Vec<_>>();
        let output =
            crate::builtins::math::trigonometry::inverse_helpers::upload_value_like_protected(
                resident.provider,
                value,
                "grp2idx",
                &resident.prototype,
                &protected,
            );
        let Ok(output) = output else {
            free_outputs(&restored, &resident.prototype);
            return Ok((host[0].clone(), host[1].clone()));
        };
        let valid = handle(&output).is_some_and(|candidate| {
            !same(&candidate, &resident.prototype)
                && restored
                    .iter()
                    .filter_map(handle)
                    .all(|prior| !same(&candidate, &prior))
        });
        if !valid {
            if let Some(candidate) = handle(&output) {
                free_if_fresh(&candidate, &resident.prototype, &restored);
            }
            free_outputs(&restored, &resident.prototype);
            return Ok((host[0].clone(), host[1].clone()));
        }
        restored.push(output);
    }
    Ok((restored.remove(0), restored.remove(0)))
}

fn handle(value: &Value) -> Option<GpuTensorHandle> {
    match value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    }
}

fn same(left: &GpuTensorHandle, right: &GpuTensorHandle) -> bool {
    left.device_id == right.device_id && left.buffer_id == right.buffer_id
}

fn free_if_fresh(handle: &GpuTensorHandle, prototype: &GpuTensorHandle, protected: &[Value]) {
    if same(handle, prototype)
        || protected
            .iter()
            .filter_map(self::handle)
            .any(|other| same(handle, &other))
    {
        return;
    }
    if let Some(owner) = runmat_accelerate_api::provider_for_handle(handle) {
        let _ = owner.free(handle);
    }
}

fn free_outputs(outputs: &[Value], prototype: &GpuTensorHandle) {
    let mut freed = std::collections::BTreeSet::new();
    for handle in outputs.iter().filter_map(self::handle) {
        if same(&handle, prototype) || !freed.insert((handle.device_id, handle.buffer_id)) {
            continue;
        }
        if let Some(owner) = runmat_accelerate_api::provider_for_handle(&handle) {
            let _ = owner.free(&handle);
        }
    }
}
