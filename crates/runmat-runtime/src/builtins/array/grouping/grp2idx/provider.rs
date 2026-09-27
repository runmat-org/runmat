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
        let output = crate::builtins::common::provider_restore::upload_value_like_protected(
            resident.provider,
            value,
            "grp2idx",
            &resident.prototype,
            &protected,
        );
        let Ok(output) = output else {
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
    left.shares_allocation_with(right)
}
