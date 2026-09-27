//! Collect only resident identities and prove visibility before any release.
use runmat_accelerate_api::{handle_identity, AccelProvider, GpuHandleIdentity, GpuTensorHandle};
use runmat_gc::RootId;
use runmat_value::Value;
use std::collections::{HashMap, HashSet};

pub(super) struct Allocation {
    pub(super) handle: GpuTensorHandle,
    pub(super) owner: &'static dyn AccelProvider,
}

#[derive(Default)]
pub(super) struct AllocationCollector(HashMap<GpuHandleIdentity, GpuTensorHandle>);
impl AllocationCollector {
    pub(super) fn visit(&mut self, value: &Value) -> Result<(), String> {
        runmat_runtime::dispatcher::try_visit_gpu_handles(value, &mut |handle| {
            self.0
                .entry(handle_identity(handle))
                .or_insert_with(|| handle.clone());
        })
        .map_err(|error| error.to_string())
    }
    pub(super) fn finish(self) -> Result<Vec<Allocation>, String> {
        self.0
            .into_values()
            .map(|handle| {
                let owner = runmat_accelerate_api::provider_for_handle(&handle)
                    .ok_or_else(|| "resident value owner is unavailable".to_string())?;
                Ok(Allocation { handle, owner })
            })
            .collect()
    }
}

pub(super) fn allocations(values: &[Value]) -> Result<Vec<Allocation>, String> {
    let mut collector = AllocationCollector::default();
    for value in values {
        collector.visit(value)?;
    }
    collector.finish()
}

pub(super) fn retain_value(value: &Value, keep: &mut HashSet<GpuHandleIdentity>) -> Option<()> {
    runmat_runtime::dispatcher::try_visit_gpu_handles(value, &mut |handle| {
        keep.insert(handle_identity(handle));
    })
    .ok()
}

pub(super) fn retained(
    excluded: &[RootId],
    extra: &HashSet<GpuHandleIdentity>,
) -> Result<HashSet<GpuHandleIdentity>, String> {
    let mut keep = extra.clone();
    let mut visit = |value: &Value| retain_value(value, &mut keep);
    runmat_gc::gc_visit_retained_values(excluded, &mut visit)
        .ok_or_else(|| "registered value roots are unavailable".to_string())?;
    runmat_runtime::workspace::session::visit_roots(&mut visit)
        .ok_or_else(|| "session retained values are unavailable".to_string())?;
    runmat_runtime::class_registry::visit_static_property_retained_values(&mut visit)
        .ok_or_else(|| "class retained values are unavailable".to_string())?;
    if let Some(runtime) = runmat_runtime::context::legacy::active() {
        if let Some(calls) = runtime.service_ports().call() {
            calls
                .visit_retained_values(&mut visit)
                .ok_or_else(|| "call service retained values are unavailable".to_string())?;
        }
        if let Some(foreign) = runtime.service_ports().foreign() {
            foreign
                .visit_retained_values(&mut visit)
                .ok_or_else(|| "foreign service retained values are unavailable".to_string())?;
        }
        if let Some(distributed) = runtime.service_ports().distributed() {
            distributed
                .visit_retained_values(&mut visit)
                .ok_or_else(|| "distributed service retained values are unavailable".to_string())?;
        }
        runtime
            .execution()
            .visit_retained_values(&mut visit)
            .ok_or_else(|| "execution retained values are unavailable".to_string())?;
        if let Some(objects) = runtime.service_ports().object() {
            objects
                .visit_retained_values(&mut visit)
                .ok_or_else(|| "object service retained values are unavailable".to_string())?;
        }
        if let Some(workspace) = runtime.service_ports().workspace() {
            workspace
                .visit_retained_values(&mut visit)
                .ok_or_else(|| "workspace retained values are unavailable".to_string())?;
        }
    }
    Ok(keep)
}

pub(super) fn release(
    owned: Vec<Allocation>,
    excluded: &[RootId],
    extra: &HashSet<GpuHandleIdentity>,
) {
    if owned.is_empty() {
        return;
    }
    let keep = match retained(excluded, extra) {
        Ok(keep) => keep,
        Err(reason) => {
            log::debug!(
                "VM resident cleanup skipped because alias visibility is unavailable: {reason}"
            );
            return;
        }
    };
    for allocation in owned {
        if !keep.contains(&handle_identity(&allocation.handle)) {
            match allocation.owner.free(&allocation.handle) {
                Ok(()) => runmat_accelerate_api::clear_handle_metadata(&allocation.handle),
                Err(error) => log::warn!("VM resident release failed: {error}"),
            }
        }
    }
}
