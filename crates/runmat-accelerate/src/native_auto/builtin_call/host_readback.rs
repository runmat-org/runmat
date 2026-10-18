//! Invocation-private automatic resident preparation, using the same physical
//! source and resource owner as VM overlays. No workspace or VM right is made.
use super::*;
use crate::fusion_exec::host_readback::{CpuHostResources, CpuReadbackPlan, Source};
use runmat_execution::Digest;
use runmat_runtime::context::preparation::AdmittedBuiltinPreparation;

pub(super) struct Admission {
    pub plan: CpuReadbackPlan,
    pub resources: CpuHostResources,
    pub envelope: runmat_runtime::context::preparation::AdmittedHostResources,
    pub identity: Digest,
}

pub(super) fn admit(
    auto: &NativeAutoOffload,
    context: &RuntimeContext,
    request: &AdmittedBuiltinPreparation,
) -> Option<Admission> {
    if !runmat_runtime::context::preparation::PreparedHostExecution::supports_form(request.form()) {
        return None;
    }
    // Shared canonical inference/allocation owner. This envelope does not grant
    // readback or host execution; exact automatic source admission below does.
    let envelope = request.host_resource_bound(context)?;
    let mut sources: Vec<Source> = Vec::new();
    for (offset, value) in request.operands().iter().enumerate() {
        if let Value::GpuTensor(handle) = value {
            if let Some(source) = sources
                .iter_mut()
                .find(|source| source.matches_handle(handle))
            {
                source.offsets.push(offset);
            } else {
                sources.push(Source::admit(handle, auto.provider, vec![offset], vec![])?);
            }
        } else if runmat_runtime::dispatcher::value_contains_gpu(value) {
            return None;
        }
    }
    if sources.is_empty() {
        return None;
    }
    let managed =
        runmat_runtime::context::host_allocation::HostAllocationReceipt::policy(Some(context))
            .ok()?
            == runmat_runtime::context::host_allocation::HostAllocationPolicy::Managed;
    if managed
        && sources
            .iter()
            .any(|source| !matches!(&source.capabilities, std::borrow::Cow::Borrowed(_)))
    {
        return None;
    }
    let resources = crate::fusion_exec::host_readback::resources(
        &sources,
        envelope.bound.retained_bytes,
        envelope.bound.scratch_bytes,
        envelope.operand_metadata,
        managed,
    )?;
    let identity = Digest::sha256_serialized(&(
        "prepared-host-envelope-v2",
        envelope.identity,
        resources.resources,
    ))
    .ok()?;
    let resident_offsets = sources
        .iter()
        .flat_map(|source| source.offsets.iter().copied())
        .collect();
    Some(Admission {
        plan: CpuReadbackPlan {
            sources,
            stack_len: request.operands().len(),
            resident_offsets,
        },
        resources,
        envelope,
        identity,
    })
}
