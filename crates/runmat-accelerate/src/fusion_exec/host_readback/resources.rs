//! Shared sequential readback peak and retained host-body accounting.
use super::{CpuHostResources, Source};
use runmat_execution::{
    CandidateOutputPresentation, CandidateResourceDemand, ProviderReadbackDemand,
    ProviderReadbackDemands, ResourceByteBound,
};

pub(crate) fn resources(
    sources: &[Source],
    retained_outputs: u64,
    active_scratch: u64,
    operand_metadata: u64,
    managed: bool,
) -> Option<CpuHostResources> {
    let mut retained_mirrors = 0_u64;
    let mut base_mirror_bytes = 0_u64;
    let mut host_peak = 0_u64;
    let mut download_bytes = 0_u64;
    let mut demands = Vec::with_capacity(sources.len());
    for source in sources {
        let metadata = u64::try_from(source.handle.shape().len())
            .ok()?
            .checked_mul(std::mem::size_of::<usize>() as u64)?;
        let value_bytes = source.payload_bytes.checked_add(metadata)?;
        base_mirror_bytes = base_mirror_bytes.checked_add(value_bytes)?;
        host_peak = host_peak.max(
            retained_mirrors
                .checked_add(value_bytes)?
                .checked_add(source.host_workspace)?,
        );
        // Numeric storage is COW. Each live ordinary load may retain its own
        // shape clone; numeric COW copies are bounded by its canonical body.
        let occurrences =
            u64::try_from(source.offsets.len().checked_add(source.loads.len())?).ok()?;
        retained_mirrors = retained_mirrors
            .checked_add(value_bytes)?
            .checked_add(metadata.checked_mul(occurrences)?)?;
        download_bytes = download_bytes.checked_add(source.payload_bytes)?;
        demands.push(ProviderReadbackDemand {
            device_id: source.owner.device_id(),
            provider_scratch_bytes: source.provider_scratch,
            queue_slots: source.queue_slots,
        });
    }
    let body_extra = retained_mirrors
        .checked_add(active_scratch)?
        .checked_add(operand_metadata)?;
    let (scratch_bytes, retained_bytes) = if managed {
        let readback = super::managed_readback::quote(sources)?;
        let body_structural = readback.body_live_bytes.checked_sub(base_mirror_bytes)?;
        let body_peak = body_extra.checked_add(body_structural)?;
        // Output/body receipts remain live while the separate source receipt
        // stages downloads. Their scratch reservations therefore overlap,
        // even though numerical body execution follows the last readback.
        // Callback claims survive the provider frame and join retained demand.
        (
            readback
                .plan
                .scratch_peak_bytes()
                .checked_add(active_scratch)?
                .max(body_peak),
            retained_outputs.checked_add(readback.plan.retained_output_bytes())?,
        )
    } else {
        (
            crate::fusion_exec::checked_extra_host_scratch(
                retained_outputs,
                host_peak,
                body_extra,
            )?,
            retained_outputs,
        )
    };
    let presentation = if sources.is_empty() {
        CandidateOutputPresentation::Direct
    } else {
        CandidateOutputPresentation::ProviderReadbacksHost {
            readbacks: Box::new(ProviderReadbackDemands::sequential(demands)?),
        }
    };
    Some(CpuHostResources {
        resources: CandidateResourceDemand {
            cpu_millicores: 1000,
            retained_bytes: ResourceByteBound::Known(retained_bytes),
            scratch_bytes: ResourceByteBound::Known(scratch_bytes),
            queue_slots: 0,
            provider_host_workspace: None,
            provider_input_stages: None,
            peak_live: None,
        },
        presentation,
        scratch_bytes,
        download_bytes,
        readback_count: u64::try_from(sources.len()).ok()?,
    })
}
