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
) -> Option<CpuHostResources> {
    let mut retained_mirrors = 0_u64;
    let mut host_peak = 0_u64;
    let mut download_bytes = 0_u64;
    let mut demands = Vec::with_capacity(sources.len());
    for source in sources {
        let metadata = u64::try_from(source.handle.shape().len())
            .ok()?
            .checked_mul(std::mem::size_of::<usize>() as u64)?;
        let value_bytes = source.payload_bytes.checked_add(metadata)?;
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
    let scratch_bytes =
        crate::fusion_exec::checked_extra_host_scratch(retained_outputs, host_peak, body_extra)?;
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
            retained_bytes: ResourceByteBound::Known(retained_outputs),
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
