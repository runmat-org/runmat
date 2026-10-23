//! Canonical physical resource arithmetic, independent of prices or observation.
use super::account::{ResourceAccount, ResourceAdmissionError, ResourceBudgetKind};
use crate::placement::{
    CandidateExecutionLocation, CandidateOutputPresentation, CandidateOutputResidency,
    CandidateResourceDemand, ExecutionCandidateKind, PlacementResourceSnapshot,
};
use std::collections::BTreeMap;

/// Physical eviction needed to fit a legal candidate. Timing/policy remain
/// with the selected executor, not this resource calculation.
pub struct AdmittedResourceDemand {
    evictions: BTreeMap<u32, u64>,
}
impl AdmittedResourceDemand {
    pub fn provider_evictions(&self) -> impl Iterator<Item = (u32, u64)> + '_ {
        self.evictions
            .iter()
            .map(|(&device, &bytes)| (device, bytes))
    }
}

/// Resource-only admission shared by numeric graph placement and exploratory
/// whole-work selection. It never constructs a synthetic cost estimate.
pub fn single_candidate_resources_admitted(
    candidate: CandidateResourceFacts<'_>,
    resources: &PlacementResourceSnapshot,
    require_transactional_results: bool,
) -> Result<(), ResourceAdmissionError> {
    admit_candidate_resources(
        candidate,
        ResourceAdmissionContext {
            resources,
            host_retained_bytes: ResourceAccount::default(),
            provider_retained_bytes: &BTreeMap::new(),
            require_transactional_results,
        },
    )
    .map(|_| ())
}

pub struct CandidateResourceFacts<'a> {
    pub location: CandidateExecutionLocation,
    pub output_residency: CandidateOutputResidency,
    pub presentation: &'a CandidateOutputPresentation,
    pub kind: ExecutionCandidateKind,
    pub demand: CandidateResourceDemand,
    pub transactional_results: bool,
}

pub struct ResourceAdmissionContext<'a> {
    pub resources: &'a PlacementResourceSnapshot,
    pub host_retained_bytes: ResourceAccount,
    pub provider_retained_bytes: &'a BTreeMap<u32, ResourceAccount>,
    pub require_transactional_results: bool,
}

pub fn admit_candidate_resources(
    candidate: CandidateResourceFacts<'_>,
    admission: ResourceAdmissionContext<'_>,
) -> Result<AdmittedResourceDemand, ResourceAdmissionError> {
    admission
        .resources
        .validate_typed()
        .map_err(ResourceAdmissionError::InvalidSnapshot)?;
    let CandidateResourceFacts {
        location,
        output_residency,
        presentation,
        kind,
        demand,
        transactional_results,
    } = candidate;
    let ResourceAdmissionContext {
        resources,
        host_retained_bytes,
        provider_retained_bytes,
        require_transactional_results,
    } = admission;
    if location == CandidateExecutionLocation::NativeSource {
        let unknown = crate::placement::ResourceByteBound::Unavailable(
            crate::placement::ResourceBoundUnavailable::NoPeakMemoryBound,
        );
        if kind != ExecutionCandidateKind::OriginalNativeRegion
            || output_residency != CandidateOutputResidency::Unknown
            || !matches!(presentation, CandidateOutputPresentation::Direct)
            || transactional_results
            || require_transactional_results
            || demand.cpu_millicores != 0
            || demand.queue_slots != 0
            || demand.retained_bytes != unknown
            || demand.scratch_bytes != unknown
            || demand.provider_host_workspace.is_some()
            || demand.provider_input_stages.is_some()
            || demand.peak_live.is_some()
        {
            return Err(ResourceAdmissionError::Overflow);
        }
        // No single device/tail is certified. Each ordinary native operation
        // retains its own admission; an aggregate finite policy cannot infer
        // a missing whole-region bound from that deferred ownership.
        if resources.cpu_millicores_available == 0
            || resources.memory_available_bytes.is_some()
            || resources.providers.iter().any(|provider| {
                provider.capacity_bytes.is_some()
                    || provider.scratch_available_bytes.is_some()
                    || provider.queue_limit.is_some()
                    || provider.queue_depth.is_some()
            })
            || !host_retained_bytes.is_empty()
            || provider_retained_bytes
                .values()
                .any(|bytes| !bytes.is_empty())
        {
            return Err(ResourceAdmissionError::InsufficientEvidence {
                budget: ResourceBudgetKind::HostMemory,
                reason: crate::placement::ResourceBoundUnavailable::NoPeakMemoryBound,
            });
        }
        return Ok(AdmittedResourceDemand {
            evictions: BTreeMap::new(),
        });
    }
    if let CandidateExecutionLocation::VmRegion { device_id }
    | CandidateExecutionLocation::NativeRegion { device_id } = location
    {
        // The original VM route is admitted by each instruction's existing
        // host/provider owners, not by a fabricated aggregate allocation.
        // It can enter whole-work learning only where no finite whole-route
        // resource policy could be bypassed by that deferred admission.
        let provider = resources
            .provider(device_id)
            .ok_or(ResourceAdmissionError::UnavailableProvider)?;
        let matching_owner = matches!(
            (kind, location),
            (
                ExecutionCandidateKind::OriginalVmRegion,
                CandidateExecutionLocation::VmRegion { .. }
            ) | (
                ExecutionCandidateKind::OriginalNativeRegion,
                CandidateExecutionLocation::NativeRegion { .. }
            )
        );
        if !matching_owner
            || output_residency != CandidateOutputResidency::Unknown
            || !matches!(presentation, CandidateOutputPresentation::Direct)
            || transactional_results
            || require_transactional_results
            || demand.queue_slots != 1
            || demand.cpu_millicores != 0
            || demand.retained_bytes != CandidateResourceDemand::default().retained_bytes
            || matches!(
                demand.provider_host_workspace,
                Some(crate::placement::ResourceByteBound::Unavailable(_))
            )
            || (kind == ExecutionCandidateKind::OriginalNativeRegion
                && !matches!(
                    demand.provider_host_workspace,
                    Some(crate::placement::ResourceByteBound::Known(_))
                ))
            || demand.provider_input_stages.is_some()
            || demand.peak_live.is_some()
            || !matches!(
                demand.scratch_bytes,
                crate::placement::ResourceByteBound::Known(_)
            )
        {
            return Err(ResourceAdmissionError::Overflow);
        }
        if provider.lost {
            return Err(ResourceAdmissionError::UnavailableProvider);
        }
        if resources.cpu_millicores_available == 0
            || resources.memory_available_bytes.is_some()
            || provider.capacity_bytes.is_some()
            || provider.scratch_available_bytes.is_some()
            || provider.queue_limit.is_some()
            || provider.queue_depth.is_some()
            || !host_retained_bytes.is_empty()
            || provider_retained_bytes
                .values()
                .any(|value| !value.is_empty())
        {
            return Err(ResourceAdmissionError::InsufficientEvidence {
                budget: ResourceBudgetKind::HostMemory,
                reason: crate::placement::ResourceBoundUnavailable::NoPeakMemoryBound,
            });
        }
        return Ok(AdmittedResourceDemand {
            evictions: BTreeMap::new(),
        });
    }
    let mut evictions = BTreeMap::new();
    let mut host_peak = host_retained_bytes;
    let mut provider_demands = BTreeMap::<u32, ResourceAccount>::new();
    if let Some(peak) = demand.peak_live {
        let incremental = peak
            .incremental_bytes()
            .ok_or(ResourceAdmissionError::Overflow)?;
        if !matches!(location, CandidateExecutionLocation::Provider { .. })
            || matches!(demand.retained_bytes, crate::placement::ResourceByteBound::Known(retained) if retained > incremental)
        {
            return Err(ResourceAdmissionError::Overflow);
        }
    }
    match location {
        CandidateExecutionLocation::VmRegion { .. }
        | CandidateExecutionLocation::NativeRegion { .. }
        | CandidateExecutionLocation::NativeSource => unreachable!("handled above"),
        CandidateExecutionLocation::Host => {
            if demand.cpu_millicores > resources.cpu_millicores_available {
                return Err(ResourceAdmissionError::OverBudget);
            }
            host_peak = host_peak
                .checked_add(demand.scratch_bytes)
                .ok_or(ResourceAdmissionError::Overflow)?;
        }
        CandidateExecutionLocation::Provider { device_id } => {
            if demand.cpu_millicores > resources.cpu_millicores_available {
                return Err(ResourceAdmissionError::OverBudget);
            }
            if let Some(host_workspace) = demand.provider_host_workspace {
                host_peak = host_peak
                    .checked_add(host_workspace)
                    .ok_or(ResourceAdmissionError::Overflow)?;
            }
            let provider = resources
                .provider(device_id)
                .ok_or(ResourceAdmissionError::UnavailableProvider)?;
            if provider.lost
                || provider
                    .queue_depth
                    .zip(provider.queue_limit)
                    .is_some_and(|(depth, limit)| demand.queue_slots > limit.saturating_sub(depth))
            {
                return Err(ResourceAdmissionError::OverBudget);
            }
            // A physical peak already includes the published output and any
            // transient storage. Only the genuinely new portion is compared
            // with free capacity; scratch is not added a second time.
            let scratch = ResourceAccount::default()
                .checked_add(match demand.peak_live {
                    Some(peak) => peak
                        .incremental_bytes()
                        .ok_or(ResourceAdmissionError::Overflow)?
                        .into(),
                    None => demand.scratch_bytes,
                })
                .ok_or(ResourceAdmissionError::Overflow)?;
            scratch.within(
                provider.scratch_available_bytes,
                ResourceBudgetKind::ProviderScratch(device_id),
            )?;
            provider_demands.insert(device_id, scratch);
        }
    }
    if (kind.is_provider()
        || matches!(
            presentation,
            CandidateOutputPresentation::ProviderUpload { .. }
                | CandidateOutputPresentation::ProviderReadbackUpload { .. }
                | CandidateOutputPresentation::ProviderProducedReadback { .. }
                | CandidateOutputPresentation::ConditionalProviderUpload { .. }
        ))
        && require_transactional_results
        && !transactional_results
    {
        return Err(ResourceAdmissionError::TransactionalResultsRequired);
    }
    let upload_stage = match presentation {
        CandidateOutputPresentation::ProviderUpload {
            device_id,
            host_result_bytes,
            host_scratch_bytes,
            provider_scratch_bytes,
            queue_slots,
            ..
        }
        | CandidateOutputPresentation::ProviderReadbackUpload {
            device_id,
            host_result_bytes,
            host_scratch_bytes,
            provider_scratch_bytes,
            queue_slots,
            ..
        }
        | CandidateOutputPresentation::ConditionalProviderUpload {
            device_id,
            host_result_bytes,
            host_scratch_bytes,
            provider_scratch_bytes,
            queue_slots,
            ..
        } => Some((
            *device_id,
            *host_result_bytes,
            *host_scratch_bytes,
            *provider_scratch_bytes,
            *queue_slots,
        )),
        _ => None,
    };
    if let Some((
        device_id,
        host_result_bytes,
        host_scratch_bytes,
        provider_scratch_bytes,
        queue_slots,
    )) = upload_stage
    {
        let provider = resources
            .provider(device_id)
            .ok_or(ResourceAdmissionError::UnavailableProvider)?;
        if provider.lost
            || provider
                .queue_depth
                .zip(provider.queue_limit)
                .is_some_and(|(depth, limit)| queue_slots > limit.saturating_sub(depth))
        {
            return Err(ResourceAdmissionError::OverBudget);
        }
        let mut scratch = ResourceAccount::default()
            .checked_add(provider_scratch_bytes)
            .ok_or(ResourceAdmissionError::Overflow)?;
        if matches!(
            presentation,
            CandidateOutputPresentation::ConditionalProviderUpload { .. }
        ) {
            // The direct route's peak_live includes its own output. The
            // conditional upload is a different branch: its newly allocated
            // output coexists with upload staging even when peak_live is set.
            scratch = scratch
                .checked_add(demand.retained_bytes)
                .ok_or(ResourceAdmissionError::Overflow)?;
        }
        scratch.within(
            provider.scratch_available_bytes,
            ResourceBudgetKind::ProviderScratch(device_id),
        )?;
        provider_demands
            .entry(device_id)
            .and_modify(|output_stage| *output_stage = output_stage.peak(scratch))
            .or_insert(scratch);
        if matches!(
            presentation,
            CandidateOutputPresentation::ConditionalProviderUpload { .. }
        ) {
            let fallback_peak = host_retained_bytes
                .checked_add(host_scratch_bytes)
                .and_then(|peak| {
                    peak.checked_add(host_result_bytes.unwrap_or(demand.retained_bytes))
                })
                .ok_or(ResourceAdmissionError::Overflow)?;
            host_peak = host_peak.peak(fallback_peak);
        } else {
            host_peak = host_peak
                .checked_add(host_scratch_bytes)
                .ok_or(ResourceAdmissionError::Overflow)?;
            if matches!(output_residency, CandidateOutputResidency::Provider { .. }) {
                host_peak = host_peak
                    .checked_add(host_result_bytes.unwrap_or(demand.retained_bytes))
                    .ok_or(ResourceAdmissionError::Overflow)?;
            }
        }
    }
    let mut readback_stage_demands = BTreeMap::new();
    let readbacks = match presentation {
        CandidateOutputPresentation::ProviderReadbackUpload { readbacks, .. } => {
            Some(readbacks.as_ref())
        }
        CandidateOutputPresentation::ProviderReadbacksHost { readbacks } => {
            Some(readbacks.as_ref())
        }
        _ => None,
    };
    if let Some(readbacks) = readbacks {
        for readback in readbacks.iter() {
            let provider = resources
                .provider(readback.device_id)
                .ok_or(ResourceAdmissionError::UnavailableProvider)?;
            if provider.lost
                || provider
                    .queue_depth
                    .zip(provider.queue_limit)
                    .is_some_and(|(depth, limit)| {
                        readback.queue_slots > limit.saturating_sub(depth)
                    })
            {
                return Err(ResourceAdmissionError::OverBudget);
            }
            let scratch = ResourceAccount::default()
                .checked_add(readback.provider_scratch_bytes)
                .ok_or(ResourceAdmissionError::Overflow)?;
            scratch.within(
                provider.scratch_available_bytes,
                ResourceBudgetKind::ProviderScratch(readback.device_id),
            )?;
            readback_stage_demands.insert(readback.device_id, scratch);
        }
    }
    let mut provider_input_stage_demands = BTreeMap::new();
    if let Some(stages) = demand.provider_input_stages {
        if !matches!(location, CandidateExecutionLocation::Provider { .. })
            || !stages.is_canonical()
        {
            return Err(ResourceAdmissionError::Overflow);
        }
        for stage in stages.iter() {
            let provider = resources
                .provider(stage.device_id)
                .ok_or(ResourceAdmissionError::UnavailableProvider)?;
            if provider.lost
                || provider
                    .queue_depth
                    .zip(provider.queue_limit)
                    .is_some_and(|(depth, limit)| stage.queue_slots > limit.saturating_sub(depth))
            {
                return Err(ResourceAdmissionError::OverBudget);
            }
            let peak = ResourceAccount::default()
                .checked_add(stage.incremental_peak_bytes)
                .ok_or(ResourceAdmissionError::Overflow)?;
            peak.within(
                provider.scratch_available_bytes,
                ResourceBudgetKind::ProviderScratch(stage.device_id),
            )?;
            provider_input_stage_demands.insert(stage.device_id, peak);
        }
    }
    if let CandidateOutputPresentation::ProviderDownload {
        device_id,
        provider_scratch_bytes,
        queue_slots,
    }
    | CandidateOutputPresentation::ProviderProducedReadback {
        device_id,
        provider_scratch_bytes,
        queue_slots,
        ..
    } = presentation
    {
        let provider = resources
            .provider(*device_id)
            .ok_or(ResourceAdmissionError::UnavailableProvider)?;
        if provider.lost
            || provider
                .queue_depth
                .zip(provider.queue_limit)
                .is_some_and(|(depth, limit)| *queue_slots > limit.saturating_sub(depth))
        {
            return Err(ResourceAdmissionError::OverBudget);
        }
        let scratch = ResourceAccount::default()
            .checked_add(*provider_scratch_bytes)
            .ok_or(ResourceAdmissionError::Overflow)?;
        scratch.within(
            provider.scratch_available_bytes,
            ResourceBudgetKind::ProviderScratch(*device_id),
        )?;
        provider_demands
            .entry(*device_id)
            .and_modify(|existing| *existing = existing.peak(scratch))
            .or_insert(scratch);
    }
    let host_published_bound = match presentation {
        CandidateOutputPresentation::ProviderUpload {
            host_result_bytes, ..
        }
        | CandidateOutputPresentation::ProviderReadbackUpload {
            host_result_bytes, ..
        }
        | CandidateOutputPresentation::ConditionalProviderUpload {
            host_result_bytes, ..
        } => host_result_bytes.unwrap_or(demand.retained_bytes),
        CandidateOutputPresentation::Direct
        | CandidateOutputPresentation::ProviderDownload { .. }
        | CandidateOutputPresentation::ProviderProducedReadback { .. }
        | CandidateOutputPresentation::ProviderReadbacksHost { .. } => demand.retained_bytes,
    };
    match output_residency {
        CandidateOutputResidency::Host | CandidateOutputResidency::Unknown => {
            host_peak = host_peak
                .checked_add(demand.retained_bytes)
                .ok_or(ResourceAdmissionError::Overflow)?;
        }
        CandidateOutputResidency::Provider { device_id } => {
            let retained = provider_demands.entry(device_id).or_default();
            if demand.peak_live.is_none() {
                *retained = retained
                    .checked_add(demand.retained_bytes)
                    .ok_or(ResourceAdmissionError::Overflow)?;
            }
        }
        CandidateOutputResidency::Mirrored { device_id } => {
            host_peak = host_peak
                .checked_add(host_published_bound)
                .ok_or(ResourceAdmissionError::Overflow)?;
            let retained = provider_demands.entry(device_id).or_default();
            *retained = retained
                .checked_add(demand.retained_bytes)
                .ok_or(ResourceAdmissionError::Overflow)?;
        }
        CandidateOutputResidency::ProviderAndHost {
            device_id,
            host_bytes,
        } => {
            host_peak = host_peak
                .checked_add(host_bytes)
                .ok_or(ResourceAdmissionError::Overflow)?;
            let retained = provider_demands.entry(device_id).or_default();
            if demand.peak_live.is_none() {
                *retained = retained
                    .checked_add(demand.retained_bytes)
                    .ok_or(ResourceAdmissionError::Overflow)?;
            }
        }
        CandidateOutputResidency::ConditionalHostOrProvider {
            device_id,
            host_bytes,
        } => {
            // The provider-first route already includes its operation-owned
            // host workspace in host_peak. Its fallback result may be that
            // same retained buffer, so compare complete branch peaks rather
            // than counting it twice. The provider branch's physical peak is
            // independently covered by peak_live below.
            let host_fallback_peak = host_retained_bytes
                .checked_add(host_bytes)
                .ok_or(ResourceAdmissionError::Overflow)?;
            host_peak = host_peak.peak(host_fallback_peak);
            let retained = provider_demands.entry(device_id).or_default();
            if demand.peak_live.is_none() {
                *retained = retained
                    .checked_add(demand.retained_bytes)
                    .ok_or(ResourceAdmissionError::Overflow)?;
            }
        }
    }
    // A readback finishes before the output upload/allocation. Peak each
    // source-owner stage against the complete output stage (including its
    // retained allocation), then add graph predecessors that remain live
    // throughout both stages below. Summing both stages would reject a legal
    // same-owner route at an exact finite-capacity boundary.
    for (device_id, scratch) in readback_stage_demands {
        provider_demands
            .entry(device_id)
            .and_modify(|output_stage| *output_stage = output_stage.peak(scratch))
            .or_insert(scratch);
    }
    // Input preparation has completed before the final provider execution
    // stage. Each stage bound already includes operation-created allocations
    // retained across it; borrowed/provider graph values are added once below.
    for (device_id, peak) in provider_input_stage_demands {
        provider_demands
            .entry(device_id)
            .and_modify(|output_stage| *output_stage = output_stage.peak(peak))
            .or_insert(peak);
    }
    host_peak.within(
        resources.memory_available_bytes,
        ResourceBudgetKind::HostMemory,
    )?;
    for (device_id, candidate_bytes) in provider_demands {
        let provider = resources
            .provider(device_id)
            .ok_or(ResourceAdmissionError::UnavailableProvider)?;
        if provider.lost {
            return Err(ResourceAdmissionError::UnavailableProvider);
        }
        let graph_retained = provider_retained_bytes
            .get(&device_id)
            .copied()
            .unwrap_or_default();
        let required_peak = candidate_bytes
            .combine(graph_retained)
            .ok_or(ResourceAdmissionError::Overflow)?;
        if let Some(after_eviction) = provider.available_after_eviction_bytes() {
            required_peak.within(
                Some(after_eviction),
                ResourceBudgetKind::ProviderMemory(device_id),
            )?;
            let immediately_available = provider
                .immediately_available_bytes()
                .expect("known post-eviction capacity has known immediate capacity");
            if required_peak.known_bytes > immediately_available {
                let eviction_bytes = required_peak.known_bytes - immediately_available;
                evictions.insert(device_id, eviction_bytes);
            }
        }
    }
    Ok(AdmittedResourceDemand { evictions })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::{ProviderResourceSnapshot, ResourceBoundUnavailable, ResourceByteBound};

    #[test]
    fn unbound_native_source_is_not_a_finite_cpu_or_provider_certificate() {
        let check = |resources: &PlacementResourceSnapshot, kind| {
            admit_candidate_resources(
                CandidateResourceFacts {
                    location: CandidateExecutionLocation::NativeSource,
                    output_residency: CandidateOutputResidency::Unknown,
                    presentation: &CandidateOutputPresentation::Direct,
                    kind,
                    demand: CandidateResourceDemand {
                        retained_bytes: ResourceByteBound::Unavailable(
                            ResourceBoundUnavailable::NoPeakMemoryBound,
                        ),
                        scratch_bytes: ResourceByteBound::Unavailable(
                            ResourceBoundUnavailable::NoPeakMemoryBound,
                        ),
                        ..Default::default()
                    },
                    transactional_results: false,
                },
                ResourceAdmissionContext {
                    resources,
                    host_retained_bytes: ResourceAccount::default(),
                    provider_retained_bytes: &BTreeMap::new(),
                    require_transactional_results: false,
                },
            )
        };
        let mut resources = PlacementResourceSnapshot {
            cpu_millicores_available: 1000,
            ..Default::default()
        };
        assert!(check(&resources, ExecutionCandidateKind::OriginalNativeRegion).is_ok());
        assert!(check(&resources, ExecutionCandidateKind::SpecializedNativeCpu).is_err());
        resources.memory_available_bytes = Some(1024);
        assert!(matches!(
            check(&resources, ExecutionCandidateKind::OriginalNativeRegion),
            Err(ResourceAdmissionError::InsufficientEvidence {
                reason: ResourceBoundUnavailable::NoPeakMemoryBound,
                ..
            })
        ));
        assert!(ExecutionCandidateKind::OriginalNativeRegion
            .matches_location(CandidateExecutionLocation::NativeSource));
        assert!(!ExecutionCandidateKind::VectorizedNativeCpu
            .matches_location(CandidateExecutionLocation::NativeSource));
    }

    #[test]
    fn physical_eviction_and_unavailable_scratch_remain_distinct_from_timing() {
        let mut resources = PlacementResourceSnapshot {
            cpu_millicores_available: 1000,
            providers: vec![ProviderResourceSnapshot {
                device_id: 7,
                capacity_bytes: Some(100),
                live_bytes: 90,
                reclaimable_bytes: 20,
                scratch_available_bytes: Some(30),
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut demand = CandidateResourceDemand {
            retained_bytes: ResourceByteBound::Known(20),
            scratch_bytes: ResourceByteBound::Known(0),
            ..Default::default()
        };
        let check = |resources: &PlacementResourceSnapshot, demand| {
            admit_candidate_resources(
                CandidateResourceFacts {
                    location: CandidateExecutionLocation::Provider { device_id: 7 },
                    output_residency: CandidateOutputResidency::Provider { device_id: 7 },
                    presentation: &CandidateOutputPresentation::Direct,
                    kind: ExecutionCandidateKind::ProviderOperation,
                    demand,
                    transactional_results: false,
                },
                ResourceAdmissionContext {
                    resources,
                    host_retained_bytes: ResourceAccount::default(),
                    provider_retained_bytes: &BTreeMap::new(),
                    require_transactional_results: false,
                },
            )
        };
        assert_eq!(
            check(&resources, demand)
                .unwrap()
                .provider_evictions()
                .collect::<Vec<_>>(),
            [(7, 10)]
        );
        demand.scratch_bytes =
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound);
        assert!(matches!(
            check(&resources, demand),
            Err(ResourceAdmissionError::InsufficientEvidence {
                reason: ResourceBoundUnavailable::NoScratchBound,
                budget: ResourceBudgetKind::ProviderScratch(7)
            })
        ));
        demand.scratch_bytes = ResourceByteBound::Known(0);
        resources.providers[0].lost = true;
        assert!(matches!(
            check(&resources, demand),
            Err(ResourceAdmissionError::OverBudget)
        ));
        resources.providers[0].lost = false;
        let valid = resources.providers[0].clone();
        for malformed in [
            crate::placement::ProviderResourceSnapshot {
                queue_depth: Some(1),
                queue_limit: None,
                ..valid.clone()
            },
            crate::placement::ProviderResourceSnapshot {
                reclaimable_bytes: 91,
                ..valid.clone()
            },
            crate::placement::ProviderResourceSnapshot {
                live_bytes: 101,
                ..valid.clone()
            },
            crate::placement::ProviderResourceSnapshot {
                scratch_available_bytes: Some(101),
                ..valid.clone()
            },
        ] {
            resources.providers[0] = malformed;
            assert!(matches!(
                check(&resources, demand),
                Err(ResourceAdmissionError::InvalidSnapshot(
                    crate::placement::ResourceSnapshotError::InconsistentProvider { device_id: 7 }
                ))
            ));
        }
    }
}
