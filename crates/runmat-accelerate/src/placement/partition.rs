use std::cmp::Ordering;
use std::collections::BTreeMap;

use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputPresentation, CandidateOutputResidency,
    CandidateResourceDemand, ExecutionCandidateKind, PlacementGraphCandidate, PlacementPlanRequest,
    PlacementResourceSnapshot, SelectedExecutionCandidate,
};

use super::PlacementPolicy;

mod resource_account;
pub(super) use resource_account::ResourceAdmissionError;
use resource_account::{ResourceAccount, ResourceBudgetKind};

#[derive(Clone, Debug)]
pub(super) struct PartitionPlan {
    pub(super) selections: Vec<usize>,
    pub(super) total_ns: u64,
    pub(super) explored_states: u32,
    pub(super) pruned_states: u32,
    pub(super) used_local_fallback: bool,
    pub(super) alternate: Option<(Vec<usize>, u64)>,
}

#[derive(Clone, Debug)]
struct SearchState {
    selections: Vec<usize>,
    total_ns: u64,
    host_retained_bytes: ResourceAccount,
    provider_retained_bytes: BTreeMap<u32, ResourceAccount>,
}

pub(super) fn solve_partition(
    request: &PlacementPlanRequest,
    policy: PlacementPolicy,
) -> Result<PartitionPlan, &'static str> {
    request.validate()?;
    if request.resources.cancellation_requested {
        return Err("placement cancelled before dispatch");
    }
    let mut frontier = vec![SearchState {
        selections: Vec::new(),
        total_ns: 0,
        host_retained_bytes: ResourceAccount::default(),
        provider_retained_bytes: BTreeMap::new(),
    }];
    let mut explored = 0_u32;
    let mut pruned = 0_u32;
    for (node_index, node) in request.graph.nodes.iter().enumerate() {
        let mut expanded = Vec::new();
        for state in &frontier {
            for (candidate_index, candidate) in node.candidates.iter().enumerate() {
                explored = explored.saturating_add(1);
                if explored > request.limits.max_expansions {
                    return local_fallback(request, explored, pruned);
                }
                let Some(candidate_ns) = candidate_total_ns(candidate, request, state) else {
                    continue;
                };
                let Some(transition_ns) = predecessor_transition_ns(
                    request,
                    node_index,
                    &state.selections,
                    candidate.descriptor.execution_location,
                ) else {
                    continue;
                };
                let Some(total_ns) = state
                    .total_ns
                    .checked_add(candidate_ns)
                    .and_then(|total| total.checked_add(transition_ns))
                else {
                    continue;
                };
                let mut selections = state.selections.clone();
                selections.push(candidate_index);
                let Some((host_retained_bytes, provider_retained_bytes)) =
                    next_retained_state(request, node_index, state, candidate)
                else {
                    continue;
                };
                expanded.push(SearchState {
                    selections,
                    total_ns,
                    host_retained_bytes,
                    provider_retained_bytes,
                });
            }
        }
        if expanded.is_empty() {
            return local_fallback(request, explored, pruned);
        }
        expanded.sort_by(|left, right| compare_states(left, right, request));
        let frontier_limit = request.limits.max_frontier_states as usize;
        if expanded.len() > frontier_limit {
            pruned = pruned
                .saturating_add(u32::try_from(expanded.len() - frontier_limit).unwrap_or(u32::MAX));
            expanded.truncate(frontier_limit);
        }
        frontier = expanded;
    }
    frontier.sort_by(|left, right| compare_states(left, right, request));
    let best = frontier.remove(0);
    let alternate = frontier
        .into_iter()
        .find(|candidate| candidate.selections != best.selections)
        .map(|candidate| (candidate.selections, candidate.total_ns));
    let local = local_fallback(request, explored, pruned)?;
    if uses_provider(request, &best.selections) {
        let required = policy
            .required_improvement_ns(local.total_ns)
            .unwrap_or(u64::MAX);
        if best
            .total_ns
            .checked_add(required)
            .is_none_or(|cost| cost > local.total_ns)
        {
            return Ok(PartitionPlan {
                alternate: Some((best.selections, best.total_ns)),
                explored_states: explored,
                pruned_states: pruned,
                used_local_fallback: false,
                ..local
            });
        }
    }
    Ok(PartitionPlan {
        selections: best.selections,
        total_ns: best.total_ns,
        explored_states: explored,
        pruned_states: pruned,
        used_local_fallback: false,
        alternate,
    })
}

pub(super) fn evaluate_selections(
    request: &PlacementPlanRequest,
    selected: &[SelectedExecutionCandidate],
) -> Option<u64> {
    if selected.len() != request.graph.nodes.len() {
        return None;
    }
    let mut state = SearchState {
        selections: Vec::with_capacity(selected.len()),
        total_ns: 0,
        host_retained_bytes: ResourceAccount::default(),
        provider_retained_bytes: BTreeMap::new(),
    };
    for (node_index, (node, selected)) in request.graph.nodes.iter().zip(selected).enumerate() {
        if selected.node != node.identity {
            return None;
        }
        let candidate_index = node.candidates.iter().position(|candidate| {
            candidate.descriptor.identity == selected.candidate
                && candidate.descriptor.kind == selected.kind
        })?;
        let candidate = &node.candidates[candidate_index];
        let candidate_ns = candidate_total_ns(candidate, request, &state)?;
        let transition_ns = predecessor_transition_ns(
            request,
            node_index,
            &state.selections,
            candidate.descriptor.execution_location,
        )?;
        state.total_ns = state
            .total_ns
            .checked_add(candidate_ns)?
            .checked_add(transition_ns)?;
        let (host_retained_bytes, provider_retained_bytes) =
            next_retained_state(request, node_index, &state, candidate)?;
        state.selections.push(candidate_index);
        state.host_retained_bytes = host_retained_bytes;
        state.provider_retained_bytes = provider_retained_bytes;
    }
    Some(state.total_ns)
}

fn uses_provider(request: &PlacementPlanRequest, selections: &[usize]) -> bool {
    selections.iter().enumerate().any(|(node, candidate)| {
        request.graph.nodes[node].candidates[*candidate]
            .descriptor
            .kind
            .is_provider()
    })
}

fn local_fallback(
    request: &PlacementPlanRequest,
    explored_states: u32,
    pruned_states: u32,
) -> Result<PartitionPlan, &'static str> {
    let mut selections = Vec::with_capacity(request.graph.nodes.len());
    let mut total_ns = 0_u64;
    let mut resource_state = SearchState {
        selections: Vec::new(),
        total_ns: 0,
        host_retained_bytes: ResourceAccount::default(),
        provider_retained_bytes: BTreeMap::new(),
    };
    for (node_index, node) in request.graph.nodes.iter().enumerate() {
        let best = node
            .candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| !candidate.descriptor.kind.is_provider())
            .filter_map(|(index, candidate)| {
                candidate_total_ns(candidate, request, &resource_state).map(|cost| (index, cost))
            })
            .min_by(|left, right| {
                left.1.cmp(&right.1).then_with(|| {
                    node.candidates[left.0]
                        .descriptor
                        .identity
                        .cmp(&node.candidates[right.0].descriptor.identity)
                })
            })
            .ok_or("placement node has no legal local fallback")?;
        let transition = predecessor_transition_ns(
            request,
            node_index,
            &selections,
            node.candidates[best.0].descriptor.execution_location,
        )
        .ok_or("local fallback transition overflow")?;
        total_ns = total_ns
            .checked_add(best.1)
            .and_then(|total| total.checked_add(transition))
            .ok_or("local fallback cost overflow")?;
        selections.push(best.0);
        let (host_retained_bytes, provider_retained_bytes) = next_retained_state(
            request,
            node_index,
            &resource_state,
            &node.candidates[best.0],
        )
        .ok_or("local fallback resource accounting overflow")?;
        resource_state.selections.push(best.0);
        resource_state.host_retained_bytes = host_retained_bytes;
        resource_state.provider_retained_bytes = provider_retained_bytes;
    }
    Ok(PartitionPlan {
        selections,
        total_ns,
        explored_states,
        pruned_states,
        used_local_fallback: true,
        alternate: None,
    })
}

fn candidate_total_ns(
    candidate: &PlacementGraphCandidate,
    request: &PlacementPlanRequest,
    state: &SearchState,
) -> Option<u64> {
    candidate.descriptor.validate().ok()?;
    let mut total = candidate.descriptor.cost.checked_risk_adjusted_ns()?;
    total = total.checked_add(
        candidate_resource_adjustment_ns(
            CandidateResourceFacts {
                location: candidate.descriptor.execution_location,
                output_residency: candidate.descriptor.output_residency,
                presentation: candidate.descriptor.presentation,
                kind: candidate.descriptor.kind,
                demand: candidate.resources,
                transactional_results: candidate.transactional_results,
            },
            ResourceAdmissionContext {
                resources: &request.resources,
                host_retained_bytes: state.host_retained_bytes,
                provider_retained_bytes: &state.provider_retained_bytes,
                require_transactional_results: request.require_transactional_results,
            },
        )
        .ok()?,
    )?;
    Some(total)
}

/// Resource-only admission shared by numeric graph placement and exploratory
/// whole-work selection. It never constructs a synthetic cost estimate.
pub(super) fn single_candidate_resources_admitted(
    candidate: CandidateResourceFacts,
    resources: &PlacementResourceSnapshot,
    require_transactional_results: bool,
) -> Result<(), ResourceAdmissionError> {
    candidate_resource_adjustment_ns(
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

pub(super) struct CandidateResourceFacts {
    pub(super) location: CandidateExecutionLocation,
    pub(super) output_residency: CandidateOutputResidency,
    pub(super) presentation: CandidateOutputPresentation,
    pub(super) kind: ExecutionCandidateKind,
    pub(super) demand: CandidateResourceDemand,
    pub(super) transactional_results: bool,
}

struct ResourceAdmissionContext<'a> {
    resources: &'a PlacementResourceSnapshot,
    host_retained_bytes: ResourceAccount,
    provider_retained_bytes: &'a BTreeMap<u32, ResourceAccount>,
    require_transactional_results: bool,
}

fn candidate_resource_adjustment_ns(
    candidate: CandidateResourceFacts,
    admission: ResourceAdmissionContext<'_>,
) -> Result<u64, ResourceAdmissionError> {
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
    let mut adjustment = 0_u64;
    let mut host_peak = host_retained_bytes;
    let mut provider_demands = BTreeMap::<u32, ResourceAccount>::new();
    if let Some(peak) = demand.peak_live {
        let incremental = peak
            .incremental_bytes()
            .ok_or(ResourceAdmissionError::Overflow)?;
        if !matches!(location, CandidateExecutionLocation::Provider { .. })
            || matches!(demand.retained_bytes, runmat_execution::ResourceByteBound::Known(retained) if retained > incremental)
        {
            return Err(ResourceAdmissionError::Overflow);
        }
    }
    match location {
        CandidateExecutionLocation::Host => {
            if demand.cpu_millicores > resources.cpu_millicores_available {
                return Err(ResourceAdmissionError::OverBudget);
            }
            host_peak = host_peak
                .add(demand.scratch_bytes)
                .ok_or(ResourceAdmissionError::Overflow)?;
        }
        CandidateExecutionLocation::Provider { device_id } => {
            if demand.cpu_millicores > resources.cpu_millicores_available {
                return Err(ResourceAdmissionError::OverBudget);
            }
            if let Some(host_workspace) = demand.provider_host_workspace {
                host_peak = host_peak
                    .add(host_workspace)
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
                .add(match demand.peak_live {
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
        } => Some((
            device_id,
            host_result_bytes,
            host_scratch_bytes,
            provider_scratch_bytes,
            queue_slots,
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
        let scratch = ResourceAccount::default()
            .add(provider_scratch_bytes)
            .ok_or(ResourceAdmissionError::Overflow)?;
        scratch.within(
            provider.scratch_available_bytes,
            ResourceBudgetKind::ProviderScratch(device_id),
        )?;
        provider_demands.insert(device_id, scratch);
        host_peak = host_peak
            .add(host_scratch_bytes)
            .ok_or(ResourceAdmissionError::Overflow)?;
        // The host-produced value remains live while its provider presentation
        // is uploaded. The provider's retained output is accounted below.
        if matches!(output_residency, CandidateOutputResidency::Provider { .. }) {
            host_peak = host_peak
                .add(host_result_bytes.unwrap_or(demand.retained_bytes))
                .ok_or(ResourceAdmissionError::Overflow)?;
        }
    }
    let mut readback_stage_demands = BTreeMap::new();
    if let CandidateOutputPresentation::ProviderReadbackUpload { readbacks, .. } = presentation {
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
                .add(readback.provider_scratch_bytes)
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
                .add(stage.incremental_peak_bytes)
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
    } = presentation
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
        let scratch = ResourceAccount::default()
            .add(provider_scratch_bytes)
            .ok_or(ResourceAdmissionError::Overflow)?;
        scratch.within(
            provider.scratch_available_bytes,
            ResourceBudgetKind::ProviderScratch(device_id),
        )?;
        provider_demands.insert(device_id, scratch);
    }
    let host_published_bound = match presentation {
        CandidateOutputPresentation::ProviderUpload {
            host_result_bytes, ..
        }
        | CandidateOutputPresentation::ProviderReadbackUpload {
            host_result_bytes, ..
        } => host_result_bytes.unwrap_or(demand.retained_bytes),
        CandidateOutputPresentation::Direct
        | CandidateOutputPresentation::ProviderDownload { .. } => demand.retained_bytes,
    };
    match output_residency {
        CandidateOutputResidency::Host | CandidateOutputResidency::Unknown => {
            host_peak = host_peak
                .add(demand.retained_bytes)
                .ok_or(ResourceAdmissionError::Overflow)?;
        }
        CandidateOutputResidency::Provider { device_id } => {
            let retained = provider_demands.entry(device_id).or_default();
            if demand.peak_live.is_none() {
                *retained = retained
                    .add(demand.retained_bytes)
                    .ok_or(ResourceAdmissionError::Overflow)?;
            }
        }
        CandidateOutputResidency::Mirrored { device_id } => {
            host_peak = host_peak
                .add(host_published_bound)
                .ok_or(ResourceAdmissionError::Overflow)?;
            let retained = provider_demands.entry(device_id).or_default();
            *retained = retained
                .add(demand.retained_bytes)
                .ok_or(ResourceAdmissionError::Overflow)?;
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
                .add(host_bytes)
                .ok_or(ResourceAdmissionError::Overflow)?;
            host_peak = host_peak.peak(host_fallback_peak);
            let retained = provider_demands.entry(device_id).or_default();
            if demand.peak_live.is_none() {
                *retained = retained
                    .add(demand.retained_bytes)
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
                adjustment = eviction_adjustment_ns(adjustment, eviction_bytes)
                    .ok_or(ResourceAdmissionError::Overflow)?;
            }
        }
    }
    Ok(adjustment)
}

fn eviction_adjustment_ns(current: u64, eviction_bytes: u64) -> Option<u64> {
    current
        .checked_add(50_000)?
        .checked_add(eviction_bytes.div_ceil(8))
}

fn next_retained_state(
    request: &PlacementPlanRequest,
    node_index: usize,
    state: &SearchState,
    candidate: &PlacementGraphCandidate,
) -> Option<(ResourceAccount, BTreeMap<u32, ResourceAccount>)> {
    let mut host = state.host_retained_bytes;
    let mut providers = state.provider_retained_bytes.clone();
    for predecessor in 0..node_index {
        let last_consumer = request
            .graph
            .edges
            .iter()
            .filter(|edge| edge.from as usize == predecessor)
            .map(|edge| edge.to as usize)
            .max();
        if last_consumer != Some(node_index) {
            continue;
        }
        let Some(selected) = state
            .selections
            .get(predecessor)
            .and_then(|selection| request.graph.nodes[predecessor].candidates.get(*selection))
        else {
            continue;
        };
        release_retained(&mut host, &mut providers, selected)?;
    }
    let has_future_consumer = request
        .graph
        .edges
        .iter()
        .any(|edge| edge.from as usize == node_index && edge.to as usize > node_index);
    if has_future_consumer {
        retain_output(&mut host, &mut providers, candidate)?;
    }
    Some((host, providers))
}

fn retain_output(
    host: &mut ResourceAccount,
    providers: &mut BTreeMap<u32, ResourceAccount>,
    candidate: &PlacementGraphCandidate,
) -> Option<()> {
    match candidate.descriptor.output_residency {
        CandidateOutputResidency::Host | CandidateOutputResidency::Unknown => {
            *host = host.add(candidate.resources.retained_bytes)?;
        }
        CandidateOutputResidency::Provider { device_id } => {
            let retained = providers.entry(device_id).or_default();
            *retained = retained.add(candidate.resources.retained_bytes)?;
        }
        CandidateOutputResidency::Mirrored { device_id } => {
            *host = host.add(candidate.resources.retained_bytes)?;
            let retained = providers.entry(device_id).or_default();
            *retained = retained.add(candidate.resources.retained_bytes)?;
        }
        CandidateOutputResidency::ConditionalHostOrProvider {
            device_id,
            host_bytes,
        } => {
            // The static graph cannot know which branch publishes at runtime;
            // keep both possible values live for future-consumer admission.
            *host = host.add(host_bytes)?;
            let retained = providers.entry(device_id).or_default();
            *retained = retained.add(candidate.resources.retained_bytes)?;
        }
    }
    Some(())
}

fn release_retained(
    host: &mut ResourceAccount,
    providers: &mut BTreeMap<u32, ResourceAccount>,
    candidate: &PlacementGraphCandidate,
) -> Option<()> {
    match candidate.descriptor.output_residency {
        CandidateOutputResidency::Host | CandidateOutputResidency::Unknown => {
            *host = host.subtract(candidate.resources.retained_bytes)?;
        }
        CandidateOutputResidency::Provider { device_id } => {
            if let Some(retained) = providers.get_mut(&device_id) {
                *retained = retained.subtract(candidate.resources.retained_bytes)?;
                if retained.known_bytes == 0 && retained.unavailable_bounds.is_empty() {
                    providers.remove(&device_id);
                }
            }
        }
        CandidateOutputResidency::Mirrored { device_id } => {
            *host = host.subtract(candidate.resources.retained_bytes)?;
            if let Some(retained) = providers.get_mut(&device_id) {
                *retained = retained.subtract(candidate.resources.retained_bytes)?;
                if retained.known_bytes == 0 && retained.unavailable_bounds.is_empty() {
                    providers.remove(&device_id);
                }
            }
        }
        CandidateOutputResidency::ConditionalHostOrProvider {
            device_id,
            host_bytes,
        } => {
            *host = host.subtract(host_bytes)?;
            if let Some(retained) = providers.get_mut(&device_id) {
                *retained = retained.subtract(candidate.resources.retained_bytes)?;
                if retained.known_bytes == 0 && retained.unavailable_bounds.is_empty() {
                    providers.remove(&device_id);
                }
            }
        }
    }
    Some(())
}

fn predecessor_transition_ns(
    request: &PlacementPlanRequest,
    node_index: usize,
    selections: &[usize],
    execution_location: CandidateExecutionLocation,
) -> Option<u64> {
    request
        .graph
        .edges
        .iter()
        .filter(|edge| edge.to as usize == node_index)
        .try_fold(0_u64, |total, edge| {
            let predecessor = request
                .graph
                .nodes
                .get(edge.from as usize)?
                .candidates
                .get(*selections.get(edge.from as usize)?)?;
            total.checked_add(transition_ns(
                predecessor.descriptor.output_residency,
                execution_location,
                edge,
            ))
        })
}

fn transition_ns(
    from: CandidateOutputResidency,
    to: CandidateExecutionLocation,
    edge: &runmat_execution::PlacementGraphEdge,
) -> u64 {
    use CandidateExecutionLocation::{Host as ExecuteHost, Provider as ExecuteProvider};
    use CandidateOutputResidency::{Host, Mirrored, Provider, Unknown};
    match (from, to) {
        (Host, ExecuteHost) | (Mirrored { .. }, ExecuteHost) => 0,
        (Host, ExecuteProvider { .. }) => edge.host_to_provider_ns,
        (Provider { .. }, ExecuteHost) => edge.provider_to_host_ns,
        (Provider { device_id: left }, ExecuteProvider { device_id: right }) if left == right => 0,
        (Mirrored { device_id: left }, ExecuteProvider { device_id: right }) if left == right => 0,
        (Provider { .. }, ExecuteProvider { .. }) | (Mirrored { .. }, ExecuteProvider { .. }) => {
            edge.cross_provider_ns
        }
        (Unknown, ExecuteHost) => edge.provider_to_host_ns,
        (Unknown, ExecuteProvider { .. }) => edge.cross_provider_ns,
    }
}

fn compare_states(
    left: &SearchState,
    right: &SearchState,
    request: &PlacementPlanRequest,
) -> Ordering {
    left.total_ns.cmp(&right.total_ns).then_with(|| {
        candidate_identity_sequence(left, request).cmp(&candidate_identity_sequence(right, request))
    })
}

fn candidate_identity_sequence<'a>(
    state: &'a SearchState,
    request: &'a PlacementPlanRequest,
) -> Vec<&'a str> {
    state
        .selections
        .iter()
        .enumerate()
        .filter_map(|(node, candidate)| {
            request.graph.nodes[node]
                .candidates
                .get(*candidate)
                .map(|candidate| candidate.descriptor.identity.as_str())
        })
        .collect()
}

#[cfg(test)]
mod eviction_boundary_tests {
    use super::eviction_adjustment_ns;

    #[test]
    fn maximum_eviction_bytes_round_up_without_saturation() {
        assert_eq!(u64::MAX.div_ceil(8), 2_305_843_009_213_693_952);
        assert_eq!(
            eviction_adjustment_ns(0, u64::MAX),
            Some(2_305_843_009_213_743_952)
        );
        assert_eq!(eviction_adjustment_ns(u64::MAX, 1), None);
    }
}
