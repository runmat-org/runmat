use std::cmp::Ordering;
use std::collections::BTreeMap;

use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputResidency, PlacementGraphCandidate,
    PlacementPlanRequest, SelectedExecutionCandidate,
};

use super::PlacementPolicy;

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
    host_retained_bytes: u64,
    provider_retained_bytes: BTreeMap<u32, u64>,
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
        host_retained_bytes: 0,
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
                let (host_retained_bytes, provider_retained_bytes) =
                    next_retained_state(request, node_index, state, candidate);
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
        host_retained_bytes: 0,
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
            next_retained_state(request, node_index, &state, candidate);
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
        host_retained_bytes: 0,
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
        );
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
    let demand = candidate.resources;
    let mut host_peak = state.host_retained_bytes;
    let mut provider_demands = BTreeMap::<u32, u64>::new();
    match candidate.descriptor.execution_location {
        CandidateExecutionLocation::Host => {
            if demand.cpu_millicores > request.resources.cpu_millicores_available {
                return None;
            }
            host_peak = host_peak.checked_add(demand.scratch_bytes)?;
        }
        CandidateExecutionLocation::Provider { device_id } => {
            let provider = request.resources.provider(device_id)?;
            if provider.lost
                || provider
                    .scratch_available_bytes
                    .is_some_and(|available| demand.scratch_bytes > available)
                || provider
                    .queue_depth
                    .zip(provider.queue_limit)
                    .is_some_and(|(depth, limit)| demand.queue_slots > limit.saturating_sub(depth))
            {
                return None;
            }
            *provider_demands.entry(device_id).or_default() = demand.scratch_bytes;
        }
    }
    if candidate.descriptor.kind.is_provider()
        && request.require_transactional_results
        && !candidate.transactional_results
    {
        return None;
    }
    match candidate.descriptor.output_residency {
        CandidateOutputResidency::Host | CandidateOutputResidency::Unknown => {
            host_peak = host_peak.checked_add(demand.retained_bytes)?;
        }
        CandidateOutputResidency::Provider { device_id } => {
            let retained = provider_demands.entry(device_id).or_default();
            *retained = retained.checked_add(demand.retained_bytes)?;
        }
        CandidateOutputResidency::Mirrored { device_id } => {
            host_peak = host_peak.checked_add(demand.retained_bytes)?;
            let retained = provider_demands.entry(device_id).or_default();
            *retained = retained.checked_add(demand.retained_bytes)?;
        }
    }
    if request
        .resources
        .memory_available_bytes
        .is_some_and(|available| host_peak > available)
    {
        return None;
    }
    for (device_id, candidate_bytes) in provider_demands {
        let provider = request.resources.provider(device_id)?;
        if provider.lost {
            return None;
        }
        let graph_retained = state
            .provider_retained_bytes
            .get(&device_id)
            .copied()
            .unwrap_or(0);
        let required_peak = candidate_bytes.checked_add(graph_retained)?;
        if let Some(after_eviction) = provider.available_after_eviction_bytes() {
            if required_peak > after_eviction {
                return None;
            }
            let immediately_available = provider
                .immediately_available_bytes()
                .expect("known post-eviction capacity has known immediate capacity");
            if required_peak > immediately_available {
                let eviction_bytes = required_peak - immediately_available;
                total = total
                    .checked_add(50_000)?
                    .checked_add(eviction_bytes.saturating_add(7) / 8)?;
            }
        }
    }
    Some(total)
}

fn next_retained_state(
    request: &PlacementPlanRequest,
    node_index: usize,
    state: &SearchState,
    candidate: &PlacementGraphCandidate,
) -> (u64, BTreeMap<u32, u64>) {
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
        release_retained(&mut host, &mut providers, selected);
    }
    let has_future_consumer = request
        .graph
        .edges
        .iter()
        .any(|edge| edge.from as usize == node_index && edge.to as usize > node_index);
    if has_future_consumer {
        retain_output(&mut host, &mut providers, candidate);
    }
    (host, providers)
}

fn retain_output(
    host: &mut u64,
    providers: &mut BTreeMap<u32, u64>,
    candidate: &PlacementGraphCandidate,
) {
    match candidate.descriptor.output_residency {
        CandidateOutputResidency::Host | CandidateOutputResidency::Unknown => {
            *host = host.saturating_add(candidate.resources.retained_bytes);
        }
        CandidateOutputResidency::Provider { device_id } => {
            let retained = providers.entry(device_id).or_default();
            *retained = retained.saturating_add(candidate.resources.retained_bytes);
        }
        CandidateOutputResidency::Mirrored { device_id } => {
            *host = host.saturating_add(candidate.resources.retained_bytes);
            let retained = providers.entry(device_id).or_default();
            *retained = retained.saturating_add(candidate.resources.retained_bytes);
        }
    }
}

fn release_retained(
    host: &mut u64,
    providers: &mut BTreeMap<u32, u64>,
    candidate: &PlacementGraphCandidate,
) {
    match candidate.descriptor.output_residency {
        CandidateOutputResidency::Host | CandidateOutputResidency::Unknown => {
            *host = host.saturating_sub(candidate.resources.retained_bytes);
        }
        CandidateOutputResidency::Provider { device_id } => {
            if let Some(retained) = providers.get_mut(&device_id) {
                *retained = retained.saturating_sub(candidate.resources.retained_bytes);
                if *retained == 0 {
                    providers.remove(&device_id);
                }
            }
        }
        CandidateOutputResidency::Mirrored { device_id } => {
            *host = host.saturating_sub(candidate.resources.retained_bytes);
            if let Some(retained) = providers.get_mut(&device_id) {
                *retained = retained.saturating_sub(candidate.resources.retained_bytes);
                if *retained == 0 {
                    providers.remove(&device_id);
                }
            }
        }
    }
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
