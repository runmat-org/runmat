use std::cmp::Ordering;
use std::collections::BTreeMap;

use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputResidency, PlacementGraphCandidate,
    PlacementPlanRequest, PlacementResourceSnapshot, SelectedExecutionCandidate,
    SourceOutputOrdinal,
};

use super::{session::PreparedPlacementRequest, PlacementPolicy};

mod retention;
use retention::RetentionSchedule;
use runmat_execution::{admit_candidate_resources, ResourceAccount, ResourceAdmissionContext};
pub(super) use runmat_execution::{CandidateResourceFacts, ResourceAdmissionError};

#[derive(Clone, Debug)]
pub(super) struct PartitionPlan {
    pub(super) selections: Vec<usize>,
    pub(super) total_ns: u64,
    pub(super) explored_states: u32,
    pub(super) pruned_states: u32,
    pub(super) used_local_fallback: bool,
    /// Feasible alternatives in canonical cost order, bounded by the search frontier.
    pub(super) alternates: Vec<(Vec<usize>, u64)>,
}

#[derive(Clone, Debug)]
struct SearchState {
    selections: Vec<usize>,
    total_ns: u64,
    host_retained_bytes: ResourceAccount,
    provider_retained_bytes: BTreeMap<u32, ResourceAccount>,
}

pub(super) fn solve_partition(
    prepared: &PreparedPlacementRequest,
    policy: PlacementPolicy,
) -> Result<PartitionPlan, &'static str> {
    // The private carrier proves the original graph passed canonical
    // validation before feedback could adjust costs or prune candidates.
    let request = prepared.request();
    if request.resources.cancellation_requested {
        return Err("placement cancelled before dispatch");
    }
    let retention = RetentionSchedule::new(request);
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
                    return local_fallback(request, &retention, explored, pruned)
                        .ok_or("bounded search has no executable host fallback");
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
                    next_retained_state(request, &retention, node_index, state, candidate)
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
            return local_fallback(request, &retention, explored, pruned)
                .ok_or("no complete placement route has admissible resource evidence");
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
    let mut alternates = frontier
        .into_iter()
        .filter(|candidate| candidate.selections != best.selections)
        .map(|candidate| (candidate.selections, candidate.total_ns))
        .collect::<Vec<_>>();
    let local = local_fallback(request, &retention, explored, pruned);
    if let Some(local) = local.filter(|_| uses_provider(request, &best.selections)) {
        let required = policy
            .required_improvement_ns(local.total_ns)
            .unwrap_or(u64::MAX);
        if best
            .total_ns
            .checked_add(required)
            .is_none_or(|cost| cost > local.total_ns)
        {
            alternates.insert(0, (best.selections, best.total_ns));
            alternates.retain(|(selections, _)| selections != &local.selections);
            return Ok(PartitionPlan {
                alternates,
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
        alternates,
    })
}

pub(super) fn evaluate_selections(
    prepared: &PreparedPlacementRequest,
    selected: &[SelectedExecutionCandidate],
) -> Option<u64> {
    let request = prepared.request();
    if selected.len() != request.graph.nodes.len() {
        return None;
    }
    let retention = RetentionSchedule::new(request);
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
            next_retained_state(request, &retention, node_index, &state, candidate)?;
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
    retention: &RetentionSchedule,
    explored_states: u32,
    pruned_states: u32,
) -> Option<PartitionPlan> {
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
            })?;
        let transition = predecessor_transition_ns(
            request,
            node_index,
            &selections,
            node.candidates[best.0].descriptor.execution_location,
        )?;
        total_ns = total_ns
            .checked_add(best.1)
            .and_then(|total| total.checked_add(transition))?;
        selections.push(best.0);
        let (host_retained_bytes, provider_retained_bytes) = next_retained_state(
            request,
            retention,
            node_index,
            &resource_state,
            &node.candidates[best.0],
        )?;
        resource_state.selections.push(best.0);
        resource_state.host_retained_bytes = host_retained_bytes;
        resource_state.provider_retained_bytes = provider_retained_bytes;
    }
    Some(PartitionPlan {
        selections,
        total_ns,
        explored_states,
        pruned_states,
        used_local_fallback: true,
        alternates: Vec::new(),
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
                presentation: &candidate.descriptor.presentation,
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
    candidate: CandidateResourceFacts<'_>,
    resources: &PlacementResourceSnapshot,
    require_transactional_results: bool,
) -> Result<(), ResourceAdmissionError> {
    single_candidate_resource_adjustment_ns(candidate, resources, require_transactional_results)
        .map(|_| ())
}

/// The same physical admission and eviction adjustment used by the graph
/// planner, evaluated for a complete one-node route without constructing a
/// graph or a portable descriptor. The empty provider-retention map has no
/// backing allocation.
pub(super) fn single_candidate_resource_adjustment_ns(
    candidate: CandidateResourceFacts<'_>,
    resources: &PlacementResourceSnapshot,
    require_transactional_results: bool,
) -> Result<u64, ResourceAdmissionError> {
    candidate_resource_adjustment_ns(
        candidate,
        ResourceAdmissionContext {
            resources,
            host_retained_bytes: ResourceAccount::default(),
            provider_retained_bytes: &BTreeMap::new(),
            require_transactional_results,
        },
    )
}

fn candidate_resource_adjustment_ns(
    candidate: CandidateResourceFacts<'_>,
    admission: ResourceAdmissionContext<'_>,
) -> Result<u64, ResourceAdmissionError> {
    admit_candidate_resources(candidate, admission)?
        .provider_evictions()
        .try_fold(0_u64, |current, (_, bytes)| {
            eviction_adjustment_ns(current, bytes).ok_or(ResourceAdmissionError::Overflow)
        })
}

fn eviction_adjustment_ns(current: u64, eviction_bytes: u64) -> Option<u64> {
    current
        .checked_add(50_000)?
        .checked_add(eviction_bytes.div_ceil(8))
}

fn next_retained_state(
    request: &PlacementPlanRequest,
    retention: &RetentionSchedule,
    node_index: usize,
    state: &SearchState,
    candidate: &PlacementGraphCandidate,
) -> Option<(ResourceAccount, BTreeMap<u32, ResourceAccount>)> {
    let mut host = state.host_retained_bytes;
    let mut providers = state.provider_retained_bytes.clone();
    for &(predecessor, output) in &retention.release_at[node_index] {
        let Some(selected) = state
            .selections
            .get(predecessor)
            .and_then(|selection| request.graph.nodes[predecessor].candidates.get(*selection))
        else {
            continue;
        };
        release_public_output(&mut host, &mut providers, selected, output)?;
    }
    for &output in &retention.retain_at[node_index] {
        retain_public_output(&mut host, &mut providers, candidate, output)?;
    }
    Some((host, providers))
}

fn retain_public_output(
    host: &mut ResourceAccount,
    providers: &mut BTreeMap<u32, ResourceAccount>,
    candidate: &PlacementGraphCandidate,
    output: SourceOutputOrdinal,
) -> Option<()> {
    let Some(facts) = candidate.output_facts.as_ref() else {
        if output.index() != 0 {
            return None;
        }
        return retain_output(host, providers, candidate);
    };
    let fact = facts.get(output.index() as usize)?;
    match fact.residency {
        CandidateOutputResidency::Host => *host = host.checked_add(fact.retained_bytes)?,
        CandidateOutputResidency::Provider { device_id } => {
            let retained = providers.entry(device_id).or_default();
            *retained = retained.checked_add(fact.retained_bytes)?;
        }
        CandidateOutputResidency::Mirrored { device_id } => {
            *host = host.checked_add(fact.retained_bytes)?;
            let retained = providers.entry(device_id).or_default();
            *retained = retained.checked_add(fact.retained_bytes)?;
        }
        _ => return None,
    }
    Some(())
}

fn release_public_output(
    host: &mut ResourceAccount,
    providers: &mut BTreeMap<u32, ResourceAccount>,
    candidate: &PlacementGraphCandidate,
    output: SourceOutputOrdinal,
) -> Option<()> {
    let Some(facts) = candidate.output_facts.as_ref() else {
        if output.index() != 0 {
            return None;
        }
        return release_retained(host, providers, candidate);
    };
    let fact = facts.get(output.index() as usize)?;
    match fact.residency {
        CandidateOutputResidency::Host => *host = host.subtract(fact.retained_bytes)?,
        CandidateOutputResidency::Provider { device_id } => {
            release_provider(providers, device_id, fact.retained_bytes)?;
        }
        CandidateOutputResidency::Mirrored { device_id } => {
            *host = host.subtract(fact.retained_bytes)?;
            release_provider(providers, device_id, fact.retained_bytes)?;
        }
        _ => return None,
    }
    Some(())
}

fn release_provider(
    providers: &mut BTreeMap<u32, ResourceAccount>,
    device_id: u32,
    bytes: runmat_execution::ResourceByteBound,
) -> Option<()> {
    let retained = providers.get_mut(&device_id)?;
    *retained = retained.subtract(bytes)?;
    if retained.is_empty() {
        providers.remove(&device_id);
    }
    Some(())
}

fn retain_output(
    host: &mut ResourceAccount,
    providers: &mut BTreeMap<u32, ResourceAccount>,
    candidate: &PlacementGraphCandidate,
) -> Option<()> {
    match candidate.descriptor.output_residency {
        CandidateOutputResidency::Host | CandidateOutputResidency::Unknown => {
            *host = host.checked_add(candidate.resources.retained_bytes)?;
        }
        CandidateOutputResidency::Provider { device_id } => {
            let retained = providers.entry(device_id).or_default();
            *retained = retained.checked_add(candidate.resources.retained_bytes)?;
        }
        CandidateOutputResidency::Mirrored { device_id } => {
            *host = host.checked_add(candidate.resources.retained_bytes)?;
            let retained = providers.entry(device_id).or_default();
            *retained = retained.checked_add(candidate.resources.retained_bytes)?;
        }
        CandidateOutputResidency::ProviderAndHost {
            device_id,
            host_bytes,
        } => {
            *host = host.checked_add(host_bytes)?;
            let retained = providers.entry(device_id).or_default();
            *retained = retained.checked_add(candidate.resources.retained_bytes)?;
        }
        CandidateOutputResidency::ConditionalHostOrProvider {
            device_id,
            host_bytes,
        } => {
            // The static graph cannot know which branch publishes at runtime;
            // keep both possible values live for future-consumer admission.
            *host = host.checked_add(host_bytes)?;
            let retained = providers.entry(device_id).or_default();
            *retained = retained.checked_add(candidate.resources.retained_bytes)?;
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
                if retained.is_empty() {
                    providers.remove(&device_id);
                }
            }
        }
        CandidateOutputResidency::Mirrored { device_id } => {
            *host = host.subtract(candidate.resources.retained_bytes)?;
            if let Some(retained) = providers.get_mut(&device_id) {
                *retained = retained.subtract(candidate.resources.retained_bytes)?;
                if retained.is_empty() {
                    providers.remove(&device_id);
                }
            }
        }
        CandidateOutputResidency::ProviderAndHost {
            device_id,
            host_bytes,
        } => {
            *host = host.subtract(host_bytes)?;
            if let Some(retained) = providers.get_mut(&device_id) {
                *retained = retained.subtract(candidate.resources.retained_bytes)?;
                if retained.is_empty() {
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
                if retained.is_empty() {
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
            let ordinal = edge.source_output.unwrap_or(SourceOutputOrdinal::new(0));
            let output = predecessor.source_output_residency(ordinal)?;
            total.checked_add(transition_ns(output, execution_location, edge)?)
        })
}

fn transition_ns(
    from: CandidateOutputResidency,
    to: CandidateExecutionLocation,
    edge: &runmat_execution::PlacementGraphEdge,
) -> Option<u64> {
    use CandidateExecutionLocation::{Host as ExecuteHost, Provider as ExecuteProvider, VmRegion};
    use CandidateOutputResidency::{
        ConditionalHostOrProvider, Host, Mirrored, Provider, ProviderAndHost, Unknown,
    };
    let transition = match (from, to) {
        // An original instruction region has no concrete graph-input
        // residency. Its whole-work placement must not acquire a fabricated
        // graph-edge transfer estimate, including a free same-device one.
        (
            _,
            VmRegion { .. }
            | CandidateExecutionLocation::NativeRegion { .. }
            | CandidateExecutionLocation::NativeSource,
        ) => return None,
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
        // Defensive aggregate envelope only. Validated consumed graph edges
        // resolve a typed per-output role before this function; an unresolved
        // mixed aggregate cannot be a source. Keep conservative costs here
        // for direct aggregate callers without inventing a free mirror.
        (ProviderAndHost { .. }, ExecuteHost) => edge.provider_to_host_ns,
        (
            ProviderAndHost {
                device_id: left, ..
            },
            ExecuteProvider { device_id: right },
        ) if left == right => edge.host_to_provider_ns,
        (ProviderAndHost { .. }, ExecuteProvider { .. }) => {
            edge.host_to_provider_ns.max(edge.cross_provider_ns)
        }
        // Until execution resolves the conditional publication, charge the
        // greater of its admitted host/provider transition costs. Neither
        // branch is a mirrored copy available to the next operation.
        (ConditionalHostOrProvider { .. }, ExecuteHost) => edge.provider_to_host_ns,
        (
            ConditionalHostOrProvider {
                device_id: left, ..
            },
            ExecuteProvider { device_id: right },
        ) if left == right => edge.host_to_provider_ns,
        (ConditionalHostOrProvider { .. }, ExecuteProvider { .. }) => {
            edge.host_to_provider_ns.max(edge.cross_provider_ns)
        }
    };
    Some(transition)
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
mod conditional_transition_tests;
#[cfg(test)]
mod mixed_output_tests;

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
