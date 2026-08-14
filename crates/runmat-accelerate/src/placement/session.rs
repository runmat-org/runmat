use std::cell::RefCell;
use std::collections::BTreeMap;

use runmat_execution::{
    Digest, EstimateConfidence, EstimateSource, ExecutionCostComponents, PlacementDecision,
    PlacementFeedback, PlacementInvalidation, PlacementPlanRequest, SelectedExecutionCandidate,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{partition::evaluate_selections, partition::solve_partition, PlacementPolicy};

pub const PLACEMENT_PROFILE_SCHEMA_VERSION: u32 = 1;
const MAX_FEEDBACK_OBSERVATIONS: u64 = 1_048_576;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExplorationMode {
    Disabled,
    TransactionalEvery { interval: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementSessionConfig {
    pub max_cached_decisions: usize,
    pub max_feedback_series: usize,
    pub leave_absolute_margin_ns: u64,
    pub leave_relative_margin_basis_points: u32,
    pub exploration: ExplorationMode,
    pub exploration_max_overhead_basis_points: u32,
}

impl Default for PlacementSessionConfig {
    fn default() -> Self {
        Self {
            max_cached_decisions: 256,
            max_feedback_series: 512,
            leave_absolute_margin_ns: 10_000,
            leave_relative_margin_basis_points: 500,
            exploration: ExplorationMode::Disabled,
            exploration_max_overhead_basis_points: 1_000,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[error("{code}: {message}")]
pub struct PlacementPlanError {
    pub code: &'static str,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementProfileSnapshot {
    pub schema_version: u32,
    pub feedback: Vec<PlacementFeedbackSeries>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementFeedbackSeries {
    pub signature: runmat_execution::PlacementSignature,
    pub candidate: String,
    #[serde(with = "u64_decimal")]
    pub samples: u64,
    #[serde(with = "u64_decimal")]
    pub failures: u64,
    #[serde(with = "u128_decimal")]
    pub total_elapsed_ns: u128,
    #[serde(with = "u128_decimal")]
    pub total_squared_elapsed_ns: u128,
    #[serde(with = "u64_decimal")]
    pub latest_tick: u64,
}

mod u64_decimal {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

mod u128_decimal {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &u128, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u128, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

impl PlacementFeedbackSeries {
    pub fn mean_ns(&self) -> Option<u64> {
        (self.samples > 0).then(|| {
            u64::try_from(self.total_elapsed_ns / u128::from(self.samples)).unwrap_or(u64::MAX)
        })
    }

    pub fn variance_ns2(&self) -> Option<u128> {
        if self.samples < 2 {
            return None;
        }
        let mean = self.total_elapsed_ns / u128::from(self.samples);
        Some(
            (self.total_squared_elapsed_ns / u128::from(self.samples))
                .saturating_sub(mean.saturating_mul(mean)),
        )
    }

    pub fn estimated_p95_ns(&self) -> Option<u64> {
        let mean = self.mean_ns()?;
        let deviation = integer_sqrt(self.variance_ns2().unwrap_or(0));
        Some(mean.saturating_add(u64::try_from(deviation.saturating_mul(2)).unwrap_or(u64::MAX)))
    }
}

#[derive(Clone, Debug)]
struct CachedDecision {
    decision: PlacementDecision,
    resources: runmat_execution::PlacementResourceSnapshot,
    last_used: u64,
}

#[derive(Default)]
struct PlacementSessionState {
    tick: u64,
    decisions: BTreeMap<Digest, CachedDecision>,
    feedback: BTreeMap<(Digest, String), PlacementFeedbackSeries>,
}

pub struct PlacementSession {
    config: PlacementSessionConfig,
    state: RefCell<PlacementSessionState>,
}

impl Default for PlacementSession {
    fn default() -> Self {
        Self::new(PlacementSessionConfig::default())
    }
}

impl PlacementSession {
    pub fn new(config: PlacementSessionConfig) -> Self {
        Self {
            config,
            state: RefCell::new(PlacementSessionState::default()),
        }
    }

    pub fn plan(
        &self,
        mut request: PlacementPlanRequest,
    ) -> Result<PlacementDecision, PlacementPlanError> {
        request.validate().map_err(|message| PlacementPlanError {
            code: "RunMat:Placement:InvalidRequest",
            message: message.into(),
        })?;
        let key = request.signature.cache_key();
        let mut state = self.state.borrow_mut();
        state.decisions.retain(|_, entry| {
            entry.decision.signature.operation != request.signature.operation
                || entry.decision.signature.revision == request.signature.revision
        });
        state.tick = state.tick.saturating_add(1);
        let tick = state.tick;
        apply_feedback_estimates(&mut request, key, tick, &state.feedback);
        let partition =
            solve_partition(&request, PlacementPolicy::default()).map_err(|message| {
                PlacementPlanError {
                    code: if request.resources.cancellation_requested {
                        "RunMat:Placement:Cancelled"
                    } else {
                        "RunMat:Placement:NoLegalCandidate"
                    },
                    message: message.into(),
                }
            })?;
        let mut fresh = decision_from_partition(&request, &partition);
        if let Some(cached) = state.decisions.get_mut(&key) {
            if cached.resources == request.resources
                && cached.decision.signature.revision == request.signature.revision
            {
                if let Some(incumbent_ns) =
                    evaluate_selections(&request, &cached.decision.selections)
                {
                    if !clears_leave_margin(incumbent_ns, fresh.predicted_total_ns, self.config) {
                        cached.last_used = tick;
                        let mut decision = cached.decision.clone();
                        decision.predicted_total_ns = incumbent_ns;
                        decision.from_cache = true;
                        return Ok(decision);
                    }
                }
            }
        }
        if should_explore(self.config, tick, request.deterministic) {
            if let Some((selections, alternate_ns)) = partition.alternate.as_ref() {
                if alternate_is_transactional(&request, &partition.selections, selections)
                    && within_exploration_bound(
                        partition.total_ns,
                        *alternate_ns,
                        self.config.exploration_max_overhead_basis_points,
                    )
                {
                    fresh = decision_from_selections(
                        &request,
                        selections,
                        *alternate_ns,
                        partition.explored_states,
                        partition.pruned_states,
                        false,
                    );
                }
            }
        }
        state.decisions.insert(
            key,
            CachedDecision {
                decision: fresh.clone(),
                resources: request.resources.clone(),
                last_used: tick,
            },
        );
        evict_oldest_decisions(&mut state, self.config.max_cached_decisions);
        Ok(fresh)
    }

    pub fn observe(&self, feedback: PlacementFeedback) -> Result<(), PlacementPlanError> {
        if feedback.signature.validate().is_err()
            || feedback.candidate.is_empty()
            || feedback.candidate.len() > 128
            || feedback.candidate.chars().any(char::is_control)
        {
            return Err(PlacementPlanError {
                code: "RunMat:Placement:InvalidFeedback",
                message: "feedback candidate identity is invalid".into(),
            });
        }
        let mut state = self.state.borrow_mut();
        let observation_tick = state.tick;
        let signature_key = feedback.signature.cache_key();
        let key = (signature_key, feedback.candidate.clone());
        let series = state
            .feedback
            .entry(key)
            .or_insert_with(|| PlacementFeedbackSeries {
                signature: feedback.signature.clone(),
                candidate: feedback.candidate.clone(),
                samples: 0,
                failures: 0,
                total_elapsed_ns: 0,
                total_squared_elapsed_ns: 0,
                latest_tick: 0,
            });
        if feedback.succeeded {
            if series.samples >= MAX_FEEDBACK_OBSERVATIONS {
                series.samples = series.samples.div_ceil(2);
                series.total_elapsed_ns = series.total_elapsed_ns.div_ceil(2);
                series.total_squared_elapsed_ns = series.total_squared_elapsed_ns.div_ceil(2);
            }
            let elapsed_ns = if series.samples >= 4 {
                let mean = series.mean_ns().unwrap_or(feedback.total_elapsed_ns).max(1);
                feedback
                    .total_elapsed_ns
                    .clamp((mean / 4).max(1), mean.saturating_mul(4))
            } else {
                feedback.total_elapsed_ns.max(1)
            };
            series.samples = series.samples.saturating_add(1);
            series.total_elapsed_ns = series
                .total_elapsed_ns
                .saturating_add(u128::from(elapsed_ns));
            series.total_squared_elapsed_ns = series
                .total_squared_elapsed_ns
                .saturating_add(u128::from(elapsed_ns).saturating_mul(u128::from(elapsed_ns)));
        } else {
            series.failures = series
                .failures
                .saturating_add(1)
                .min(MAX_FEEDBACK_OBSERVATIONS);
        }
        series.latest_tick = observation_tick;
        let needs_replan =
            !feedback.succeeded || series.samples <= 2 || series.samples.is_power_of_two();
        if needs_replan {
            state.decisions.remove(&signature_key);
        }
        evict_oldest_feedback(&mut state, self.config.max_feedback_series);
        Ok(())
    }

    pub fn invalidate(&self, invalidation: PlacementInvalidation) {
        let mut state = self.state.borrow_mut();
        match invalidation {
            PlacementInvalidation::All => {
                state.decisions.clear();
                state.feedback.clear();
            }
            PlacementInvalidation::Program { revision } => {
                state.decisions.retain(|_, entry| {
                    entry.decision.signature.revision.program.as_ref() != Some(&revision)
                });
                state.feedback.retain(|_, series| {
                    series.signature.revision.program.as_ref() != Some(&revision)
                });
            }
            PlacementInvalidation::Provider { digest } => {
                state
                    .decisions
                    .retain(|_, entry| entry.decision.signature.revision.provider != digest);
                state
                    .feedback
                    .retain(|_, series| series.signature.revision.provider != digest);
            }
            PlacementInvalidation::Policy { digest } => {
                state
                    .decisions
                    .retain(|_, entry| entry.decision.signature.revision.policy != digest);
                state
                    .feedback
                    .retain(|_, series| series.signature.revision.policy != digest);
            }
            PlacementInvalidation::Signature { key } => {
                state.decisions.remove(&key);
                state
                    .feedback
                    .retain(|_, series| series.signature.cache_key() != key);
            }
        }
    }

    pub fn profile_snapshot(&self) -> PlacementProfileSnapshot {
        PlacementProfileSnapshot {
            schema_version: PLACEMENT_PROFILE_SCHEMA_VERSION,
            feedback: self.state.borrow().feedback.values().cloned().collect(),
        }
    }

    pub fn restore_profile(
        &self,
        snapshot: PlacementProfileSnapshot,
    ) -> Result<(), PlacementPlanError> {
        if snapshot.schema_version != PLACEMENT_PROFILE_SCHEMA_VERSION {
            return Err(PlacementPlanError {
                code: "RunMat:Placement:ProfileVersion",
                message: "placement profile schema version is not supported".into(),
            });
        }
        if snapshot.feedback.len() > self.config.max_feedback_series {
            return Err(PlacementPlanError {
                code: "RunMat:Placement:ProfileLimit",
                message: "placement profile exceeds the configured series limit".into(),
            });
        }
        let mut feedback = snapshot.feedback;
        feedback.sort_by(|left, right| {
            left.latest_tick
                .cmp(&right.latest_tick)
                .then_with(|| left.signature.cache_key().cmp(&right.signature.cache_key()))
                .then_with(|| left.candidate.cmp(&right.candidate))
        });
        let mut restored = BTreeMap::new();
        let mut restored_tick = 0_u64;
        for mut series in feedback {
            if !valid_profile_series(&series) {
                return Err(PlacementPlanError {
                    code: "RunMat:Placement:ProfileEntry",
                    message: "placement profile contains an invalid feedback series".into(),
                });
            }
            restored_tick = restored_tick.saturating_add(1);
            series.latest_tick = restored_tick;
            let key = (series.signature.cache_key(), series.candidate.clone());
            if restored.insert(key, series).is_some() {
                return Err(PlacementPlanError {
                    code: "RunMat:Placement:ProfileEntry",
                    message: "placement profile contains a duplicate feedback series".into(),
                });
            }
        }
        let mut state = self.state.borrow_mut();
        state.tick = restored_tick;
        state.decisions.clear();
        state.feedback = restored;
        evict_oldest_feedback(&mut state, self.config.max_feedback_series);
        Ok(())
    }
}

fn valid_profile_series(series: &PlacementFeedbackSeries) -> bool {
    series.signature.validate().is_ok()
        && !series.candidate.is_empty()
        && series.candidate.len() <= 128
        && !series.candidate.chars().any(char::is_control)
        && series.samples <= MAX_FEEDBACK_OBSERVATIONS
        && series.failures <= MAX_FEEDBACK_OBSERVATIONS
        && if series.samples == 0 {
            series.failures > 0
                && series.total_elapsed_ns == 0
                && series.total_squared_elapsed_ns == 0
        } else {
            series.total_elapsed_ns >= u128::from(series.samples)
                && series.total_elapsed_ns
                    <= u128::from(u64::MAX).saturating_mul(u128::from(series.samples))
                && series.total_squared_elapsed_ns >= series.total_elapsed_ns
        }
}

impl runmat_runtime::context::RuntimePlacementService for PlacementSession {
    fn plan(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<PlacementDecision, runmat_runtime::RuntimeError> {
        PlacementSession::plan(self, request).map_err(runtime_error)
    }

    fn observe(&self, feedback: PlacementFeedback) -> Result<(), runmat_runtime::RuntimeError> {
        PlacementSession::observe(self, feedback).map_err(runtime_error)
    }

    fn invalidate(&self, invalidation: PlacementInvalidation) {
        PlacementSession::invalidate(self, invalidation);
    }
}

fn runtime_error(error: PlacementPlanError) -> runmat_runtime::RuntimeError {
    runmat_runtime::build_runtime_error(error.message)
        .with_builtin("placement")
        .with_identifier(error.code)
        .build()
}

fn apply_feedback_estimates(
    request: &mut PlacementPlanRequest,
    signature_key: Digest,
    tick: u64,
    feedback: &BTreeMap<(Digest, String), PlacementFeedbackSeries>,
) {
    for node in &mut request.graph.nodes {
        for candidate in &mut node.candidates {
            let key = (signature_key, candidate.descriptor.identity.clone());
            let Some(series) = feedback.get(&key) else {
                continue;
            };
            if tick.saturating_sub(series.latest_tick) > 1_024 {
                continue;
            }
            if series.failures >= 2 && candidate.descriptor.kind.is_provider() {
                candidate.resources.queue_slots = u32::MAX;
                continue;
            }
            if series.samples >= 2 {
                if let Some(mean) = series.mean_ns() {
                    candidate.descriptor.cost.components = ExecutionCostComponents {
                        execution_ns: mean,
                        ..ExecutionCostComponents::default()
                    };
                    let deviation = integer_sqrt(series.variance_ns2().unwrap_or(0));
                    let variation_basis_points = if mean == 0 {
                        u128::MAX
                    } else {
                        deviation.saturating_mul(10_000) / u128::from(mean)
                    };
                    candidate.descriptor.cost.confidence =
                        if series.samples >= 8 && variation_basis_points <= 1_000 {
                            EstimateConfidence::High
                        } else if variation_basis_points > 5_000 {
                            EstimateConfidence::Low
                        } else {
                            EstimateConfidence::Medium
                        };
                    candidate.descriptor.cost.source = EstimateSource::Observation;
                }
            }
        }
    }
}

fn integer_sqrt(value: u128) -> u128 {
    if value < 2 {
        return value;
    }
    let mut low = 1_u128;
    let mut high = value.min(u128::from(u64::MAX));
    while low <= high {
        let middle = low + (high - low) / 2;
        if middle <= value / middle {
            low = middle.saturating_add(1);
        } else {
            high = middle - 1;
        }
    }
    high
}

fn decision_from_partition(
    request: &PlacementPlanRequest,
    partition: &super::partition::PartitionPlan,
) -> PlacementDecision {
    decision_from_selections(
        request,
        &partition.selections,
        partition.total_ns,
        partition.explored_states,
        partition.pruned_states,
        partition.used_local_fallback,
    )
}

fn decision_from_selections(
    request: &PlacementPlanRequest,
    selections: &[usize],
    total_ns: u64,
    explored_states: u32,
    pruned_states: u32,
    used_local_fallback: bool,
) -> PlacementDecision {
    PlacementDecision {
        signature: request.signature.clone(),
        selections: selections
            .iter()
            .enumerate()
            .map(|(node_index, candidate_index)| {
                let node = &request.graph.nodes[node_index];
                let candidate = &node.candidates[*candidate_index].descriptor;
                SelectedExecutionCandidate {
                    node: node.identity.clone(),
                    candidate: candidate.identity.clone(),
                    kind: candidate.kind,
                }
            })
            .collect(),
        predicted_total_ns: total_ns,
        from_cache: false,
        used_local_fallback,
        explored_states,
        pruned_states,
    }
}

fn clears_leave_margin(incumbent: u64, challenger: u64, config: PlacementSessionConfig) -> bool {
    let relative = incumbent
        .saturating_mul(u64::from(config.leave_relative_margin_basis_points))
        .saturating_add(9_999)
        / 10_000;
    challenger.saturating_add(config.leave_absolute_margin_ns.max(relative)) <= incumbent
}

fn should_explore(config: PlacementSessionConfig, tick: u64, deterministic: bool) -> bool {
    if deterministic {
        return false;
    }
    match config.exploration {
        ExplorationMode::Disabled => false,
        ExplorationMode::TransactionalEvery { interval } => {
            interval > 0 && tick.is_multiple_of(u64::from(interval))
        }
    }
}

fn alternate_is_transactional(
    request: &PlacementPlanRequest,
    best: &[usize],
    alternate: &[usize],
) -> bool {
    alternate.iter().enumerate().all(|(node, candidate)| {
        candidate == &best[node]
            || request.graph.nodes[node].candidates[*candidate].transactional_results
    })
}

fn within_exploration_bound(best: u64, alternate: u64, overhead_basis_points: u32) -> bool {
    alternate <= best.saturating_add(best.saturating_mul(u64::from(overhead_basis_points)) / 10_000)
}

fn evict_oldest_decisions(state: &mut PlacementSessionState, limit: usize) {
    while state.decisions.len() > limit {
        let Some(key) = state
            .decisions
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| *key)
        else {
            break;
        };
        state.decisions.remove(&key);
    }
}

fn evict_oldest_feedback(state: &mut PlacementSessionState, limit: usize) {
    while state.feedback.len() > limit {
        let Some(key) = state
            .feedback
            .iter()
            .min_by_key(|(_, series)| series.latest_tick)
            .map(|(key, _)| key.clone())
        else {
            break;
        };
        state.feedback.remove(&key);
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use runmat_execution::{
        CandidateExecutionLocation, CandidateOutputResidency, CandidatePreparationState,
        CandidateResourceDemand, ExecutionCandidateDescriptor, ExecutionCandidateKind,
        ExecutionCostComponents, ExecutionCostEstimate, PlacementGraph, PlacementGraphCandidate,
        PlacementGraphEdge, PlacementGraphLimits, PlacementGraphNode, PlacementResourceSnapshot,
        PlacementRevision, PlacementSignature, ProviderResourceSnapshot,
    };

    use super::*;

    fn digest(token: &str) -> Digest {
        Digest::sha256(token.as_bytes())
    }

    fn signature(token: &str) -> PlacementSignature {
        PlacementSignature {
            region: None,
            operation: token.into(),
            runtime_facts: digest(token),
            revision: PlacementRevision {
                program: None,
                catalog: digest("catalog"),
                compiler: digest("compiler"),
                provider: digest("provider"),
                policy: digest("policy"),
            },
        }
    }

    fn candidate(
        identity: &str,
        kind: ExecutionCandidateKind,
        cost_ns: u64,
        retained_bytes: u64,
        transactional: bool,
    ) -> PlacementGraphCandidate {
        PlacementGraphCandidate {
            descriptor: ExecutionCandidateDescriptor {
                identity: identity.into(),
                region: None,
                kind,
                execution_location: if kind.is_provider() {
                    CandidateExecutionLocation::Provider { device_id: 7 }
                } else {
                    CandidateExecutionLocation::Host
                },
                preparation: CandidatePreparationState::Warm,
                cost: ExecutionCostEstimate {
                    components: ExecutionCostComponents {
                        execution_ns: cost_ns,
                        ..ExecutionCostComponents::default()
                    },
                    scratch_bytes: 0,
                    confidence: EstimateConfidence::Exact,
                    source: EstimateSource::Synthetic,
                },
                output_residency: if kind.is_provider() {
                    CandidateOutputResidency::Provider { device_id: 7 }
                } else {
                    CandidateOutputResidency::Host
                },
                guards: Vec::new(),
            },
            resources: CandidateResourceDemand {
                cpu_millicores: if kind.is_provider() { 0 } else { 1_000 },
                retained_bytes,
                scratch_bytes: 0,
                queue_slots: if kind.is_provider() { 1 } else { 0 },
            },
            transactional_results: transactional,
        }
    }

    fn resources(capacity: u64) -> PlacementResourceSnapshot {
        PlacementResourceSnapshot {
            cpu_millicores_available: 1_000,
            memory_available_bytes: None,
            cancellation_requested: false,
            providers: vec![ProviderResourceSnapshot {
                device_id: 7,
                capacity_bytes: Some(capacity),
                live_bytes: 0,
                reclaimable_bytes: 0,
                scratch_available_bytes: Some(capacity),
                queue_depth: Some(0),
                queue_limit: Some(1),
                lost: false,
                epoch: 1,
            }],
            epoch: 1,
        }
    }

    fn two_node_request(
        token: &str,
        first: (u64, u64),
        second: (u64, u64),
    ) -> PlacementPlanRequest {
        PlacementPlanRequest {
            signature: signature(token),
            graph: PlacementGraph {
                nodes: vec![
                    PlacementGraphNode {
                        identity: "first".into(),
                        candidates: vec![
                            candidate(
                                "cpu.first",
                                ExecutionCandidateKind::SharedRuntime,
                                first.0,
                                0,
                                true,
                            ),
                            candidate(
                                "gpu.first",
                                ExecutionCandidateKind::ProviderFusion,
                                first.1,
                                64,
                                true,
                            ),
                        ],
                    },
                    PlacementGraphNode {
                        identity: "second".into(),
                        candidates: vec![
                            candidate(
                                "cpu.second",
                                ExecutionCandidateKind::SharedRuntime,
                                second.0,
                                0,
                                true,
                            ),
                            candidate(
                                "gpu.second",
                                ExecutionCandidateKind::ProviderFusion,
                                second.1,
                                64,
                                true,
                            ),
                        ],
                    },
                ],
                edges: vec![PlacementGraphEdge {
                    from: 0,
                    to: 1,
                    bytes: 64,
                    host_to_provider_ns: 100_000,
                    provider_to_host_ns: 100_000,
                    cross_provider_ns: 100_000,
                }],
            },
            limits: PlacementGraphLimits::default(),
            resources: resources(1_024),
            deterministic: true,
            require_transactional_results: true,
        }
    }

    #[test]
    fn bounded_graph_planning_keeps_profitable_resident_chain() {
        let session = PlacementSession::default();
        let decision = session
            .plan(two_node_request(
                "resident-chain",
                (100_000, 45_000),
                (100_000, 5_000),
            ))
            .unwrap();
        assert!(decision
            .selections
            .iter()
            .all(|selected| selected.kind.is_provider()));
        assert_eq!(decision.predicted_total_ns, 50_000);
        assert!(!decision.used_local_fallback);
    }

    #[test]
    fn graph_planning_accounts_for_transfer_boundaries() {
        let session = PlacementSession::default();
        let decision = session
            .plan(two_node_request(
                "transfer-boundary",
                (60_000, 130_000),
                (60_000, 20_000),
            ))
            .unwrap();
        assert!(decision
            .selections
            .iter()
            .all(|selected| !selected.kind.is_provider()));
    }

    #[test]
    fn transfers_target_execution_location_not_output_residency() {
        let session = PlacementSession::default();
        let mut request = two_node_request(
            "materialized-provider-output",
            (100_000, 1_000),
            (100_000, 1_000),
        );
        request.graph.nodes[0].candidates[1]
            .descriptor
            .output_residency = CandidateOutputResidency::Host;
        request.graph.edges[0].host_to_provider_ns = 250_000;

        let decision = session.plan(request).unwrap();
        assert!(decision.selections[0].kind.is_provider());
        assert!(!decision.selections[1].kind.is_provider());
        assert_eq!(decision.predicted_total_ns, 101_000);
    }

    #[test]
    fn adversarial_search_budget_uses_cpu_fallback() {
        let session = PlacementSession::default();
        let mut request = two_node_request("bounded", (100_000, 10_000), (100_000, 10_000));
        request.limits.max_expansions = 1;
        let decision = session.plan(request).unwrap();
        assert!(decision.used_local_fallback);
        assert_eq!(decision.predicted_total_ns, 200_000);
        assert!(decision
            .selections
            .iter()
            .all(|selected| !selected.kind.is_provider()));
    }

    #[test]
    fn resource_pressure_rejects_nominally_faster_provider() {
        let session = PlacementSession::default();
        let mut request = two_node_request("memory", (100_000, 1_000), (100_000, 1_000));
        for node in &mut request.graph.nodes {
            node.candidates[1].resources.retained_bytes = 2_048;
        }
        let decision = session.plan(request).unwrap();
        assert!(decision
            .selections
            .iter()
            .all(|selected| !selected.kind.is_provider()));
    }

    #[test]
    fn mirrored_outputs_are_admitted_against_host_and_provider_memory() {
        let session = PlacementSession::default();
        let mut request = two_node_request("mirrored-memory", (100_000, 1_000), (100_000, 1_000));
        request.resources.memory_available_bytes = Some(500);
        for node in &mut request.graph.nodes {
            node.candidates[1].descriptor.output_residency =
                CandidateOutputResidency::Mirrored { device_id: 7 };
            node.candidates[1].resources.retained_bytes = 600;
        }

        let decision = session.plan(request).unwrap();
        assert!(decision
            .selections
            .iter()
            .all(|selected| !selected.kind.is_provider()));
    }

    #[test]
    fn cached_decisions_require_an_exact_resource_snapshot() {
        let session = PlacementSession::default();
        let request = two_node_request("resource-cache", (100_000, 1_000), (100_000, 1_000));
        assert!(session
            .plan(request.clone())
            .unwrap()
            .selections
            .iter()
            .all(|selected| selected.kind.is_provider()));

        let mut constrained = request;
        constrained.resources.providers[0].capacity_bytes = Some(32);
        constrained.resources.providers[0].scratch_available_bytes = Some(32);
        let decision = session.plan(constrained).unwrap();
        assert!(!decision.from_cache);
        assert!(decision
            .selections
            .iter()
            .all(|selected| !selected.kind.is_provider()));
    }

    #[test]
    fn cached_decisions_revalidate_selected_candidate_demand() {
        let session = PlacementSession::default();
        let request = two_node_request("candidate-cache", (100_000, 1_000), (100_000, 1_000));
        assert!(session
            .plan(request.clone())
            .unwrap()
            .selections
            .iter()
            .all(|selected| selected.kind.is_provider()));

        let mut constrained = request;
        for node in &mut constrained.graph.nodes {
            node.candidates[1].resources.queue_slots = 2;
        }
        let decision = session.plan(constrained).unwrap();
        assert!(!decision.from_cache);
        assert!(decision
            .selections
            .iter()
            .all(|selected| !selected.kind.is_provider()));
    }

    #[test]
    fn provider_loss_and_queue_pressure_fail_over_before_dispatch() {
        let session = PlacementSession::default();
        let request = two_node_request("provider-loss", (100_000, 1_000), (100_000, 1_000));

        let mut lost = request.clone();
        lost.resources.providers[0].lost = true;
        assert!(session
            .plan(lost)
            .unwrap()
            .selections
            .iter()
            .all(|selected| !selected.kind.is_provider()));

        let mut saturated = request;
        saturated.resources.providers[0].queue_depth = Some(1);
        saturated.resources.providers[0].queue_limit = Some(1);
        assert!(session
            .plan(saturated)
            .unwrap()
            .selections
            .iter()
            .all(|selected| !selected.kind.is_provider()));
    }

    #[test]
    fn invalid_candidate_execution_location_fails_closed() {
        let session = PlacementSession::default();
        let mut request = two_node_request("invalid-location", (100_000, 1_000), (100_000, 1_000));
        request.graph.nodes[0].candidates[0]
            .descriptor
            .execution_location = CandidateExecutionLocation::Provider { device_id: 7 };
        assert_eq!(
            session.plan(request).unwrap_err().code,
            "RunMat:Placement:InvalidRequest"
        );
    }

    #[test]
    fn resource_admission_accounts_for_simultaneously_live_intermediates() {
        let session = PlacementSession::default();
        let mut request =
            two_node_request("live-intermediates", (100_000, 1_000), (100_000, 1_000));
        request.graph.nodes.push(PlacementGraphNode {
            identity: "join".into(),
            candidates: vec![
                candidate(
                    "cpu.join",
                    ExecutionCandidateKind::SharedRuntime,
                    100_000,
                    0,
                    true,
                ),
                candidate(
                    "gpu.join",
                    ExecutionCandidateKind::ProviderFusion,
                    1_000,
                    0,
                    true,
                ),
            ],
        });
        request.graph.edges = vec![
            PlacementGraphEdge {
                from: 0,
                to: 2,
                bytes: 600,
                host_to_provider_ns: 0,
                provider_to_host_ns: 0,
                cross_provider_ns: 0,
            },
            PlacementGraphEdge {
                from: 1,
                to: 2,
                bytes: 600,
                host_to_provider_ns: 0,
                provider_to_host_ns: 0,
                cross_provider_ns: 0,
            },
        ];
        request.resources = resources(1_000);
        request.graph.nodes[0].candidates[1]
            .resources
            .retained_bytes = 600;
        request.graph.nodes[1].candidates[1]
            .resources
            .retained_bytes = 600;

        let decision = session.plan(request).unwrap();
        let provider_count = decision
            .selections
            .iter()
            .filter(|selected| selected.kind.is_provider())
            .count();
        assert_eq!(provider_count, 2);
        assert!(
            !decision.selections[0].kind.is_provider()
                || !decision.selections[1].kind.is_provider()
        );
    }

    #[test]
    fn automatic_fallback_requires_transactional_provider_results() {
        let session = PlacementSession::default();
        let mut request = two_node_request("transactional", (100_000, 1_000), (100_000, 1_000));
        for node in &mut request.graph.nodes {
            node.candidates[1].transactional_results = false;
        }
        let decision = session.plan(request).unwrap();
        assert!(decision
            .selections
            .iter()
            .all(|selected| !selected.kind.is_provider()));
    }

    #[test]
    fn bounded_exploration_is_disabled_in_deterministic_mode() {
        let session = PlacementSession::new(PlacementSessionConfig {
            exploration: ExplorationMode::TransactionalEvery { interval: 1 },
            ..PlacementSessionConfig::default()
        });
        let deterministic = session
            .plan(two_node_request(
                "deterministic-exploration",
                (100_000, 94_000),
                (100_000, 94_000),
            ))
            .unwrap();
        assert!(deterministic
            .selections
            .iter()
            .all(|selected| selected.kind.is_provider()));

        let mut adaptive_request =
            two_node_request("adaptive-exploration", (100_000, 94_000), (100_000, 94_000));
        adaptive_request.deterministic = false;
        let explored = session.plan(adaptive_request).unwrap();
        assert_ne!(explored.selections, deterministic.selections);
    }

    #[test]
    fn cached_decision_is_sticky_until_leave_margin_clears() {
        let session = PlacementSession::default();
        let first = session
            .plan(two_node_request(
                "sticky",
                (100_000, 35_000),
                (100_000, 35_000),
            ))
            .unwrap();
        assert!(first.selections[0].kind.is_provider());

        let close = session
            .plan(two_node_request(
                "sticky",
                (32_500, 35_000),
                (32_500, 35_000),
            ))
            .unwrap();
        assert!(close.from_cache);
        assert!(close.selections[0].kind.is_provider());

        let clear = session
            .plan(two_node_request(
                "sticky",
                (20_000, 35_000),
                (20_000, 35_000),
            ))
            .unwrap();
        assert!(!clear.from_cache);
        assert!(!clear.selections[0].kind.is_provider());
    }

    #[test]
    fn feedback_replaces_priors_and_round_trips_bounded_profile() {
        let session = PlacementSession::default();
        let request = two_node_request("feedback", (100_000, 20_000), (100_000, 20_000));
        let initial = session.plan(request.clone()).unwrap();
        assert!(initial.selections[0].kind.is_provider());
        for _ in 1..=2 {
            for candidate in ["gpu.first", "gpu.second"] {
                session
                    .observe(PlacementFeedback {
                        signature: request.signature.clone(),
                        candidate: candidate.into(),
                        total_elapsed_ns: 250_000,
                        succeeded: true,
                    })
                    .unwrap();
            }
        }
        let adapted = session.plan(request).unwrap();
        assert!(adapted
            .selections
            .iter()
            .all(|selected| !selected.kind.is_provider()));

        let encoded = serde_json::to_string(&session.profile_snapshot()).unwrap();
        assert!(encoded.contains("\"total_elapsed_ns\":\""));
        assert!(encoded.contains("\"samples\":\"2\""));
        let snapshot: PlacementProfileSnapshot = serde_json::from_str(&encoded).unwrap();
        let restored = PlacementSession::default();
        restored.restore_profile(snapshot).unwrap();
        assert_eq!(restored.profile_snapshot().feedback.len(), 2);
    }

    #[test]
    fn invalidation_and_profile_restore_clear_matching_cached_state() {
        let session = PlacementSession::default();
        let request = two_node_request("strict-invalidation", (100_000, 20_000), (100_000, 20_000));
        let key = request.signature.cache_key();
        assert!(!session.plan(request.clone()).unwrap().from_cache);
        assert!(session.plan(request.clone()).unwrap().from_cache);
        session
            .observe(PlacementFeedback {
                signature: request.signature.clone(),
                candidate: "gpu.first".into(),
                total_elapsed_ns: 20_000,
                succeeded: true,
            })
            .unwrap();
        assert_eq!(session.profile_snapshot().feedback.len(), 1);

        session.invalidate(PlacementInvalidation::Signature { key });
        assert!(session.profile_snapshot().feedback.is_empty());
        assert!(!session.plan(request.clone()).unwrap().from_cache);
        assert!(session.plan(request.clone()).unwrap().from_cache);

        session
            .restore_profile(PlacementProfileSnapshot {
                schema_version: PLACEMENT_PROFILE_SCHEMA_VERSION,
                feedback: Vec::new(),
            })
            .unwrap();
        assert!(!session.plan(request).unwrap().from_cache);
    }

    #[test]
    fn failure_only_feedback_survives_profile_round_trip() {
        let request = two_node_request("failed-profile", (100_000, 20_000), (100_000, 20_000));
        let session = PlacementSession::default();
        for _ in 1..=2 {
            session
                .observe(PlacementFeedback {
                    signature: request.signature.clone(),
                    candidate: "gpu.first".into(),
                    total_elapsed_ns: 0,
                    succeeded: false,
                })
                .unwrap();
        }
        let encoded = serde_json::to_string(&session.profile_snapshot()).unwrap();
        assert!(encoded.contains("\"failures\":\"2\""));

        let restored = PlacementSession::default();
        restored
            .restore_profile(serde_json::from_str(&encoded).unwrap())
            .unwrap();
        assert_eq!(restored.profile_snapshot().feedback.len(), 1);
        assert!(!restored.plan(request).unwrap().selections[0]
            .kind
            .is_provider());
    }

    #[test]
    fn cancellation_and_stale_profile_fail_closed() {
        let session = PlacementSession::default();
        let mut request = two_node_request("cancel", (100_000, 20_000), (100_000, 20_000));
        request.resources.cancellation_requested = true;
        assert_eq!(
            session.plan(request).unwrap_err().code,
            "RunMat:Placement:Cancelled"
        );
        assert_eq!(
            session
                .restore_profile(PlacementProfileSnapshot {
                    schema_version: PLACEMENT_PROFILE_SCHEMA_VERSION + 1,
                    feedback: Vec::new(),
                })
                .unwrap_err()
                .code,
            "RunMat:Placement:ProfileVersion"
        );
    }

    #[test]
    fn runtime_contexts_keep_adaptive_state_session_owned() {
        let first_session = Rc::new(PlacementSession::default());
        let second_session = Rc::new(PlacementSession::default());
        let first = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default().with_placement(first_session),
        );
        let second = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default().with_placement(second_session),
        );
        let request = two_node_request("isolated", (100_000, 45_000), (100_000, 45_000));
        let first_plan = first
            .service_ports()
            .placement()
            .unwrap()
            .plan(request.clone())
            .unwrap();
        let first_cached = first
            .service_ports()
            .placement()
            .unwrap()
            .plan(request.clone())
            .unwrap();
        let second_plan = second
            .service_ports()
            .placement()
            .unwrap()
            .plan(request)
            .unwrap();
        assert!(!first_plan.from_cache);
        assert!(first_cached.from_cache);
        assert!(!second_plan.from_cache);
    }
}
