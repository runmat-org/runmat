use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use runmat_execution::{
    CandidateCostUnavailable, CandidateObservationEvidence, Digest, DigestStream, EstimateSource,
    ExecutionCandidateKind, ExecutionCostComponents, PlacementDecision, PlacementFeedback,
    PlacementInvalidation, PlacementPlanRequest, PlacementSignature, SelectedExecutionCandidate,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{partition::evaluate_selections, partition::solve_partition, PlacementPolicy};
mod evidence;
mod feedback_observation;
mod exploration;
#[cfg(test)]
mod graph_permit_fixture;
mod managed_feedback;
mod observation_permits;
mod pending;
mod prepared_request;
mod whole_work;
pub(super) use prepared_request::PreparedPlacementRequest;
pub(crate) use whole_work::select_unobserved;
pub(in crate::placement) use whole_work::unobserved_admitted_record_bytes;
pub(super) use whole_work::{candidate_resources_admitted, no_resource_candidate};
#[cfg(test)]
#[path = "session/authority_tests.rs"]
mod authority_tests;
mod runtime_binding_authority;

pub const PLACEMENT_PROFILE_SCHEMA_VERSION: u32 = 3;
const MAX_FEEDBACK_OBSERVATIONS: u64 = 1_048_576;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum FeedbackCandidateKey {
    Canonical(Digest),
    Text(Digest),
}

fn feedback_candidate_key(candidate: &str) -> FeedbackCandidateKey {
    if let Some(identity) = Digest::parse_canonical(candidate) {
        return FeedbackCandidateKey::Canonical(identity);
    }
    let mut identity = DigestStream::new();
    identity.update(b"runmat.placement.feedback-candidate.v1\0");
    identity.update(candidate.as_bytes());
    FeedbackCandidateKey::Text(identity.finish())
}

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

/// Explicit session setup capacities for requested Rust-managed backing.
/// These are slot counts and serialized-signature bytes, not an RSS estimate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ManagedPlacementCapacity {
    pub in_flight_slots: usize,
    pub feedback_series: usize,
    pub signature_bytes_per_series: usize,
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
    pub kind: ExecutionCandidateKind,
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
        evidence::FeedbackNumbers::from_series(self).mean_ns()
    }

    pub fn variance_ns2(&self) -> Option<u128> {
        evidence::FeedbackNumbers::from_series(self).variance_ns2()
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
    feedback:
        BTreeMap<(Digest, FeedbackCandidateKey, ExecutionCandidateKind), PlacementFeedbackSeries>,
}

pub struct PlacementSession {
    runtime_binding_session: runtime_binding_authority::SessionRuntimeBindingAuthority,
    config: PlacementSessionConfig,
    state: RefCell<PlacementSessionState>,
    pending: pending::PendingFeedback,
    observation_permits: observation_permits::ObservationPermits,
    managed_observations: Option<runmat_runtime::context::ManagedObservationPool>,
    managed_feedback: Option<managed_feedback::ManagedFeedbackPool>,
    whole_work_probes: Arc<Mutex<BTreeSet<whole_work::ProbeKey>>>,
}

impl Default for PlacementSession {
    fn default() -> Self {
        Self::new(PlacementSessionConfig::default())
    }
}

impl PlacementSession {
    pub fn new(config: PlacementSessionConfig) -> Self {
        Self {
            runtime_binding_session: runtime_binding_authority::SessionRuntimeBindingAuthority::new(
            ),
            config,
            state: RefCell::new(PlacementSessionState::default()),
            pending: pending::PendingFeedback::default(),
            observation_permits: observation_permits::ObservationPermits::default(),
            managed_observations: None,
            managed_feedback: None,
            whole_work_probes: Arc::default(),
        }
    }

    /// Install a finite in-flight observation backing at session setup. The
    /// existing unbounded/external constructor retains its dynamic registry.
    pub fn new_with_managed_observations(
        config: PlacementSessionConfig,
        ledger: &runmat_runtime::context::host_effect::ManagedHostEffectLedger,
        capacity: ManagedPlacementCapacity,
    ) -> Result<Self, runmat_runtime::context::host_effect::HostEffectAllocationError> {
        let managed_observations =
            runmat_runtime::context::ManagedObservationPool::new(ledger, capacity.in_flight_slots)?;
        let managed_feedback = managed_feedback::ManagedFeedbackPool::new(
            ledger,
            capacity.feedback_series,
            capacity.signature_bytes_per_series,
        )?;
        let mut session = Self::new(config);
        session.managed_observations = Some(managed_observations);
        session.managed_feedback = Some(managed_feedback);
        Ok(session)
    }

    pub fn plan(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<PlacementDecision, PlacementPlanError> {
        self.drain_observations();
        let mut prepared =
            PreparedPlacementRequest::new(request).map_err(|message| PlacementPlanError {
                code: "RunMat:Placement:InvalidRequest",
                message: message.into(),
            })?;
        let key = prepared.request().signature.cache_key();
        let mut state = self.state.borrow_mut();
        state.decisions.retain(|_, entry| {
            entry.decision.signature.operation != prepared.request().signature.operation
                || entry.decision.signature.revision == prepared.request().signature.revision
        });
        state.tick = state.tick.saturating_add(1);
        let tick = state.tick;
        prepared.apply_feedback(key, tick, &state.feedback);
        let partition =
            solve_partition(&prepared, PlacementPolicy::default()).map_err(|message| {
                PlacementPlanError {
                    code: if prepared.request().resources.cancellation_requested {
                        "RunMat:Placement:Cancelled"
                    } else {
                        "RunMat:Placement:NoLegalCandidate"
                    },
                    message: message.into(),
                }
            })?;
        let request = prepared.request();
        let mut fresh = decision_from_partition(request, &partition);
        if let Some(cached) = state.decisions.get_mut(&key) {
            if cached.resources == request.resources
                && cached.decision.signature.revision == request.signature.revision
            {
                if let Some(incumbent_ns) =
                    evaluate_selections(&prepared, &cached.decision.selections)
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
            if let Some((selections, alternate_ns)) = exploration::choose_alternate(
                request,
                &partition,
                self.config,
                key,
                &state.feedback,
                self.managed_feedback.as_ref(),
                tick,
            ) {
                fresh = decision_from_selections(
                    request,
                    selections,
                    *alternate_ns,
                    partition.explored_states,
                    partition.pruned_states,
                    false,
                );
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
        self.drain_observations();
        self.record_feedback(feedback)
    }

    pub fn observed_candidate_duration(
        &self,
        signature: &PlacementSignature,
        candidate: &str,
        kind: ExecutionCandidateKind,
    ) -> CandidateObservationEvidence {
        self.observed_candidate_duration_key(signature, feedback_candidate_key(candidate), kind)
    }

    fn observed_candidate_duration_digest(
        &self,
        signature: runmat_execution::PlacementSignatureRef<'_>,
        candidate: Digest,
        kind: ExecutionCandidateKind,
        drain_pending: bool,
    ) -> CandidateObservationEvidence {
        self.observed_candidate_duration_key_digest(
            signature.cache_key(),
            FeedbackCandidateKey::Canonical(candidate),
            kind,
            drain_pending,
        )
    }

    fn observed_candidate_duration_key(
        &self,
        signature: &PlacementSignature,
        candidate: FeedbackCandidateKey,
        kind: ExecutionCandidateKind,
    ) -> CandidateObservationEvidence {
        self.observed_candidate_duration_key_with_drain(signature, candidate, kind, true)
    }

    fn observed_candidate_duration_key_with_drain(
        &self,
        signature: &PlacementSignature,
        candidate: FeedbackCandidateKey,
        kind: ExecutionCandidateKind,
        drain_pending: bool,
    ) -> CandidateObservationEvidence {
        self.observed_candidate_duration_key_digest(
            signature.cache_key(),
            candidate,
            kind,
            drain_pending,
        )
    }

    fn observed_candidate_duration_key_digest(
        &self,
        signature: Digest,
        candidate: FeedbackCandidateKey,
        kind: ExecutionCandidateKind,
        drain_pending: bool,
    ) -> CandidateObservationEvidence {
        if drain_pending {
            self.drain_observations();
        }
        let state = self.state.borrow();
        if let FeedbackCandidateKey::Canonical(candidate_digest) = candidate {
            if let Some(observation) = self
                .managed_feedback
                .as_ref()
                .and_then(|pool| pool.observed(signature, candidate_digest, kind, state.tick))
            {
                return observation;
            }
        }
        let key = (signature, candidate, kind);
        state.feedback.get(&key).map_or(
            CandidateObservationEvidence::Unavailable(CandidateCostUnavailable::NoObservations),
            |series| evidence::from_series(series, state.tick),
        )
    }

    fn record_feedback(&self, feedback: PlacementFeedback) -> Result<(), PlacementPlanError> {
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
        let key = (
            signature_key,
            feedback_candidate_key(&feedback.candidate),
            feedback.kind,
        );
        let series = state
            .feedback
            .entry(key)
            .or_insert_with(|| PlacementFeedbackSeries {
                signature: feedback.signature.clone(),
                candidate: feedback.candidate.clone(),
                kind: feedback.kind,
                samples: 0,
                failures: 0,
                total_elapsed_ns: 0,
                total_squared_elapsed_ns: 0,
                latest_tick: 0,
            });
        let mut numbers = evidence::FeedbackNumbers::from_series(series);
        let needs_replan = numbers.record(
            feedback.total_elapsed_ns,
            feedback.succeeded,
            observation_tick,
        );
        series.samples = numbers.samples;
        series.failures = numbers.failures;
        series.total_elapsed_ns = numbers.total_elapsed_ns;
        series.total_squared_elapsed_ns = numbers.total_squared_elapsed_ns;
        series.latest_tick = numbers.latest_tick;
        if needs_replan {
            state.decisions.remove(&signature_key);
        }
        evict_oldest_feedback(&mut state, self.config.max_feedback_series);
        Ok(())
    }

    pub fn invalidate(&self, invalidation: PlacementInvalidation) {
        self.runtime_binding_session
            .invalidate_vm_generations(&invalidation);
        self.observation_permits.invalidate(&invalidation);
        if let Some(pool) = &self.managed_observations {
            pool.invalidate(&invalidation);
        }
        if let Some(pool) = &self.managed_feedback {
            pool.invalidate(&invalidation);
        }
        self.pending.invalidate(&invalidation);
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
                let provider = PlacementInvalidation::Provider { digest };
                state
                    .decisions
                    .retain(|_, entry| !provider.matches_signature(&entry.decision.signature));
                state
                    .feedback
                    .retain(|_, series| !provider.matches_signature(&series.signature));
            }
            PlacementInvalidation::HostService { digest } => {
                state.decisions.retain(|_, entry| {
                    entry.decision.signature.revision.authority
                        != runmat_execution::PlacementExecutionAuthorityRevision::HostService(
                            digest,
                        )
                });
                state.feedback.retain(|_, series| {
                    series.signature.revision.authority
                        != runmat_execution::PlacementExecutionAuthorityRevision::HostService(
                            digest,
                        )
                });
            }
            PlacementInvalidation::RuntimeBindingSession { digest } => {
                state.decisions.retain(|_, entry| {
                    entry.decision.signature.revision.authority
                        != runmat_execution::PlacementExecutionAuthorityRevision::RuntimeBindingSession(digest)
                });
                state.feedback.retain(|_, series| {
                    series.signature.revision.authority
                        != runmat_execution::PlacementExecutionAuthorityRevision::RuntimeBindingSession(digest)
                });
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

    /// Inspect ordinary session-local measurements without the persistence
    /// filter. Binding-session feedback must never enter an exported profile.
    #[cfg(test)]
    pub(crate) fn ordinary_feedback_for_test(&self) -> Vec<PlacementFeedbackSeries> {
        self.drain_observations();
        self.state.borrow().feedback.values().cloned().collect()
    }

    pub fn profile_snapshot(&self) -> PlacementProfileSnapshot {
        self.drain_observations();
        let mut feedback: Vec<_> = self.state.borrow().feedback.values().cloned().collect();
        if let Some(pool) = &self.managed_feedback {
            let managed = pool.profile_snapshot();
            feedback.retain(|ordinary| {
                !managed.iter().any(|series| {
                    ordinary.signature.cache_key() == series.signature.cache_key()
                        && ordinary.candidate == series.candidate
                        && ordinary.kind == series.kind
                })
            });
            feedback.extend(managed);
        }
        feedback.retain(|series| {
            !matches!(
                series.signature.revision.authority,
                runmat_execution::PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
            )
        });
        PlacementProfileSnapshot {
            schema_version: PLACEMENT_PROFILE_SCHEMA_VERSION,
            feedback,
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
                .then_with(|| left.kind.cmp(&right.kind))
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
            let key = (
                series.signature.cache_key(),
                feedback_candidate_key(&series.candidate),
                series.kind,
            );
            if restored.insert(key, series).is_some() {
                return Err(PlacementPlanError {
                    code: "RunMat:Placement:ProfileEntry",
                    message: "placement profile contains a duplicate feedback series".into(),
                });
            }
        }
        // A successful import replaces the observation epoch. In-flight samples
        // from the previous profile must not repopulate the replacement later.
        // Validate the entire import first, so rejected imports change nothing.
        self.runtime_binding_session
            .invalidate_vm_generations(&PlacementInvalidation::All);
        self.pending.invalidate(&PlacementInvalidation::All);
        self.observation_permits
            .invalidate(&PlacementInvalidation::All);
        if let Some(pool) = &self.managed_observations {
            pool.invalidate(&PlacementInvalidation::All);
        }
        if let Some(pool) = &self.managed_feedback {
            pool.invalidate(&PlacementInvalidation::All);
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
    series.signature.provider_provenance.is_none()
        && !matches!(
            series.signature.revision.authority,
            runmat_execution::PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
        )
        && series.signature.validate().is_ok()
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
    fn whole_work_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        provider: Digest,
    ) -> Option<runmat_execution::PlacementRevision> {
        Some(super::revision::whole_work(program, provider, self.config))
    }

    fn whole_work_revision_ref<'a>(
        &self,
        program: Option<&'a runmat_execution::ProgramRevision>,
        provider: Digest,
    ) -> Option<runmat_execution::PlacementRevisionRef<'a>> {
        Some(super::revision::whole_work_ref(
            program,
            provider,
            self.config,
        ))
    }

    fn whole_work_runtime_binding_session_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        binding: runmat_builtins::BuiltinBindingIdentity,
        registration: Digest,
    ) -> Option<runmat_execution::PlacementRevision> {
        let (revision, replaced) = self.runtime_binding_session.revision_and_replaced(
            program,
            binding,
            registration,
            self.config,
        );
        if let Some(digest) = replaced {
            self.invalidate(PlacementInvalidation::RuntimeBindingSession { digest });
        }
        Some(revision)
    }

    fn whole_work_vm_fusion_region_session_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        region: runmat_runtime::context::VmFusionRegionRegistration,
        liveness: std::sync::Weak<()>,
    ) -> Option<runmat_runtime::context::VmFusionSessionRegistration> {
        self.runtime_binding_session
            .vm_registration(program, region, liveness, self.config)
    }

    fn whole_work_host_service_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        host_service: Digest,
    ) -> Option<runmat_execution::PlacementRevision> {
        Some(super::revision::whole_work_host_service(
            program,
            host_service,
            self.config,
        ))
    }

    fn whole_work_host_service_revision_ref<'a>(
        &self,
        program: Option<&'a runmat_execution::ProgramRevision>,
        host_service: Digest,
    ) -> Option<runmat_execution::PlacementRevisionRef<'a>> {
        Some(super::revision::whole_work_host_service_ref(
            program,
            host_service,
            self.config,
        ))
    }

    fn select_whole_work(
        &self,
        request: runmat_execution::WholeWorkPlacementRequest,
    ) -> Result<runmat_runtime::context::RuntimeWholeWorkSelection, runmat_runtime::RuntimeError>
    {
        PlacementSession::select_whole_work(self, request).map_err(runtime_error)
    }

    fn try_select_whole_work(
        &self,
        request: runmat_execution::WholeWorkPlacementRequest,
    ) -> Result<
        Option<runmat_runtime::context::RuntimeWholeWorkSelection>,
        runmat_runtime::RuntimeError,
    > {
        PlacementSession::try_select_whole_work(self, request).map_err(runtime_error)
    }

    fn try_select_whole_work_borrowed_managed(
        &self,
        request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<
        Option<runmat_runtime::context::RuntimeManagedWholeWorkSelection>,
        runmat_runtime::RuntimeError,
    > {
        PlacementSession::try_select_whole_work_borrowed_managed(self, request)
            .map_err(runtime_error)
    }

    fn select_whole_work_borrowed(
        &self,
        request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<runmat_runtime::context::RuntimeWholeWorkSelection, runmat_runtime::RuntimeError>
    {
        PlacementSession::select_whole_work_borrowed(self, request).map_err(runtime_error)
    }

    fn select_whole_work_borrowed_managed(
        &self,
        request: runmat_execution::WholeWorkPlacementRequestRef<'_>,
    ) -> Result<
        runmat_runtime::context::RuntimeManagedWholeWorkSelection,
        runmat_runtime::RuntimeError,
    > {
        PlacementSession::select_whole_work_borrowed_managed(self, request).map_err(runtime_error)
    }

    fn managed_whole_work_available(&self) -> bool {
        self.managed_observations.is_some() && self.managed_feedback.is_some()
    }

    fn observed_candidate_duration(
        &self,
        signature: &PlacementSignature,
        candidate: &str,
        kind: ExecutionCandidateKind,
    ) -> CandidateObservationEvidence {
        PlacementSession::observed_candidate_duration(self, signature, candidate, kind)
    }
    fn observe_selected_whole_work(
        &self,
        selected: &runmat_runtime::context::RuntimeWholeWorkSelection,
        elapsed_ns: u64,
        outcome: runmat_execution::WholeWorkObservationOutcome,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        if !selected.observation_is_current()
            || matches!(
                outcome,
                runmat_execution::WholeWorkObservationOutcome::Cancelled
                    | runmat_execution::WholeWorkObservationOutcome::Abandoned
            )
        {
            return Ok(());
        }
        self.observe_whole_work(runmat_execution::WholeWorkObservation {
            signature: selected.decision.signature.clone(),
            selected: selected.decision.selected.clone(),
            elapsed_ns,
            outcome,
        })
    }
    fn observe_managed_whole_work(
        &self,
        selected: &runmat_runtime::context::RuntimeManagedWholeWorkSelection,
        elapsed_ns: u64,
        outcome: runmat_execution::WholeWorkObservationOutcome,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        if !selected.observation_is_current()
            || matches!(
                outcome,
                runmat_execution::WholeWorkObservationOutcome::Cancelled
                    | runmat_execution::WholeWorkObservationOutcome::Abandoned
            )
        {
            return Ok(());
        }
        let pool = self.managed_observations.as_ref().ok_or_else(|| {
            runtime_error(PlacementPlanError {
                code: "RunMat:Placement:InvalidFeedback",
                message: "managed observation owner is absent".into(),
            })
        })?;
        if !selected.observation_lease.same_owner(pool) {
            return Err(runtime_error(PlacementPlanError {
                code: "RunMat:Placement:InvalidFeedback",
                message: "managed observation belongs to another session".into(),
            }));
        }
        let key = managed_feedback::Key {
            signature: selected.identity.signature,
            permit: selected.identity,
            candidate: selected.selected.candidate,
            kind: selected.selected.kind,
        };
        let tick = self.state.borrow().tick;
        let needs_replan = self
            .managed_feedback
            .as_ref()
            .expect("managed feedback backing")
            .record(
                key,
                elapsed_ns,
                matches!(
                    outcome,
                    runmat_execution::WholeWorkObservationOutcome::Success
                ),
                tick,
            )
            .map_err(|source| {
                runtime_error(PlacementPlanError {
                    code: "RunMat:Placement:InvalidFeedback",
                    message: source.to_string(),
                })
            })?;
        if needs_replan {
            self.state.borrow_mut().decisions.remove(&key.signature);
        }
        Ok(())
    }
    fn defer_feedback(
        &self,
        source: Box<dyn runmat_runtime::context::RuntimePlacementFeedbackSource>,
    ) -> Result<(), Box<dyn runmat_runtime::context::RuntimePlacementFeedbackSource>> {
        self.drain_observations();
        self.pending.push(source)
    }

    fn drain_feedback(&self) {
        self.drain_observations();
    }
    fn plan(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<PlacementDecision, runmat_runtime::RuntimeError> {
        PlacementSession::plan(self, request).map_err(runtime_error)
    }

    fn plan_with_observation_permit(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<runmat_runtime::context::RuntimeGraphPlacementSelection, runmat_runtime::RuntimeError>
    {
        let decision = PlacementSession::plan(self, request).map_err(runtime_error)?;
        // Both operations run on this Rc-owned service before it returns to
        // the executor. No effect/publication can interleave this boundary.
        let observation_permit = self.observation_permits.register_graph(&decision);
        Ok(runmat_runtime::context::RuntimeGraphPlacementSelection {
            decision,
            observation_permit,
        })
    }

    fn observe(&self, feedback: PlacementFeedback) -> Result<(), runmat_runtime::RuntimeError> {
        PlacementSession::observe(self, feedback).map_err(runtime_error)
    }

    fn invalidate(&self, invalidation: PlacementInvalidation) {
        PlacementSession::invalidate(self, invalidation);
    }
}

impl PlacementSession {
    fn drain_observations(&self) {
        for feedback in self.pending.take_ready() {
            if let Err(error) =
                runmat_runtime::context::RuntimePlacementService::observe_deferred_feedback(
                    self, feedback,
                )
            {
                log::warn!("placement observation rejected: {error}");
            }
        }
    }
}

impl Drop for PlacementSession {
    fn drop(&mut self) {
        self.drain_observations();
        self.runtime_binding_session
            .invalidate_vm_generations(&PlacementInvalidation::All);
    }
}

fn runtime_error(error: PlacementPlanError) -> runmat_runtime::RuntimeError {
    runmat_runtime::build_runtime_error(error.message)
        .with_builtin("placement")
        .with_identifier(error.code)
        .build()
}

fn apply_feedback_evidence(
    request: &mut PlacementPlanRequest,
    signature_key: Digest,
    tick: u64,
    feedback: &BTreeMap<
        (Digest, FeedbackCandidateKey, ExecutionCandidateKind),
        PlacementFeedbackSeries,
    >,
) {
    for node in &mut request.graph.nodes {
        node.candidates.retain_mut(|candidate| {
            let key = (
                signature_key,
                feedback_candidate_key(&candidate.descriptor.identity),
                candidate.descriptor.kind,
            );
            let Some(series) = feedback.get(&key) else {
                return true;
            };
            match evidence::from_series(series, tick) {
                CandidateObservationEvidence::Observed(duration) => {
                    candidate.descriptor.cost.components = ExecutionCostComponents {
                        execution_ns: duration.elapsed_ns,
                        ..ExecutionCostComponents::default()
                    };
                    candidate.descriptor.cost.confidence = duration.confidence;
                    candidate.descriptor.cost.source = EstimateSource::Observation;
                }
                CandidateObservationEvidence::Unavailable(
                    CandidateCostUnavailable::RepeatedFailures { .. },
                ) if candidate.descriptor.kind.is_provider() => {
                    // Failure evidence makes this route ineligible. It is not
                    // a claim about queue capacity or resource demand.
                    return false;
                }
                CandidateObservationEvidence::Unavailable(_) => {}
            }
            true
        });
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
            .map(|(key, _)| *key)
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
        CandidateResourceDemand, EstimateConfidence, ExecutionCandidateDescriptor,
        ExecutionCandidateKind, ExecutionCostComponents, ExecutionCostEstimate, PlacementGraph,
        PlacementGraphCandidate, PlacementGraphEdge, PlacementGraphLimits, PlacementGraphNode,
        PlacementResourceSnapshot, PlacementRevision, PlacementSignature, ProviderResourceSnapshot,
        SourceOutputOrdinal,
    };

    use super::*;

    fn digest(token: &str) -> Digest {
        Digest::sha256(token.as_bytes())
    }

    pub(super) fn signature(token: &str) -> PlacementSignature {
        PlacementSignature {
            region: None,
            operation: token.into(),
            runtime_facts: digest(token),
            provider_provenance: None,
            graph_output_authority: None,
            revision: PlacementRevision {
                program: None,
                catalog: digest("catalog"),
                compiler: digest("compiler"),
                authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(digest(
                    "provider",
                )),
                policy: digest("policy"),
            },
        }
    }

    #[test]
    fn typed_feedback_key_preserves_exact_profile_candidate_text() {
        let session = PlacementSession::default();
        let signature = signature("feedback");
        let candidate = "candidate-λ";
        session
            .observe(PlacementFeedback {
                signature: signature.clone(),
                candidate: candidate.into(),
                kind: ExecutionCandidateKind::SharedRuntime,
                total_elapsed_ns: 17,
                succeeded: true,
            })
            .expect("record candidate");
        assert!(matches!(
            session.observed_candidate_duration(
                &signature,
                candidate,
                ExecutionCandidateKind::SharedRuntime
            ),
            CandidateObservationEvidence::Unavailable(
                CandidateCostUnavailable::InsufficientObservations {
                    successful: 1,
                    failed: 0
                }
            )
        ));
        assert_eq!(
            session.observed_candidate_duration(
                &signature,
                "candidate-other",
                ExecutionCandidateKind::SharedRuntime
            ),
            CandidateObservationEvidence::Unavailable(CandidateCostUnavailable::NoObservations)
        );
        assert_eq!(session.profile_snapshot().feedback[0].candidate, candidate);
    }

    #[test]
    fn canonical_candidate_digest_uses_the_same_observation_key_without_text() {
        let session = PlacementSession::default();
        let signature = signature("whole-work");
        let candidate = digest("typed-route");
        session
            .observe(PlacementFeedback {
                signature: signature.clone(),
                candidate: candidate.to_string(),
                kind: ExecutionCandidateKind::SharedRuntime,
                total_elapsed_ns: 31,
                succeeded: true,
            })
            .expect("record canonical candidate");
        assert_eq!(
            session.observed_candidate_duration(
                &signature,
                &candidate.to_string(),
                ExecutionCandidateKind::SharedRuntime
            ),
            session.observed_candidate_duration_digest(
                (&signature).into(),
                candidate,
                ExecutionCandidateKind::SharedRuntime,
                true,
            ),
        );
    }

    pub(super) fn candidate(
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
                    scratch_bytes: Some(0),
                    confidence: EstimateConfidence::Exact,
                    source: EstimateSource::Synthetic,
                },
                output_residency: if kind.is_provider() {
                    CandidateOutputResidency::Provider { device_id: 7 }
                } else {
                    CandidateOutputResidency::Host
                },
                presentation: runmat_execution::CandidateOutputPresentation::Direct,
                guards: Vec::new(),
            },
            resources: CandidateResourceDemand {
                cpu_millicores: if kind.is_provider() { 0 } else { 1_000 },
                retained_bytes: retained_bytes.into(),
                scratch_bytes: 0.into(),
                provider_host_workspace: None,
                provider_input_stages: None,
                queue_slots: if kind.is_provider() { 1 } else { 0 },
                peak_live: None,
            },
            output_facts: None,
            transactional_results: transactional,
        }
    }

    pub(super) fn resources(capacity: u64) -> PlacementResourceSnapshot {
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
                    source_output: None,
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
    fn transactional_provider_only_whole_work_needs_no_fake_host_baseline() {
        let mut request = two_node_request("provider-only", (100_000, 1_000), (100_000, 1_000));
        request.graph.nodes.truncate(1);
        request.graph.edges.clear();
        request.graph.nodes[0]
            .candidates
            .retain(|candidate| candidate.descriptor.kind.is_provider());
        let selected = PlacementSession::default()
            .plan(request.clone())
            .expect("one executable transactional provider route");
        assert_eq!(selected.selections.len(), 1);
        assert!(selected.selections[0].kind.is_provider());
        assert!(!selected.used_local_fallback);

        request.resources.providers[0].lost = true;
        let error = PlacementSession::default()
            .plan(request.clone())
            .expect_err("lost provider has no executable route");
        assert_eq!(error.code, "RunMat:Placement:NoLegalCandidate");

        request.resources.providers[0].lost = false;
        request.resources.cancellation_requested = true;
        let error = PlacementSession::default()
            .plan(request)
            .expect_err("cancellation cannot select an invented host route");
        assert_eq!(error.code, "RunMat:Placement:Cancelled");
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
            node.candidates[1].resources.retained_bytes = 2_048.into();
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
            node.candidates[1].resources.retained_bytes = 600.into();
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
                source_output: None,
                bytes: 600,
                host_to_provider_ns: 0,
                provider_to_host_ns: 0,
                cross_provider_ns: 0,
            },
            PlacementGraphEdge {
                from: 1,
                to: 2,
                source_output: None,
                bytes: 600,
                host_to_provider_ns: 0,
                provider_to_host_ns: 0,
                cross_provider_ns: 0,
            },
        ];
        request.resources = resources(1_000);
        request.graph.nodes[0].candidates[1]
            .resources
            .retained_bytes = 600.into();
        request.graph.nodes[1].candidates[1]
            .resources
            .retained_bytes = 600.into();

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
            for selected in &initial.selections {
                session
                    .observe(PlacementFeedback {
                        signature: request.signature.clone(),
                        candidate: selected.candidate.clone(),
                        kind: selected.kind,
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
        assert!(encoded.contains("\"kind\":\"provider_fusion\""));
        let snapshot: PlacementProfileSnapshot = serde_json::from_str(&encoded).unwrap();
        let restored = PlacementSession::default();
        restored.restore_profile(snapshot.clone()).unwrap();
        assert_eq!(restored.profile_snapshot().feedback.len(), 2);
        let mut legacy = serde_json::to_value(snapshot).unwrap();
        legacy["schema_version"] = serde_json::json!(1);
        legacy["feedback"][0]
            .as_object_mut()
            .unwrap()
            .remove("kind");
        assert!(serde_json::from_value::<PlacementProfileSnapshot>(legacy).is_err());
    }

    #[test]
    fn replanning_and_profile_restore_do_not_mix_same_label_across_candidate_kinds() {
        use runmat_execution::{CandidateCostUnavailable, CandidateObservationEvidence};

        let mut request = two_node_request("kind-separated", (10_000, 100_000), (100_000, 10_000));
        let edge = &mut request.graph.edges[0];
        edge.host_to_provider_ns = 0;
        edge.provider_to_host_ns = 0;
        edge.cross_provider_ns = 0;
        let host = &request.graph.nodes[0].candidates[0].descriptor;
        let shared_label = host.identity.clone();
        let host_kind = host.kind;
        let provider = &mut request.graph.nodes[1].candidates[1].descriptor;
        provider.identity = shared_label.clone();
        let provider_kind = provider.kind;
        assert_ne!(host_kind, provider_kind);

        let session = PlacementSession::default();
        let initial = session.plan(request.clone()).unwrap();
        assert_eq!(initial.selections[0].kind, host_kind);
        assert_eq!(initial.selections[1].kind, provider_kind);
        for _ in 0..2 {
            session
                .observe(PlacementFeedback {
                    signature: request.signature.clone(),
                    candidate: shared_label.clone(),
                    kind: provider_kind,
                    total_elapsed_ns: 250_000,
                    succeeded: true,
                })
                .unwrap();
        }
        assert_eq!(
            session.observed_candidate_duration(&request.signature, &shared_label, host_kind),
            CandidateObservationEvidence::Unavailable(CandidateCostUnavailable::NoObservations)
        );
        let replanned = session.plan(request.clone()).unwrap();
        assert_eq!(replanned.selections[0].kind, host_kind);
        assert_eq!(replanned.selections[1].kind, host_kind);

        let restored = PlacementSession::default();
        restored
            .restore_profile(session.profile_snapshot())
            .unwrap();
        let restored_plan = restored.plan(request).unwrap();
        assert_eq!(restored_plan.selections[0].kind, host_kind);
        assert_eq!(restored_plan.selections[1].kind, host_kind);
    }

    #[test]
    fn observed_duration_requires_two_successes_and_tracks_revision_invalidation() {
        use runmat_execution::{
            CandidateCostUnavailable as Missing, CandidateObservationEvidence as E,
        };

        let session = PlacementSession::default();
        let signature = signature("whole-route");
        let candidate = "provider.whole-route";
        assert_eq!(
            session.observed_candidate_duration(
                &signature,
                candidate,
                ExecutionCandidateKind::ProviderGraph
            ),
            E::Unavailable(Missing::NoObservations)
        );
        session
            .observe(PlacementFeedback {
                signature: signature.clone(),
                candidate: candidate.into(),
                kind: ExecutionCandidateKind::ProviderGraph,
                total_elapsed_ns: 8,
                succeeded: false,
            })
            .unwrap();
        assert_eq!(
            session.observed_candidate_duration(
                &signature,
                candidate,
                ExecutionCandidateKind::ProviderGraph
            ),
            E::Unavailable(Missing::InsufficientObservations {
                successful: 0,
                failed: 1
            })
        );
        for elapsed_ns in [100, 120] {
            session
                .observe(PlacementFeedback {
                    signature: signature.clone(),
                    candidate: candidate.into(),
                    kind: ExecutionCandidateKind::ProviderGraph,
                    total_elapsed_ns: elapsed_ns,
                    succeeded: true,
                })
                .unwrap();
            if elapsed_ns == 100 {
                assert_eq!(
                    session.observed_candidate_duration(
                        &signature,
                        candidate,
                        ExecutionCandidateKind::ProviderGraph
                    ),
                    E::Unavailable(Missing::InsufficientObservations {
                        successful: 1,
                        failed: 1
                    })
                );
            }
        }
        assert_eq!(
            session.observed_candidate_duration(
                &signature,
                candidate,
                ExecutionCandidateKind::ProviderGraph
            ),
            E::Observed(runmat_execution::ObservedCandidateDuration {
                elapsed_ns: 110,
                confidence: EstimateConfidence::Medium,
            })
        );
        let port: &dyn runmat_runtime::context::RuntimePlacementService = &session;
        assert_eq!(
            port.observed_candidate_duration(
                &signature,
                candidate,
                ExecutionCandidateKind::ProviderGraph
            ),
            session.observed_candidate_duration(
                &signature,
                candidate,
                ExecutionCandidateKind::ProviderGraph
            )
        );
        let series = session.profile_snapshot().feedback.pop().unwrap();
        assert_eq!(
            evidence::from_series(&series, series.latest_tick + 1_025),
            E::Unavailable(Missing::StaleObservations)
        );
        for _ in 0..6 {
            session
                .observe(PlacementFeedback {
                    signature: signature.clone(),
                    candidate: candidate.into(),
                    kind: ExecutionCandidateKind::ProviderGraph,
                    total_elapsed_ns: 110,
                    succeeded: true,
                })
                .unwrap();
        }
        assert!(matches!(
            session.observed_candidate_duration(
                &signature,
                candidate,
                ExecutionCandidateKind::ProviderGraph
            ),
            E::Observed(runmat_execution::ObservedCandidateDuration {
                confidence: EstimateConfidence::High,
                ..
            })
        ));
        let mut different_revision = signature.clone();
        different_revision.revision.authority =
            runmat_execution::PlacementExecutionAuthorityRevision::Provider(digest("new-provider"));
        assert_eq!(
            session.observed_candidate_duration(
                &different_revision,
                candidate,
                ExecutionCandidateKind::ProviderGraph
            ),
            E::Unavailable(Missing::NoObservations)
        );
        session.invalidate(PlacementInvalidation::Provider {
            digest: match signature.revision.authority {
                runmat_execution::PlacementExecutionAuthorityRevision::Provider(digest) => digest,
                runmat_execution::PlacementExecutionAuthorityRevision::HostService(_)
                | runmat_execution::PlacementExecutionAuthorityRevision::RuntimeBindingSession(_) =>
                {
                    unreachable!("provider fixture")
                }
            },
        });
        assert_eq!(
            session.observed_candidate_duration(
                &signature,
                candidate,
                ExecutionCandidateKind::ProviderGraph
            ),
            E::Unavailable(Missing::NoObservations)
        );
    }

    #[test]
    fn repeated_failures_suppress_observation_without_fabricating_cost() {
        use runmat_execution::{
            CandidateCostUnavailable as Missing, CandidateObservationEvidence as E,
        };

        let session = PlacementSession::default();
        let signature = signature("failed-route");
        let candidate = "provider.failed-route";
        for succeeded in [true, true, false, false] {
            session
                .observe(PlacementFeedback {
                    signature: signature.clone(),
                    candidate: candidate.into(),
                    kind: ExecutionCandidateKind::ProviderGraph,
                    total_elapsed_ns: 100,
                    succeeded,
                })
                .unwrap();
        }
        assert_eq!(
            session.observed_candidate_duration(
                &signature,
                candidate,
                ExecutionCandidateKind::ProviderGraph
            ),
            E::Unavailable(Missing::RepeatedFailures { failed: 2 })
        );
        assert_eq!(
            session.observed_candidate_duration(
                &signature,
                candidate,
                ExecutionCandidateKind::SharedRuntime
            ),
            E::Unavailable(Missing::NoObservations)
        );
        for _ in 0..2 {
            session
                .observe(PlacementFeedback {
                    signature: signature.clone(),
                    candidate: candidate.into(),
                    kind: ExecutionCandidateKind::SharedRuntime,
                    total_elapsed_ns: 75,
                    succeeded: true,
                })
                .unwrap();
        }
        assert!(matches!(
            session.observed_candidate_duration(
                &signature,
                candidate,
                ExecutionCandidateKind::SharedRuntime
            ),
            E::Observed(_)
        ));
        assert_eq!(
            session.observed_candidate_duration(
                &signature,
                candidate,
                ExecutionCandidateKind::ProviderGraph
            ),
            E::Unavailable(Missing::RepeatedFailures { failed: 2 })
        );
    }

    #[test]
    fn failed_provider_is_ineligible_with_unknown_queue_bounds_until_evidence_stales() {
        let mut request =
            two_node_request("failed-eligibility", (100_000, 10_000), (100_000, 10_000));
        request.resources.providers[0].queue_depth = None;
        request.resources.providers[0].queue_limit = None;
        let edge = &mut request.graph.edges[0];
        edge.host_to_provider_ns = 0;
        edge.provider_to_host_ns = 0;
        edge.cross_provider_ns = 0;
        let failed = &request.graph.nodes[0].candidates[1].descriptor;
        let failed_label = failed.identity.clone();
        let failed_kind = failed.kind;
        request.graph.nodes[1].candidates[0].descriptor.identity = failed_label.clone();

        let session = PlacementSession::default();
        let initial = session.plan(request.clone()).unwrap();
        assert!(initial
            .selections
            .iter()
            .all(|selected| selected.kind.is_provider()));
        for _ in 0..2 {
            session
                .observe(PlacementFeedback {
                    signature: request.signature.clone(),
                    candidate: failed_label.clone(),
                    kind: failed_kind,
                    total_elapsed_ns: 0,
                    succeeded: false,
                })
                .unwrap();
        }
        let key = request.signature.cache_key();
        let mut admitted = request.clone();
        {
            let mut state = session.state.borrow_mut();
            apply_feedback_evidence(&mut admitted, key, state.tick, &state.feedback);
            // A stale cached provider ticket must also fail revalidation.
            let tick = state.tick;
            state.decisions.insert(
                key,
                CachedDecision {
                    decision: initial,
                    resources: request.resources.clone(),
                    last_used: tick,
                },
            );
        }
        assert_eq!(admitted.graph.nodes[0].candidates.len(), 1);
        assert_eq!(
            admitted.graph.nodes[0].candidates[0].resources.queue_slots,
            0
        );
        assert_eq!(admitted.graph.nodes[1].candidates.len(), 2);
        assert_eq!(
            admitted.graph.nodes[1].candidates[1].resources.queue_slots,
            1
        );

        let after_failure = session.plan(request.clone()).unwrap();
        assert!(!after_failure.from_cache);
        assert!(!after_failure.selections[0].kind.is_provider());
        assert!(after_failure.selections[1].kind.is_provider());

        let restored = PlacementSession::default();
        restored
            .restore_profile(session.profile_snapshot())
            .unwrap();
        let restored_plan = restored.plan(request.clone()).unwrap();
        assert!(!restored_plan.selections[0].kind.is_provider());
        assert!(restored_plan.selections[1].kind.is_provider());

        let latest = session.profile_snapshot().feedback[0].latest_tick;
        session.state.borrow_mut().tick = latest + 1_025;
        let readmitted = session.plan(request).unwrap();
        assert!(readmitted
            .selections
            .iter()
            .all(|selected| selected.kind.is_provider()));
    }

    #[test]
    fn graph_output_authority_separates_failure_profiles_and_cached_decisions() {
        let mut old = two_node_request("output-feedback", (100_000, 10_000), (100_000, 10_000));
        old.resources.providers[0].queue_depth = None;
        old.resources.providers[0].queue_limit = None;
        old.graph.edges[0].host_to_provider_ns = 0;
        old.graph.edges[0].provider_to_host_ns = 0;
        old.graph.edges[0].cross_provider_ns = 0;
        let failed = &old.graph.nodes[0].candidates[1].descriptor;
        let candidate = failed.identity.clone();
        let kind = failed.kind;
        let session = PlacementSession::default();
        assert!(session.plan(old.clone()).unwrap().selections[0]
            .kind
            .is_provider());
        for _ in 0..2 {
            session
                .observe(PlacementFeedback {
                    signature: old.signature.clone(),
                    candidate: candidate.clone(),
                    kind,
                    total_elapsed_ns: 0,
                    succeeded: false,
                })
                .unwrap();
        }
        assert!(!session.plan(old.clone()).unwrap().selections[0]
            .kind
            .is_provider());
        let mut changed = old;
        changed.graph.edges[0].source_output = Some(SourceOutputOrdinal::new(0));
        assert_eq!(
            session.plan(changed.clone()).unwrap_err().code,
            "RunMat:Placement:InvalidRequest"
        );
        changed.bind_graph_output_authority();
        assert_ne!(
            changed.signature.cache_key(),
            session.profile_snapshot().feedback[0].signature.cache_key()
        );
        assert!(session.plan(changed.clone()).unwrap().selections[0]
            .kind
            .is_provider());
        let restored = PlacementSession::default();
        restored
            .restore_profile(session.profile_snapshot())
            .unwrap();
        assert!(restored.plan(changed).unwrap().selections[0]
            .kind
            .is_provider());
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
                candidate: request.graph.nodes[0].candidates[1]
                    .descriptor
                    .identity
                    .clone(),
                kind: request.graph.nodes[0].candidates[1].descriptor.kind,
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
        let provider = &request.graph.nodes[0].candidates[1].descriptor;
        for _ in 1..=2 {
            session
                .observe(PlacementFeedback {
                    signature: request.signature.clone(),
                    candidate: provider.identity.clone(),
                    kind: provider.kind,
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
