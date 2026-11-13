//! Display timing evidence for exact checked workloads. No synthetic throughput,
//! CPU-name table, deadline-as-ETA, or core-count speedup is used here.
//!
//! This is an adapter over the session's existing bounded placement profile,
//! not another timing store. Session authority prevents reuse on a different
//! machine/runtime. Freshness and invalidation remain with the existing profile
//! policy, so long-running studies can accumulate applicable observations.
use super::{CheckedFeaSessionSource, RunMatSession};
use runmat_accelerate::placement::PlacementSession;
use runmat_execution::{
    CandidateObservationEvidence, Digest, EstimateConfidence, ExecutionCandidateKind,
    PlacementExecutionAuthorityRevision, PlacementFeedback, PlacementInvalidation,
    PlacementRevision, PlacementSignature,
};
use runmat_runtime::analysis::AnalysisExecutionDuration;
use runmat_runtime::context::RuntimeTimingService;
use std::{rc::Rc, time::Duration};

#[path = "fea_duration/statistics.rs"]
mod statistics;

const CANDIDATE: &str = "fea.complete-local-route.v1";

#[derive(Clone, Copy, Debug)]
pub enum FeaTimingRoute {
    /// Includes native preparation, execution, assessment, and publication.
    NativeCheckedSource,
    /// Includes mesh preparation, worker startup, execution, and publication.
    BrowserCheckedSource,
}

/// An explicit host handle; never embedded in portable checked-source products.
#[derive(Clone)]
pub struct FeaExecutionTimingProfile {
    placement: Rc<PlacementSession>,
    clock: Option<Rc<dyn RuntimeTimingService>>,
    authority: Digest,
}

pub struct FeaExecutionTimingKey {
    profile: FeaExecutionTimingProfile,
    clock: Rc<dyn RuntimeTimingService>,
    signature: PlacementSignature,
    kind: ExecutionCandidateKind,
}

/// Dropping an observation (including cancellation) records no successful time.
pub struct FeaExecutionTimingObservation {
    key: FeaExecutionTimingKey,
    started: Duration,
    permit: Option<std::sync::Arc<dyn runmat_runtime::context::RuntimeWholeWorkObservationPermit>>,
}

impl RunMatSession {
    /// Missing Clock authority leaves advisory estimates unavailable; it does
    /// not install a fallback clock or prevent untimed execution.
    pub fn fea_execution_timing_profile(&self) -> FeaExecutionTimingProfile {
        FeaExecutionTimingProfile {
            placement: self.placement_session.clone(),
            clock: self.runtime_context.service_ports().timing().ok().cloned(),
            authority: Digest::sha256(self.abi_workspace_handle.0.as_bytes()),
        }
    }
}

impl FeaExecutionTimingProfile {
    pub fn for_source(
        &self,
        source: &CheckedFeaSessionSource,
        route: FeaTimingRoute,
        cores: u32,
    ) -> Result<FeaExecutionTimingKey, String> {
        self.for_revision(
            source.program_revision()?,
            source.check_result().revision.0,
            route,
            cores,
        )
    }

    fn for_revision(
        &self,
        program: runmat_execution::ProgramRevision,
        checked_revision: [u8; 32],
        route: FeaTimingRoute,
        cores: u32,
    ) -> Result<FeaExecutionTimingKey, String> {
        let clock = self
            .clock
            .clone()
            .ok_or_else(|| "Timing is unavailable for this session".to_owned())?;
        if cores == 0
            || cores
                > u32::from(runmat_analysis_fea::planning::resource::MAX_LOCAL_EXECUTION_THREADS)
        {
            return Err("Invalid timing thread allocation".into());
        }
        let kind = match route {
            FeaTimingRoute::NativeCheckedSource => ExecutionCandidateKind::GenericNativeCpu,
            FeaTimingRoute::BrowserCheckedSource => ExecutionCandidateKind::SharedRuntime,
        };
        let route = match route {
            FeaTimingRoute::NativeCheckedSource => "native-checked-source",
            FeaTimingRoute::BrowserCheckedSource => "browser-checked-source-worker",
        };
        let facts = Digest::sha256_serialized(&(checked_revision, route, cores))
            .map_err(|error| error.to_string())?;
        Ok(FeaExecutionTimingKey {
            profile: self.clone(),
            clock,
            kind,
            signature: PlacementSignature {
                region: None,
                operation: "fea.local-duration".into(),
                runtime_facts: facts,
                provider_provenance: None,
                graph_output_authority: None,
                revision: PlacementRevision {
                    program: Some(program),
                    catalog: Digest::sha256(b"fea-local-duration/v1"),
                    compiler: Digest::sha256(b"complete-successful-route/v1"),
                    authority: PlacementExecutionAuthorityRevision::HostService(self.authority),
                    policy: Digest::sha256(b"three-samples-cv25/v1"),
                },
            },
        })
    }
}

impl FeaExecutionTimingKey {
    /// Fail closed for unknown, failed, stale, sparse, or unstable observations.
    /// These are observed repeat-run timings, not an extrapolation to other meshes.
    pub fn estimate(&self) -> Option<(u64, AnalysisExecutionDuration)> {
        let CandidateObservationEvidence::Observed(observed) = self
            .profile
            .placement
            .observed_candidate_duration(&self.signature, CANDIDATE, self.kind)
        else {
            return None;
        };
        if !matches!(
            observed.confidence,
            EstimateConfidence::Medium | EstimateConfidence::High
        ) {
            return None;
        }
        let snapshot = self.profile.placement.profile_snapshot();
        let series = snapshot.feedback.iter().find(|series| {
            series.signature == self.signature
                && series.candidate == CANDIDATE
                && series.kind == self.kind
        })?;
        let summary = statistics::summarize(
            series.samples,
            series.failures,
            series.total_elapsed_ns,
            series.total_squared_elapsed_ns,
        )?;
        Some((
            summary.mean_millis,
            AnalysisExecutionDuration {
                lower_millis: summary.lower_millis,
                upper_millis: summary.upper_millis,
                sample_count: summary.sample_count,
            },
        ))
    }

    pub fn start(self) -> FeaExecutionTimingObservation {
        let permit = self
            .profile
            .placement
            .feedback_observation_permit(&self.signature)
            .ok();
        let started = self.clock.monotonic_elapsed();
        FeaExecutionTimingObservation {
            key: self,
            started,
            permit,
        }
    }
}

impl FeaExecutionTimingObservation {
    /// Call only after complete validated execution and durable publication.
    /// A failed route retires old evidence; a dropped/cancelled route adds none.
    pub fn complete(self, succeeded: bool) {
        if !self
            .permit
            .as_ref()
            .is_some_and(|permit| permit.is_current())
        {
            return;
        }
        if !succeeded {
            self.key
                .profile
                .placement
                .invalidate(PlacementInvalidation::Signature {
                    key: self.key.signature.cache_key(),
                });
            return;
        }
        let Some(elapsed) = self.key.clock.monotonic_elapsed().checked_sub(self.started) else {
            return;
        };
        let Ok(elapsed) = u64::try_from(elapsed.as_nanos()) else {
            return;
        };
        // Zero-resolution clocks must not manufacture a sub-millisecond sample.
        if elapsed == 0 {
            return;
        }
        tracing::debug!(target: "runmat.fea.timing", elapsed_ns = elapsed, "recording successful exact-route duration");
        let _ = self.key.profile.placement.observe(PlacementFeedback {
            signature: self.key.signature,
            candidate: CANDIDATE.into(),
            kind: self.key.kind,
            total_elapsed_ns: elapsed,
            succeeded: true,
        });
    }
}

#[cfg(test)]
#[path = "fea_duration/tests.rs"]
mod tests;
