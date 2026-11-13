//! Owning-session delivery of provider-completed candidate totals.
use super::local::LocalPlacementFeedbackAuthority;
use super::LocalPlacementFeedback;
use runmat_accelerate_api::observation::{
    ProviderObservationFailure, ProviderObservationStatus, ProviderOperationReceipt,
};
use runmat_execution::profiling::{CommandProviderContribution, ProviderObservationTerminal};
#[cfg(test)]
use runmat_runtime::context::RuntimeGraphObservationPermit;
use runmat_runtime::context::RuntimePlacementFeedbackSource;
use runmat_runtime::context::VmFusionFeedbackLease;

/// Publication follows output validation/adoption and temporary cleanup. The
/// deferred source retains only compact identity and cancellation, not runtime
/// or session, so a session owning that source cannot form a reference cycle.
pub(crate) fn publish_provider_feedback(
    receipt: ProviderOperationReceipt,
    feedback: LocalPlacementFeedback,
    command_contribution: Option<CommandProviderContribution>,
) {
    publish_provider_feedback_with_boundary(
        receipt,
        feedback,
        command_contribution,
        CandidateTimingBoundary::PreparationBodyCleanup,
        None,
    );
}

/// The outer fence has finished before publication. The receipt's own default
/// monotonic clock covers staging through later provider completion; the typed
/// prelude covers the earlier selected call. Custom-clock receipts are rejected
/// before this function is called.
pub(crate) fn publish_provider_feedback_full_call(
    receipt: ProviderOperationReceipt,
    feedback: LocalPlacementFeedback,
    prelude: std::time::Duration,
    command_contribution: Option<CommandProviderContribution>,
) {
    publish_provider_feedback_with_boundary(
        receipt,
        feedback,
        command_contribution,
        CandidateTimingBoundary::FullCall { prelude },
        None,
    );
}

/// VM fusion retains only a Send-safe installation liveness lease until the
/// selected queue fence resolves. No RuntimeContext or service Rc is retained.
pub(crate) fn publish_provider_feedback_vm_full_call(
    receipt: ProviderOperationReceipt,
    feedback: LocalPlacementFeedback,
    prelude: std::time::Duration,
    installation: VmFusionFeedbackLease,
    command_contribution: Option<CommandProviderContribution>,
) {
    publish_provider_feedback_with_boundary(
        receipt,
        feedback,
        command_contribution,
        CandidateTimingBoundary::FullCall { prelude },
        Some(installation),
    );
}

#[derive(Clone, Copy)]
enum CandidateTimingBoundary {
    PreparationBodyCleanup,
    FullCall { prelude: std::time::Duration },
}

fn publish_provider_feedback_with_boundary(
    mut receipt: ProviderOperationReceipt,
    feedback: LocalPlacementFeedback,
    command_contribution: Option<CommandProviderContribution>,
    boundary: CandidateTimingBoundary,
    installation: Option<VmFusionFeedbackLease>,
) {
    if !feedback.is_current() {
        return;
    }
    let cancellation = feedback.runtime.cancellation();
    if cancellation.is_cancelled() {
        receipt.fail(ProviderObservationFailure::Cancelled);
    } else if receipt.publish().is_err() {
        receipt.fail(ProviderObservationFailure::PublicationFailed);
    }
    let Some(service) = feedback.runtime.service_ports().placement() else {
        return;
    };
    let mut source = DeferredProviderFeedback {
        receipt,
        command_contribution,
        signature: feedback.signature,
        candidate: feedback.candidate,
        kind: feedback.kind,
        authority: feedback.authority,
        cancellation,
        terminal_without_feedback: false,
        boundary,
        installation,
    };
    if let Some(ready) = source.take_ready() {
        let _ = service.observe_deferred_feedback(ready);
    } else if !source.terminal_without_feedback()
        && service.defer_feedback(Box::new(source)).is_err()
    {
        log::debug!(
            "provider observation discarded: placement service cannot retain pending completion"
        );
    }
}

struct DeferredProviderFeedback {
    receipt: ProviderOperationReceipt,
    command_contribution: Option<CommandProviderContribution>,
    signature: runmat_execution::PlacementSignature,
    candidate: String,
    kind: runmat_execution::ExecutionCandidateKind,
    authority: LocalPlacementFeedbackAuthority,
    cancellation: runmat_execution::cancellation::CancellationToken,
    terminal_without_feedback: bool,
    boundary: CandidateTimingBoundary,
    installation: Option<VmFusionFeedbackLease>,
}

impl DeferredProviderFeedback {
    fn retire_unattributable(&mut self) {
        if let Some(contribution) = self.command_contribution.take() {
            contribution.complete(ProviderObservationTerminal::Missing);
        }
        self.terminal_without_feedback = true;
    }
}

impl RuntimePlacementFeedbackSource for DeferredProviderFeedback {
    fn signature(&self) -> &runmat_execution::PlacementSignature {
        &self.signature
    }
    fn take_ready(&mut self) -> Option<runmat_runtime::context::RuntimePlacementFeedbackDelivery> {
        if !self
            .authority
            .authorizes(&self.signature, &self.candidate, self.kind)
        {
            self.receipt.fail(ProviderObservationFailure::Abandoned);
            let _ = self.receipt.take_completed();
            self.retire_unattributable();
            return None;
        }
        if self
            .installation
            .as_ref()
            .is_some_and(|lease| !lease.is_current())
        {
            self.receipt.fail(ProviderObservationFailure::Abandoned);
            let _ = self.receipt.take_completed();
            self.retire_unattributable();
            return None;
        }
        if self.cancellation.is_cancelled() {
            self.receipt.fail(ProviderObservationFailure::Cancelled);
        }
        let observation = self.receipt.take_completed()?;
        if !self
            .authority
            .authorizes(&self.signature, &self.candidate, self.kind)
        {
            self.retire_unattributable();
            return None;
        }
        if self
            .installation
            .as_ref()
            .is_some_and(|lease| !lease.is_current())
        {
            self.retire_unattributable();
            return None;
        }
        if let Some(contribution) = self.command_contribution.take() {
            let terminal = match observation.status {
                ProviderObservationStatus::Completed { .. } => {
                    ProviderObservationTerminal::Completed {
                        gpu_execution: observation.verified_wgpu_compute_elapsed,
                        kernel_count: observation.verified_wgpu_dispatches,
                    }
                }
                ProviderObservationStatus::Failed(ProviderObservationFailure::ProviderFailed) => {
                    ProviderObservationTerminal::Failed
                }
                ProviderObservationStatus::Failed(_) | ProviderObservationStatus::Pending => {
                    ProviderObservationTerminal::Missing
                }
            };
            contribution.complete(terminal);
        }
        let (elapsed, succeeded) = match observation.status {
            ProviderObservationStatus::Completed { total_elapsed, .. } => {
                let total = match self.boundary {
                    CandidateTimingBoundary::PreparationBodyCleanup => total_elapsed,
                    CandidateTimingBoundary::FullCall { prelude } => prelude
                        .checked_add(total_elapsed)
                        .unwrap_or(std::time::Duration::MAX),
                };
                (runmat_time::duration_ns_saturating(total), true)
            }
            ProviderObservationStatus::Failed(failure) => match failure {
                ProviderObservationFailure::Cancelled | ProviderObservationFailure::Abandoned => {
                    self.terminal_without_feedback = true;
                    return None;
                }
                // A FullCall receipt failure has no measured outer duration;
                // preserve negative receipt accounting only for v2.
                ProviderObservationFailure::ProviderFailed
                    if matches!(self.boundary, CandidateTimingBoundary::FullCall { .. }) =>
                {
                    self.terminal_without_feedback = true;
                    return None;
                }
                ProviderObservationFailure::ProviderFailed => (0, false),
                _ if matches!(self.boundary, CandidateTimingBoundary::FullCall { .. }) => {
                    self.terminal_without_feedback = true;
                    return None;
                }
                // Preserve the existing v2 receipt-failure contract.
                _ => (0, false),
            },
            ProviderObservationStatus::Pending => return None,
        };
        let delivery = self.authority.delivery(
            &self.signature,
            &self.candidate,
            self.kind,
            elapsed,
            succeeded,
        );
        if delivery.is_none() {
            self.terminal_without_feedback = true;
        }
        delivery
    }

    fn terminal_without_feedback(&self) -> bool {
        self.terminal_without_feedback
    }
}

#[cfg(test)]
#[derive(Debug)]
pub(super) struct ReceiptFixturePermit(pub(super) runmat_execution::Digest);

#[cfg(test)]
impl RuntimeGraphObservationPermit for ReceiptFixturePermit {
    fn authorizes(
        &self,
        signature: &runmat_execution::PlacementSignature,
        candidate: &str,
        kind: runmat_execution::ExecutionCandidateKind,
    ) -> bool {
        signature.cache_key() == self.0
            && candidate == "provider.fixture"
            && kind == runmat_execution::ExecutionCandidateKind::ProviderOperation
    }
}

#[cfg(test)]
pub(in crate::placement) fn whole_work_source_for_test(
    receipt: ProviderOperationReceipt,
    selected: runmat_runtime::context::RuntimeWholeWorkSelection,
) -> Box<dyn RuntimePlacementFeedbackSource> {
    let mut source = source_for_test(
        receipt,
        CandidateTimingBoundary::FullCall {
            prelude: std::time::Duration::ZERO,
        },
    );
    source.signature = selected.decision.signature.clone();
    source.candidate = selected.decision.selected.candidate.clone();
    source.kind = selected.decision.selected.kind;
    source.authority = LocalPlacementFeedbackAuthority::WholeWork(std::sync::Arc::new(
        std::sync::Mutex::new(Some(selected)),
    ));
    Box::new(source)
}

#[cfg(test)]
pub(in crate::placement) fn managed_whole_work_source_for_test(
    receipt: ProviderOperationReceipt,
    selected: runmat_runtime::context::RuntimeManagedWholeWorkSelection,
    signature: runmat_execution::PlacementSignature,
) -> Box<dyn RuntimePlacementFeedbackSource> {
    let mut source = source_for_test(
        receipt,
        CandidateTimingBoundary::FullCall {
            prelude: std::time::Duration::ZERO,
        },
    );
    source.signature = signature;
    source.candidate = selected.selected.candidate.to_string();
    source.kind = selected.selected.kind;
    source.authority = LocalPlacementFeedbackAuthority::Managed(std::sync::Arc::new(
        std::sync::Mutex::new(Some(selected)),
    ));
    Box::new(source)
}

#[cfg(test)]
pub(in crate::placement) fn failed_source_for_test(
    failure: ProviderObservationFailure,
) -> Box<dyn RuntimePlacementFeedbackSource> {
    let mut receipt = ProviderOperationReceipt::new();
    receipt.fail(failure);
    Box::new(source_for_test(
        receipt,
        CandidateTimingBoundary::PreparationBodyCleanup,
    ))
}

#[cfg(test)]
fn source_for_test(
    receipt: ProviderOperationReceipt,
    boundary: CandidateTimingBoundary,
) -> DeferredProviderFeedback {
    use runmat_execution::{
        Digest, ExecutionCandidateKind, PlacementExecutionAuthorityRevision, PlacementRevision,
        PlacementSignature,
    };
    let digest = Digest::sha256(b"provider-completion-fixture");
    let signature = PlacementSignature {
        region: None,
        operation: "provider-completion-fixture".into(),
        runtime_facts: digest,
        provider_provenance: None,
        graph_output_authority: None,
        revision: PlacementRevision {
            program: None,
            authority: PlacementExecutionAuthorityRevision::Provider(digest),
            policy: digest,
            compiler: digest,
            catalog: digest,
        },
    };
    DeferredProviderFeedback {
        receipt,
        command_contribution: None,
        authority: LocalPlacementFeedbackAuthority::Graph(std::sync::Arc::new(
            ReceiptFixturePermit(signature.cache_key()),
        )),
        signature,
        candidate: "provider.fixture".into(),
        kind: ExecutionCandidateKind::ProviderOperation,
        cancellation: runmat_execution::cancellation::CancellationSource::new().token(),
        terminal_without_feedback: false,
        boundary,
        installation: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::observation::{ProviderObservationClock, WgpuSubmissionMeasurement};
    use runmat_execution::profiling::{
        CommandProfilingRoot, ObservedMetricCoverage, ProviderObservationCounts,
    };
    use runmat_execution::{
        CandidateCostUnavailable, CandidateObservationEvidence, Digest, ExecutionCandidateKind,
        PlacementExecutionAuthorityRevision, PlacementInvalidation, PlacementRevision,
        PlacementSignature,
    };
    use runmat_runtime::context::{RuntimeContext, RuntimePlacementService, RuntimeServicePorts};
    use std::rc::Rc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    #[derive(Default)]
    struct TestClock(AtomicU64);
    impl ProviderObservationClock for TestClock {
        fn now(&self) -> Option<std::time::Duration> {
            Some(std::time::Duration::from_nanos(
                self.0.load(Ordering::SeqCst),
            ))
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn invalidated_selected_receipts_cannot_publish_and_fresh_selection_can() {
        let session = Rc::new(crate::placement::PlacementSession::default());
        let command = CommandProfilingRoot::new();
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_placement(session.clone()));
        let provider = Digest::sha256(b"capability A");
        let revision = Digest::sha256(b"VM installation");
        let signature = PlacementSignature {
            region: None,
            operation: "fusion.reduction".into(),
            runtime_facts: Digest::sha256(b"selected facts"),
            provider_provenance: Some(provider),
            graph_output_authority: None,
            revision: PlacementRevision {
                program: None,
                catalog: revision,
                compiler: revision,
                authority: PlacementExecutionAuthorityRevision::RuntimeBindingSession(revision),
                policy: revision,
            },
        };
        let candidate = "provider.physical";
        let kind = ExecutionCandidateKind::ProviderFusion;
        let selected = |signature: PlacementSignature| LocalPlacementFeedback {
            authority: super::LocalPlacementFeedbackAuthority::Graph(
                session.test_graph_observation_permit(signature.clone(), candidate, kind),
            ),
            runtime: runtime.clone(),
            signature,
            candidate: candidate.into(),
            kind,
        };

        let pending = ProviderOperationReceipt::new();
        let completion = pending.recorder().submission().unwrap();
        publish_provider_feedback(
            pending,
            selected(signature.clone()),
            command.receipt().begin_provider_operation(),
        );
        session.invalidate(PlacementInvalidation::Provider { digest: provider });
        completion.complete();
        RuntimePlacementService::drain_feedback(session.as_ref());
        assert_eq!(command.seal().observations.expect("valid").missing, 1);
        assert_eq!(
            session.observed_candidate_duration(&signature, candidate, kind),
            CandidateObservationEvidence::Unavailable(CandidateCostUnavailable::NoObservations)
        );

        let ready = ProviderOperationReceipt::new();
        ready.recorder().submission().unwrap().complete();
        let invalidated = selected(signature.clone());
        session.invalidate(PlacementInvalidation::Provider { digest: provider });
        publish_provider_feedback(ready, invalidated, None);
        assert_eq!(
            session.observed_candidate_duration(&signature, candidate, kind),
            CandidateObservationEvidence::Unavailable(CandidateCostUnavailable::NoObservations)
        );

        let fresh = ProviderOperationReceipt::new();
        fresh.recorder().submission().unwrap().complete();
        publish_provider_feedback(fresh, selected(signature.clone()), None);
        assert_eq!(
            session.observed_candidate_duration(&signature, candidate, kind),
            CandidateObservationEvidence::Unavailable(
                CandidateCostUnavailable::InsufficientObservations {
                    successful: 1,
                    failed: 0,
                }
            )
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn deferred_total_adds_exact_prelude_without_counting_queue_drain_delay() {
        let command = CommandProfilingRoot::new();
        let clock = Arc::new(TestClock::default());
        let mut receipt = ProviderOperationReceipt::with_clock(clock.clone());
        let completion = receipt.recorder().submission().unwrap();
        clock.0.store(5, Ordering::SeqCst);
        receipt.publish().unwrap();
        let mut source = source_for_test(
            receipt,
            CandidateTimingBoundary::FullCall {
                prelude: std::time::Duration::from_nanos(80),
            },
        );
        source.command_contribution = command.receipt().begin_provider_operation();
        assert!(source.take_ready().is_none());
        assert!(!source.terminal_without_feedback());
        clock.0.store(20, Ordering::SeqCst);
        completion
            .capture_completion()
            .complete_with_wgpu(WgpuSubmissionMeasurement {
                device_elapsed: None,
                compute_elapsed: Some(std::time::Duration::from_nanos(7)),
                dispatches: None,
                query_allocation: None,
            });
        clock.0.store(1_000, Ordering::SeqCst);
        let feedback = source
            .take_ready()
            .expect("provider completion is ready")
            .snapshot()
            .expect("sample");
        assert!(feedback.succeeded);
        assert_eq!(feedback.total_elapsed_ns, 100);
        let snapshot = command.seal();
        assert_eq!(
            snapshot.observations,
            Some(ProviderObservationCounts {
                incomplete: 1,
                ..ProviderObservationCounts::default()
            })
        );
        assert_eq!(snapshot.gpu_execution, None);
        assert_eq!(snapshot.kernel_count, None);
        assert_eq!(
            snapshot.observed_metric_coverage,
            Some(ObservedMetricCoverage {
                gpu_execution: true,
                kernel_count: false,
            })
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn v3_discards_missing_proof_without_changing_v2_failure_accounting() {
        for failure in [
            ProviderObservationFailure::ExecutionUnproven,
            ProviderObservationFailure::InvalidTiming,
            ProviderObservationFailure::TrackingOverflow,
            ProviderObservationFailure::ResourceLimit,
            ProviderObservationFailure::PublicationFailed,
            ProviderObservationFailure::SharedRuntimeFallback,
        ] {
            let mut receipt = ProviderOperationReceipt::new();
            receipt.fail(failure);
            let mut v3 = source_for_test(
                receipt,
                CandidateTimingBoundary::FullCall {
                    prelude: std::time::Duration::ZERO,
                },
            );
            assert!(v3.take_ready().is_none());
            assert!(v3.terminal_without_feedback());

            let mut receipt = ProviderOperationReceipt::new();
            receipt.fail(failure);
            let mut v2 = source_for_test(receipt, CandidateTimingBoundary::PreparationBodyCleanup);
            assert!(
                !v2.take_ready()
                    .expect("v2 negative sample")
                    .snapshot()
                    .expect("sample")
                    .succeeded
            );
        }
        let mut receipt = ProviderOperationReceipt::new();
        receipt.fail(ProviderObservationFailure::ProviderFailed);
        let mut full_call_failure = source_for_test(
            receipt,
            CandidateTimingBoundary::FullCall {
                prelude: std::time::Duration::ZERO,
            },
        );
        assert!(full_call_failure.take_ready().is_none());
        assert!(full_call_failure.terminal_without_feedback());

        let mut receipt = ProviderOperationReceipt::new();
        receipt.fail(ProviderObservationFailure::ProviderFailed);
        let mut preparation_failure =
            source_for_test(receipt, CandidateTimingBoundary::PreparationBodyCleanup);
        let feedback = preparation_failure
            .take_ready()
            .expect("existing preparation-body failure remains attributable")
            .snapshot()
            .expect("sample");
        assert!(!feedback.succeeded);
        assert_eq!(feedback.total_elapsed_ns, 0);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn cancellation_and_abandonment_are_terminal_without_a_false_sample() {
        for failure in [
            ProviderObservationFailure::Cancelled,
            ProviderObservationFailure::Abandoned,
        ] {
            let mut source = failed_source_for_test(failure);
            assert!(source.take_ready().is_none());
            assert!(source.terminal_without_feedback());
            assert!(source.take_ready().is_none());
        }
        let mut completed_error =
            failed_source_for_test(ProviderObservationFailure::ProviderFailed);
        let feedback = completed_error
            .take_ready()
            .expect("completed operational error")
            .snapshot()
            .expect("sample");
        assert!(!feedback.succeeded);
        assert!(!completed_error.terminal_without_feedback());
    }
}
