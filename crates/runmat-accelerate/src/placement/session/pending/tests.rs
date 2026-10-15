use super::*;
use runmat_accelerate_api::observation::ProviderObservationFailure;
use runmat_execution::{Digest, PlacementFeedback, PlacementRevision, PlacementSignature};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

struct Source {
    signature: PlacementSignature,
    ready: Arc<AtomicBool>,
    dropped: Arc<AtomicUsize>,
}
impl RuntimePlacementFeedbackSource for Source {
    fn signature(&self) -> &PlacementSignature {
        &self.signature
    }
    fn take_ready(&mut self) -> Option<RuntimePlacementFeedbackDelivery> {
        self.ready
            .load(Ordering::Acquire)
            .then(|| RuntimePlacementFeedbackDelivery::Graph {
                feedback: PlacementFeedback {
                    signature: self.signature.clone(),
                    candidate: "provider.fixture".into(),
                    kind: runmat_execution::ExecutionCandidateKind::ProviderOperation,
                    total_elapsed_ns: 19,
                    succeeded: true,
                },
                permit: Arc::new(crate::placement::completion::ReceiptFixturePermit(
                    self.signature.cache_key(),
                )),
            })
    }
}
impl Drop for Source {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::AcqRel);
    }
}
fn source(ready: Arc<AtomicBool>, dropped: Arc<AtomicUsize>) -> Box<Source> {
    let digest = Digest::sha256(b"pending-test");
    Box::new(Source {
        ready,
        dropped,
        signature: PlacementSignature {
            region: None,
            operation: "test".into(),
            runtime_facts: digest,
            provider_provenance: None,
            graph_output_authority: None,
            revision: PlacementRevision {
                program: None,
                authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(digest),
                policy: digest,
                compiler: digest,
                catalog: digest,
            },
        },
    })
}

struct TerminalSource(Source);
impl RuntimePlacementFeedbackSource for TerminalSource {
    fn signature(&self) -> &PlacementSignature {
        &self.0.signature
    }
    fn take_ready(&mut self) -> Option<RuntimePlacementFeedbackDelivery> {
        None
    }
    fn terminal_without_feedback(&self) -> bool {
        true
    }
}

#[test]
fn cancelled_or_abandoned_completion_retires_without_a_failure_sample() {
    let queue = PendingFeedback::default();
    let dropped = Arc::new(AtomicUsize::new(0));
    let pending = source(Arc::new(AtomicBool::new(false)), dropped.clone());
    assert!(queue.push(Box::new(TerminalSource(*pending))).is_ok());
    assert!(queue.take_ready().is_empty());
    assert!(queue.sources.borrow().is_empty());
    assert_eq!(dropped.load(Ordering::Acquire), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn concrete_provider_receipt_cancellation_and_abandonment_retire_from_queue() {
    for failure in [
        ProviderObservationFailure::Cancelled,
        ProviderObservationFailure::Abandoned,
    ] {
        let queue = PendingFeedback::default();
        assert!(queue
            .push(crate::placement::completion::failed_source_for_test(
                failure
            ))
            .is_ok());
        assert!(queue.has_sources());
        assert!(queue.take_ready().is_empty());
        assert!(!queue.has_sources());
    }
}

#[test]
fn queue_is_bounded_and_ready_sources_deliver_once() {
    let queue = PendingFeedback::default();
    let ready = Arc::new(AtomicBool::new(false));
    let dropped = Arc::new(AtomicUsize::new(0));
    for _ in 0..MAX_PENDING {
        assert!(queue.push(source(ready.clone(), dropped.clone())).is_ok());
    }
    assert!(queue.push(source(ready.clone(), dropped.clone())).is_err());
    assert_eq!(queue.discarded.get(), 1);
    assert!(queue.take_ready().is_empty());
    ready.store(true, Ordering::Release);
    assert_eq!(queue.take_ready().len(), MAX_PENDING);
    assert!(queue.take_ready().is_empty());
    assert_eq!(dropped.load(Ordering::Acquire), MAX_PENDING + 1);
}

#[test]
fn invalidated_receipts_cannot_restore_old_profiles() {
    for invalidation in [
        PlacementInvalidation::All,
        PlacementInvalidation::Provider {
            digest: Digest::sha256(b"pending-test"),
        },
        PlacementInvalidation::Policy {
            digest: Digest::sha256(b"pending-test"),
        },
    ] {
        let queue = PendingFeedback::default();
        let ready = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicUsize::new(0));
        assert!(queue.push(source(ready.clone(), dropped.clone())).is_ok());
        queue.invalidate(&invalidation);
        ready.store(true, Ordering::Release);
        assert!(queue.take_ready().is_empty());
        assert_eq!(dropped.load(Ordering::Acquire), 1);
    }
}

#[test]
fn same_digest_invalidation_retires_only_matching_authority_receipts() {
    use runmat_execution::PlacementExecutionAuthorityRevision;

    let queue = PendingFeedback::default();
    let ready = Arc::new(AtomicBool::new(false));
    let dropped = Arc::new(AtomicUsize::new(0));
    let provider = source(ready.clone(), dropped.clone());
    let mut host = source(ready.clone(), dropped.clone());
    let digest = match provider.signature.revision.authority {
        PlacementExecutionAuthorityRevision::Provider(digest) => digest,
        PlacementExecutionAuthorityRevision::HostService(_)
        | PlacementExecutionAuthorityRevision::RuntimeBindingSession(_) => {
            unreachable!("provider fixture")
        }
    };
    host.signature.revision.authority = PlacementExecutionAuthorityRevision::HostService(digest);
    assert!(queue.push(provider).is_ok());
    assert!(queue.push(host).is_ok());
    queue.invalidate(&PlacementInvalidation::Provider { digest });
    assert_eq!(dropped.load(Ordering::Acquire), 1);
    assert_eq!(queue.sources.borrow().len(), 1);
    ready.store(true, Ordering::Release);
    assert_eq!(queue.take_ready().len(), 1);
    assert_eq!(dropped.load(Ordering::Acquire), 2);

    let host = source(Arc::new(AtomicBool::new(false)), dropped.clone());
    let mut host = host;
    host.signature.revision.authority = PlacementExecutionAuthorityRevision::HostService(digest);
    assert!(queue.push(host).is_ok());
    queue.invalidate(&PlacementInvalidation::HostService { digest });
    assert!(queue.sources.borrow().is_empty());
    assert_eq!(dropped.load(Ordering::Acquire), 3);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn provider_invalidation_retires_only_matching_session_bound_pending_source() {
    use runmat_execution::PlacementExecutionAuthorityRevision;

    let queue = PendingFeedback::default();
    let ready = Arc::new(AtomicBool::new(false));
    let dropped = Arc::new(AtomicUsize::new(0));
    let provider_a = Digest::sha256(b"provider A");
    let provider_b = Digest::sha256(b"provider B");
    let session = Digest::sha256(b"same VM session");
    let mut first = source(ready.clone(), dropped.clone());
    first.signature.revision.authority =
        PlacementExecutionAuthorityRevision::RuntimeBindingSession(session);
    first.signature.provider_provenance = Some(provider_a);
    let mut second = source(ready.clone(), dropped.clone());
    second.signature.revision.authority =
        PlacementExecutionAuthorityRevision::RuntimeBindingSession(session);
    second.signature.provider_provenance = Some(provider_b);
    assert!(queue.push(first).is_ok());
    assert!(queue.push(second).is_ok());

    queue.invalidate(&PlacementInvalidation::Provider { digest: provider_a });
    assert_eq!(dropped.load(Ordering::Acquire), 1);
    assert_eq!(queue.sources.borrow().len(), 1);
    ready.store(true, Ordering::Release);
    assert_eq!(queue.take_ready().len(), 1);
    assert_eq!(dropped.load(Ordering::Acquire), 2);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn runtime_binding_session_invalidation_retires_only_its_pending_receipt() {
    use runmat_execution::PlacementExecutionAuthorityRevision;

    let queue = PendingFeedback::default();
    let ready = Arc::new(AtomicBool::new(false));
    let dropped = Arc::new(AtomicUsize::new(0));
    let digest = Digest::sha256(b"same-authority-digest");
    let mut local = source(ready.clone(), dropped.clone());
    local.signature.revision.authority =
        PlacementExecutionAuthorityRevision::RuntimeBindingSession(digest);
    let mut provider = source(ready.clone(), dropped.clone());
    provider.signature.revision.authority = PlacementExecutionAuthorityRevision::Provider(digest);
    assert!(queue.push(local).is_ok());
    assert!(queue.push(provider).is_ok());
    queue.invalidate(&PlacementInvalidation::RuntimeBindingSession { digest });
    assert_eq!(dropped.load(Ordering::Acquire), 1);
    assert_eq!(queue.sources.borrow().len(), 1);
    ready.store(true, Ordering::Release);
    assert_eq!(queue.take_ready().len(), 1);
    assert_eq!(dropped.load(Ordering::Acquire), 2);
}
