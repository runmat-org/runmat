use super::*;
use runmat_execution::{Digest, PlacementRevision, PlacementSignature};
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
    fn take_ready(&mut self) -> Option<PlacementFeedback> {
        self.ready
            .load(Ordering::Acquire)
            .then(|| PlacementFeedback {
                signature: self.signature.clone(),
                candidate: "provider.test".into(),
                kind: runmat_execution::ExecutionCandidateKind::ProviderOperation,
                total_elapsed_ns: 19,
                succeeded: true,
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
    fn take_ready(&mut self) -> Option<PlacementFeedback> {
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
