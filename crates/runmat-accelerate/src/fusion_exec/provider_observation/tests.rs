use super::*;
use crate::placement::PlacementSession;
use runmat_accelerate_api::observation::ProviderObservationClock;
use runmat_execution::{Digest, PlacementRevision, PlacementSignature};
use runmat_runtime::context::{RuntimeContext, RuntimeServicePorts};
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::Duration;

struct Clock(AtomicU64);
impl ProviderObservationClock for Clock {
    fn now(&self) -> Option<Duration> {
        Some(Duration::from_nanos(self.0.load(Ordering::SeqCst)))
    }
}
fn signature() -> PlacementSignature {
    let digest = Digest::sha256(b"observed-fusion");
    PlacementSignature {
        region: None,
        operation: "fusion.elementwise".into(),
        runtime_facts: digest,
        revision: PlacementRevision {
            program: None,
            catalog: digest,
            compiler: digest,
            provider: digest,
            policy: digest,
        },
    }
}
fn runtime(session: Rc<PlacementSession>) -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(session))
}
fn selected(
    runtime: RuntimeContext,
    receipt: ProviderOperationReceipt,
) -> FusionPlacementSelection {
    FusionPlacementSelection {
        observer: FusionPlacementObserver::new(None),
        started: Instant::now(),
        feedback: Some(LocalPlacementFeedback {
            runtime,
            signature: signature(),
            candidate: "provider.fusion".into(),
        }),
        receipt: Some(receipt),
    }
}

#[test]
fn selected_fusion_defers_until_completion_and_keeps_full_staging_total() {
    let session = Rc::new(PlacementSession::default());
    let runtime = runtime(session.clone());
    let clock = Arc::new(Clock(AtomicU64::new(100)));
    let receipt = ProviderOperationReceipt::with_clock(clock.clone());
    let token = receipt.recorder().submission().unwrap();
    let selection = selected(runtime.clone(), receipt);
    clock.0.store(125, Ordering::SeqCst); // upload/preparation included, no timer reset
    selection.submitted();
    assert!(session.profile_snapshot().feedback.is_empty());
    clock.0.store(150, Ordering::SeqCst);
    std::thread::spawn(move || token.complete()).join().unwrap();
    clock.0.store(10000, Ordering::SeqCst);
    // Actual RuntimeContext final-evaluation drain, no subsequent placement plan.
    futures::executor::block_on(runtime.scope(async {}));
    let profile = session.profile_snapshot();
    assert_eq!(profile.feedback.len(), 1);
    assert_eq!(profile.feedback[0].samples, 1);
    assert_eq!(profile.feedback[0].mean_ns(), Some(50));
    assert_eq!(session.profile_snapshot().feedback[0].samples, 1);
}

#[test]
fn cancellation_after_publication_before_completion_is_not_positive_feedback() {
    let session = Rc::new(PlacementSession::default());
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = runtime(session.clone()).with_cancellation_token(cancellation.token());
    let receipt = ProviderOperationReceipt::new();
    let token = receipt.recorder().submission().unwrap();
    selected(runtime, receipt).submitted();
    cancellation.cancel(runmat_execution::CancellationReason::User);
    token.complete();
    let profile = session.profile_snapshot();
    assert_eq!(profile.feedback[0].samples, 0);
    assert_eq!(profile.feedback[0].failures, 1);
}

#[test]
fn failed_or_abandoned_selected_candidate_records_failure_not_partial_total() {
    let session = Rc::new(PlacementSession::default());
    let receipt = ProviderOperationReceipt::new();
    let token = receipt.recorder().submission().unwrap();
    drop(selected(runtime(session.clone()), receipt));
    token.complete();
    let profile = session.profile_snapshot();
    assert_eq!(profile.feedback[0].samples, 0);
    assert_eq!(profile.feedback[0].failures, 1);
}

#[test]
fn pending_feedback_does_not_retain_runtime_session_and_late_callback_is_inert() {
    let session = Rc::new(PlacementSession::default());
    let weak = Rc::downgrade(&session);
    let receipt = ProviderOperationReceipt::new();
    let recorder = receipt.recorder();
    let token = recorder.submission().unwrap();
    selected(runtime(session.clone()), receipt).submitted();
    drop(session);
    assert!(weak.upgrade().is_none());
    token.complete();
    assert!(matches!(
        recorder.submission(),
        Err(ProviderObservationFailure::Abandoned)
    ));
}

#[test]
fn successful_profile_replacement_invalidates_pending_observations() {
    let session = Rc::new(PlacementSession::default());
    let receipt = ProviderOperationReceipt::new();
    let token = receipt.recorder().submission().unwrap();
    selected(runtime(session.clone()), receipt).submitted();
    session
        .restore_profile(crate::placement::PlacementProfileSnapshot {
            schema_version: crate::placement::PLACEMENT_PROFILE_SCHEMA_VERSION,
            feedback: Vec::new(),
        })
        .unwrap();
    token.complete();
    assert!(session.profile_snapshot().feedback.is_empty());
}

#[test]
fn final_scope_drop_drains_completion_after_inner_destructor() {
    struct CompleteOnDrop(Option<runmat_accelerate_api::observation::ProviderSubmission>);
    impl Drop for CompleteOnDrop {
        fn drop(&mut self) {
            self.0.take().unwrap().complete();
        }
    }
    let session = Rc::new(PlacementSession::default());
    let context = runtime(session.clone());
    let receipt = ProviderOperationReceipt::new();
    let token = CompleteOnDrop(Some(receipt.recorder().submission().unwrap()));
    selected(context.clone(), receipt).submitted();
    let future = context.scope(async move {
        let _keep = token;
        futures::future::pending::<()>().await;
    });
    drop(future);
    assert_eq!(session.profile_snapshot().feedback[0].samples, 1);
}
