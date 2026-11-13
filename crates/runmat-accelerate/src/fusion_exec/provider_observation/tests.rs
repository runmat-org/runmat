use super::*;
use crate::placement::{LocalPlacementFeedbackAuthority, PlacementSession};
use runmat_accelerate_api::observation::{ProviderObservationClock, ProviderObservationFailure};
use runmat_execution::profiling::{CommandProfilingRoot, ProviderObservationCounts};
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
        provider_provenance: None,
        graph_output_authority: None,
        revision: PlacementRevision {
            program: None,
            catalog: digest,
            compiler: digest,
            authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(digest),
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

fn vm_semantic(label: &[u8]) -> runmat_runtime::context::VmFusionSemanticIdentity {
    runmat_runtime::context::VmFusionSemanticIdentity::from_validated_digest(Digest::sha256(label))
}
fn graph_feedback(
    runtime: RuntimeContext,
    session: &PlacementSession,
    signature: PlacementSignature,
    candidate: &str,
) -> LocalPlacementFeedback {
    let kind = runmat_execution::ExecutionCandidateKind::ProviderFusion;
    let permit = session.test_graph_observation_permit(signature.clone(), candidate, kind);
    LocalPlacementFeedback {
        runtime,
        signature,
        candidate: candidate.into(),
        kind,
        authority: LocalPlacementFeedbackAuthority::Graph(permit),
    }
}
fn selected(
    runtime: RuntimeContext,
    session: &PlacementSession,
    receipt: ProviderOperationReceipt,
) -> FusionPlacementSelection {
    let command_contribution = runtime
        .command_profiling()
        .and_then(|command| command.begin_provider_operation());
    let signature = signature();
    let candidate = "provider.fusion";
    let kind = runmat_execution::ExecutionCandidateKind::ProviderFusion;
    let permit = session.test_graph_observation_permit(signature.clone(), candidate, kind);
    FusionPlacementSelection {
        observer: FusionPlacementObserver::new(None),
        started: Instant::now(),
        drop_reports_failure: true,
        feedback: Some(LocalPlacementFeedback {
            runtime,
            signature,
            candidate: candidate.into(),
            kind,
            authority: LocalPlacementFeedbackAuthority::Graph(permit),
        }),
        receipt: Some(receipt),
        command_contribution,
    }
}

#[test]
fn selected_fusion_defers_until_completion_and_keeps_full_staging_total() {
    let session = Rc::new(PlacementSession::default());
    let command = CommandProfilingRoot::new();
    let runtime = runtime(session.clone()).with_command_profiling(command.receipt());
    let clock = Arc::new(Clock(AtomicU64::new(100)));
    let receipt = ProviderOperationReceipt::with_clock(clock.clone());
    let token = receipt.recorder().submission().unwrap();
    let selection = selected(runtime.clone(), &session, receipt);
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
    let snapshot = command.seal();
    assert_eq!(
        snapshot.observations,
        Some(ProviderObservationCounts {
            incomplete: 1,
            ..ProviderObservationCounts::default()
        })
    );
    assert_eq!(snapshot.gpu_execution, None);
}

#[test]
fn cancellation_after_publication_before_completion_has_no_feedback() {
    let session = Rc::new(PlacementSession::default());
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = runtime(session.clone()).with_cancellation_token(cancellation.token());
    let receipt = ProviderOperationReceipt::new();
    let recorder = receipt.recorder();
    let token = recorder.submission().unwrap();
    selected(runtime.clone(), &session, receipt).submitted();
    assert!(session.profile_snapshot().feedback.is_empty());
    cancellation.cancel(runmat_execution::CancellationReason::User);
    token.complete();
    // The final scope drain retires cancellation without a positive or negative sample.
    futures::executor::block_on(runtime.scope(async {}));
    assert!(session.profile_snapshot().feedback.is_empty());
    assert!(matches!(
        recorder.submission(),
        Err(ProviderObservationFailure::Cancelled)
    ));
}

#[test]
fn failed_or_abandoned_selected_candidate_records_failure_not_partial_total() {
    let session = Rc::new(PlacementSession::default());
    let receipt = ProviderOperationReceipt::new();
    let token = receipt.recorder().submission().unwrap();
    drop(selected(runtime(session.clone()), &session, receipt));
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
    selected(runtime(session.clone()), &session, receipt).submitted();
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
    selected(runtime(session.clone()), &session, receipt).submitted();
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
    selected(context.clone(), &session, receipt).submitted();
    let future = context.scope(async move {
        let _keep = token;
        futures::future::pending::<()>().await;
    });
    drop(future);
    assert_eq!(session.profile_snapshot().feedback[0].samples, 1);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn vm_full_call_custom_clock_receipt_cannot_publish_a_positive_sample() {
    let session = Rc::new(PlacementSession::default());
    let runtime = runtime(session.clone());
    let installed = runmat_runtime::context::VmFusionInstalledRegion::install(
        &runtime,
        vm_semantic(b"custom clock"),
        7,
        0,
        1,
    )
    .expect("same-context VM installation");
    let right = installed
        .right(&runtime)
        .expect("same-context completion right");
    let mut feedback_signature = signature();
    feedback_signature.revision = right
        .placement_revision()
        .cloned()
        .expect("real VM session revision");
    let feedback = graph_feedback(
        runtime.clone(),
        &session,
        feedback_signature,
        "provider.fusion.custom-clock",
    );
    let clock = Arc::new(Clock(AtomicU64::new(10)));
    let receipt = ProviderOperationReceipt::with_clock(clock.clone());
    let submission = receipt.recorder().submission().unwrap();
    clock.0.store(20, Ordering::SeqCst);
    submission.complete();
    let selected = FusionPlacementSelection {
        observer: FusionPlacementObserver::new(None),
        feedback: Some(feedback),
        receipt: Some(receipt),
        command_contribution: None,
        started: Instant::now(),
        drop_reports_failure: false,
    };
    let completion = selected
        .into_vm_completion(right)
        .expect("matching VM right retains the completed receipt");
    assert!(session.profile_snapshot().feedback.is_empty());
    completion.complete(Instant::now());
    futures::executor::block_on(runtime.scope(async {}));
    assert!(session.profile_snapshot().feedback.is_empty());
}

#[test]
fn published_pending_receipt_survives_owner_drop_and_concurrent_same_code_installation() {
    let session = Rc::new(PlacementSession::default());
    let runtime = runtime(session.clone());
    let semantic = vm_semantic(b"concurrent pending receipt");
    let installed =
        runmat_runtime::context::VmFusionInstalledRegion::install(&runtime, semantic, 3, 4, 6)
            .unwrap();
    let right = installed.right(&runtime).unwrap();
    let stale_sibling = installed.right(&runtime).unwrap();
    let mut feedback_signature = signature();
    feedback_signature.revision = right.placement_revision().unwrap().clone();
    let feedback = graph_feedback(
        runtime.clone(),
        &session,
        feedback_signature,
        "provider.fusion.concurrent",
    );
    let selected_signature = feedback.signature.clone();
    let outer_started = Instant::now();
    let receipt = ProviderOperationReceipt::new();
    let submission = receipt.recorder().submission().unwrap();
    let selected = FusionPlacementSelection {
        observer: FusionPlacementObserver::new(None),
        feedback: Some(feedback),
        receipt: Some(receipt),
        command_contribution: None,
        started: Instant::now(),
        drop_reports_failure: false,
    };
    selected
        .into_vm_completion(right)
        .unwrap()
        .complete(outer_started);
    drop(installed);
    assert!(stale_sibling.planning_revision(&runtime).is_none());

    let concurrent =
        runmat_runtime::context::VmFusionInstalledRegion::install(&runtime, semantic, 3, 4, 6)
            .unwrap();
    let concurrent_right = concurrent.right(&runtime).unwrap();
    assert_eq!(
        concurrent_right.placement_revision(),
        Some(&selected_signature.revision)
    );
    let concurrent_outer_started = Instant::now();
    let concurrent_receipt = ProviderOperationReceipt::new();
    let concurrent_submission = concurrent_receipt.recorder().submission().unwrap();
    let concurrent_feedback = LocalPlacementFeedback {
        runtime: runtime.clone(),
        signature: selected_signature.clone(),
        candidate: "provider.fusion.concurrent".into(),
        kind: runmat_execution::ExecutionCandidateKind::ProviderFusion,
        authority: LocalPlacementFeedbackAuthority::Graph(session.test_graph_observation_permit(
            selected_signature.clone(),
            "provider.fusion.concurrent",
            runmat_execution::ExecutionCandidateKind::ProviderFusion,
        )),
    };
    FusionPlacementSelection {
        observer: FusionPlacementObserver::new(None),
        feedback: Some(concurrent_feedback),
        receipt: Some(concurrent_receipt),
        command_contribution: None,
        started: Instant::now(),
        drop_reports_failure: false,
    }
    .into_vm_completion(concurrent_right)
    .unwrap()
    .complete(concurrent_outer_started);
    concurrent_submission.complete();
    submission.complete();
    futures::executor::block_on(runtime.scope(async {}));
    assert!(matches!(
        session.observed_candidate_duration(
            &selected_signature,
            "provider.fusion.concurrent",
            runmat_execution::ExecutionCandidateKind::ProviderFusion,
        ),
        runmat_execution::CandidateObservationEvidence::Observed(_)
    ));
}

#[test]
fn profile_replacement_retires_vm_pending_receipt_but_fresh_right_is_admitted() {
    let session = Rc::new(PlacementSession::default());
    let runtime = runtime(session.clone());
    let semantic = vm_semantic(b"pending profile replacement");
    let installed =
        runmat_runtime::context::VmFusionInstalledRegion::install(&runtime, semantic, 5, 8, 9)
            .unwrap();
    let right = installed.right(&runtime).unwrap();
    let revision = right.placement_revision().unwrap().clone();
    let outer_started = Instant::now();
    let receipt = ProviderOperationReceipt::new();
    let submission = receipt.recorder().submission().unwrap();
    let mut feedback_signature = signature();
    feedback_signature.revision = revision.clone();
    let feedback = graph_feedback(
        runtime.clone(),
        &session,
        feedback_signature,
        "provider.fusion.reset",
    );
    let selected = FusionPlacementSelection {
        observer: FusionPlacementObserver::new(None),
        feedback: Some(feedback),
        receipt: Some(receipt),
        command_contribution: None,
        started: Instant::now(),
        drop_reports_failure: false,
    };
    selected
        .into_vm_completion(right)
        .unwrap()
        .complete(outer_started);
    session
        .restore_profile(crate::placement::PlacementProfileSnapshot {
            schema_version: crate::placement::PLACEMENT_PROFILE_SCHEMA_VERSION,
            feedback: Vec::new(),
        })
        .unwrap();
    submission.complete();
    futures::executor::block_on(runtime.scope(async {}));
    assert_eq!(
        session.observed_candidate_duration(
            &PlacementSignature {
                revision: revision.clone(),
                ..signature()
            },
            "provider.fusion.reset",
            runmat_execution::ExecutionCandidateKind::ProviderFusion,
        ),
        runmat_execution::CandidateObservationEvidence::Unavailable(
            runmat_execution::CandidateCostUnavailable::NoObservations
        )
    );
    assert_eq!(
        installed.right(&runtime).unwrap().placement_revision(),
        Some(&revision)
    );
}
