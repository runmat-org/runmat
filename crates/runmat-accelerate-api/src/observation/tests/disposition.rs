use super::*;

#[test]
fn transfers_and_resident_publication_do_not_prove_selected_execution() {
    let mut receipt = ProviderOperationReceipt::new();
    receipt.require_selected_execution();
    receipt.recorder().submission().unwrap().complete();
    assert_eq!(
        receipt.publish(),
        Err(ProviderObservationFailure::ExecutionUnproven)
    );
    assert_eq!(
        receipt.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::ExecutionUnproven)
    );
}

#[test]
fn selected_route_witness_still_waits_for_actual_completion() {
    let mut receipt = ProviderOperationReceipt::new();
    receipt.require_selected_execution();
    let recorder = receipt.recorder();
    let token = recorder.submission().unwrap();
    recorder.execution_disposition(ProviderExecutionDisposition::SelectedProvider);
    receipt.publish().unwrap();
    assert_eq!(receipt.status(), ProviderObservationStatus::Pending);
    token.complete();
    assert!(matches!(
        receipt.status(),
        ProviderObservationStatus::Completed { submissions: 1, .. }
    ));
}

#[test]
fn selected_image_upload_is_device_bound_but_not_a_stage_witness() {
    let mut receipt = ProviderOperationReceipt::new();
    receipt.require_selected_execution();
    receipt.require_selected_image_upload(7);
    let recorder = receipt.recorder();
    assert!(recorder.observes_selected_image_upload(7));
    assert!(!recorder.observes_selected_image_upload(8));
    recorder.submission().unwrap().complete();
    // The upload fence cannot turn the selected stage into an observed run.
    assert_eq!(
        receipt.publish(),
        Err(ProviderObservationFailure::ExecutionUnproven)
    );
}

#[test]
fn fallback_cannot_be_repaired_by_later_provider_restore_or_late_route_witness() {
    for after_publication in [false, true] {
        let mut receipt = ProviderOperationReceipt::new();
        receipt.require_selected_execution();
        let recorder = receipt.recorder();
        let token = recorder.submission().unwrap();
        recorder.execution_disposition(ProviderExecutionDisposition::SelectedProvider);
        if after_publication {
            receipt.publish().unwrap();
        }
        recorder.execution_disposition(ProviderExecutionDisposition::SharedRuntimeFallback);
        recorder.execution_disposition(ProviderExecutionDisposition::SelectedProvider);
        token.complete();
        assert_eq!(
            receipt.status(),
            ProviderObservationStatus::Failed(ProviderObservationFailure::SharedRuntimeFallback)
        );
    }
}

#[test]
fn late_route_requirement_fails_instead_of_relabelling_started_work() {
    let mut receipt = ProviderOperationReceipt::new();
    receipt.recorder().submission().unwrap().complete();
    receipt.require_selected_execution();
    assert_eq!(
        receipt.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::ExecutionUnproven)
    );
}

#[test]
fn exact_device_and_operation_are_required_and_mismatch_is_terminal() {
    let operation = crate::ProviderOperationIdentity::new("fixture.cast.single");
    for invalid in 0..5 {
        let mut receipt = ProviderOperationReceipt::new();
        receipt.require_provider_execution(7, operation.clone());
        let recorder = receipt.recorder();
        match invalid {
            0 => recorder.confirm_provider_execution(8, &operation),
            1 => recorder.confirm_provider_execution(
                7,
                &crate::ProviderOperationIdentity::new("fixture.cast.double"),
            ),
            2 => recorder.execution_disposition(ProviderExecutionDisposition::SelectedProvider),
            3 => {
                recorder.execution_disposition(ProviderExecutionDisposition::SharedRuntimeFallback)
            }
            _ => {
                assert!(receipt.publish().is_err());
            }
        }
        // Neither a later correct device nor a same-owner facade can repair an
        // already invalidated receipt or a missing publication witness.
        recorder.confirm_provider_execution(7, &operation);
        assert!(receipt.publish().is_err());
        assert!(matches!(
            receipt.status(),
            ProviderObservationStatus::Failed(_)
        ));
    }
}

#[test]
fn exact_identity_accepts_same_owner_facade_without_pointer_identity() {
    let operation = crate::ProviderOperationIdentity::new("fixture.cast.single");
    let mut receipt = ProviderOperationReceipt::new();
    receipt.require_provider_execution(7, operation.clone());
    let recorder = receipt.recorder();
    // A cloned provider facade carries the same existing device registration;
    // the witness is independent of Rust object/vtable addresses.
    recorder
        .clone()
        .confirm_provider_execution(7, &operation.clone());
    receipt.publish().unwrap();
    assert!(matches!(
        receipt.status(),
        ProviderObservationStatus::Completed { submissions: 0, .. }
    ));
}

#[test]
fn submitted_operation_witness_requires_a_real_registered_submission() {
    let sum = crate::ProviderOperationIdentity::new("fixture.sum");
    let other = crate::ProviderOperationIdentity::new("fixture.other");
    let mut unsubmitted = ProviderOperationReceipt::new();
    unsubmitted.require_provider_execution(7, sum.clone());
    let recorder = unsubmitted.recorder();
    assert!(recorder.expects_operation(&sum));
    assert!(!recorder.expects_operation(&other));
    recorder.confirm_submitted_provider_execution_after(7, &sum, recorder.submission_checkpoint());
    assert!(unsubmitted.publish().is_err());
    assert!(matches!(
        unsubmitted.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::ExecutionUnproven)
    ));

    let mut submitted = ProviderOperationReceipt::new();
    submitted.require_provider_execution(7, sum.clone());
    let recorder = submitted.recorder();
    let checkpoint = recorder.submission_checkpoint();
    recorder.submission().unwrap().complete();
    recorder.confirm_submitted_provider_execution_after(7, &sum, checkpoint);
    submitted.publish().unwrap();
    assert!(matches!(
        submitted.status(),
        ProviderObservationStatus::Completed { submissions: 1, .. }
    ));
}

#[test]
fn prior_upload_submission_cannot_certify_later_operation() {
    let sum = crate::ProviderOperationIdentity::new("fixture.sum");
    let mut receipt = ProviderOperationReceipt::new();
    receipt.require_provider_execution(7, sum.clone());
    let recorder = receipt.recorder();
    recorder.submission().unwrap().complete();
    let after_upload = recorder.submission_checkpoint();
    assert_eq!(after_upload, 1);
    recorder.confirm_submitted_provider_execution_after(7, &sum, after_upload);
    assert_eq!(
        receipt.publish(),
        Err(ProviderObservationFailure::ExecutionUnproven)
    );
}

#[test]
fn selected_upload_is_pending_until_its_post_publication_fence_completes() {
    let upload = crate::ProviderOperationIdentity::new("transfer.upload");
    let mut receipt = ProviderOperationReceipt::new();
    receipt.require_provider_execution(7, upload.clone());
    let recorder = receipt.recorder();
    let checkpoint = recorder.submission_checkpoint();
    let fence = recorder.submission().unwrap();
    recorder.confirm_submitted_provider_execution_after(7, &upload, checkpoint);
    receipt.publish().unwrap();
    assert_eq!(receipt.status(), ProviderObservationStatus::Pending);
    fence.complete();
    assert!(matches!(
        receipt.status(),
        ProviderObservationStatus::Completed { submissions: 1, .. }
    ));

    let mut cancelled = ProviderOperationReceipt::new();
    cancelled.require_provider_execution(7, upload.clone());
    let recorder = cancelled.recorder();
    let checkpoint = recorder.submission_checkpoint();
    let fence = recorder.submission().unwrap();
    recorder.confirm_submitted_provider_execution_after(7, &upload, checkpoint);
    cancelled.publish().unwrap();
    cancelled.fail(ProviderObservationFailure::Cancelled);
    fence.complete();
    assert_eq!(
        cancelled.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::Cancelled)
    );
}

#[test]
fn conditional_host_fallback_is_one_completed_route_not_provider_execution() {
    let operation = crate::ProviderOperationIdentity::new("fspecial");
    let mut receipt = ProviderOperationReceipt::new();
    receipt.require_provider_execution(7, operation.clone());
    receipt.allow_conditional_host_fallback();
    let fence = receipt.recorder().submission().unwrap();
    receipt.confirm_conditional_host_fallback();
    receipt.publish().unwrap();
    assert_eq!(receipt.status(), ProviderObservationStatus::Pending);
    fence.complete();
    let observation = receipt.take_completed().unwrap();
    assert!(matches!(
        observation.status,
        ProviderObservationStatus::Completed { submissions: 1, .. }
    ));
    assert_eq!(
        observation.route_outcome,
        Some(ProviderRouteOutcome::ConditionalHostFallback)
    );
    assert!(receipt.take_completed().is_none());

    let mut forbidden = ProviderOperationReceipt::new();
    forbidden.require_provider_execution(7, operation);
    forbidden.confirm_conditional_host_fallback();
    assert_eq!(
        forbidden.publish(),
        Err(ProviderObservationFailure::ExecutionUnproven)
    );
}

#[test]
fn conditional_provider_upload_needs_its_own_admitted_outcome_and_completion() {
    let operation = crate::ProviderOperationIdentity::new("imfilter");
    let mut receipt = ProviderOperationReceipt::new();
    receipt.require_provider_execution(7, operation.clone());
    receipt.allow_conditional_provider_upload();
    assert!(!receipt.recorder().observes_conditional_provider_upload(7));
    receipt.begin_conditional_provider_upload();
    assert!(receipt.recorder().observes_conditional_provider_upload(7));
    assert!(!receipt.recorder().observes_conditional_provider_upload(8));
    let fence = receipt.recorder().submission().unwrap();
    receipt.confirm_conditional_provider_upload();
    receipt.publish().unwrap();
    assert_eq!(receipt.status(), ProviderObservationStatus::Pending);
    fence.complete();
    let observation = receipt.take_completed().unwrap();
    assert!(matches!(
        observation.status,
        ProviderObservationStatus::Completed { submissions: 1, .. }
    ));
    assert_eq!(
        observation.route_outcome,
        Some(ProviderRouteOutcome::ConditionalProviderUpload)
    );
    assert!(receipt.take_completed().is_none());

    let mut forbidden = ProviderOperationReceipt::new();
    forbidden.require_provider_execution(7, operation.clone());
    forbidden.allow_conditional_provider_upload();
    forbidden.confirm_conditional_provider_upload();
    assert_eq!(
        forbidden.publish(),
        Err(ProviderObservationFailure::ExecutionUnproven)
    );

    let mut selected = ProviderOperationReceipt::new();
    selected.require_provider_execution(7, operation);
    selected.allow_conditional_provider_upload();
    selected.begin_conditional_provider_upload();
    selected
        .recorder()
        .confirm_provider_execution(7, &crate::ProviderOperationIdentity::new("imfilter"));
    selected.confirm_conditional_provider_upload();
    assert_eq!(
        selected.publish(),
        Err(ProviderObservationFailure::ExecutionUnproven)
    );
}

#[test]
fn preparation_unsupported_is_not_relabelled_as_hook_execution() {
    let operation = crate::ProviderOperationIdentity::new("imfilter");
    let mut host = ProviderOperationReceipt::new();
    host.require_provider_execution(7, operation.clone());
    host.allow_conditional_host_fallback();
    host.confirm_preparation_host_fallback();
    host.publish().unwrap();
    assert_eq!(
        host.take_completed().unwrap().route_outcome,
        Some(ProviderRouteOutcome::PreparationHostFallback)
    );

    let mut restored = ProviderOperationReceipt::new();
    restored.require_provider_execution(7, operation.clone());
    restored.allow_conditional_provider_upload();
    restored.begin_preparation_provider_upload();
    assert!(restored.recorder().observes_conditional_provider_upload(7));
    let fence = restored.recorder().submission().unwrap();
    restored.confirm_preparation_provider_upload();
    restored.publish().unwrap();
    assert_eq!(restored.status(), ProviderObservationStatus::Pending);
    fence.complete();
    assert_eq!(
        restored.take_completed().unwrap().route_outcome,
        Some(ProviderRouteOutcome::PreparationProviderUpload)
    );

    let mut wrong = ProviderOperationReceipt::new();
    wrong.require_provider_execution(7, operation);
    wrong.allow_conditional_provider_upload();
    wrong.begin_preparation_provider_upload();
    wrong.confirm_conditional_provider_upload();
    assert_eq!(
        wrong.publish(),
        Err(ProviderObservationFailure::ExecutionUnproven)
    );
}

#[test]
fn provider_success_cannot_be_relabelled_conditional_fallback() {
    let operation = crate::ProviderOperationIdentity::new("fspecial");
    let mut receipt = ProviderOperationReceipt::new();
    receipt.require_provider_execution(7, operation.clone());
    receipt.allow_conditional_host_fallback();
    receipt.recorder().confirm_provider_execution(7, &operation);
    receipt.confirm_conditional_host_fallback();
    assert_eq!(
        receipt.publish(),
        Err(ProviderObservationFailure::ExecutionUnproven)
    );
}
