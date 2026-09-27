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
