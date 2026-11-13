use super::*;

fn scope(receipt: &ProviderOperationReceipt) -> ProviderResourceScope {
    let scope = receipt
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([1000; 4]).unwrap())
        .unwrap();
    for index in 0..4 {
        assert_eq!(scope.begin_next_phase().unwrap().index(), index);
    }
    scope
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn issued_owner_tickets_hold_completion_until_final_retirement() {
    let mut receipt = ProviderOperationReceipt::new();
    assert_eq!(receipt.resource_snapshot(), None);
    let scope = scope(&receipt);
    let first = scope.issue_resource_ticket(320).unwrap();
    let alias = first.clone();
    let second = scope.issue_resource_ticket(8).unwrap();
    assert_eq!(first.requested_bytes(), 320);
    assert_eq!(
        receipt.resource_snapshot().unwrap().requested_peak_bytes,
        328
    );
    scope.finish().unwrap();
    receipt.publish().unwrap();
    drop(first);
    assert_eq!(receipt.status(), ProviderObservationStatus::Pending);
    drop(second);
    assert_eq!(
        receipt.resource_snapshot().unwrap().requested_live_bytes,
        320
    );
    drop(alias);
    let evidence = receipt.take_completed().unwrap();
    assert!(matches!(
        evidence.status,
        ProviderObservationStatus::Completed { .. }
    ));
    assert_eq!(
        evidence.resources,
        Some(ProviderResourceSnapshot {
            requested_live_bytes: 0,
            requested_peak_bytes: 328,
            issued: 2,
            retired: 2,
            pooled_handoffs: 0,
            scopes: vec![ProviderScopedResourceSnapshot {
                scope_id: 1,
                requested_live_bytes: 0,
                requested_peak_bytes: 328,
                issued: 2,
                retired: 2,
                pooled_handoffs: 0,
                phase_peaks: vec![Some(0), Some(0), Some(0), Some(328)],
                phase_limits: vec![1000; 4],
                finished: true,
            }],
        })
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancelled_receipt_cannot_be_resurrected_by_late_resource_retirement() {
    let mut receipt = ProviderOperationReceipt::new();
    let scope = scope(&receipt);
    let ticket = scope.issue_resource_ticket(8).unwrap();
    scope.finish().unwrap();
    receipt.publish().unwrap();
    receipt.fail(ProviderObservationFailure::Cancelled);
    drop(ticket);
    assert_eq!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::Cancelled)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pooled_handoff_is_not_misreported_as_buffer_destruction() {
    let mut receipt = ProviderOperationReceipt::new();
    let scope = scope(&receipt);
    let ticket = scope.issue_resource_ticket(320).unwrap();
    ticket.mark_pooled_handoff();
    scope.finish().unwrap();
    receipt.publish().unwrap();
    drop(ticket);
    let resources = receipt.take_completed().unwrap().resources.unwrap();
    assert_eq!(resources.pooled_handoffs, 1);
    assert_eq!(resources.requested_live_bytes, 0);
    assert_eq!(resources.requested_peak_bytes, 320);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn phase_peaks_follow_actual_ticket_issue_and_retirement_order() {
    let mut receipt = ProviderOperationReceipt::new();
    let scope = receipt
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([336, 328, 336, 336]).unwrap())
        .unwrap();
    scope.begin_next_phase().unwrap();
    let flag = scope.issue_resource_ticket(320).unwrap();
    let uniform = scope.issue_resource_ticket(16).unwrap();
    drop(uniform);
    scope.begin_next_phase().unwrap();
    let first = scope.issue_resource_ticket(8).unwrap();
    scope.begin_next_phase().unwrap();
    let second = scope.issue_resource_ticket(8).unwrap();
    drop(first);
    scope.begin_next_phase().unwrap();
    let staging = scope.issue_resource_ticket(8).unwrap();
    scope.finish().unwrap();
    receipt.publish().unwrap();
    assert_eq!(receipt.status(), ProviderObservationStatus::Pending);
    drop(staging);
    drop(second);
    drop(flag);
    let evidence = receipt.take_completed().unwrap();
    assert_eq!(
        evidence.resources.unwrap().scopes[0].phase_peaks,
        vec![Some(336), Some(328), Some(336), Some(336)]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn unfinished_or_out_of_order_scope_fails_canonical_receipt() {
    let mut receipt = ProviderOperationReceipt::new();
    let scope = receipt
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([8; 4]).unwrap())
        .unwrap();
    assert_eq!(
        scope.finish(),
        Err(ProviderObservationFailure::TrackingOverflow)
    );
    drop(scope);
    assert_eq!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::TrackingOverflow)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn failed_owner_cleanup_cannot_be_reinterpreted_as_retirement_success() {
    let mut receipt = ProviderOperationReceipt::new();
    let scope = scope(&receipt);
    let staging = scope.issue_resource_ticket(8).unwrap();
    scope.finish().unwrap();
    receipt.publish().unwrap();
    staging.fail_owner();
    drop(staging);
    let evidence = receipt.take_completed().unwrap();
    assert_eq!(
        evidence.status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::ProviderFailed)
    );
    assert_eq!(evidence.resources.unwrap().requested_live_bytes, 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn actual_ticket_cannot_exceed_prepared_phase_limit() {
    let mut receipt = ProviderOperationReceipt::new();
    let scope = receipt
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([8; 4]).unwrap())
        .unwrap();
    scope.begin_next_phase().unwrap();
    assert!(matches!(
        scope.issue_resource_ticket(9),
        Err(ProviderObservationFailure::ResourceLimit)
    ));
    assert_eq!(receipt.resource_snapshot().unwrap().issued, 0);
    assert_eq!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::ResourceLimit)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sequential_scopes_share_one_receipt_without_mixing_peaks() {
    let mut receipt = ProviderOperationReceipt::new();
    let first = receipt
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([16, 24]).unwrap())
        .unwrap();
    first.begin_next_phase().unwrap();
    let first_ticket = first.issue_resource_ticket(16).unwrap();
    first.begin_next_phase().unwrap();
    first.finish().unwrap();
    drop(first_ticket);
    drop(first);
    let second = receipt
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([32]).unwrap())
        .unwrap();
    second.begin_next_phase().unwrap();
    let second_ticket = second.issue_resource_ticket(32).unwrap();
    second.finish().unwrap();
    receipt.publish().unwrap();
    assert_eq!(receipt.status(), ProviderObservationStatus::Pending);
    drop(second_ticket);
    let resources = receipt.take_completed().unwrap().resources.unwrap();
    assert_eq!(resources.scopes.len(), 2);
    assert_eq!(resources.scopes[0].scope_id, 1);
    assert_eq!(resources.scopes[0].phase_peaks, vec![Some(16), Some(16)]);
    assert_eq!(resources.scopes[1].scope_id, 2);
    assert_eq!(resources.scopes[1].phase_peaks, vec![Some(32)]);
    assert_eq!(resources.requested_peak_bytes, 32);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nested_scopes_keep_identity_through_late_retirement_and_cancellation() {
    let mut receipt = ProviderOperationReceipt::new();
    let outer = receipt
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([40, 40]).unwrap())
        .unwrap();
    outer.begin_next_phase().unwrap();
    let outer_ticket = outer.issue_resource_ticket(40).unwrap();
    let inner = receipt
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([24]).unwrap())
        .unwrap();
    inner.begin_next_phase().unwrap();
    let inner_ticket = inner.issue_resource_ticket(24).unwrap();
    inner.finish().unwrap();
    outer.begin_next_phase().unwrap();
    outer.finish().unwrap();
    receipt.publish().unwrap();
    assert_eq!(
        receipt.resource_snapshot().unwrap().requested_peak_bytes,
        64
    );
    receipt.fail(ProviderObservationFailure::Cancelled);
    drop(outer_ticket);
    let pending_snapshot = receipt.resource_snapshot().unwrap();
    assert_eq!(pending_snapshot.scopes[0].requested_live_bytes, 0);
    assert_eq!(pending_snapshot.scopes[1].requested_live_bytes, 24);
    drop(inner_ticket);
    let evidence = receipt.take_completed().unwrap();
    assert_eq!(
        evidence.status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::Cancelled)
    );
    assert_eq!(evidence.resources.unwrap().requested_live_bytes, 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn issued_scope_ids_resolve_contiguously_without_aliasing() {
    let mut state = ResourceState::default();
    let first = state
        .begin_scope(ProviderResourcePhasePlan::new([8]).unwrap())
        .unwrap();
    let second = state
        .begin_scope(ProviderResourcePhasePlan::new([16]).unwrap())
        .unwrap();
    assert_eq!((first, second), (1, 2));
    assert!(state.scope_mut(0).is_err());
    assert!(state.scope_mut(3).is_err());
    assert_eq!(state.scope_mut(first).unwrap().id, first);
    assert_eq!(state.scope_mut(second).unwrap().id, second);
}
