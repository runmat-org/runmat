use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn ticket_distinguishes_absent_pending_completed_and_failed() {
    let absent = ProbeFenceTicket::armed();
    assert_eq!(absent.state(), ProbeFenceState::Armed);
    absent.submitted_without_fence(false);
    assert_eq!(absent.state(), ProbeFenceState::Absent);
    assert_eq!(absent.fence_requested_bytes(), None);
    absent.complete();
    assert_eq!(absent.state(), ProbeFenceState::Absent);

    let pending = ProbeFenceTicket::armed();
    let alias = pending.clone();
    pending.submitted_with_fence(8);
    assert_eq!(alias.state(), ProbeFenceState::Pending);
    assert_eq!(alias.fence_requested_bytes(), Some(8));
    pending.complete();
    assert_eq!(alias.state(), ProbeFenceState::Completed);
    pending.fail();
    assert_eq!(alias.state(), ProbeFenceState::Completed);

    let failed = ProbeFenceTicket::armed();
    failed.submitted_with_fence(8);
    failed.fail();
    failed.complete();
    assert_eq!(failed.state(), ProbeFenceState::Failed);
    assert_eq!(failed.fence_requested_bytes(), Some(8));

    let reservation_failed = ProbeFenceTicket::armed();
    reservation_failed.submitted_without_fence(true);
    assert_eq!(reservation_failed.state(), ProbeFenceState::Failed);
    assert_eq!(reservation_failed.fence_requested_bytes(), None);
}
