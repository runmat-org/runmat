use super::*;
use crate::profiling::{CommandProfilingRoot, CommandRouteCoverage, PhysicalSubmissionCounts};

#[test]
fn explicit_success_failure_and_unsettled_drop_are_distinct_terminal_facts() {
    let root = CommandProfilingRoot::new();
    let receipt = root.receipt();
    receipt
        .begin_deferred_work()
        .expect("first future admitted")
        .complete(DeferredWorkTerminal::Completed);
    receipt
        .begin_deferred_work()
        .expect("second future admitted")
        .complete(DeferredWorkTerminal::Failed);
    drop(
        receipt
            .begin_deferred_work()
            .expect("third future admitted"),
    );
    let snapshot = root.seal();
    assert_eq!(
        snapshot.deferred_work,
        Some(DeferredWorkCounts {
            admitted: 3,
            completed: 1,
            failed: 1,
            missing: 1,
            ..DeferredWorkCounts::default()
        })
    );
    assert_eq!(
        snapshot.physical_submissions,
        Some(PhysicalSubmissionCounts::default())
    );
    assert_eq!(snapshot.route_coverage, CommandRouteCoverage::Incomplete);
    assert_eq!(
        (snapshot.gpu_execution, snapshot.kernel_count),
        (None, None)
    );
}

#[test]
fn pending_at_seal_and_late_settlement_cannot_change_published_snapshot() {
    let root = CommandProfilingRoot::new();
    let borrowed = root.receipt();
    let pending = borrowed.begin_deferred_work().expect("queued before seal");
    let published = root.seal();
    assert_eq!(
        published.deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            pending: 1,
            ..DeferredWorkCounts::default()
        })
    );
    pending.complete(DeferredWorkTerminal::Completed);
    assert!(borrowed.begin_deferred_work().is_none());
    assert_eq!(published.deferred_work.expect("snapshot").pending, 1);
    assert_eq!(published.route_coverage, CommandRouteCoverage::Incomplete);
    assert_eq!(
        (published.gpu_execution, published.kernel_count),
        (None, None)
    );
}

#[test]
fn root_drop_seals_cancellation_without_turning_it_into_success() {
    let root = CommandProfilingRoot::new();
    let borrowed = root.receipt();
    let unsettled = borrowed.begin_deferred_work().expect("queued future");
    drop(root);
    drop(unsettled);
    assert!(borrowed.begin_deferred_work().is_none());
}

#[test]
fn interleaved_command_roots_do_not_share_deferred_counts() {
    let first = CommandProfilingRoot::new();
    let second = CommandProfilingRoot::new();
    let first_pending = first.receipt().begin_deferred_work().expect("first");
    second
        .receipt()
        .begin_deferred_work()
        .expect("second")
        .complete(DeferredWorkTerminal::Completed);
    assert_eq!(second.seal().deferred_work.expect("valid").completed, 1);
    assert_eq!(first.seal().deferred_work.expect("valid").pending, 1);
    drop(first_pending);
}

#[test]
fn empty_admission_does_not_certify_coverage() {
    let snapshot = CommandProfilingRoot::new().seal();
    assert_eq!(snapshot.deferred_work, Some(DeferredWorkCounts::default()));
    assert_eq!(snapshot.route_coverage, CommandRouteCoverage::Incomplete);
    assert_eq!(
        (snapshot.gpu_execution, snapshot.kernel_count),
        (None, None)
    );
}

#[test]
fn checked_admission_and_terminal_overflow_fail_closed() {
    let admission = CommandProfilingRoot::new();
    admission
        .receipt
        .0
        .lock()
        .expect("test state")
        .deferred
        .counts
        .admitted = u64::MAX;
    assert!(admission.receipt().begin_deferred_work().is_none());
    assert_eq!(admission.seal().deferred_work, None);

    let pending_overflow = CommandProfilingRoot::new();
    pending_overflow
        .receipt
        .0
        .lock()
        .expect("test state")
        .deferred
        .counts
        .pending = u64::MAX;
    assert!(pending_overflow.receipt().begin_deferred_work().is_none());
    assert_eq!(pending_overflow.seal().deferred_work, None);

    let terminal = CommandProfilingRoot::new();
    let work = terminal
        .receipt()
        .begin_deferred_work()
        .expect("one future");
    terminal
        .receipt
        .0
        .lock()
        .expect("test state")
        .deferred
        .counts
        .completed = u64::MAX;
    work.complete(DeferredWorkTerminal::Completed);
    let snapshot = terminal.seal();
    assert_eq!(snapshot.deferred_work, None);
    assert_eq!(snapshot.physical_submissions, None);
    assert_eq!(
        (snapshot.gpu_execution, snapshot.kernel_count),
        (None, None)
    );

    let pending_underflow = CommandProfilingRoot::new();
    let work = pending_underflow
        .receipt()
        .begin_deferred_work()
        .expect("admitted before corrupt fixture");
    pending_underflow
        .receipt
        .0
        .lock()
        .expect("test state")
        .deferred
        .counts
        .pending = 0;
    work.complete(DeferredWorkTerminal::Completed);
    assert_eq!(pending_underflow.seal().deferred_work, None);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn lease_is_send_safe_and_poisoned_state_never_looks_empty() {
    fn require_send<T: Send>() {}
    require_send::<CommandDeferredWork>();
    let root = CommandProfilingRoot::new();
    let work = root.receipt().begin_deferred_work().expect("future");
    std::thread::spawn(move || work.complete(DeferredWorkTerminal::Failed))
        .join()
        .expect("worker settled");
    assert_eq!(root.seal().deferred_work.expect("valid").failed, 1);

    let poisoned = CommandProfilingRoot::new();
    let state = Arc::clone(&poisoned.receipt.0);
    let _ = std::thread::spawn(move || {
        let _guard = state.lock().expect("test state");
        panic!("poison the accounting lock");
    })
    .join();
    assert!(poisoned.receipt().begin_deferred_work().is_none());
    assert_eq!(poisoned.seal().deferred_work, None);
}
