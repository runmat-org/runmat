use super::*;
#[test]
fn adapter_keeps_live_parent_and_deadline_without_mutating_primary_flag() {
    let owner = Arc::new(AtomicBool::new(false));
    let caller = Arc::new(AtomicBool::new(false));
    let shared =
        runmat_execution::CancellationToken::from_shared(owner.clone()).with_source(caller.clone());
    let adapter = FeaCancellationToken::from_token(shared.clone());
    assert!(!adapter.is_cancelled());
    caller.store(true, std::sync::atomic::Ordering::Release);
    assert!(adapter.is_cancelled());
    assert!(!owner.load(std::sync::atomic::Ordering::Acquire));
    caller.store(false, std::sync::atomic::Ordering::Release);
    assert!(FeaCancellationToken::from_token(shared.with_deadline(Some(0))).is_cancelled());
    assert!(!owner.load(std::sync::atomic::Ordering::Acquire));
}

fn context(
    token: &FeaCancellationToken,
    handler: Option<FeaProgressHandler>,
) -> FeaProgressContext {
    FeaProgressContext {
        handler,
        cancellation: token.clone(),
        attempt_id: None,
    }
}

#[test]
fn nested_context_restores_parent_cancellation_and_event_owner() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let parent = FeaCancellationToken::new();
    let child = FeaCancellationToken::new();
    let parent_events = Arc::new(AtomicUsize::new(0));
    let child_events = Arc::new(AtomicUsize::new(0));
    let handler = |count: &Arc<AtomicUsize>| {
        let count = count.clone();
        Arc::new(move |_| {
            count.fetch_add(1, Ordering::Relaxed);
        }) as FeaProgressHandler
    };
    let outer = replace_fea_progress_context(context(&parent, Some(handler(&parent_events))));
    parent.cancel();
    assert!(is_cancelled());
    {
        let _inner = replace_fea_progress_context(context(&child, Some(handler(&child_events))));
        assert!(!is_cancelled());
        emit_phase(
            "child",
            FeaProgressPhase::Solve,
            FeaProgressStatus::Started,
            "child",
            None,
            None,
        );
        child.cancel();
        assert!(is_cancelled());
    }
    assert!(is_cancelled());
    emit_phase(
        "parent",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "parent",
        None,
        None,
    );
    assert_eq!(parent_events.load(Ordering::Relaxed), 1);
    assert_eq!(child_events.load(Ordering::Relaxed), 1);
    drop(outer);
    assert!(!is_cancelled());
}

#[test]
fn panic_unwind_restores_original_thread_context() {
    let parent = FeaCancellationToken::new();
    let child = FeaCancellationToken::new();
    child.cancel();
    let _parent = replace_fea_progress_context(context(&parent, None));
    let result = std::panic::catch_unwind(|| {
        let _child = replace_fea_progress_context(context(&child, None));
        assert!(is_cancelled());
        panic!("controlled worker failure");
    });
    assert!(result.is_err());
    assert!(!is_cancelled());
    parent.cancel();
    assert!(is_cancelled());
}

#[test]
fn workers_install_their_own_context_and_share_live_cancellation() {
    let shared = FeaCancellationToken::new();
    let _parent = replace_fea_progress_context(context(&shared, None));
    let entered = Arc::new(std::sync::Barrier::new(2));
    let cancelled = Arc::new(std::sync::Barrier::new(2));
    let worker_token = shared.clone();
    let worker_entered = entered.clone();
    let worker_cancelled = cancelled.clone();
    let worker = std::thread::spawn(move || {
        assert!(!is_cancelled());
        let guard = replace_fea_progress_context(context(&worker_token, None));
        assert!(!is_cancelled());
        worker_entered.wait();
        worker_cancelled.wait();
        assert!(is_cancelled());
        drop(guard);
        assert!(!is_cancelled());
    });
    entered.wait();
    shared.cancel();
    assert!(is_cancelled());
    cancelled.wait();
    worker.join().unwrap();
    assert!(is_cancelled());
}
