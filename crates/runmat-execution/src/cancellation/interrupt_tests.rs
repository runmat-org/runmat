use super::*;

#[test]
fn child_deadline_does_not_change_parent() {
    let parent = CancellationToken::default();
    let child = parent.clone().with_deadline(Some(0));
    assert_eq!(parent.stop_reason_at(0), None);
    assert_eq!(child.stop_reason_at(0), Some(CancellationReason::Deadline));
}

#[test]
fn original_interrupt_flags_remain_live_and_deduplicated_after_clone() {
    let session = Arc::new(AtomicBool::new(false));
    let caller = Arc::new(AtomicBool::new(false));
    let token = CancellationToken::from_shared(session.clone())
        .with_source(caller.clone())
        .with_source(caller.clone())
        .with_source(session.clone());
    assert_eq!(token.interrupts.len(), 2);
    let child = token.clone();
    assert_eq!(child.stop_reason_at(0), None);
    caller.store(true, Ordering::Release);
    assert_eq!(child.stop_reason_at(0), Some(CancellationReason::User));
    assert!(!session.load(Ordering::Acquire));
    caller.store(false, Ordering::Release);
    assert!(token.cancel());
    assert!(session.load(Ordering::Acquire));
    assert_eq!(child.stop_reason_at(0), Some(CancellationReason::User));
}

#[test]
fn child_cannot_extend_or_clear_parent_deadline() {
    let parent = CancellationToken::default().with_deadline(Some(100));
    for child in [
        parent.clone().with_deadline(Some(200)),
        parent.clone().with_deadline(None),
    ] {
        assert_eq!(child.stop_reason_at(99), None);
        assert_eq!(
            child.stop_reason_at(100),
            Some(CancellationReason::Deadline)
        );
    }
    let child = parent.with_deadline(Some(50));
    assert_eq!(child.stop_reason_at(50), Some(CancellationReason::Deadline));
}

#[test]
fn generation_tokens_are_read_only_and_compose_with_interrupts() {
    let source = CancellationSource::new();
    let generation = source.token();
    assert!(!generation.cancel());
    let flag = Arc::new(AtomicBool::new(false));
    let combined = generation
        .with_deadline(Some(500))
        .or(&CancellationToken::from_shared(flag.clone()).with_deadline(Some(100)));
    assert_eq!(combined.deadline_unix_ms(), Some(100));
    assert!(!combined.cancel());
    flag.store(true, Ordering::Release);
    assert_eq!(combined.reason(), Some(CancellationReason::User));
    source.cancel(CancellationReason::Shutdown);
    assert_eq!(combined.reason(), Some(CancellationReason::Shutdown));
}
