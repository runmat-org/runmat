use super::*;
use std::rc::Rc;

fn context() -> crate::context::RuntimeContext {
    crate::context::RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
}

#[test]
fn equal_numeric_ids_in_distinct_sessions_never_exchange_source_authority() {
    let first = context();
    let second = context();
    let first_lease = lease_source_context_in_context(&first, "first.m", None, "error('first')")
        .expect("first source");
    let second_lease =
        lease_source_context_in_context(&second, "second.m", None, "error('second')")
            .expect("second source");
    assert_eq!(first_lease.source_id(), second_lease.source_id());
    assert_ne!(first_lease.source_ref(), second_lease.source_ref());
    assert_eq!(
        first_lease.source_ref().resolve_in(&first),
        Some(first_lease.source_id())
    );
    assert_eq!(
        second_lease.source_ref().resolve_in(&second),
        Some(second_lease.source_id())
    );
    assert_eq!(first_lease.source_ref().resolve_in(&second), None);
    assert_eq!(second_lease.source_ref().resolve_in(&first), None);
}

#[test]
fn fork_keeps_issued_origin_and_exact_text_after_parent_lease_retires() {
    let parent = context();
    let lease = lease_source_context_in_context(&parent, "captured.m", None, "captured = 42;")
        .expect("parent source");
    let reference = lease.source_ref();
    let child = parent.fork_parallel_lab(crate::context::RuntimeServicePorts::default());
    assert_eq!(reference.resolve_in(&child), Some(lease.source_id()));
    drop(lease);
    assert_eq!(reference.resolve_in(&parent), None);
    assert_eq!(reference.resolve_in(&child), Some(reference.source_id()));
    assert_eq!(
        source_info_in_context(&child, reference.source_id())
            .unwrap()
            .text
            .as_ref(),
        "captured = 42;"
    );
    let unrelated = context();
    let collision =
        lease_source_context_in_context(&unrelated, "captured.m", None, "captured = 42;")
            .expect("same text, other owner");
    assert_eq!(collision.source_id(), reference.source_id());
    assert_eq!(reference.resolve_in(&unrelated), None);
}
