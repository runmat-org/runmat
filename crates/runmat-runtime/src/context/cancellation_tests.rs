use std::rc::Rc;

use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;

use super::{RuntimeContext, RuntimeServicePorts};
use crate::execution::RuntimeExecutionService;

#[test]
fn rebinding_invocation_does_not_change_cached_context_generation() {
    let old = CancellationSource::new();
    let context =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), old.token());
    let cached = context.clone();
    let fresh = CancellationSource::new();
    let rebound = context.with_cancellation_token(fresh.token());
    assert_eq!(cached.local_identity(), rebound.local_identity());
    old.cancel(CancellationReason::User);
    assert!(cached.is_cancelled());
    assert!(!rebound.is_cancelled());
    fresh.cancel(CancellationReason::Deadline);
    assert_eq!(
        cached.cancellation().reason(),
        Some(CancellationReason::User)
    );
    assert_eq!(
        rebound.cancellation().reason(),
        Some(CancellationReason::Deadline)
    );
}

#[test]
fn scoped_interrupt_changes_active_clone_family_and_restores_original() {
    let original = CancellationSource::new();
    let override_source = CancellationSource::new();
    let context = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        original.token(),
    );
    let clone = context.clone();
    let _active = context.enter();
    {
        let _scope = crate::interrupt::replace_interrupt(Some(override_source.token()));
        override_source.cancel(CancellationReason::User);
        assert!(clone.is_cancelled());
        assert!(crate::interrupt::is_cancelled());
    }
    assert!(!clone.is_cancelled());
    original.cancel(CancellationReason::Shutdown);
    assert!(crate::interrupt::is_cancelled());
}

#[test]
fn parallel_child_retains_origin_token_across_parent_rebinding_and_scoped_override() {
    let source = CancellationSource::new();
    let context =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    let child = context.fork_parallel_lab(RuntimeServicePorts::default());
    let override_source = CancellationSource::new();
    let _active = context.enter();
    {
        let _scope = crate::interrupt::replace_interrupt(Some(override_source.token()));
        override_source.cancel(CancellationReason::User);
        assert!(context.is_cancelled());
        assert!(!child.is_cancelled());
    }
    source.cancel(CancellationReason::ParentScope);
    assert!(context.is_cancelled());
    assert!(child.is_cancelled());
}
