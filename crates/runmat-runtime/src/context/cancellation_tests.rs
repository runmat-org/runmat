use std::rc::Rc;

use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;

use super::{RuntimeContext, RuntimeServicePorts};
use crate::execution::RuntimeExecutionService;

#[test]
fn retained_callback_uses_same_session_caller_environment_and_debug_ancestry() {
    let captured = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    captured
        .install_environment_snapshot(vec![("CALL_ENV".into(), "registration".into())])
        .unwrap();
    let caller_environment =
        runmat_execution::EnvironmentSnapshot::new(vec![("CALL_ENV".into(), "invocation".into())])
            .unwrap();
    let caller = captured
        .clone()
        .with_invocation_environment(&caller_environment)
        .unwrap()
        .with_debug_frame(crate::debug_context::DebugFrame {
            function: "caller".into(),
            source_id: None,
            span: None,
        });
    let projected = captured
        .clone()
        .try_with_inherited_callback_invocation_authority_from(&caller)
        .unwrap()
        .with_debug_frame(crate::debug_context::DebugFrame {
            function: "callback".into(),
            source_id: None,
            span: None,
        });
    {
        let _scope = projected.enter();
        assert_eq!(
            crate::builtins::common::env::var("CALL_ENV").unwrap(),
            "invocation"
        );
        assert_eq!(
            crate::debug_context::current_frames()
                .iter()
                .map(|frame| frame.function.as_str())
                .collect::<Vec<_>>(),
            ["callback", "caller"]
        );
    }
    {
        let _scope = captured.enter();
        assert_eq!(
            crate::builtins::common::env::var("CALL_ENV").unwrap(),
            "registration"
        );
        assert!(crate::debug_context::current_frames().is_empty());
    }
    let foreign = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    assert!(captured
        .try_with_inherited_callback_invocation_authority_from(&foreign)
        .is_none());
}

#[test]
fn fresh_contexts_register_network_property_protocols_without_construction() {
    let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    assert_ne!(first.local_identity(), second.local_identity());
    for context in [first, second] {
        let _guard = context.enter();
        for identity in [
            runmat_types::standard::DL_NETWORK,
            runmat_types::standard::SERIES_NETWORK,
            runmat_types::standard::DAG_NETWORK,
        ] {
            let (learnables, owner) = crate::class_registry::lookup_property(
                &identity.owned(),
                &runmat_types::MemberName::from("Learnables"),
            )
            .expect("fresh context registers dependent Learnables dispatch");
            assert_eq!(owner, identity.owned());
            assert!(learnables.is_dependent);
            assert_eq!(learnables.set_access, runmat_types::MemberAccess::Public);
            let (layers, _) = crate::class_registry::lookup_property(
                &identity.owned(),
                &runmat_types::MemberName::from("Layers"),
            )
            .expect("fresh context registers canonical Layers access");
            assert_eq!(layers.set_access, runmat_types::MemberAccess::Private);
        }
    }
}

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
fn cancellation_scopes_restore_parent_and_retire_without_resetting_invocations() {
    let parent = CancellationSource::new();
    let context =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), parent.token());
    let outer = CancellationSource::new();
    let scope = context.scoped_cancellation_token(outer.token());
    let retained = context.clone().with_pinned_cancellation();
    let inner = CancellationSource::new();
    let nested = context.scoped_cancellation_token(inner.token());
    inner.cancel(CancellationReason::Deadline);
    assert_eq!(
        context.cancellation().stop_reason(),
        Some(CancellationReason::Deadline)
    );
    drop(nested);
    assert!(!context.is_cancelled());
    outer.cancel(CancellationReason::User);
    assert_eq!(
        context.cancellation().stop_reason(),
        Some(CancellationReason::User)
    );
    let successor = CancellationSource::new();
    scope.retire_to(successor.token());
    assert!(!context.is_cancelled());
    assert_eq!(
        retained.cancellation().stop_reason(),
        Some(CancellationReason::User)
    );
    let restored = context.scoped_cancellation_token(parent.token());
    parent.cancel(CancellationReason::Shutdown);
    assert_eq!(
        context.cancellation().stop_reason(),
        Some(CancellationReason::Shutdown)
    );
    drop(restored);
    assert!(!context.is_cancelled());
}

#[cfg(feature = "plot-core")]
#[test]
fn background_selection_keeps_original_policy_cancellation_and_weak_session() {
    use super::{RuntimeGraphicsCallbackQueue, RuntimeLanguageMode, RuntimeLanguagePolicy};
    use std::num::NonZeroUsize;

    let cancellation = CancellationSource::new();
    let original = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_language_policy(RuntimeLanguagePolicy {
        language_mode: RuntimeLanguageMode::Strict,
        extensions_enabled: false,
        top_level_await_enabled: false,
        dynamic_eval_enabled: false,
    });
    let state = Rc::downgrade(original.state());
    let execution = Rc::downgrade(original.execution());
    let selected = original.capture_background_callback(false).unwrap();
    let queue = RuntimeGraphicsCallbackQueue::new_for_selected(
        &original,
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(1).unwrap(),
    );
    let fresh = CancellationSource::new();
    let attachment = original
        .clone()
        .with_cancellation_token(fresh.token())
        .with_language_policy(RuntimeLanguagePolicy {
            language_mode: RuntimeLanguageMode::RunMat,
            extensions_enabled: true,
            top_level_await_enabled: true,
            dynamic_eval_enabled: true,
        })
        .with_service_ports(
            RuntimeServicePorts::default().with_graphics_callback_queue(queue.downgrade()),
        );
    let restored = selected(&attachment).unwrap();
    assert_eq!(restored.language_mode(), RuntimeLanguageMode::Strict);
    assert!(!restored.runmat_extensions_enabled());
    assert!(!restored.top_level_await_enabled());
    assert!(!restored.dynamic_eval_enabled());
    assert!(restored
        .service_ports()
        .graphics_callback_queue()
        .unwrap()
        .points_to(&queue));
    let different_executor = attachment
        .clone()
        .with_execution(Rc::new(RuntimeExecutionService::new()));
    assert!(selected(&different_executor).is_err());
    let foreign = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    assert!(selected(&foreign).is_err());
    cancellation.cancel(CancellationReason::User);
    assert!(
        selected(&attachment).is_err(),
        "later cancellation cannot replace the original token"
    );
    assert!(!attachment.is_cancelled());
    drop((restored, different_executor, attachment, original));
    assert!(
        state.upgrade().is_none(),
        "the retained selection must not own session state"
    );
    assert!(
        execution.upgrade().is_none(),
        "the retained selection must not own the executor"
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
    let child = context
        .fork_parallel_lab(RuntimeServicePorts::default())
        .expect("idle session fork");
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
