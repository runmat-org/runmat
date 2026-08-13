use super::RuntimeContext;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

runmat_thread_local! {
    static ACTIVE_CONTEXTS: RefCell<Vec<RuntimeContext>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn active_runtime_context() -> Option<RuntimeContext> {
    ACTIVE_CONTEXTS.with(|contexts| contexts.borrow().last().cloned())
}

#[must_use]
pub(crate) struct RuntimeContextGuard {
    state_identity: *const super::RuntimeContextState,
}

impl RuntimeContextGuard {
    pub(crate) fn enter(context: RuntimeContext) -> Self {
        let state_identity = context.state_identity();
        ACTIVE_CONTEXTS.with(|contexts| contexts.borrow_mut().push(context));
        Self { state_identity }
    }
}

impl Drop for RuntimeContextGuard {
    fn drop(&mut self) {
        ACTIVE_CONTEXTS.with(|contexts| {
            let popped = contexts.borrow_mut().pop();
            debug_assert_eq!(
                popped.as_ref().map(RuntimeContext::state_identity),
                Some(self.state_identity),
                "runtime contexts must unwind in stack order"
            );
        });
    }
}

/// Future adapter that activates the same context for every poll and while the
/// inner future is destroyed. It never keeps ambient state installed across a
/// yield, so independently polled sessions cannot observe one another.
pub struct ContextFuture<F> {
    context: RuntimeContext,
    future: Option<Pin<Box<F>>>,
}

impl<F> ContextFuture<F> {
    pub(super) fn new(context: RuntimeContext, future: F) -> Self {
        Self {
            context,
            future: Some(Box::pin(future)),
        }
    }
}

impl<F: Future> Future for ContextFuture<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        let _guard = RuntimeContextGuard::enter(this.context.clone());
        this.future
            .as_mut()
            .expect("context future polled after completion")
            .as_mut()
            .poll(cx)
    }
}

impl<F> Drop for ContextFuture<F> {
    fn drop(&mut self) {
        if let Some(future) = self.future.take() {
            let _guard = RuntimeContextGuard::enter(self.context.clone());
            drop(future);
        }
    }
}

impl<F> Unpin for ContextFuture<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;
    use runmat_types::SourceId;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::task::Wake;

    struct NoopWake;

    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn interleaved_context_futures_restore_the_active_session() {
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let first_scope = first.execution().scope_id();
        let second_scope = second.execution().scope_id();

        let mut first_future = Box::pin(first.scope(std::future::poll_fn(move |_| {
            assert_eq!(
                active_runtime_context()
                    .expect("first context active")
                    .execution()
                    .scope_id(),
                first_scope
            );
            Poll::<()>::Pending
        })));
        let mut second_future = Box::pin(second.scope(std::future::poll_fn(move |_| {
            assert_eq!(
                active_runtime_context()
                    .expect("second context active")
                    .execution()
                    .scope_id(),
                second_scope
            );
            Poll::<()>::Pending
        })));
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);

        assert!(first_future.as_mut().poll(&mut cx).is_pending());
        assert!(active_runtime_context().is_none());
        assert!(second_future.as_mut().poll(&mut cx).is_pending());
        assert!(active_runtime_context().is_none());
    }

    #[test]
    fn interleaved_futures_isolate_source_output_and_cancellation_state() {
        async fn session_future(source_id: SourceId, source: &'static str, outputs: usize) {
            let _catalog = crate::source_context::replace_source_catalog(vec![(
                source_id,
                format!("session-{}.m", source_id.0),
                source.to_string(),
            )]);
            let _source = crate::source_context::replace_current_source_id(Some(source_id));
            let _outputs = crate::output_count::push_output_count(Some(outputs));
            let mut yielded = false;
            std::future::poll_fn(move |_| {
                if yielded {
                    Poll::Ready(())
                } else {
                    yielded = true;
                    Poll::Pending
                }
            })
            .await;
            assert_eq!(
                crate::source_context::current_source().as_deref(),
                Some(source)
            );
            assert_eq!(crate::output_count::current_output_count(), Some(outputs));
            assert!(!crate::interrupt::is_cancelled());
        }

        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        first
            .cancellation()
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let mut first_future = Box::pin(first.scope(session_future(SourceId(1), "first", 1)));
        let mut second_future = Box::pin(second.scope(session_future(SourceId(2), "second", 2)));
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);

        assert!(first_future.as_mut().poll(&mut cx).is_pending());
        assert!(second_future.as_mut().poll(&mut cx).is_pending());
        first
            .cancellation()
            .store(false, std::sync::atomic::Ordering::Relaxed);
        assert!(first_future.as_mut().poll(&mut cx).is_ready());
        assert!(second_future.as_mut().poll(&mut cx).is_ready());
        assert!(crate::source_context::current_source().is_none());
        assert_eq!(crate::output_count::current_output_count(), None);
    }

    #[test]
    fn interleaved_futures_isolate_host_interaction_handlers() {
        async fn session_future(label: &'static str) {
            let handler: Arc<crate::interaction::AsyncInteractionHandler> = Arc::new(move |_| {
                Box::pin(async move {
                    Ok(crate::interaction::InteractionResponse::Line(
                        label.to_string(),
                    ))
                })
            });
            let _handler = crate::interaction::replace_async_handler(Some(handler));
            let mut yielded = false;
            std::future::poll_fn(move |_| {
                if yielded {
                    Poll::Ready(())
                } else {
                    yielded = true;
                    Poll::Pending
                }
            })
            .await;
            assert_eq!(
                crate::interaction::request_line_async("", false)
                    .await
                    .unwrap(),
                label
            );
        }

        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let mut first_future = Box::pin(first.scope(session_future("first")));
        let mut second_future = Box::pin(second.scope(session_future("second")));
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);
        assert!(first_future.as_mut().poll(&mut cx).is_pending());
        assert!(second_future.as_mut().poll(&mut cx).is_pending());
        assert!(first_future.as_mut().poll(&mut cx).is_ready());
        assert!(second_future.as_mut().poll(&mut cx).is_ready());
    }

    #[test]
    fn dropping_scoped_future_activates_its_context_for_guard_cleanup() {
        struct DropProbe(runmat_execution::ExecutionScopeId);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                assert_eq!(
                    active_runtime_context()
                        .expect("context active while inner future drops")
                        .execution()
                        .scope_id(),
                    self.0
                );
            }
        }

        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let scope_id = runtime.execution().scope_id();
        let future = async move {
            let _probe = DropProbe(scope_id);
            std::future::pending::<()>().await;
        };
        let mut future = Box::pin(runtime.scope(future));
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);
        assert!(future.as_mut().poll(&mut cx).is_pending());
        drop(future);
        assert!(active_runtime_context().is_none());
    }

    #[test]
    fn workspace_resolver_registration_is_context_scoped_in_production_shape() {
        fn first_lookup(name: &str) -> Option<runmat_value::Value> {
            (name == "x").then_some(runmat_value::Value::Num(1.0))
        }
        fn second_lookup(name: &str) -> Option<runmat_value::Value> {
            (name == "x").then_some(runmat_value::Value::Num(2.0))
        }
        fn empty_snapshot() -> Vec<(String, runmat_value::Value)> {
            Vec::new()
        }
        fn empty_globals() -> Vec<String> {
            Vec::new()
        }

        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        {
            let _scope = RuntimeContextGuard::enter(first.clone());
            crate::workspace::register_workspace_resolver(crate::workspace::WorkspaceResolver {
                lookup: first_lookup,
                snapshot: empty_snapshot,
                globals: empty_globals,
                assign: None,
                clear: None,
                remove: None,
            });
        }
        {
            let _scope = RuntimeContextGuard::enter(second.clone());
            crate::workspace::register_workspace_resolver(crate::workspace::WorkspaceResolver {
                lookup: second_lookup,
                snapshot: empty_snapshot,
                globals: empty_globals,
                assign: None,
                clear: None,
                remove: None,
            });
        }
        {
            let _scope = RuntimeContextGuard::enter(first);
            assert_eq!(
                crate::workspace::lookup("x"),
                Some(runmat_value::Value::Num(1.0))
            );
        }
        {
            let _scope = RuntimeContextGuard::enter(second);
            assert_eq!(
                crate::workspace::lookup("x"),
                Some(runmat_value::Value::Num(2.0))
            );
        }
    }

    #[test]
    fn class_and_static_property_state_is_session_owned() {
        fn class(name: &str) -> crate::class_registry::RuntimeClass {
            crate::class_registry::RuntimeClass {
                name: name.to_string(),
                parent: None,
                properties: std::collections::HashMap::from([(
                    "answer".to_string(),
                    crate::class_registry::RuntimeProperty {
                        name: "answer".to_string(),
                        is_static: true,
                        is_constant: false,
                        is_dependent: false,
                        get_access: runmat_types::MemberAccess::Public,
                        set_access: runmat_types::MemberAccess::Public,
                        default_value: None,
                    },
                )]),
                methods: std::collections::HashMap::new(),
            }
        }

        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        {
            let _scope = RuntimeContextGuard::enter(first.clone());
            crate::class_registry::register_class(class("OnlyFirst"));
            crate::class_registry::set_static_property_value(
                "OnlyFirst",
                "answer",
                runmat_value::Value::Num(1.0),
            );
        }
        {
            let _scope = RuntimeContextGuard::enter(second);
            assert!(crate::class_registry::get_class("OnlyFirst").is_none());
            assert!(
                crate::class_registry::get_static_property_value("OnlyFirst", "answer").is_none()
            );
        }
        {
            let _scope = RuntimeContextGuard::enter(first);
            assert!(crate::class_registry::get_class("OnlyFirst").is_some());
            assert_eq!(
                crate::class_registry::get_static_property_value("OnlyFirst", "answer"),
                Some(runmat_value::Value::Num(1.0))
            );
        }
    }

    #[test]
    fn lazy_builtin_class_registration_is_session_owned() {
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        {
            let _scope = RuntimeContextGuard::enter(first.clone());
            crate::testing::ensure_testing_classes();
            assert!(crate::class_registry::get_class(crate::testing::TEST_CASE_CLASS).is_some());
        }
        {
            let _scope = RuntimeContextGuard::enter(second.clone());
            assert!(crate::class_registry::get_class(crate::testing::TEST_CASE_CLASS).is_none());
            crate::testing::ensure_testing_classes();
            assert!(crate::class_registry::get_class(crate::testing::TEST_CASE_CLASS).is_some());
        }
        {
            let _scope = RuntimeContextGuard::enter(first);
            assert!(crate::class_registry::get_class(crate::testing::TEST_CASE_CLASS).is_some());
        }
    }

    #[test]
    fn error_and_compatibility_policy_is_session_owned() {
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        first.set_error_namespace("First");
        first.set_callstack_limit(11);
        second.set_error_namespace("Second");
        second.set_callstack_limit(22);
        {
            let _scope = RuntimeContextGuard::enter(first);
            crate::compatibility::set_runmat_extensions_enabled(true);
            assert_eq!(active_runtime_context().unwrap().error_namespace(), "First");
            assert_eq!(active_runtime_context().unwrap().callstack_limit(), 11);
            assert!(crate::compatibility::runmat_extensions_enabled());
        }
        {
            let _scope = RuntimeContextGuard::enter(second);
            assert_eq!(
                active_runtime_context().unwrap().error_namespace(),
                "Second"
            );
            assert_eq!(active_runtime_context().unwrap().callstack_limit(), 22);
            assert!(!crate::compatibility::runmat_extensions_enabled());
        }
    }
}
