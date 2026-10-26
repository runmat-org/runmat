use super::super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pending_public_getfield_restores_enclosing_session_and_drops_getter() {
    use std::future::Future as _;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::task::{Context, Poll};

    struct DropProbe(Arc<AtomicUsize>);
    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let wrong = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    wrong.set_error_namespace("EnclosingGetfield");
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let selected = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_cancellation_token(cancellation.token());
    selected.set_error_namespace("SelectedGetfield");
    selected.set_runmat_extensions_enabled(true);
    let expected_session = selected.session_lease();
    let _wrong_scope = wrong.enter();
    let _wrong_outputs =
        crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let _selected_outputs = {
        let _selected_scope = selected.enter();
        crate::output_count::push_output_count(Some(1)).expect("selected output-count scope")
    };
    let selected_setup_scope = selected.enter();
    let class = runmat_types::ClassIdentity::from("PendingSelectedGetfieldSession");
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: class.clone(),
        parent: None,
        properties: std::collections::HashMap::from([(
            runmat_types::MemberName::from("p"),
            crate::class_registry::RuntimeProperty {
                name: "p".into(),
                is_static: false,
                is_constant: false,
                is_dependent: true,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: None,
            },
        )]),
        methods: std::collections::HashMap::from([(
            runmat_types::MethodName::from("get.p"),
            crate::class_registry::RuntimeMethod {
                name: "get.p".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: "PendingSelectedGetfieldSession.get.p".into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        )]),
    });
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "PendingSelectedGetfieldSession.get.p").then_some(943)
        })))
        .expect("call-port scope");
    let dropped = Arc::new(AtomicUsize::new(0));
    let getter_dropped = dropped.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, args, requested_outputs| {
            assert_eq!(function, 943);
            assert_eq!(requested_outputs, 1);
            assert!(matches!(args, [Value::Object(_)]));
            let active = crate::context::legacy::active().expect("selected getter poll");
            assert!(std::sync::Weak::ptr_eq(
                &active.session_lease(),
                &expected_session
            ));
            let budget = active.cpu_cooperation();
            let dropped = getter_dropped.clone();
            Box::pin(async move {
                let _probe = DropProbe(dropped);
                let error = budget
                    .await_cancellable(std::future::pending::<()>())
                    .await
                    .expect_err("pending getter ends only on cancellation");
                Err(error.into_runtime_error())
            })
        },
    )))
    .expect("call-port scope");
    drop(selected_setup_scope);
    let mut work = Box::pin(getfield_builtin(
        Some(selected.clone()),
        Value::Object(ObjectInstance::new(class.clone())),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("p")]),
    ));
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(work.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(crate::output_count::current_output_count(), Some(2));
    assert!(std::sync::Weak::ptr_eq(
        &crate::context::legacy::active()
            .expect("enclosing scope restored after pending poll")
            .session_lease(),
        &wrong.session_lease()
    ));
    drop(work);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(crate::output_count::current_output_count(), Some(2));

    let mut cancelled_work = Box::pin(getfield_builtin(
        Some(selected),
        Value::Object(ObjectInstance::new(class)),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("p")]),
    ));
    assert!(matches!(
        cancelled_work.as_mut().poll(&mut context),
        Poll::Pending
    ));
    assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
    let Poll::Ready(Err(error)) = cancelled_work.as_mut().poll(&mut context) else {
        panic!("selected cancelled getter must finish with a typed error")
    };
    // The admitted continuation fences cancellation before repolling the
    // getter. Its captured selected namespace, not the enclosing scope or
    // the inner cooperation adapter's default namespace, owns this error.
    assert_eq!(error.identifier(), Some("SelectedGetfield:Cancelled"));
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
    assert!(matches!(
        std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>()),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
    drop(cancelled_work);
    assert_eq!(dropped.load(Ordering::SeqCst), 2);
    assert_eq!(crate::output_count::current_output_count(), Some(2));
    assert!(std::sync::Weak::ptr_eq(
        &crate::context::legacy::active()
            .expect("enclosing scope restored after cancellation")
            .session_lease(),
        &wrong.session_lease()
    ));
}
