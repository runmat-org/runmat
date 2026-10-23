use super::*;
use futures::executor::block_on;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

#[test]
fn rhs_operator_callee_failures_are_not_retried_or_reordered() {
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _activation = runtime.enter();
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "RhsFailureFixture.plus").then_some(939)
        })))
        .expect("call-port scope");
    let class = ClassIdentity::from("RhsFailureFixture");
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: class.clone(),
        parent: None,
        properties: Default::default(),
        methods: std::collections::HashMap::from([(
            runmat_types::MethodName::from("plus"),
            crate::class_registry::RuntimeMethod {
                name: "plus".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: runmat_types::MemberAccess::Public,
                function_name: "RhsFailureFixture.plus".into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        )]),
    });

    let calls = Arc::new(AtomicUsize::new(0));
    let diagnostic = Arc::new(Mutex::new("UndefinedFunction"));
    let observed_calls = calls.clone();
    let observed_diagnostic = diagnostic.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, args, requested_outputs| {
            assert_eq!(function, 939);
            assert_eq!(requested_outputs, 1);
            assert!(matches!(args, [Value::Num(2.0), Value::Object(_)]));
            observed_calls.fetch_add(1, Ordering::SeqCst);
            let code = *observed_diagnostic.lock().expect("diagnostic fixture lock");
            Box::pin(async move {
                Err(crate::runtime_error::semantic_error(
                    code,
                    "selected operator callee failed",
                ))
            })
        },
    )))
    .expect("call-port scope");

    for code in ["UndefinedFunction", "ProbeReceiverInvalid"] {
        *diagnostic.lock().expect("diagnostic fixture lock") = code;
        calls.store(0, Ordering::SeqCst);
        let rhs = Value::Object(runmat_value::ObjectInstance::new(class.clone()));
        let err = block_on(try_call_rhs_object_operator_method_ordered(
            Value::Num(2.0),
            rhs,
            "plus",
        ))
        .expect_err("selected callee error must propagate");
        assert_eq!(err.identifier(), Some(format!("RunMat:{code}").as_str()));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn missing_operator_is_certified_absent_without_invocation() {
    let object = Value::Object(runmat_value::ObjectInstance::new("MissingOperatorFixture"));
    let outcome = block_on(try_call_object_operator_method(
        object,
        "plus",
        Value::Num(2.0),
    ))
    .expect("missing operator resolution");
    assert!(outcome.is_none());
}

#[test]
fn rhs_operator_uses_selected_session_not_unrelated_ambient_session() {
    let wrong = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _wrong_activation = wrong.enter();
    let selected = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let expected_session = selected.session_lease();
    let selected_setup = selected.enter();
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "SelectedSessionRhs.plus").then_some(940)
        })))
        .expect("call-port scope");
    let class = ClassIdentity::from("SelectedSessionRhs");
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: class.clone(),
        parent: None,
        properties: Default::default(),
        methods: std::collections::HashMap::from([(
            runmat_types::MethodName::from("plus"),
            crate::class_registry::RuntimeMethod {
                name: "plus".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: runmat_types::MemberAccess::Public,
                function_name: "SelectedSessionRhs.plus".into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        )]),
    });
    let calls = Arc::new(AtomicUsize::new(0));
    let observed_calls = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, args, requested_outputs| {
            assert_eq!(function, 940);
            assert_eq!(requested_outputs, 1);
            assert!(matches!(args, [Value::Num(2.0), Value::Object(_)]));
            let active = crate::context::legacy::active().expect("selected callback scope");
            assert!(std::sync::Weak::ptr_eq(
                &active.session_lease(),
                &expected_session
            ));
            observed_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(12.0))
            })
        },
    )))
    .expect("call-port scope");
    drop(selected_setup);
    let rhs = Value::Object(runmat_value::ObjectInstance::new(class));
    let value = block_on(call_rhs_object_operator_method_ordered_in_context(
        &selected,
        Value::Num(2.0),
        rhs,
        "plus",
    ))
    .expect("selected session RHS overload");
    assert_eq!(value, Value::Num(12.0));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(std::sync::Weak::ptr_eq(
        &crate::context::legacy::active()
            .expect("wrong outer scope restored")
            .session_lease(),
        &wrong.session_lease()
    ));
}

#[test]
fn dependent_property_setter_uses_selected_session_not_unrelated_ambient_session() {
    let wrong = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _wrong_activation = wrong.enter();
    let selected = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let expected_session = selected.session_lease();
    let selected_setup = selected.enter();
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "SelectedSessionProperty.set.p").then_some(941)
        })))
        .expect("call-port scope");
    let class = ClassIdentity::from("SelectedSessionProperty");
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
                get_access: runmat_types::MemberAccess::Public,
                set_access: runmat_types::MemberAccess::Public,
                default_value: None,
            },
        )]),
        methods: std::collections::HashMap::from([(
            runmat_types::MethodName::from("set.p"),
            crate::class_registry::RuntimeMethod {
                name: "set.p".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: runmat_types::MemberAccess::Public,
                function_name: "SelectedSessionProperty.set.p".into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        )]),
    });
    let calls = Arc::new(AtomicUsize::new(0));
    let observed_calls = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, args, requested_outputs| {
            assert_eq!(function, 941);
            assert_eq!(requested_outputs, 1);
            let [Value::Object(original), Value::Num(9.0)] = args else {
                panic!("dependent setter receiver and value")
            };
            let active = crate::context::legacy::active().expect("selected setter scope");
            assert!(std::sync::Weak::ptr_eq(
                &active.session_lease(),
                &expected_session
            ));
            observed_calls.fetch_add(1, Ordering::SeqCst);
            let mut replacement = original
                .clone()
                .into_unbounded_builder()
                .expect("ordinary selected receiver fixture");
            replacement.insert("p_backing".into(), Value::Num(9.0));
            let replacement = replacement.finish();
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Object(
                    replacement,
                ))
            })
        },
    )))
    .expect("call-port scope");
    drop(selected_setup);
    let original = Value::Object(runmat_value::ObjectInstance::new(class));
    let updated = block_on(crate::object::resolve::store_member_traced_in_context(
        &selected,
        original,
        "p".into(),
        Value::Num(9.0),
        None,
    ))
    .expect("selected dependent setter");
    let Value::Object(updated) = updated.value() else {
        panic!("dependent setter returns replacement object")
    };
    assert_eq!(updated.property("p_backing"), Some(&Value::Num(9.0)));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
