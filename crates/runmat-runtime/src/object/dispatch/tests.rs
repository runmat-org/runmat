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
        })));
    let class = ClassIdentity::from("RhsFailureFixture");
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
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
    )));

    for code in ["UndefinedFunction", "ProbeReceiverInvalid"] {
        *diagnostic.lock().expect("diagnostic fixture lock") = code;
        calls.store(0, Ordering::SeqCst);
        let rhs = Value::Object(runmat_value::ObjectInstance::new(class.clone()));
        let err = block_on(call_rhs_operator_method_ordered_with_outputs(
            Value::Num(2.0),
            rhs,
            "plus".into(),
            1,
            None,
        ))
        .expect_err("selected callee error must propagate");
        assert_eq!(err.identifier(), Some(format!("RunMat:{code}").as_str()));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
