use super::*;
use crate::callsite::{current_function_input_callsite, CallsiteInfo};
use runmat_types::{Span, SymbolName};
use std::cell::Cell;
use std::rc::Rc;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn runtime_named_resolution_installs_caller_only_for_selected_user_function() {
    let _resolver = install_semantic_function_resolver(Some(Arc::new(|name| {
        (name == "late_user").then_some(17)
    })));
    let _invoker = install_semantic_function_invoker(Some(Arc::new(|function, _, _| {
        assert_eq!(function, 17);
        Box::pin(async {
            let callsite = current_function_input_callsite().expect("selected user caller facts");
            assert_eq!(callsite.source_id, Some(SourceId(13)));
            assert_eq!(callsite.arg_spans, vec![Span { start: 4, end: 9 }]);
            crate::sequence::single_value_sequence(Value::Num(17.0))
        })
    })));
    let request = CallableRequest::resolved(
        CallableIdentity::DynamicName(SymbolName("late_user".into())),
        CallableFallbackPolicy::RuntimeNameResolution,
        vec![Value::Num(1.0)],
        1,
    )
    .with_function_input_callsite(Some(CallsiteInfo {
        source_id: Some(SourceId(13)),
        arg_spans: vec![Span { start: 4, end: 9 }],
    }));
    let result = futures::executor::block_on(try_call_semantic_descriptor(request))
        .expect("user function selected")
        .expect("user function completed");
    assert_eq!(result.into_values(), vec![Value::Num(17.0)]);
    assert!(current_function_input_callsite().is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn scoped_name_binding_does_not_redirect_same_numbered_numeric_call() {
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _entry = runtime.enter();
    let _resolver = install_semantic_function_resolver(Some(Arc::new(|name| {
        (name == "private_helper").then_some(17)
    })));
    let leased = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let republished = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let binding = runmat_value::BoundFunctionBinding {
        function: 17,
        authority: runmat_value::BoundFunctionAuthority::local(
            runtime.session_lease(),
            runmat_value::FunctionDefinitionOwner::issue(&leased, 17),
            Vec::new(),
            runmat_value::LocalExport::Ephemeral,
        ),
    };
    let _scoped = install_scoped_name_target_resolver(Some(Arc::new(move |name| {
        (name == "private_helper").then(|| ScopedSemanticNameTarget::Retained(binding.clone()))
    })));
    let _numeric = install_local_semantic_function_invoker(std::rc::Rc::new(|function, _, _| {
        assert_eq!(function, 17);
        Box::pin(async { crate::sequence::single_value_sequence(Value::Num(22.0)) })
    }));
    let _bound = install_local_bound_function_invoker(std::rc::Rc::new(move |_, binding, _, _| {
        assert_eq!(binding.function, 17);
        let lease = binding.authority.local_lease(17).expect("selected lease");
        assert!(lease.matches(&leased));
        assert!(!lease.matches(&republished));
        Box::pin(async { crate::sequence::single_value_sequence(Value::Num(33.0)) })
    }));
    let named =
        futures::executor::block_on(try_call_semantic_function_by_name("private_helper", &[], 1))
            .expect("name route")
            .expect("leased name result");
    assert_eq!(named.into_values(), vec![Value::Num(33.0)]);
    let numeric = futures::executor::block_on(try_call_semantic_function(17, &[], 1))
        .expect("numeric route")
        .expect("published numeric result");
    assert_eq!(numeric.into_values(), vec![Value::Num(22.0)]);
    let unbound_closure = runmat_value::Closure {
        function_name: "private_helper".into(),
        binding: None,
        source_role: runmat_types::FunctionSourceRole::Unknown,
        source_file: None,
        captures: Vec::new(),
    };
    let feval = futures::executor::block_on(crate::call_feval_sequence_async_with_outputs(
        Value::Closure(unbound_closure.clone()),
        &[],
        1,
    ))
    .expect("unbound feval keeps leased name selection");
    assert_eq!(feval.into_values(), vec![Value::Num(33.0)]);
    let mapped = futures::executor::block_on(
        crate::builtins::common::mapped_callable::MappedCallable::Closure(unbound_closure)
            .invoke_with_outputs(&[], 1),
    )
    .unwrap_or_else(|_| panic!("mapped unbound closure keeps leased name selection"));
    assert_eq!(mapped, Value::Num(33.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn only_caller_private_targets_preempt_before_semantic_loader() {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _entry = runtime.enter();
    let loader_calls = Rc::new(Cell::new(0));
    let observed = Rc::clone(&loader_calls);
    runtime.set_dynamic_function_loader(Some(Rc::new(move |_, _, _, _, phase, _| {
        if phase == DynamicFunctionLoadPhase::BeforeSemantic {
            observed.set(observed.get() + 1);
            Box::pin(async { Some(crate::sequence::single_value_sequence(Value::Num(44.0))) })
        } else {
            Box::pin(async { None })
        }
    })));
    let _resolver = install_semantic_function_resolver(Some(Arc::new(|name| {
        matches!(name, "private_current" | "private_leased" | "public_name").then_some(17)
    })));
    let leased = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let binding = runmat_value::BoundFunctionBinding {
        function: 17,
        authority: runmat_value::BoundFunctionAuthority::local(
            runtime.session_lease(),
            runmat_value::FunctionDefinitionOwner::issue(&leased, 17),
            Vec::new(),
            runmat_value::LocalExport::Ephemeral,
        ),
    };
    let _scoped = install_scoped_name_target_resolver(Some(Arc::new(move |name| match name {
        "private_current" => Some(ScopedSemanticNameTarget::Published(
            runmat_types::FunctionId(17),
        )),
        "private_leased" => Some(ScopedSemanticNameTarget::Retained(binding.clone())),
        _ => None,
    })));
    let _numeric = install_local_semantic_function_invoker(Rc::new(|_, _, _| {
        Box::pin(async { crate::sequence::single_value_sequence(Value::Num(33.0)) })
    }));
    let _bound = install_local_bound_function_invoker(Rc::new(move |_, binding, _, _| {
        assert!(binding
            .authority
            .local_lease(17)
            .expect("lease")
            .matches(&leased));
        Box::pin(async { crate::sequence::single_value_sequence(Value::Num(55.0)) })
    }));
    for (name, expected, loader_count) in [
        ("private_current", 33.0, 0),
        ("private_leased", 55.0, 0),
        ("public_name", 44.0, 1),
    ] {
        let request = CallableRequest::resolved(
            CallableIdentity::DynamicName(SymbolName(name.into())),
            CallableFallbackPolicy::RuntimeNameResolution,
            Vec::new(),
            1,
        );
        let result = futures::executor::block_on(try_call_semantic_descriptor(request))
            .expect("selected callable")
            .expect("call completed");
        assert_eq!(result.into_values(), vec![Value::Num(expected)]);
        assert_eq!(loader_calls.get(), loader_count);
    }
}
