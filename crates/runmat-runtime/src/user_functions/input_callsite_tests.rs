use super::*;
use crate::callsite::{current_function_input_callsite, CallsiteInfo};
use runmat_types::{Span, SymbolName};
use std::cell::Cell;
use std::rc::Rc;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn dynamic_probe_returns_the_same_argument_owner_and_keeps_selected_outcomes_terminal() {
    use std::ops::ControlFlow;

    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _entry = runtime.enter();
    runtime.set_dynamic_function_loader(Some(Rc::new(
        |runtime, _, arguments, requested_outputs, phase, _| {
            assert_eq!(requested_outputs, 0);
            crate::execution::admit_dynamic_function_load_with(&runtime, move || async move {
                match phase {
                    DynamicFunctionLoadPhase::BeforeSemantic => ControlFlow::Continue(arguments),
                    DynamicFunctionLoadPhase::AfterSemantic => ControlFlow::Break(
                        crate::execution::RootedValueSequence::new(
                            runmat_value::ValueSequence::empty(),
                        )
                        .map_err(|error| error.into_runtime_error()),
                    ),
                    _ => ControlFlow::Break(Err(crate::build_runtime_error(
                        "selected source failed",
                    )
                    .with_identifier("RunMat:Test:SelectedSource")
                    .build())),
                }
            })
        },
    )));
    let text = String::from("original argument owner");
    let payload = text.as_ptr();
    let first = futures::executor::block_on(try_load_and_call_dynamic_function(
        "late".into(),
        crate::call::RootedFunctionArguments::ordinary(
            crate::execution::RootedValues::in_context(vec![Value::String(text.into())], &runtime)
                .expect("root the original argument"),
        ),
        0,
        DynamicFunctionLoadPhase::BeforeSemantic,
        None,
    ));
    let ControlFlow::Continue(arguments) = first else {
        panic!("missing source returns its original arguments");
    };
    assert_eq!(arguments.len(), 1);
    let Some(Value::String(text)) = arguments.first() else {
        panic!("original string argument retained");
    };
    assert_eq!(text.as_str().expect("nonmissing text").as_ptr(), payload);
    let selected = futures::executor::block_on(try_load_and_call_dynamic_function(
        "late".into(),
        arguments,
        0,
        DynamicFunctionLoadPhase::AfterSemantic,
        None,
    ));
    assert_eq!(
        selected
            .break_value()
            .expect("selected zero-output call")
            .expect("output")
            .value()
            .len(),
        0
    );
    let error = futures::executor::block_on(try_load_and_call_dynamic_function(
        "late".into(),
        crate::call::RootedFunctionArguments::ordinary(
            crate::execution::RootedValues::in_context(Vec::new(), &runtime)
                .expect("root the selected loader arguments"),
        ),
        0,
        DynamicFunctionLoadPhase::RegisteredConstructorSource(
            runmat_types::ClassIdentity::new("SelectedCtor").expect("class identity"),
        ),
        None,
    ))
    .break_value()
    .expect("selected error cannot fall through")
    .expect_err("selected source error");
    assert_eq!(error.identifier(), Some("RunMat:Test:SelectedSource"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn runtime_named_resolution_installs_caller_only_for_selected_user_function() {
    let _resolver = install_semantic_function_resolver(Some(Arc::new(|name| {
        (name == "late_user").then_some(17)
    })))
    .expect("call-port scope");
    let _invoker = install_semantic_function_invoker(Some(Arc::new(|function, _, _| {
        assert_eq!(function, 17);
        Box::pin(async {
            let callsite = current_function_input_callsite().expect("selected user caller facts");
            assert_eq!(callsite.source_id, Some(SourceId(13)));
            assert_eq!(callsite.arg_spans, vec![Span { start: 4, end: 9 }]);
            crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(17.0))
        })
    })))
    .expect("call-port scope");
    let request = CallableRequest::resolved(
        CallableIdentity::DynamicName(SymbolName("late_user".into())),
        CallableFallbackPolicy::RuntimeNameResolution,
        vec![Value::Num(1.0)],
        1,
    )
    .with_function_input_callsite(Some(CallsiteInfo {
        source_id: Some(SourceId(13)),
        arg_spans: vec![Span { start: 4, end: 9 }].into(),
    }));
    let result = futures::executor::block_on(try_call_semantic_descriptor(request))
        .expect("user function selected")
        .expect("user function completed");
    assert_eq!(result.as_slice(), &[Value::Num(17.0)]);
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
    })))
    .expect("call-port scope");
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
    let _scoped = install_scoped_name_target_resolver(Some(Rc::new(move |name| {
        (name == "private_helper").then(|| ScopedSemanticNameTarget::Retained(binding.clone()))
    })))
    .expect("call-port scope");
    let _numeric =
        install_local_semantic_function_invoker(std::rc::Rc::new(|_runtime, function, _, _| {
            assert_eq!(function, 17);
            crate::user_functions::admit_user_function_with(_runtime.as_ref(), move || async {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(22.0))
            })
        }))
        .expect("call-port scope");
    let _bound = install_local_bound_function_invoker(std::rc::Rc::new(
        move |bound_runtime, binding, _, _, _lexical_environment| {
            assert_eq!(binding.function, 17);
            let lease = binding.authority.local_lease(17).expect("selected lease");
            assert!(lease.matches(&leased));
            assert!(!lease.matches(&republished));
            crate::execution::admit_executable_sequence_with(
                &bound_runtime.clone(),
                move || async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        33.0,
                    ))
                },
            )
        },
    ))
    .expect("call-port scope");
    let named =
        futures::executor::block_on(try_call_semantic_function_by_name("private_helper", &[], 1))
            .expect("name route")
            .expect("leased name result");
    assert_eq!(named.as_slice(), &[Value::Num(33.0)]);
    let numeric = futures::executor::block_on(try_call_semantic_function(17, &[], 1))
        .expect("numeric route")
        .expect("published numeric result");
    assert_eq!(numeric.as_slice(), &[Value::Num(22.0)]);
    let unbound_closure = runmat_value::Closure::from_data(runmat_value::ClosureData {
        lexical_environment: None,
        function_name: "private_helper".into(),
        binding: None,
        source_role: runmat_types::FunctionSourceRole::Unknown,
        source_file: None,
        captures: Vec::new().into(),
    });
    let feval = futures::executor::block_on(crate::call_feval_sequence_async_with_outputs(
        Value::Closure(unbound_closure.clone()),
        &[],
        1,
    ))
    .expect("unbound feval keeps leased name selection");
    assert_eq!(feval.as_slice(), &[Value::Num(33.0)]);
    let mapped = futures::executor::block_on(
        crate::builtins::common::mapped_callable::MappedCallable::Closure(
            crate::call::descriptor::RetainedClosureValue::new(unbound_closure),
        )
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
    runtime.set_dynamic_function_loader(Some(Rc::new(
        move |runtime, _, arguments, _, phase, _| {
            if phase == DynamicFunctionLoadPhase::BeforeSemantic {
                observed.set(observed.get() + 1);
                crate::execution::admit_dynamic_function_load_with(&runtime, || async {
                    std::ops::ControlFlow::Break(
                        crate::user_functions::test_support::rooted_single_value_sequence(
                            Value::Num(44.0),
                        ),
                    )
                })
            } else {
                crate::execution::admit_dynamic_function_load_with(&runtime, move || async move {
                    std::ops::ControlFlow::Continue(arguments)
                })
            }
        },
    )));
    let _resolver = install_semantic_function_resolver(Some(Arc::new(|name| {
        matches!(name, "private_current" | "private_leased" | "public_name").then_some(17)
    })))
    .expect("call-port scope");
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
    let _scoped = install_scoped_name_target_resolver(Some(Rc::new(move |name| match name {
        "private_current" => Some(ScopedSemanticNameTarget::Published(
            runmat_types::FunctionId(17),
        )),
        "private_leased" => Some(ScopedSemanticNameTarget::Retained(binding.clone())),
        _ => None,
    })))
    .expect("call-port scope");
    let _numeric = install_local_semantic_function_invoker(Rc::new(|_runtime, _, _, _| {
        crate::user_functions::admit_user_function_with(_runtime.as_ref(), move || async {
            crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(33.0))
        })
    }))
    .expect("call-port scope");
    let _bound = install_local_bound_function_invoker(Rc::new(
        move |bound_runtime, binding, _, _, _lexical_environment| {
            assert!(binding
                .authority
                .local_lease(17)
                .expect("lease")
                .matches(&leased));
            crate::execution::admit_executable_sequence_with(
                &bound_runtime.clone(),
                move || async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        55.0,
                    ))
                },
            )
        },
    ))
    .expect("call-port scope");
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
        assert_eq!(result.as_slice(), &[Value::Num(expected)]);
        assert_eq!(loader_calls.get(), loader_count);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn registered_constructor_loads_only_an_unresolved_declared_source() {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _entry = runtime.enter();
    let phases = Rc::new(std::cell::RefCell::new(Vec::new()));
    let observed = Rc::clone(&phases);
    runtime.set_dynamic_function_loader(Some(Rc::new(move |runtime, name, _, _, phase, _| {
        assert!(matches!(name.as_str(), "SourceCtor" | "pkg.SourceCtor"));
        observed.borrow_mut().push(phase.clone());
        crate::execution::admit_dynamic_function_load_with(&runtime, || async {
            std::ops::ControlFlow::Break(
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(7.0)),
            )
        })
    })));
    let register = |name: &str, constructor: bool| {
        let identity = runmat_types::ClassIdentity::new(name).expect("class identity");
        let leaf = identity
            .qualified_name()
            .0
            .last()
            .expect("class leaf")
            .0
            .clone();
        let mut methods = std::collections::HashMap::new();
        if constructor {
            methods.insert(
                runmat_types::MethodName::from(leaf.as_str()),
                crate::class_registry::RuntimeMethod {
                    name: runmat_types::MethodName::from(leaf.as_str()),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: runmat_types::MemberAccess::Public,
                    function_name: format!("{name}.{leaf}"),
                    implicit_class_argument: None,
                    role: runmat_types::ClassMethodRole::Constructor,
                },
            );
        }
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: identity,
            parent: None,
            properties: std::collections::HashMap::new(),
            methods,
            events: std::collections::HashMap::new(),
        });
    };
    register("DefaultCtor", false);
    let request = CallableRequest::resolved(
        CallableIdentity::DynamicName(SymbolName("DefaultCtor".into())),
        CallableFallbackPolicy::RuntimeNameResolution,
        Vec::new(),
        1,
    );
    assert!(futures::executor::block_on(try_call_semantic_descriptor(request)).is_none());
    assert!(
        phases.borrow().is_empty(),
        "default classes must not consult source"
    );

    register("SourceCtor", true);
    let request = CallableRequest::resolved(
        CallableIdentity::DynamicName(SymbolName("SourceCtor".into())),
        CallableFallbackPolicy::RuntimeNameResolution,
        Vec::new(),
        1,
    );
    let result = futures::executor::block_on(try_call_semantic_descriptor(request))
        .expect("source constructor selected")
        .expect("source constructor completed");
    assert_eq!(result.as_slice(), &[Value::Num(7.0)]);
    assert_eq!(
        phases.borrow().as_slice(),
        &[DynamicFunctionLoadPhase::RegisteredConstructorSource(
            runmat_types::ClassIdentity::new("SourceCtor").expect("class identity")
        )]
    );

    register("pkg.SourceCtor", true);
    let request = CallableRequest::resolved(
        crate::call::identity::external_qualified_identity("pkg", "SourceCtor"),
        CallableFallbackPolicy::ExternalBoundary,
        Vec::new(),
        1,
    );
    let result = futures::executor::block_on(try_call_semantic_descriptor(request))
        .expect("package source constructor selected")
        .expect("package source constructor completed");
    assert_eq!(result.as_slice(), &[Value::Num(7.0)]);
    assert_eq!(
        phases.borrow().last(),
        Some(&DynamicFunctionLoadPhase::RegisteredConstructorSource(
            runmat_types::ClassIdentity::new("pkg.SourceCtor").expect("class identity")
        ))
    );

    let semantic = install_semantic_function_resolver(Some(Arc::new(|name| {
        (name == "SourceCtor.SourceCtor").then_some(13)
    })))
    .expect("call-port scope");
    let request = CallableRequest::resolved(
        CallableIdentity::DynamicName(SymbolName("SourceCtor".into())),
        CallableFallbackPolicy::RuntimeNameResolution,
        Vec::new(),
        1,
    );
    assert!(futures::executor::block_on(try_call_semantic_descriptor(request)).is_none());
    assert_eq!(
        phases.borrow().len(),
        2,
        "semantic constructor retains priority"
    );
    drop(semantic);
    runtime.set_dynamic_function_loader(None);
    let request = CallableRequest::resolved(
        CallableIdentity::DynamicName(SymbolName("SourceCtor".into())),
        CallableFallbackPolicy::RuntimeNameResolution,
        Vec::new(),
        1,
    );
    assert!(futures::executor::block_on(try_call_semantic_descriptor(request)).is_none());
}
