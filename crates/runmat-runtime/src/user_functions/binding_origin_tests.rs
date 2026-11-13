use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn exact_origin_resolves_relocated_id_not_reused_numeric_id() {
    let original = runmat_execution::FunctionBindingOrigin {
        program: runmat_execution::Digest::sha256("original program"),
        function: runmat_types::ProgramFunctionId(3),
    };
    let unrelated = runmat_execution::FunctionBindingOrigin {
        program: runmat_execution::Digest::sha256("unrelated program"),
        function: runmat_types::ProgramFunctionId(3),
    };
    let catalog = Arc::new(vec![
        SourceFunctionInfo {
            source_id: SourceId(1),
            name: "original".into(),
            function: 77,
            role: runmat_types::FunctionSourceRole::Primary,
            binding_origin: Some(original),
            source_file: None,
            definition_owners: None,
        },
        SourceFunctionInfo {
            source_id: SourceId(2),
            name: "unrelated".into(),
            function: 3,
            role: runmat_types::FunctionSourceRole::Primary,
            binding_origin: Some(unrelated),
            source_file: None,
            definition_owners: None,
        },
    ]);
    let _catalog = install_source_function_catalog(Some(catalog)).expect("call-port scope");
    assert_eq!(resolve_binding_origin(original), Some(77));
    assert_eq!(resolve_binding_origin(unrelated), Some(3));
    assert_eq!(
        resolve_binding_origin(runmat_execution::FunctionBindingOrigin {
            program: original.program,
            function: runmat_types::ProgramFunctionId(4),
        }),
        None
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn exact_origin_binding_owns_the_selected_definition_graph() {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let origin = runmat_execution::FunctionBindingOrigin {
        program: runmat_execution::Digest::sha256("imported program"),
        function: runmat_types::ProgramFunctionId(4),
    };
    let root = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let helper = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let entry = runtime.enter();
    let catalog = install_source_function_catalog(Some(Arc::new(vec![SourceFunctionInfo {
        source_id: SourceId(7),
        name: "imported".into(),
        function: 41,
        role: runmat_types::FunctionSourceRole::Primary,
        binding_origin: Some(origin),
        source_file: Some("imported.m".into()),
        definition_owners: Some(Arc::new(vec![
            (41, Arc::clone(&root)),
            (42, Arc::clone(&helper)),
        ])),
    }])))
    .expect("call-port scope");

    let binding = resolve_binding_origin_binding(origin).expect("exact binding");
    let copied = binding.clone();
    drop(binding);
    drop(catalog);
    assert!(copied.authority.admits_local(&runtime.session_lease()));
    assert!(copied
        .authority
        .local_lease(41)
        .is_some_and(|lease| lease.matches(&root)));
    assert!(copied
        .authority
        .local_lease(42)
        .is_some_and(|lease| lease.matches(&helper)));
    assert!(root.has_handles());
    assert!(helper.has_handles());

    drop(entry);
    drop(runtime);
    let foreign = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let foreign_entry = foreign.enter();
    let calls = Rc::new(std::cell::Cell::new(0));
    let observed = Rc::clone(&calls);
    let invoker = install_local_bound_function_invoker(Rc::new(
        move |bound_runtime, _, _, _, _lexical_environment| {
            observed.set(observed.get() + 1);
            crate::execution::admit_executable_sequence_with(
                &bound_runtime.clone(),
                move || async move {
                    crate::execution::RootedValueSequence::new(runmat_value::ValueSequence::empty())
                        .map_err(|error| error.into_runtime_error())
                },
            )
        },
    ))
    .expect("call-port scope");
    let result = futures::executor::block_on(try_call_bound_function(copied.clone(), &[], 0))
        .expect("bound route is installed");
    assert!(
        result.is_err(),
        "foreign session must reject the local lease"
    );
    assert_eq!(calls.get(), 0, "foreign rejection must precede invocation");
    drop(invoker);
    drop(foreign_entry);
    drop(foreign);
    drop(copied);
    assert!(!root.has_handles());
    assert!(!helper.has_handles());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn selected_execution_facts_resolve_only_their_exact_origins() {
    let published_origin = runmat_execution::FunctionBindingOrigin {
        program: runmat_execution::Digest::sha256("published"),
        function: runmat_types::ProgramFunctionId(1),
    };
    let published = SourceFunctionInfo {
        source_id: SourceId(2),
        name: "same_id".into(),
        function: 9,
        role: runmat_types::FunctionSourceRole::Primary,
        binding_origin: Some(published_origin),
        source_file: Some("/new/same_id.m".into()),
        definition_owners: None,
    };
    let old_origin = runmat_execution::FunctionBindingOrigin {
        program: runmat_execution::Digest::sha256("retained"),
        function: runmat_types::ProgramFunctionId(1),
    };
    let old_owner = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let old = SourceFunctionInfo {
        source_id: SourceId(1),
        name: "same_id".into(),
        function: 9,
        role: runmat_types::FunctionSourceRole::Local,
        binding_origin: Some(old_origin),
        source_file: Some("/old/same_id.m".into()),
        definition_owners: Some(Arc::new(vec![(9, Arc::clone(&old_owner))])),
    };
    let _catalog =
        install_source_function_catalog(Some(Arc::new(vec![published]))).expect("call-port scope");
    let selected_old = old.clone();
    let old_group = old.clone();
    let old_by_origin = old.clone();
    let _execution = install_execution_source_resolver(Some(Rc::new(ExecutionSourceFacts::new(
        Rc::new(move |function| (function == 9).then(|| selected_old.clone())),
        Rc::new(move |source| {
            (source == SourceId(1))
                .then(|| old_group.clone())
                .into_iter()
                .collect()
        }),
        Rc::new(|_| None),
        Rc::new(move |origin| (origin == old_origin).then(|| old_by_origin.clone())),
    ))))
    .expect("call-port scope");
    assert_eq!(
        source_function_for(9).unwrap().role,
        runmat_types::FunctionSourceRole::Local
    );
    assert_eq!(source_file_for(9).as_deref(), Some("/old/same_id.m"));
    assert_eq!(resolve_binding_origin(old_origin), Some(9));
    assert_eq!(resolve_binding_origin(published_origin), None);
    assert!(source_function_for(10).is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn selected_old_definition_issues_old_owner_under_one_runtime() {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let old_owner = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let new_owner = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let old = SourceFunctionInfo {
        source_id: SourceId(1),
        name: "reused".into(),
        function: 9,
        role: runmat_types::FunctionSourceRole::Local,
        binding_origin: None,
        source_file: None,
        definition_owners: Some(Arc::new(vec![(9, Arc::clone(&old_owner))])),
    };
    let published = SourceFunctionInfo {
        source_id: SourceId(2),
        name: "reused".into(),
        function: 9,
        role: runmat_types::FunctionSourceRole::Primary,
        binding_origin: None,
        source_file: None,
        definition_owners: Some(Arc::new(vec![(9, new_owner)])),
    };
    let entry = runtime.enter();
    let catalog =
        install_source_function_catalog(Some(Arc::new(vec![published]))).expect("call-port scope");
    let old_group = old.clone();
    let execution = install_execution_source_resolver(Some(Rc::new(ExecutionSourceFacts::new(
        Rc::new(move |function| (function == 9).then(|| old.clone())),
        Rc::new(move |source| {
            (source == SourceId(1))
                .then(|| old_group.clone())
                .into_iter()
                .collect()
        }),
        Rc::new(|_| None),
        Rc::new(|_| None),
    ))))
    .expect("call-port scope");
    let Value::BoundFunctionHandle(handle) = bound_function_handle("reused".into(), 9) else {
        panic!("bound handle");
    };
    let authority = handle.into_data().authority;
    assert!(authority.admits_local(&runtime.session_lease()));
    assert!(authority
        .local_lease(9)
        .is_some_and(|lease| lease.matches(&old_owner)));
    drop(execution);
    drop(catalog);
    drop(entry);
    drop(runtime);
    assert!(!authority.admits_local(&std::sync::Weak::new()));
    let next = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _next_entry = next.enter();
    let calls = Rc::new(std::cell::Cell::new(0));
    let observed = Rc::clone(&calls);
    let _invoker = install_local_bound_function_invoker(Rc::new(
        move |bound_runtime, _, _, _, _lexical_environment| {
            observed.set(observed.get() + 1);
            crate::execution::admit_executable_sequence_with(
                &bound_runtime.clone(),
                move || async move {
                    crate::execution::RootedValueSequence::new(runmat_value::ValueSequence::empty())
                        .map_err(|error| error.into_runtime_error())
                },
            )
        },
    ))
    .expect("call-port scope");
    let result = futures::executor::block_on(try_call_bound_function(
        runmat_value::BoundFunctionBinding {
            function: 9,
            authority,
        },
        &[],
        0,
    ));
    assert!(result.expect("typed unavailable outcome").is_err());
    assert_eq!(
        calls.get(),
        0,
        "cross-session handle must fail before effects"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn legacy_entry_context_inherits_exact_bound_invoker() {
    let exact: Rc<BoundFunctionInvoker> = Rc::new(
        |runtime,
         _binding: runmat_value::BoundFunctionBinding,
         _args: crate::call::RootedFunctionArguments,
         _outputs: usize,
         _lexical_environment: Option<runmat_value::LexicalEnvironment>| {
            crate::execution::admit_executable_sequence_with(&runtime, move || async move {
                crate::execution::RootedValueSequence::new(runmat_value::ValueSequence::empty())
                    .map_err(|error| error.into_runtime_error())
            })
        },
    );
    let _guard = install_local_bound_function_invoker(exact).expect("call-port scope");
    let context = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    inherit_legacy_call_environment(&context);
    let _entry = context.enter();
    let origin = runmat_execution::FunctionBindingOrigin {
        program: runmat_execution::Digest::sha256("legacy exact route"),
        function: runmat_types::ProgramFunctionId(1),
    };
    let result = futures::executor::block_on(try_call_bound_function(
        runmat_value::BoundFunctionBinding::portable(9, origin),
        &[],
        0,
    ))
    .expect("inherited exact invoker")
    .expect("exact call succeeds");
    assert_eq!(result, runmat_value::ValueSequence::empty());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn selected_bound_future_roots_arguments_until_drop() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let binding = test_support::exact_local_binding(&runtime, 17);
        let reached = Rc::new(std::cell::Cell::new(false));
        let observed = Rc::clone(&reached);
        let entry = runtime.enter();
        let _invoker = install_local_bound_function_invoker(Rc::new(
            move |bound_runtime, _binding, arguments, _outputs, _lexical_environment| {
                observed.set(true);
                crate::execution::admit_executable_sequence_with(
                    &bound_runtime.clone(),
                    move || async move {
                        let _arguments = arguments;
                        std::future::pending::<()>().await;
                        crate::execution::RootedValueSequence::new(
                            runmat_value::ValueSequence::empty(),
                        )
                        .map_err(|error| error.into_runtime_error())
                    },
                )
            },
        ))
        .expect("call-port scope");
        drop(entry);

        let initial = runmat_gc::gc_allocate_rooted(Value::String("argument".into()))
            .expect("allocate argument target");
        let handle = initial.handle();
        let arguments = [Value::HandleObject(runmat_value::HandleRef {
            class_name: "BoundArgument".into(),
            target: handle,
            valid: true,
        })];
        let mut call = Box::pin(try_call_bound_function_in_context(
            &runtime, binding, &arguments, 0,
        ));
        let waker = futures::task::noop_waker();
        let mut context = std::task::Context::from_waker(&waker);
        assert!(matches!(
            std::future::Future::poll(call.as_mut(), &mut context),
            std::task::Poll::Pending
        ));
        assert!(
            reached.get(),
            "selected invoker must reach its pending body"
        );
        initial.unroot().expect("transfer root to invocation");
        runmat_gc::gc_collect_major().expect("collect pending invocation");
        assert_eq!(
            runmat_gc::gc_clone_value(&handle).expect("pending argument remains rooted"),
            Value::String("argument".into())
        );

        drop(call);
        runmat_gc::gc_collect_major().expect("collect dropped invocation");
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}
