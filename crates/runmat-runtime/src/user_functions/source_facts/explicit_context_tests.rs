use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn explicit_admitted_origin_uses_selected_issuer_not_foreign_ambient() {
    let selected = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let foreign = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let origin = runmat_execution::FunctionBindingOrigin {
        program: runmat_execution::Digest::sha256(b"selected executable"),
        function: runmat_types::ProgramFunctionId(4),
    };
    let absent = runmat_execution::FunctionBindingOrigin {
        program: origin.program,
        function: runmat_types::ProgramFunctionId(5),
    };
    let owner = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let info = SourceFunctionInfo {
        source_id: SourceId(3),
        name: "selected".into(),
        function: 41,
        role: runmat_types::FunctionSourceRole::Primary,
        binding_origin: Some(origin),
        source_file: Some("selected.m".into()),
        definition_owners: Some(Arc::new(vec![(41, Arc::clone(&owner))])),
    };
    let by_function = info.clone();
    let by_source = info.clone();
    let mut wrong_info = info.clone();
    wrong_info.binding_origin = Some(absent);
    let by_origin = info;
    let owners = Arc::new(vec![(41, Arc::clone(&owner))]);
    let resolver = Rc::new(ExecutionSourceFacts::new(
        Rc::new(move |function| (function == 41).then(|| by_function.clone())),
        Rc::new(move |source| {
            (source == SourceId(3))
                .then(|| by_source.clone())
                .into_iter()
                .collect()
        }),
        Rc::new(move |function| (function == 41).then(|| Arc::clone(&owners))),
        Rc::new(move |candidate| (candidate == origin).then(|| by_origin.clone())),
    ));
    // Keep a session-state selection underneath the invocation projection so
    // the explicit-absence check below also detects accidental state fallback.
    selected.state().call.borrow_mut().execution_source_resolver = Some(Rc::clone(&resolver));
    let _selected_facts = {
        let _selected_scope = selected.enter();
        install_execution_source_resolver(Some(Rc::clone(&resolver)))
            .expect("install selected invocation facts")
    };
    let _foreign_ambient = foreign.enter();
    let binding = resolve_binding_origin_binding_in_context(&resolver, origin, &selected)
        .expect("selected exact binding");
    assert_eq!(binding.function, 41);
    assert!(binding.authority.admits_local(&selected.session_lease()));
    assert!(!binding.authority.admits_local(&foreign.session_lease()));
    assert!(binding
        .authority
        .local_lease(41)
        .is_some_and(|lease| lease.matches(&owner)));
    assert!(resolve_binding_origin_binding_in_context(&resolver, absent, &selected).is_none());
    let runtime_binding = resolve_binding_origin_binding_from_runtime(&selected, origin)
        .expect("explicit runtime ignores the foreign ambient context");
    assert_eq!(runtime_binding, binding);
    assert!(resolve_binding_origin_binding_from_runtime(&selected, absent).is_none());
    assert!(resolve_binding_origin_binding_from_runtime(&foreign, origin).is_none());
    {
        let _selected_scope = selected.enter();
        assert!(resolve_binding_origin_binding_from_runtime(&foreign, origin).is_none());
        let _absent_projection =
            install_execution_source_resolver(None).expect("mask session-state facts");
        assert!(resolve_binding_origin_binding_from_runtime(&selected, origin).is_none());
    }
    assert_eq!(
        resolve_binding_origin_binding_from_runtime(&selected, origin),
        Some(binding)
    );
    // A scoped poll owns a projection derived from this original handle.
    // Exact lookup must see that invocation's facts without reading another
    // runtime or losing an explicit absence mask to session-state fallback.
    let polling = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    futures::executor::block_on(polling.scope(async {
        let _facts = install_execution_source_resolver(Some(Rc::clone(&resolver)))
            .expect("install facts on the current poll projection");
        let binding = resolve_binding_origin_binding_from_runtime(&polling, origin)
            .expect("original handle selects its own active poll projection");
        assert!(binding.authority.admits_local(&polling.session_lease()));
        let _mask = install_execution_source_resolver(None).expect("mask the current poll");
        assert!(resolve_binding_origin_binding_from_runtime(&polling, origin).is_none());
        assert!(resolve_binding_origin_binding_from_runtime(&foreign, origin).is_none());
    }));
    assert!(resolve_binding_origin_binding_from_runtime(&polling, origin).is_none());
    let mismatched = ExecutionSourceFacts::new(
        Rc::new(|_| None),
        Rc::new(|_| Vec::new()),
        Rc::new(|_| None),
        Rc::new(move |_| Some(wrong_info.clone())),
    );
    assert!(resolve_binding_origin_binding_in_context(&mismatched, origin, &selected).is_none());
}
