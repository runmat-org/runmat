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
    let resolver = ExecutionSourceFacts::new(
        Rc::new(move |function| (function == 41).then(|| by_function.clone())),
        Rc::new(move |source| {
            (source == SourceId(3))
                .then(|| by_source.clone())
                .into_iter()
                .collect()
        }),
        Rc::new(move |function| (function == 41).then(|| Arc::clone(&owners))),
        Rc::new(move |candidate| (candidate == origin).then(|| by_origin.clone())),
    );
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
    let mismatched = ExecutionSourceFacts::new(
        Rc::new(|_| None),
        Rc::new(|_| Vec::new()),
        Rc::new(|_| None),
        Rc::new(move |_| Some(wrong_info.clone())),
    );
    assert!(resolve_binding_origin_binding_in_context(&mismatched, origin, &selected).is_none());
}
