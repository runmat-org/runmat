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
            definition_owners: None,
        },
        SourceFunctionInfo {
            source_id: SourceId(2),
            name: "unrelated".into(),
            function: 3,
            role: runmat_types::FunctionSourceRole::Primary,
            binding_origin: Some(unrelated),
            definition_owners: None,
        },
    ]);
    let _catalog = install_source_function_catalog(Some(catalog));
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
fn retained_execution_facts_do_not_replace_published_portable_resolution() {
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
        definition_owners: None,
    };
    let old_owner = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let old = SourceFunctionInfo {
        source_id: SourceId(1),
        name: "same_id".into(),
        function: 9,
        role: runmat_types::FunctionSourceRole::Local,
        binding_origin: None,
        definition_owners: Some(vec![(9, Arc::clone(&old_owner))]),
    };
    let _catalog = install_source_function_catalog(Some(Arc::new(vec![published])));
    let _execution = install_execution_source_resolver(Some(Rc::new(move |function| {
        (function == 9).then(|| old.clone())
    })));
    assert_eq!(
        source_function_for(9).unwrap().role,
        runmat_types::FunctionSourceRole::Local
    );
    assert_eq!(resolve_binding_origin(published_origin), Some(9));
    assert!(source_function_for(10).is_none());
}
