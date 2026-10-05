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
        },
        SourceFunctionInfo {
            source_id: SourceId(2),
            name: "unrelated".into(),
            function: 3,
            role: runmat_types::FunctionSourceRole::Primary,
            binding_origin: Some(unrelated),
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
