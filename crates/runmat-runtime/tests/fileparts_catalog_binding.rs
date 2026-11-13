//! The compiled runtime must join the catalog without a second legacy authority.
use futures::executor::block_on;
use runmat_builtins::FILEPARTS_CATALOG_ENTRY;
use runmat_runtime::call_builtin_async_with_outputs;
use runmat_value::{CharArray, Value};

#[test]
fn fileparts_has_exactly_its_required_canonical_binding() {
    let entry = &FILEPARTS_CATALOG_ENTRY;
    let bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name("fileparts");
    assert_eq!(entry.bindings.len(), 1);
    assert_eq!(bindings.len(), 1);
    assert_eq!(
        bindings[0].identity,
        entry.binding_identity(&entry.bindings[0])
    );
    assert!(runmat_builtins::builtin_function_by_name("fileparts").is_none());
    let errors = runmat_runtime::builtin::validate_runtime_builtin_registry()
        .into_iter()
        .filter(|error| error.identity == "fileparts" || error.identity.starts_with("fileparts#"))
        .collect::<Vec<_>>();
    assert!(errors.is_empty(), "{errors:#?}");
}

#[test]
fn fileparts_dispatch_preserves_requested_outputs_and_text_admission() {
    let text = |value: &str| Value::CharArray(CharArray::new_row(value));
    let path = format!("folder{}archive.part.tar", std::path::MAIN_SEPARATOR);
    let expected = [text("folder"), text("archive.part"), text(".tar")];
    for count in 0..=3 {
        let result = block_on(call_builtin_async_with_outputs(
            "fileparts",
            &[text(&path)],
            count,
        ))
        .unwrap();
        assert_eq!(result, Value::output_list(expected[..count].to_vec()));
    }
    for count in [0, 3] {
        let error = block_on(call_builtin_async_with_outputs(
            "fileparts",
            &[Value::Num(1.0)],
            count,
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:fileparts:InvalidFilename"));
        let error = block_on(call_builtin_async_with_outputs("fileparts", &[], count)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:fileparts:InvalidArity"));
    }
}
