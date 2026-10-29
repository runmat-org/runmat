//! LSP projection witnesses consume the canonical catalog, never executor links.
pub fn assert_catalog_signatures(name: &str, help: &lsp_types::SignatureHelp) -> Vec<&'static str> {
    let entry = runmat_builtins::public_builtin_catalog_entry_by_name(name)
        .unwrap_or_else(|| panic!("public canonical builtin {name}"));
    let expected = entry
        .descriptor
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect::<Vec<_>>();
    assert!(!expected.is_empty(), "canonical signatures for {name}");
    let actual = help
        .signatures
        .iter()
        .map(|signature| signature.label.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        actual, expected,
        "complete canonical signature projection for {name}"
    );
    assert_eq!(help.active_signature, Some(0));
    expected
}
