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
    let active = help.active_signature.expect("context-selected signature") as usize;
    assert!(
        active < expected.len(),
        "canonical active signature for {name}"
    );
    for (actual, canonical) in help.signatures.iter().zip(entry.descriptor.signatures) {
        assert_eq!(
            actual
                .parameters
                .as_ref()
                .expect("canonical parameters")
                .len(),
            canonical.inputs.len(),
            "complete parameter projection for {name}"
        );
    }
    expected
}
