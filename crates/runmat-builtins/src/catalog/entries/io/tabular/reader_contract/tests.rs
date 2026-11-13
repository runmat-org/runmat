use crate::*;

#[test]
fn promoted_runtime_leaves_keep_focused_contracts_and_catalog_documentation_authority() {
    for (name, signatures, extensions, integer_forms) in [
        ("dataTipTextRow", 2, 2, 3),
        ("datacursormode", 3, 3, 3),
        ("pan", 5, 1, 2),
        ("zoom", 6, 1, 2),
        ("readtable", 2, 1, 3),
        ("readcell", 1, 1, 2),
        ("spreadsheetImportOptions", 2, 1, 2),
        ("detectImportOptions", 2, 1, 2),
    ] {
        let entry = builtin_catalog_primary_entry_by_name(name).expect("canonical leaf");
        assert_eq!(entry.identity.name, name);
        assert_eq!(entry.descriptor.signatures.len(), signatures, "{name}");
        assert_eq!(entry.extensions.len(), extensions, "{name}");
        assert_eq!(entry.integer_capabilities.len(), integer_forms, "{name}");
        assert!(!entry.descriptor.errors.is_empty(), "{name}");
        assert_eq!(entry.bindings.len(), 1, "{name}");
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog,
            "reviewed prose owners must retain canonical documentation authority: {name}"
        );
    }
    assert_eq!(READCELL_DESCRIPTOR.signatures[0].inputs[0].name, "filename");
    assert_eq!(
        READCELL_DESCRIPTOR.signatures[0].inputs[1].arity,
        BuiltinParamArity::Variadic
    );
    assert!(READCELL_DESCRIPTOR
        .errors
        .iter()
        .map(|error| error.code)
        .eq(TABLE_ERRORS.iter().map(|error| error.code)));
}
