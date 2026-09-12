#[test]
fn runtime_and_static_constant_identity_sets_match() {
    let runtime = runmat_builtins::constants()
        .into_iter()
        .map(|constant| constant.name)
        .collect::<std::collections::BTreeSet<_>>();
    let catalog = runmat_builtins::builtin_constant_catalog_entries()
        .iter()
        .map(|constant| constant.name)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(runtime, catalog);
}
