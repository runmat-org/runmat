//! Candidate integration test for `crates/runmat-runtime/tests/search_engine_binding.rs`.
//! One linked runtime library: catalog identity, compiled binding and canonical
//! provenance must agree before engine examples can receive identity credit.
use runmat_builtins::{
    builtin_catalog_entry_by_name, with_builtin_implementation_provenance_by_name,
    BuiltinImplementationAuthority,
};
use runmat_runtime::builtin::runtime_builtin_bindings_by_name;

#[test]
fn search_quartet_has_one_exact_compiled_default_binding_each() {
    for name in ["contains", "endsWith", "startsWith", "strfind"] {
        let entry = builtin_catalog_entry_by_name(name).expect("canonical catalog entry");
        assert_eq!(
            entry.bindings.len(),
            1,
            "{name}: catalog binding cardinality"
        );
        let expected = entry.binding_identity(&entry.bindings[0]);
        assert_eq!(expected.variant, "default");

        let compiled = runtime_builtin_bindings_by_name(name);
        assert_eq!(compiled.len(), 1, "{name}: compiled binding cardinality");
        assert_eq!(compiled[0].identity, expected, "{name}: exact binding join");

        let owners = with_builtin_implementation_provenance_by_name(name, |all| all.to_vec());
        assert_eq!(
            owners.len(),
            1,
            "{name}: no duplicate or legacy implementation owner"
        );
        assert_eq!(
            owners[0].authority,
            BuiltinImplementationAuthority::CanonicalBinding
        );
        assert_eq!(owners[0].binding_variant, Some("default"));
    }
}
