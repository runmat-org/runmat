//! Read-only projections of the two registration mechanisms during cutover.
//! No dispatch, runtime context, alternate inventory, or cached authority lives here.

use runmat_builtins::{
    BuiltinExtensionDescriptor, BuiltinIntegerAuditDescriptor, BuiltinIntegerCapabilityDescriptor,
};

pub(super) struct IntegerMetadata {
    pub name: &'static str,
    pub integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    pub integer_audit: Option<&'static BuiltinIntegerAuditDescriptor>,
    pub extensions: &'static [BuiltinExtensionDescriptor],
}

pub(super) fn integer_metadata() -> Vec<IntegerMetadata> {
    // Canonical definitions win name lookups, but every legacy registration is
    // retained for the caller's per-definition invariants and duplicate audit
    // consistency checks. Do not silently hide stale legacy metadata.
    runmat_builtins::builtin_catalog_entries()
        .iter()
        .map(|entry| IntegerMetadata {
            name: entry.identity.name,
            integer_capabilities: entry.integer_capabilities,
            integer_audit: entry.integer_audit,
            extensions: entry.extensions,
        })
        .chain(
            runmat_builtins::builtin_functions()
                .into_iter()
                .map(|entry| IntegerMetadata {
                    name: entry.name,
                    integer_capabilities: entry.integer_capabilities,
                    integer_audit: entry.integer_audit,
                    extensions: entry.extensions,
                }),
        )
        .collect()
}

#[test]
fn projection_keeps_both_registration_mechanisms_without_dispatching() {
    let actual = integer_metadata();
    let canonical = runmat_builtins::builtin_catalog_entries();
    let legacy = runmat_builtins::builtin_functions();
    assert_eq!(actual.len(), canonical.len() + legacy.len());
    for (actual, entry) in actual.iter().zip(canonical.iter()) {
        assert_eq!(actual.name, entry.identity.name);
        assert_eq!(
            serde_json::to_value(actual.integer_capabilities).unwrap(),
            serde_json::to_value(entry.integer_capabilities).unwrap()
        );
        assert_eq!(actual.integer_audit, entry.integer_audit);
        assert_eq!(
            serde_json::to_value(actual.extensions).unwrap(),
            serde_json::to_value(entry.extensions).unwrap()
        );
    }
    for (actual, entry) in actual.iter().skip(canonical.len()).zip(legacy.iter()) {
        assert_eq!(actual.name, entry.name);
        assert_eq!(
            serde_json::to_value(actual.integer_capabilities).unwrap(),
            serde_json::to_value(entry.integer_capabilities).unwrap()
        );
        assert_eq!(actual.integer_audit, entry.integer_audit);
        assert_eq!(
            serde_json::to_value(actual.extensions).unwrap(),
            serde_json::to_value(entry.extensions).unwrap()
        );
    }
}

#[test]
fn duplicate_metadata_definitions_are_consistent() {
    let mut seen = std::collections::BTreeMap::new();
    for entry in integer_metadata() {
        let metadata = (
            serde_json::to_value(entry.integer_capabilities).unwrap(),
            serde_json::to_value(entry.integer_audit).unwrap(),
            serde_json::to_value(entry.extensions).unwrap(),
        );
        if let Some(previous) = seen.insert(entry.name, metadata.clone()) {
            assert_eq!(
                previous, metadata,
                "{} conflicting integer metadata",
                entry.name
            );
        }
    }
}
