//! Read-only projections of the two registration mechanisms during cutover.
//! No dispatch, runtime context, alternate inventory, or cached authority lives here.

use runmat_builtins::{
    BuiltinCatalogAudience, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinExtensionDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerCapabilityDescriptor,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum IntegerMetadataAudience {
    Catalog(BuiltinCatalogAudience),
    LegacyRegistration,
}

pub(super) struct IntegerMetadata {
    pub name: &'static str,
    pub audience: IntegerMetadataAudience,
    pub descriptor: Option<&'static BuiltinDescriptor>,
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
            audience: IntegerMetadataAudience::Catalog(entry.audience),
            descriptor: Some(entry.descriptor),
            integer_capabilities: entry.integer_capabilities,
            integer_audit: entry.integer_audit,
            extensions: entry.extensions,
        })
        .chain(
            runmat_builtins::builtin_functions()
                .into_iter()
                .map(|entry| IntegerMetadata {
                    name: entry.name,
                    audience: IntegerMetadataAudience::LegacyRegistration,
                    descriptor: entry.descriptor,
                    integer_capabilities: entry.integer_capabilities,
                    integer_audit: entry.integer_audit,
                    extensions: entry.extensions,
                }),
        )
        .collect()
}

/// Metadata represented by the default public descriptor export. The raw
/// inventory above remains available to registry and duplicate-consistency
/// tests; this projection is only for comparisons with the public artifact.
pub(super) fn public_completion_metadata() -> Vec<IntegerMetadata> {
    integer_metadata()
        .into_iter()
        .filter(|metadata| {
            let public_audience = matches!(
                metadata.audience,
                IntegerMetadataAudience::Catalog(BuiltinCatalogAudience::Public)
                    | IntegerMetadataAudience::LegacyRegistration
            );
            public_audience
                && metadata.descriptor.is_some_and(|descriptor| {
                    descriptor.completion_policy == BuiltinCompletionPolicy::Public
                })
        })
        .collect()
}

/// Complete non-fixture inventory, including method and hidden-internal rows.
pub(super) fn non_fixture_metadata() -> Vec<IntegerMetadata> {
    integer_metadata()
        .into_iter()
        .filter(|metadata| {
            !matches!(
                metadata.audience,
                IntegerMetadataAudience::Catalog(BuiltinCatalogAudience::Fixture)
            )
        })
        .collect()
}

/// Language-facing audit rows preserve method completions while excluding
/// executable fixtures and hidden internals. Raw rows remain independently
/// validated through [`integer_metadata`].
pub(super) fn reviewed_audit_metadata() -> Vec<IntegerMetadata> {
    non_fixture_metadata()
        .into_iter()
        .filter(|metadata| {
            metadata.descriptor.is_some_and(|descriptor| {
                descriptor.completion_policy != BuiltinCompletionPolicy::HiddenInternal
            })
        })
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
            actual.audience,
            IntegerMetadataAudience::Catalog(entry.audience)
        );
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
        assert_eq!(actual.audience, IntegerMetadataAudience::LegacyRegistration);
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
fn public_completion_projection_keeps_raw_fixture_and_method_inventory_separate() {
    let raw = integer_metadata();
    let public = public_completion_metadata();
    assert!(raw.len() >= public.len());
    assert!(public.iter().all(|metadata| {
        matches!(
            metadata.audience,
            IntegerMetadataAudience::Catalog(BuiltinCatalogAudience::Public)
                | IntegerMetadataAudience::LegacyRegistration
        ) && metadata.descriptor.is_some_and(|descriptor| {
            descriptor.completion_policy == BuiltinCompletionPolicy::Public
        })
    }));
    for fixture in runmat_builtins::builtin_catalog_entries()
        .iter()
        .filter(|entry| entry.audience == BuiltinCatalogAudience::Fixture)
    {
        assert!(raw
            .iter()
            .any(|metadata| metadata.name == fixture.identity.name));
        assert!(!public
            .iter()
            .any(|metadata| metadata.name == fixture.identity.name));
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
