use sha2::{Digest, Sha256};

use super::{migration_inventory, migration_inventory_json, MIGRATION_INVENTORY_SCHEMA_VERSION};

#[test]
fn compiled_inventory_is_deterministic_and_self_digesting() {
    let first = migration_inventory_json().expect("first inventory");
    let second = migration_inventory_json().expect("second inventory");
    assert_eq!(first, second);

    let inventory = migration_inventory();
    assert_eq!(inventory.schema_version, MIGRATION_INVENTORY_SCHEMA_VERSION);
    let encoded = serde_json::to_vec(&inventory.snapshot).expect("snapshot serialization");
    assert_eq!(inventory.digest.value, hex(&Sha256::digest(encoded)));
}

#[test]
fn compiled_inventory_preserves_runtime_registration_provenance() {
    let inventory = migration_inventory();

    let name = "struct";
    let catalog_provenance = inventory
        .snapshot
        .declared
        .catalog_provenance
        .iter()
        .find(|record| record.identity.builtin.name == name && record.identity.variant == "default")
        .unwrap_or_else(|| panic!("missing catalog provenance for {name}"));
    assert!(catalog_provenance.provenance.source_file.contains(name));
    assert!(catalog_provenance.provenance.module_path.contains(name));

    let provenance = inventory
        .snapshot
        .observed
        .implementation_provenance
        .iter()
        .find(|record| record.name == name)
        .unwrap_or_else(|| panic!("missing provenance for {name}"));
    assert_eq!(provenance.authority, "canonical_binding");
    assert_eq!(provenance.binding_variant, Some("default"));
    assert!(provenance.source_file.contains(name));
    assert!(provenance.module_path.contains(name));
    assert!(provenance.function.ends_with("_builtin"));
    assert!(provenance.builtin_path.contains(name));

    let binding = inventory
        .snapshot
        .observed
        .runtime_bindings
        .iter()
        .find(|binding| binding.name == name)
        .unwrap_or_else(|| panic!("missing binding for {name}"));
    assert_eq!(binding.variant, "default");
    assert_eq!(
        binding.native_symbol,
        runmat_builtins::native_binding_symbol(name, "default")
    );

    let constant = inventory
        .snapshot
        .observed
        .runtime_constants
        .iter()
        .find(|constant| constant.name == "pi")
        .expect("missing pi constant registration");
    assert!(constant.source_file.ends_with("builtins/constants/mod.rs"));
    assert!(constant.module_path.ends_with("builtins::constants"));
    assert_eq!(constant.builtin_path, "crate::builtins::constants");
    let readiness = &inventory.snapshot.validation.migration_readiness;
    assert_eq!(
        readiness.status,
        if readiness.findings.is_empty() {
            "ready"
        } else {
            "incomplete"
        }
    );
}

#[test]
fn catalog_provenance_is_not_part_of_the_semantic_fingerprint() {
    let entry =
        *runmat_builtins::builtin_catalog_entry_by_name("struct").expect("struct catalog entry");
    let mut relocated = entry;
    relocated.provenance = runmat_builtins::BuiltinCatalogProvenance::new(
        "different/physical/location.rs",
        "different::module",
    );
    assert_eq!(
        runmat_builtins::canonical_catalog_fingerprint(&[&entry], &[])
            .expect("original fingerprint"),
        runmat_builtins::canonical_catalog_fingerprint(&[&relocated], &[])
            .expect("relocated fingerprint")
    );
    let encoded = serde_json::to_value(entry).expect("catalog entry serialization");
    assert!(encoded.get("provenance").is_none());
}

#[test]
fn build_feature_inventory_covers_the_runtime_manifest() {
    let inventory = migration_inventory();
    let features = &inventory.snapshot.build.crate_feature_inventory;
    assert_eq!(features.crate_name, "runmat-runtime");
    assert_eq!(features.schema_version, 1);

    let manifest = include_str!("../../../Cargo.toml");
    let feature_section = manifest
        .split_once("[features]")
        .expect("runtime manifest features")
        .1
        .split_once("[dev-dependencies]")
        .expect("runtime manifest dev dependencies")
        .0;
    let declared = feature_section
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('"') {
                return None;
            }
            line.split_once('=')
                .map(|(name, _)| name.trim())
                .filter(|name| !name.is_empty() && *name != "default")
        })
        .collect::<std::collections::BTreeSet<_>>();
    let known = features
        .known_features
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(known, declared);
    assert!(features
        .enabled_features
        .iter()
        .all(|feature| known.contains(feature)));
}

#[test]
fn catalog_provenance_covers_every_canonical_binding_identity() {
    let inventory = migration_inventory();
    let expected = runmat_builtins::builtin_catalog_entries()
        .iter()
        .flat_map(|entry| {
            entry
                .bindings
                .iter()
                .map(|binding| entry.binding_identity(binding))
        })
        .collect::<std::collections::BTreeSet<_>>();
    let observed = inventory
        .snapshot
        .declared
        .catalog_provenance
        .iter()
        .map(|record| record.identity)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(observed, expected);
    assert_eq!(
        inventory.snapshot.declared.catalog_provenance.len(),
        expected.len()
    );
    assert!(inventory
        .snapshot
        .declared
        .catalog_provenance
        .iter()
        .all(|record| !record.provenance.source_file.is_empty()
            && !record.provenance.module_path.is_empty()));
}

#[test]
fn descriptorless_legacy_functions_remain_in_the_snapshot() {
    let inventory = migration_inventory();
    let expected = runmat_builtins::builtin_functions()
        .into_iter()
        .filter(|function| function.descriptor.is_none())
        .map(|function| function.name)
        .collect::<std::collections::BTreeSet<_>>();
    let observed = inventory
        .snapshot
        .declared
        .legacy_functions
        .iter()
        .filter(|function| function.descriptor.is_none())
        .map(|function| function.name)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(observed, expected);
}

#[test]
fn legacy_semantic_authority_matches_resolution_precedence() {
    let inventory = migration_inventory();
    for function in &inventory.snapshot.declared.legacy_functions {
        let expected = if runmat_builtins::builtin_catalog_entry_by_name(function.name).is_some() {
            super::LegacySemanticAuthority::Catalog
        } else if runmat_builtins::builtin_semantics_for_name(function.name).is_some() {
            super::LegacySemanticAuthority::NameTable
        } else {
            super::LegacySemanticAuthority::Derived
        };
        assert_eq!(function.semantic_authority, expected, "{}", function.name);
    }
}

#[test]
fn provider_spec_ownership_distinguishes_exact_identities_from_legacy_keys() {
    let classifier = super::projection::SpecOwnershipClassifier::from_exact_builtin_names([
        "exact_builtin",
        "data.create",
        "data.open",
    ])
    .with_builtin_ownership("builtins::io::data", ["data.create", "data.open"]);
    assert!(matches!(
        classifier.owner("exact_builtin", "crate::builtins::fixture"),
        super::SpecOwnerRecord::ExactBuiltin { identity } if identity.name == "exact_builtin"
    ));
    assert!(matches!(
        classifier.owner("data.*", "crate::builtins::io::data"),
        super::SpecOwnerRecord::LegacyGroup { raw, affected_identities }
            if raw == "data.*"
                && affected_identities.iter().map(|identity| identity.name).collect::<Vec<_>>()
                    == ["data.create", "data.open"]
    ));
}

#[test]
fn legacy_provider_owner_records_exact_compiled_identity_membership() {
    let inventory = migration_inventory();
    let callables = runmat_builtins::builtin_catalog_entries()
        .iter()
        .map(|entry| entry.identity.name)
        .chain(
            runmat_builtins::builtin_functions()
                .into_iter()
                .map(|function| function.name),
        )
        .collect::<std::collections::BTreeSet<_>>();
    let owner_key = |source: &'static str, owner: &super::SpecOwnerRecord| match owner {
        super::SpecOwnerRecord::LegacyGroup {
            raw,
            affected_identities,
        } => {
            assert!(!affected_identities.is_empty(), "{source}:{raw}");
            assert!(affected_identities
                .iter()
                .all(|identity| callables.contains(identity.name)));
            assert!(affected_identities
                .windows(2)
                .all(|pair| pair[0].name < pair[1].name));
            Some((
                source,
                *raw,
                affected_identities
                    .iter()
                    .map(|identity| identity.name)
                    .collect::<Vec<_>>(),
            ))
        }
        super::SpecOwnerRecord::ExactBuiltin { .. } => None,
    };
    let mut expected = inventory
        .snapshot
        .observed
        .gpu_specs
        .iter()
        .filter_map(|spec| owner_key("gpu_spec_registry", &spec.owner))
        .chain(
            inventory
                .snapshot
                .observed
                .fusion_specs
                .iter()
                .filter_map(|spec| owner_key("fusion_spec_registry", &spec.owner)),
        )
        .collect::<Vec<_>>();
    expected.sort_unstable();
    assert!(!expected.is_empty());

    let mut observed = inventory
        .snapshot
        .validation
        .migration_readiness
        .findings
        .iter()
        .filter_map(|finding| match &finding.affected {
            super::MigrationFindingAffected::Owner { owner } => owner_key(finding.source, owner),
            _ => None,
        })
        .collect::<Vec<_>>();
    observed.sort_unstable();
    assert_eq!(observed, expected);
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}
