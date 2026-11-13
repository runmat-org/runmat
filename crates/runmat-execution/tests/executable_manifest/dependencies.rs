use super::*;
use runmat_execution::{ExecutableDependencyProduct, EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES};

fn product(label: &str) -> ExecutableDependencyProduct {
    let mut manifest = manifest();
    manifest.identity.root_package = label.into();
    ExecutableDependencyProduct {
        manifest,
        payloads: component_payloads(),
    }
}

#[test]
fn original_inventory_is_canonical_complete_and_identity_bound() {
    let mut members = vec![product("original-a"), product("original-b")];
    members.sort_by_key(|p| p.descriptor().unwrap().identity);
    let envelope =
        ExecutableUnitEnvelope::with_dependencies(manifest(), component_payloads(), members)
            .unwrap();
    let bytes = envelope.canonical_bytes().unwrap();
    assert_eq!(
        ExecutableUnitEnvelope::from_canonical_bytes(&bytes).unwrap(),
        envelope
    );
    let baseline = ExecutableUnitEnvelope::new(manifest(), component_payloads()).unwrap();
    assert_ne!(envelope.cache_key().unwrap(), baseline.cache_key().unwrap());
    let mut missing = envelope.clone();
    missing.dependency_products.pop();
    assert!(missing.validate().is_err());
    let mut altered = envelope.clone();
    altered.dependency_products[0]
        .manifest
        .identity
        .root_package = "substitution".into();
    assert!(altered.validate().is_err());
    let mut corrupt = envelope.clone();
    corrupt.dependency_products[0].payloads[0].bytes[0] ^= 1;
    assert!(corrupt.validate().is_err());
    let mut reordered = envelope.clone();
    reordered.manifest.dependencies.reverse();
    reordered.dependency_products.reverse();
    assert!(reordered.validate().is_err());
    let mut duplicated = envelope.clone();
    duplicated.manifest.dependencies[1] = duplicated.manifest.dependencies[0].clone();
    duplicated.dependency_products[1] = duplicated.dependency_products[0].clone();
    assert!(duplicated.validate().is_err());
    let mut nested = product("nested");
    nested.manifest.dependencies = envelope.manifest.dependencies.clone();
    assert!(nested.validate().is_err());
}

#[test]
fn old_schema_missing_inventory_and_excessive_budget_reject() {
    let envelope = ExecutableUnitEnvelope::new(manifest(), component_payloads()).unwrap();
    let mut legacy = envelope.clone();
    legacy.schema_version = 3;
    assert!(legacy.validate().is_err());
    let mut missing = serde_json::to_value(&envelope).unwrap();
    missing
        .as_object_mut()
        .unwrap()
        .remove("dependency_products");
    assert!(serde_json::from_value::<ExecutableUnitEnvelope>(missing).is_err());
    let mut oversized = manifest();
    oversized.dependencies = vec![runmat_execution::ExecutableDependencyDescriptor {
        identity: Digest::sha256(b"oversized"),
        encoded_length: EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES as u64 + 1,
    }];
    assert!(oversized.validate().is_err());
    oversized.dependencies[0].encoded_length = 0;
    assert!(oversized.validate().is_err());
}
