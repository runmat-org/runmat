use runmat_runtime::builtin::migration_inventory::{migration_inventory, RegistrationKindRecord};
use sha2::{Digest, Sha256};

#[test]
fn registration_manifest_is_typed_self_digesting_and_exactly_reconciled() {
    let inventory = migration_inventory();
    let manifest = &inventory.snapshot.observed.registration_manifest;
    assert_eq!(manifest.schema_version, 1);
    assert_eq!(
        manifest.digest,
        hex(&Sha256::digest(
            serde_json::to_vec(&manifest.entries).expect("manifest serialization")
        ))
    );
    assert_eq!(
        manifest.counts.builtin,
        manifest
            .entries
            .iter()
            .filter(|entry| entry.kind == RegistrationKindRecord::Builtin)
            .count()
    );
    assert_eq!(
        manifest.counts.constant,
        manifest
            .entries
            .iter()
            .filter(|entry| entry.kind == RegistrationKindRecord::Constant)
            .count()
    );
    assert_eq!(
        inventory.snapshot.validation.status, "valid",
        "registration manifest validation errors: {:#?}",
        inventory.snapshot.validation.errors
    );
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(output, "{byte:02x}").expect("write to String");
    }
    output
}
