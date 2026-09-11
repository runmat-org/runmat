use std::collections::BTreeMap;

use runmat_builtins::builtin_catalog_entries;

use super::super::schema::{CatalogProvenanceRecord, InventoryValidationError};

pub(super) fn validate(
    errors: &mut Vec<InventoryValidationError>,
    provenance: &[CatalogProvenanceRecord],
) {
    let mut expected = BTreeMap::new();
    for entry in builtin_catalog_entries() {
        for binding in entry.bindings {
            expected.insert(entry.binding_identity(binding), entry.provenance);
        }
    }
    let mut observed = BTreeMap::new();
    for record in provenance {
        *observed.entry(record.identity).or_insert(0usize) += 1;
        if record.provenance.source_file.is_empty() || record.provenance.module_path.is_empty() {
            errors.push(error(
                record,
                "catalog binding has incomplete declaration provenance",
            ));
        }
        if expected.get(&record.identity) != Some(&record.provenance) {
            errors.push(error(
                record,
                "catalog provenance does not match its canonical declaration",
            ));
        }
    }
    for identity in expected.keys() {
        if observed.get(identity).copied().unwrap_or_default() != 1 {
            errors.push(InventoryValidationError {
                source: "catalog_provenance",
                identity: Some(format!("{}#{}", identity.builtin.name, identity.variant)),
                message: "canonical binding does not have exactly one catalog provenance record"
                    .into(),
            });
        }
    }
}

fn error(record: &CatalogProvenanceRecord, message: &str) -> InventoryValidationError {
    InventoryValidationError {
        source: "catalog_provenance",
        identity: Some(format!(
            "{}#{}",
            record.identity.builtin.name, record.identity.variant
        )),
        message: message.to_owned(),
    }
}
