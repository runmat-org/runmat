use std::collections::{BTreeMap, BTreeSet};

use runmat_builtins::builtin_constant_catalog_entries;

use super::super::schema::{InventoryValidationError, RuntimeConstantRecord};
use super::declaration::validate_registration_provenance;

pub(super) fn validate(
    errors: &mut Vec<InventoryValidationError>,
    runtime_constants: &[RuntimeConstantRecord],
) {
    let declared = builtin_constant_catalog_entries()
        .iter()
        .map(|constant| constant.name)
        .collect::<BTreeSet<_>>();
    let mut observed = BTreeMap::new();
    for constant in runtime_constants {
        validate_registration_provenance(
            errors,
            "constant_registry",
            constant.name,
            &constant.source_file,
            constant.module_path,
            constant.builtin_path,
            None,
        );
        *observed.entry(constant.name).or_insert(0usize) += 1;
    }
    for (name, count) in &observed {
        if *count > 1 {
            errors.push(InventoryValidationError {
                source: "constant_registry",
                identity: Some((*name).to_owned()),
                message: format!("{count} runtime constant registrations were observed"),
            });
        }
        if !declared.contains(name) {
            errors.push(InventoryValidationError {
                source: "constant_registry",
                identity: Some((*name).to_owned()),
                message: "runtime constant has no matching static constant declaration".into(),
            });
        }
    }
    for name in declared {
        if observed.get(name).copied().unwrap_or_default() != 1 {
            errors.push(InventoryValidationError {
                source: "constant_registry",
                identity: Some(name.to_owned()),
                message: "static constant does not have exactly one runtime registration".into(),
            });
        }
    }
}
