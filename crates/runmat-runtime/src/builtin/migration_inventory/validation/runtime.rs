use std::collections::{BTreeMap, BTreeSet};

use runmat_builtins::{
    builtin_catalog_entries, builtin_functions, BuiltinBindingAvailability, BuiltinBindingIdentity,
};

use super::super::schema::{
    InventoryValidationError, MigrationFinding, MigrationFindingCode, RuntimeBindingRecord,
};

pub(super) fn validate(
    errors: &mut Vec<InventoryValidationError>,
    findings: &mut Vec<MigrationFinding>,
    runtime_bindings: &[RuntimeBindingRecord],
) {
    let mut actual = BTreeMap::<BuiltinBindingIdentity, usize>::new();
    for binding in runtime_bindings {
        *actual.entry(binding_identity(binding)).or_default() += 1;
    }
    for (identity, count) in &actual {
        if *count > 1 {
            errors.push(error(
                identity,
                "runtime binding identity is registered more than once",
            ));
        }
    }

    let legacy_names = builtin_functions()
        .into_iter()
        .map(|function| function.name)
        .collect::<BTreeSet<_>>();
    let mut declared = BTreeSet::new();
    for entry in builtin_catalog_entries() {
        if legacy_names.contains(entry.identity.name) {
            findings.push(MigrationFinding {
                code: MigrationFindingCode::CatalogLegacyAuthorityOverlap,
                source: "runtime_binding_registry",
                identity: entry.identity.name.to_owned(),
                message: "catalog identity still has a legacy BuiltinFunction authority".into(),
            });
        }
        for binding in entry.bindings {
            let identity = entry.binding_identity(binding);
            declared.insert(identity);
            if binding.availability == BuiltinBindingAvailability::Required
                && actual.get(&identity).copied().unwrap_or_default() == 0
            {
                findings.push(MigrationFinding {
                    code: MigrationFindingCode::MissingRequiredRuntimeBinding,
                    source: "runtime_binding_registry",
                    identity: format_identity(&identity),
                    message: "required canonical runtime binding has not been cut over".into(),
                });
            }
        }
    }
    for identity in actual.keys() {
        if !declared.contains(identity) {
            errors.push(error(
                identity,
                "runtime binding has no canonical catalog declaration",
            ));
        }
    }
}

fn error(identity: &BuiltinBindingIdentity, message: &str) -> InventoryValidationError {
    InventoryValidationError {
        source: "runtime_binding_registry",
        identity: Some(format_identity(identity)),
        message: message.to_owned(),
    }
}

fn format_identity(identity: &BuiltinBindingIdentity) -> String {
    format!("{}#{}", identity.builtin.name, identity.variant)
}

fn binding_identity(binding: &RuntimeBindingRecord) -> BuiltinBindingIdentity {
    BuiltinBindingIdentity {
        builtin: runmat_builtins::BuiltinCatalogIdentity { name: binding.name },
        variant: binding.variant,
    }
}
