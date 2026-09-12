use std::collections::BTreeMap;

use runmat_builtins::builtin_functions;

use super::super::schema::{
    ImplementationProvenanceRecord, InventoryValidationError, RuntimeBindingRecord,
};
use super::declaration::validate_registration_provenance;

pub(super) fn validate(
    errors: &mut Vec<InventoryValidationError>,
    provenance: &[ImplementationProvenanceRecord],
    runtime_bindings: &[RuntimeBindingRecord],
) {
    let mut counts = BTreeMap::new();
    for record in provenance {
        validate_registration_provenance(
            errors,
            "implementation_provenance",
            record.name,
            &record.source_file,
            record.module_path,
            record.builtin_path,
            Some(record.function),
        );
        *counts
            .entry((record.name, record.binding_variant, record.authority))
            .or_insert(0usize) += 1;
    }
    for ((name, variant, authority), count) in &counts {
        if *count > 1 {
            errors.push(InventoryValidationError {
                source: "implementation_provenance",
                identity: Some(format_identity(name, *variant)),
                message: format!("{count} {authority} provenance registrations were observed"),
            });
        }
    }

    for function in builtin_functions() {
        require_exactly_one(errors, &counts, function.name, None, "legacy_function");
    }
    for binding in runtime_bindings {
        require_exactly_one(
            errors,
            &counts,
            binding.name,
            Some(binding.variant),
            "canonical_binding",
        );
    }

    for record in provenance {
        let observed = match (record.authority, record.binding_variant) {
            ("legacy_function", None) => builtin_functions()
                .into_iter()
                .any(|function| function.name == record.name),
            ("canonical_binding", Some(variant)) => runtime_bindings
                .iter()
                .any(|binding| binding.name == record.name && binding.variant == variant),
            _ => false,
        };
        if !observed {
            errors.push(InventoryValidationError {
                source: "implementation_provenance",
                identity: Some(format_identity(record.name, record.binding_variant)),
                message: "provenance record has no matching compiled registration".into(),
            });
        }
    }
}

fn require_exactly_one(
    errors: &mut Vec<InventoryValidationError>,
    counts: &BTreeMap<(&str, Option<&str>, &str), usize>,
    name: &str,
    variant: Option<&str>,
    authority: &str,
) {
    if counts
        .get(&(name, variant, authority))
        .copied()
        .unwrap_or_default()
        != 1
    {
        errors.push(InventoryValidationError {
            source: "implementation_provenance",
            identity: Some(format_identity(name, variant)),
            message: "compiled registration does not have exactly one declaration-derived provenance record"
                .into(),
        });
    }
}

fn format_identity(name: &str, variant: Option<&str>) -> String {
    variant.map_or_else(|| name.to_owned(), |variant| format!("{name}#{variant}"))
}
