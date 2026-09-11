use std::collections::{BTreeMap, BTreeSet};

use runmat_builtins::{builtin_docs, builtin_functions};

use super::super::schema::{
    FusionSpecRecord, GpuSpecRecord, ImplementationProvenanceRecord, InventoryValidationError,
    MigrationFinding, MigrationFindingCode, RuntimeBindingRecord, SpecOwnerRecord,
};

pub(super) fn validate(
    errors: &mut Vec<InventoryValidationError>,
    findings: &mut Vec<MigrationFinding>,
    callable_names: &BTreeSet<&str>,
    runtime_bindings: &[RuntimeBindingRecord],
    provenance: &[ImplementationProvenanceRecord],
    gpu_specs: &[GpuSpecRecord],
    fusion_specs: &[FusionSpecRecord],
) {
    exact_legacy_duplicates(errors);
    case_fold_collisions(errors, "callable_identity", callable_names.iter().copied());
    case_fold_collisions(
        errors,
        "runtime_binding_registry",
        runtime_bindings.iter().map(|binding| binding.name),
    );
    case_fold_collisions(
        errors,
        "implementation_provenance",
        provenance.iter().map(|record| record.name),
    );
    case_fold_collisions(
        errors,
        "gpu_spec_registry",
        gpu_specs.iter().map(|spec| spec.key),
    );
    case_fold_collisions(
        errors,
        "fusion_spec_registry",
        fusion_specs.iter().map(|spec| spec.key),
    );
    validate_specs(
        errors,
        findings,
        "gpu_spec_registry",
        gpu_specs.iter().map(|spec| (spec.key, &spec.owner)),
        callable_names,
    );
    validate_specs(
        errors,
        findings,
        "fusion_spec_registry",
        fusion_specs.iter().map(|spec| (spec.key, &spec.owner)),
        callable_names,
    );
}

fn exact_legacy_duplicates(errors: &mut Vec<InventoryValidationError>) {
    duplicate_names(
        errors,
        "legacy_function_registry",
        builtin_functions()
            .into_iter()
            .map(|function| function.name),
    );
    duplicate_names(
        errors,
        "legacy_documentation_registry",
        builtin_docs()
            .into_iter()
            .map(|documentation| documentation.name),
    );
}

fn duplicate_names<'a>(
    errors: &mut Vec<InventoryValidationError>,
    source: &'static str,
    names: impl IntoIterator<Item = &'a str>,
) {
    let mut counts = BTreeMap::new();
    for name in names {
        *counts.entry(name).or_insert(0usize) += 1;
    }
    for (name, count) in counts {
        if count > 1 {
            errors.push(InventoryValidationError {
                source,
                identity: Some(name.to_owned()),
                message: format!("{count} exact identity registrations were observed"),
            });
        }
    }
}

fn case_fold_collisions<'a>(
    errors: &mut Vec<InventoryValidationError>,
    source: &'static str,
    names: impl IntoIterator<Item = &'a str>,
) {
    let mut spellings_by_folded_name = BTreeMap::<String, BTreeSet<&str>>::new();
    for name in names {
        spellings_by_folded_name
            .entry(name.to_lowercase())
            .or_default()
            .insert(name);
    }
    for (folded, spellings) in spellings_by_folded_name {
        if spellings.len() > 1 {
            errors.push(InventoryValidationError {
                source,
                identity: Some(folded),
                message: format!(
                    "case-folded identity has conflicting spellings: {}",
                    spellings.into_iter().collect::<Vec<_>>().join(", ")
                ),
            });
        }
    }
}

fn validate_specs<'a>(
    errors: &mut Vec<InventoryValidationError>,
    findings: &mut Vec<MigrationFinding>,
    source: &'static str,
    specs: impl IntoIterator<Item = (&'a str, &'a SpecOwnerRecord)>,
    callable_names: &BTreeSet<&str>,
) {
    let mut counts = BTreeMap::new();
    for (key, owner) in specs {
        *counts.entry(key).or_insert(0usize) += 1;
        match owner {
            SpecOwnerRecord::ExactBuiltin { identity }
                if identity.name != key || !callable_names.contains(identity.name) =>
            {
                errors.push(InventoryValidationError {
                    source,
                    identity: Some(key.to_owned()),
                    message: "exact-builtin spec ownership does not match a compiled callable"
                        .into(),
                });
            }
            SpecOwnerRecord::LegacyGroup { raw } if *raw == key => {
                findings.push(MigrationFinding {
                    code: MigrationFindingCode::LegacySpecGroupRequiresDisposition,
                    source,
                    identity: key.to_owned(),
                    message: "legacy grouped spec key requires reviewed migration disposition"
                        .into(),
                });
            }
            SpecOwnerRecord::LegacyGroup { .. } => errors.push(InventoryValidationError {
                source,
                identity: Some(key.to_owned()),
                message: "legacy-group spec ownership does not preserve its raw registry key"
                    .into(),
            }),
            SpecOwnerRecord::ExactBuiltin { .. } => {}
        }
    }
    for (name, count) in counts {
        if count > 1 {
            errors.push(InventoryValidationError {
                source,
                identity: Some(name.to_owned()),
                message: format!("{count} specifications were registered for one identity"),
            });
        }
    }
}
