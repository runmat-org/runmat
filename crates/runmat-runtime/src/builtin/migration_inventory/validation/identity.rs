use std::collections::{BTreeMap, BTreeSet};

use runmat_builtins::{builtin_docs, builtin_functions};

use super::super::schema::{
    FusionSpecRecord, GpuSpecRecord, ImplementationProvenanceRecord, InventoryValidationError,
    MigrationFinding, MigrationFindingAffected, MigrationFindingCode, RuntimeBindingRecord,
    SpecOwnerRecord,
};
use super::declaration::{module_scopes_overlap, validate_registration_provenance};

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
        gpu_specs.iter().map(|spec| {
            (
                spec.key,
                spec.declaration,
                spec.source_file.as_str(),
                spec.module_path,
                spec.builtin_path,
                &spec.owner,
            )
        }),
        callable_names,
        provenance,
    );
    validate_specs(
        errors,
        findings,
        "fusion_spec_registry",
        fusion_specs.iter().map(|spec| {
            (
                spec.key,
                spec.declaration,
                spec.source_file.as_str(),
                spec.module_path,
                spec.builtin_path,
                &spec.owner,
            )
        }),
        callable_names,
        provenance,
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
    specs: impl IntoIterator<
        Item = (
            &'a str,
            &'a str,
            &'a str,
            &'a str,
            &'a str,
            &'a SpecOwnerRecord,
        ),
    >,
    callable_names: &BTreeSet<&str>,
    provenance: &[ImplementationProvenanceRecord],
) {
    let implementation_paths_by_identity =
        provenance
            .iter()
            .fold(BTreeMap::<&str, Vec<&str>>::new(), |mut paths, record| {
                paths
                    .entry(record.name)
                    .or_default()
                    .push(record.builtin_path);
                paths
            });
    let mut counts = BTreeMap::new();
    for (key, declaration, source_file, module_path, builtin_path, owner) in specs {
        validate_registration_provenance(
            errors,
            source,
            key,
            source_file,
            module_path,
            builtin_path,
            Some(declaration),
        );
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
            SpecOwnerRecord::ExactBuiltin { identity }
                if !implementation_paths_by_identity
                    .get(identity.name)
                    .is_some_and(|paths| {
                        paths
                            .iter()
                            .any(|path| module_scopes_overlap(path, builtin_path))
                    }) =>
            {
                errors.push(InventoryValidationError {
                    source,
                    identity: Some(identity.name.to_owned()),
                    message: format!(
                        "exact spec owner has no compiled implementation provenance at {builtin_path}"
                    ),
                });
            }
            SpecOwnerRecord::LegacyGroup {
                raw,
                affected_identities,
            } if *raw == key
                && !affected_identities.is_empty()
                && affected_identities
                    .iter()
                    .all(|identity| callable_names.contains(identity.name))
                && affected_identities
                    .windows(2)
                    .all(|pair| pair[0].name < pair[1].name) =>
            {
                findings.push(MigrationFinding {
                    code: MigrationFindingCode::LegacySpecGroupRequiresDisposition,
                    source,
                    affected: MigrationFindingAffected::Owner {
                        owner: owner.clone(),
                    },
                    message: "legacy grouped spec key requires reviewed migration disposition"
                        .into(),
                });
            }
            SpecOwnerRecord::LegacyGroup { raw, .. } if *raw != key => {
                errors.push(InventoryValidationError {
                    source,
                    identity: Some(key.to_owned()),
                    message: "legacy-group spec ownership does not preserve its raw registry key"
                        .into(),
                })
            }
            SpecOwnerRecord::LegacyGroup { .. } => errors.push(InventoryValidationError {
                source,
                identity: Some(key.to_owned()),
                message: "legacy-group spec ownership must have a nonempty canonical set of compiled affected identities".into(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_spec_owner_with_wrong_registration_path_is_invalid() {
        let mut errors = Vec::new();
        let mut findings = Vec::new();
        let callable_names = BTreeSet::from(["foo"]);
        let provenance = [ImplementationProvenanceRecord {
            name: "foo",
            binding_variant: Some("default"),
            source_file: "crates/runmat-runtime/src/builtins/math/foo.rs".into(),
            module_path: "runmat_runtime::builtins::math::foo",
            function: "foo_builtin",
            builtin_path: "builtins::math::foo",
            authority: "canonical_binding",
        }];
        let owner = SpecOwnerRecord::ExactBuiltin {
            identity: runmat_builtins::BuiltinCatalogIdentity { name: "foo" },
        };

        validate_specs(
            &mut errors,
            &mut findings,
            "gpu_spec_registry",
            [(
                "foo",
                "FOO_GPU_SPEC",
                "crates/runmat-runtime/src/builtins/math/bar.rs",
                "runmat_runtime::builtins::math::bar",
                "builtins::math::bar",
                &owner,
            )],
            &callable_names,
            &provenance,
        );

        assert!(findings.is_empty());
        assert!(errors.iter().any(|error| {
            error.identity.as_deref() == Some("foo")
                && error
                    .message
                    .contains("no compiled implementation provenance")
        }));
    }
}
