use std::collections::BTreeMap;

use super::super::schema::{
    FusionSpecRecord, GpuSpecRecord, ImplementationProvenanceRecord, InventoryValidationError,
    RegistrationKindRecord, RegistrationManifestRecord, RuntimeConstantRecord,
};
use super::declaration::validate_spec_path;

pub(super) fn validate(
    errors: &mut Vec<InventoryValidationError>,
    manifest: &[RegistrationManifestRecord],
    provenance: &[ImplementationProvenanceRecord],
    constants: &[RuntimeConstantRecord],
    gpu_specs: &[GpuSpecRecord],
    fusion_specs: &[FusionSpecRecord],
) {
    let mut counts = BTreeMap::new();
    for entry in manifest {
        validate_spec_path(
            errors,
            "registration_manifest",
            entry.declaration,
            entry.builtin_path,
        );
        *counts
            .entry((
                entry.kind,
                entry.declaration,
                entry.variant,
                entry.builtin_path,
            ))
            .or_insert(0usize) += 1;
    }
    for ((_, declaration, _, _), count) in counts {
        if count != 1 {
            push(
                errors,
                declaration,
                "registration manifest row is not unique",
            );
        }
    }

    let builtins = manifest
        .iter()
        .filter(|entry| entry.kind == RegistrationKindRecord::Builtin)
        .map(|entry| {
            (
                entry.declaration,
                entry.variant,
                canonical(entry.builtin_path),
            )
        })
        .collect();
    let expected_builtins = provenance
        .iter()
        .map(|entry| {
            (
                entry.name,
                entry.binding_variant,
                canonical(entry.builtin_path),
            )
        })
        .collect();
    exact_rows(errors, "builtin", builtins, expected_builtins);

    let constant_rows = manifest
        .iter()
        .filter(|entry| entry.kind == RegistrationKindRecord::Constant)
        .map(|entry| (entry.declaration, canonical(entry.builtin_path)))
        .collect();
    let expected_constants = constants
        .iter()
        .map(|entry| (entry.name, canonical(entry.builtin_path)))
        .collect();
    exact_rows(errors, "constant", constant_rows, expected_constants);

    exact_specs(
        errors,
        "GPU spec",
        manifest,
        RegistrationKindRecord::GpuSpec,
        gpu_specs
            .iter()
            .map(|entry| (entry.declaration, entry.builtin_path)),
    );
    exact_specs(
        errors,
        "fusion spec",
        manifest,
        RegistrationKindRecord::FusionSpec,
        fusion_specs
            .iter()
            .map(|entry| (entry.declaration, entry.builtin_path)),
    );
}

fn exact_rows<T: Ord>(
    errors: &mut Vec<InventoryValidationError>,
    label: &str,
    mut actual: Vec<T>,
    mut expected: Vec<T>,
) {
    actual.sort();
    expected.sort();
    if actual != expected {
        push(
            errors,
            label,
            "registration manifest rows differ from live registrations",
        );
    }
}

fn exact_specs<'a>(
    errors: &mut Vec<InventoryValidationError>,
    label: &str,
    manifest: &[RegistrationManifestRecord],
    kind: RegistrationKindRecord,
    expected: impl Iterator<Item = (&'a str, &'a str)>,
) {
    let actual = manifest
        .iter()
        .filter(|entry| entry.kind == kind)
        .map(|entry| (entry.declaration, canonical(entry.builtin_path)))
        .collect();
    let expected = expected
        .map(|(declaration, path)| (declaration, canonical(path)))
        .collect();
    exact_rows(errors, label, actual, expected);
}

fn canonical(path: &str) -> &str {
    path.strip_prefix("crate::").unwrap_or(path)
}

fn push(errors: &mut Vec<InventoryValidationError>, identity: &str, message: &'static str) {
    errors.push(InventoryValidationError {
        source: "registration_manifest",
        identity: Some(identity.to_owned()),
        message: message.to_owned(),
    });
}
