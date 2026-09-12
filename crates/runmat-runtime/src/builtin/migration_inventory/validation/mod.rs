mod catalog;
mod constants;
mod declaration;
mod identity;
mod placement;
mod provenance;
mod registration_manifest;
mod runtime;

use std::collections::BTreeSet;

use runmat_builtins::{builtin_functions, validate_complete_builtin_catalog};

use super::schema::{
    CatalogProvenanceRecord, FusionSpecRecord, GpuSpecRecord, ImplementationProvenanceRecord,
    InventoryValidation, InventoryValidationError, MigrationReadiness, RegistrationManifestRecord,
    RuntimeBindingRecord, RuntimeConstantRecord,
};

pub(super) fn canonical_compiler_source_path(path: &str) -> String {
    declaration::canonical_compiler_source_path(path)
}

pub(super) fn validate_inventory(
    catalog_provenance: &[CatalogProvenanceRecord],
    implementation_provenance: &[ImplementationProvenanceRecord],
    runtime_bindings: &[RuntimeBindingRecord],
    gpu_specs: &[GpuSpecRecord],
    fusion_specs: &[FusionSpecRecord],
    runtime_constants: &[RuntimeConstantRecord],
    registration_manifest: &[RegistrationManifestRecord],
) -> InventoryValidation {
    let mut errors = validate_complete_builtin_catalog()
        .into_iter()
        .map(|error| InventoryValidationError {
            source: "catalog",
            identity: error.identity.map(str::to_owned),
            message: error.message,
        })
        .collect::<Vec<_>>();
    let mut findings = Vec::new();

    let callable_names = builtin_catalog_entries()
        .iter()
        .map(|entry| entry.identity.name)
        .chain(
            builtin_functions()
                .into_iter()
                .map(|function| function.name),
        )
        .collect::<BTreeSet<_>>();

    catalog::validate(&mut errors, catalog_provenance);
    provenance::validate(&mut errors, implementation_provenance, runtime_bindings);
    runtime::validate(&mut errors, &mut findings, runtime_bindings);
    identity::validate(
        &mut errors,
        &mut findings,
        &callable_names,
        runtime_bindings,
        implementation_provenance,
        gpu_specs,
        fusion_specs,
    );
    placement::validate(&mut findings, gpu_specs, fusion_specs);
    constants::validate(&mut errors, runtime_constants);
    registration_manifest::validate(
        &mut errors,
        registration_manifest,
        implementation_provenance,
        runtime_constants,
        gpu_specs,
        fusion_specs,
    );

    errors.sort_unstable_by(|left, right| {
        (left.source, &left.identity, &left.message).cmp(&(
            right.source,
            &right.identity,
            &right.message,
        ))
    });
    findings.sort_unstable_by(|left, right| {
        (left.code, left.source, &left.affected, &left.message).cmp(&(
            right.code,
            right.source,
            &right.affected,
            &right.message,
        ))
    });
    InventoryValidation {
        status: if errors.is_empty() {
            "valid"
        } else {
            "invalid"
        },
        errors,
        migration_readiness: MigrationReadiness {
            status: if findings.is_empty() {
                "ready"
            } else {
                "incomplete"
            },
            findings,
        },
    }
}
