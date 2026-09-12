use runmat_builtins::{
    builtin_catalog_aliases, builtin_catalog_entries, builtin_constant_catalog_entries,
    builtin_docs, builtin_functions, builtin_implementation_provenance,
    canonical_catalog_fingerprint, constant_registrations, native_binding_symbol,
    registration_manifest_entries, AccelTag, BuiltinImplementationAuthority, RegistrationKind,
    TypeResolverKind, BUILTIN_CATALOG_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};

use crate::builtins::common::spec::{
    builtin_fusion_spec_registrations, builtin_gpu_spec_registrations,
};

use super::{
    environment::{build_configuration, hex},
    projection::SpecOwnershipClassifier,
    schema::*,
    validation::{canonical_compiler_source_path, validate_inventory},
};

pub fn migration_inventory() -> MigrationInventory<'static> {
    let catalog_entries = {
        let mut entries = builtin_catalog_entries().to_vec();
        entries.sort_unstable_by_key(|entry| entry.identity);
        entries
    };
    let catalog_aliases = {
        let mut aliases = builtin_catalog_aliases().to_vec();
        aliases.sort_unstable_by_key(|alias| alias.alias);
        aliases
    };
    let catalog_fingerprint =
        canonical_catalog_fingerprint(builtin_catalog_entries(), &catalog_aliases)
            .expect("the validated builtin catalog must serialize");
    let mut catalog_provenance = catalog_entries
        .iter()
        .flat_map(|entry| {
            entry
                .bindings
                .iter()
                .map(|binding| CatalogProvenanceRecord {
                    identity: entry.binding_identity(binding),
                    provenance: entry.provenance,
                })
        })
        .collect::<Vec<_>>();
    catalog_provenance.sort_unstable_by_key(|record| record.identity);
    let mut declared_constants = builtin_constant_catalog_entries()
        .iter()
        .collect::<Vec<_>>();
    declared_constants.sort_unstable_by_key(|constant| constant.name);

    let mut legacy_functions = builtin_functions();
    legacy_functions.sort_unstable_by_key(|function| function.name);
    let legacy_functions = legacy_functions
        .into_iter()
        .map(|function| LegacyBuiltinRecord {
            name: function.name,
            description: function.description,
            category: function.category,
            parameter_types: &function.param_types,
            return_type: &function.return_type,
            resolver: match function.type_resolver {
                None => LegacyResolverKind::None,
                Some(TypeResolverKind::Simple(_)) => LegacyResolverKind::Simple,
                Some(TypeResolverKind::WithContext(_)) => LegacyResolverKind::WithContext,
            },
            semantic_authority: if runmat_builtins::builtin_catalog_entry_by_name(function.name)
                .is_some()
            {
                LegacySemanticAuthority::Catalog
            } else if runmat_builtins::builtin_semantics_for_name(function.name).is_some() {
                LegacySemanticAuthority::NameTable
            } else {
                LegacySemanticAuthority::Derived
            },
            semantics: function.semantics(),
            accelerator_tags: function
                .accel_tags
                .iter()
                .map(|tag| match tag {
                    AccelTag::Unary => "unary",
                    AccelTag::Elementwise => "elementwise",
                    AccelTag::Reduction => "reduction",
                    AccelTag::MatMul => "matmul",
                    AccelTag::Transpose => "transpose",
                    AccelTag::ArrayConstruct => "array_construct",
                })
                .collect(),
            is_sink: function.is_sink,
            suppress_auto_output: function.suppress_auto_output,
            execution_stack: function.execution_stack,
            required_capabilities: function.required_capabilities,
            descriptor: function.descriptor,
            extensions: function.extensions,
            integer_capabilities: function.integer_capabilities,
            integer_audit: function.integer_audit,
        })
        .collect();

    let mut legacy_documentation = builtin_docs();
    legacy_documentation.sort_unstable_by_key(|documentation| documentation.name);
    let legacy_documentation = legacy_documentation
        .into_iter()
        .map(|documentation| LegacyDocumentationRecord {
            name: documentation.name,
            category: documentation.category,
            summary: documentation.summary,
            keywords: documentation.keywords,
            errors: documentation.errors,
            related: documentation.related,
            introduced: documentation.introduced,
            status: documentation.status,
            examples: documentation.examples,
        })
        .collect();

    let mut runtime_bindings = crate::builtin::runtime_builtin_bindings()
        .into_iter()
        .map(|binding| RuntimeBindingRecord {
            name: binding.identity.builtin.name,
            variant: binding.identity.variant,
            native_symbol: native_binding_symbol(
                binding.identity.builtin.name,
                binding.identity.variant,
            ),
        })
        .collect::<Vec<_>>();
    runtime_bindings.sort_unstable_by_key(|binding| (binding.name, binding.variant));

    let mut implementation_provenance = builtin_implementation_provenance()
        .into_iter()
        .map(|provenance| ImplementationProvenanceRecord {
            name: provenance.name,
            binding_variant: provenance.binding_variant,
            source_file: canonical_compiler_source_path(provenance.source_file),
            module_path: provenance.module_path,
            function: provenance.function,
            builtin_path: provenance.builtin_path,
            authority: match provenance.authority {
                BuiltinImplementationAuthority::CanonicalBinding => "canonical_binding",
                BuiltinImplementationAuthority::LegacyFunction => "legacy_function",
            },
        })
        .collect::<Vec<_>>();
    implementation_provenance.sort_unstable_by_key(|provenance| {
        (
            provenance.name,
            provenance.binding_variant,
            provenance.module_path,
            provenance.function,
        )
    });

    let spec_owners = SpecOwnershipClassifier::compiled();
    let mut gpu_specs = builtin_gpu_spec_registrations()
        .map(|registration| {
            spec_owners.gpu_spec(
                registration.spec,
                registration.declaration,
                registration.builtin_path,
                canonical_compiler_source_path(registration.source_file),
                registration.module_path,
            )
        })
        .collect::<Vec<_>>();
    gpu_specs.sort_unstable_by_key(|spec| spec.key);
    let mut fusion_specs = builtin_fusion_spec_registrations()
        .map(|registration| {
            spec_owners.fusion_spec(
                registration.spec,
                registration.declaration,
                registration.builtin_path,
                canonical_compiler_source_path(registration.source_file),
                registration.module_path,
            )
        })
        .collect::<Vec<_>>();
    fusion_specs.sort_unstable_by_key(|spec| spec.key);

    let mut runtime_constants = constant_registrations()
        .into_iter()
        .map(|registration| RuntimeConstantRecord {
            name: registration.constant.name,
            source_file: canonical_compiler_source_path(registration.source_file),
            module_path: registration.module_path,
            builtin_path: registration.builtin_path,
        })
        .collect::<Vec<_>>();
    runtime_constants.sort_unstable_by_key(|constant| constant.name);

    let mut registration_manifest = registration_manifest_entries()
        .into_iter()
        .map(|entry| RegistrationManifestRecord {
            kind: match entry.kind {
                RegistrationKind::Builtin => RegistrationKindRecord::Builtin,
                RegistrationKind::Constant => RegistrationKindRecord::Constant,
                RegistrationKind::GpuSpec => RegistrationKindRecord::GpuSpec,
                RegistrationKind::FusionSpec => RegistrationKindRecord::FusionSpec,
            },
            declaration: entry.declaration,
            variant: entry.variant,
            builtin_path: entry.builtin_path,
        })
        .collect::<Vec<_>>();
    registration_manifest.sort_unstable();
    let registration_manifest_digest = hex(&Sha256::digest(
        serde_json::to_vec(&registration_manifest).expect("registration manifest must serialize"),
    ));
    let registration_manifest_counts = RegistrationManifestCounts {
        builtin: registration_manifest
            .iter()
            .filter(|entry| entry.kind == RegistrationKindRecord::Builtin)
            .count(),
        constant: registration_manifest
            .iter()
            .filter(|entry| entry.kind == RegistrationKindRecord::Constant)
            .count(),
        gpu_spec: registration_manifest
            .iter()
            .filter(|entry| entry.kind == RegistrationKindRecord::GpuSpec)
            .count(),
        fusion_spec: registration_manifest
            .iter()
            .filter(|entry| entry.kind == RegistrationKindRecord::FusionSpec)
            .count(),
    };

    let validation = validate_inventory(
        &catalog_provenance,
        &implementation_provenance,
        &runtime_bindings,
        &gpu_specs,
        &fusion_specs,
        &runtime_constants,
        &registration_manifest,
    );
    let snapshot = MigrationInventorySnapshot {
        build: build_configuration(),
        declared: DeclaredInventory {
            namespace_scope: "function_callables_and_constants_are_reported_separately",
            catalog_schema_version: BUILTIN_CATALOG_SCHEMA_VERSION,
            catalog_fingerprint: hex(&catalog_fingerprint),
            catalog_entries,
            catalog_aliases,
            catalog_provenance,
            constants: declared_constants,
            legacy_functions,
            legacy_documentation,
        },
        observed: ObservedInventory {
            registration_manifest: RegistrationManifest {
                schema_version: 1,
                digest: registration_manifest_digest,
                counts: registration_manifest_counts,
                entries: registration_manifest,
            },
            runtime_constants,
            runtime_bindings,
            implementation_provenance,
            gpu_specs,
            fusion_specs,
        },
        validation,
    };
    let encoded = serde_json::to_vec(&snapshot).expect("migration inventory must serialize");
    let digest = Sha256::digest(encoded);
    MigrationInventory {
        schema_version: MIGRATION_INVENTORY_SCHEMA_VERSION,
        kind: "runmat-compiled-builtin-migration-inventory",
        authority: "derived-read-only-evidence",
        digest: InventoryDigest {
            algorithm: "sha256",
            value: hex(&digest),
        },
        snapshot,
    }
}

pub fn migration_inventory_json() -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&migration_inventory()).map(|json| format!("{json}\n"))
}
