use runmat_builtins::{
    builtin_catalog_entries, builtin_constant_catalog_entries, builtin_docs, builtin_functions,
    builtin_implementation_provenance, canonical_catalog_fingerprint, constants,
    native_binding_symbol, AccelTag, BuiltinImplementationAuthority, TypeResolverKind,
    BUILTIN_CATALOG_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};

use crate::builtins::common::spec::{builtin_fusion_specs, builtin_gpu_specs};

use super::{
    environment::{build_configuration, hex},
    projection::SpecOwnershipClassifier,
    schema::*,
    validation::validate_inventory,
};

pub fn migration_inventory() -> MigrationInventory<'static> {
    let catalog_entries = {
        let mut entries = builtin_catalog_entries().to_vec();
        entries.sort_unstable_by_key(|entry| entry.identity);
        entries
    };
    let catalog_fingerprint = canonical_catalog_fingerprint(builtin_catalog_entries())
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
            source_file: provenance.source_file,
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
    let mut gpu_specs = builtin_gpu_specs()
        .map(|spec| spec_owners.gpu_spec(spec))
        .collect::<Vec<_>>();
    gpu_specs.sort_unstable_by_key(|spec| spec.key);
    let mut fusion_specs = builtin_fusion_specs()
        .map(|spec| spec_owners.fusion_spec(spec))
        .collect::<Vec<_>>();
    fusion_specs.sort_unstable_by_key(|spec| spec.key);

    let mut runtime_constants = constants()
        .into_iter()
        .map(|constant| RuntimeConstantRecord {
            name: constant.name,
        })
        .collect::<Vec<_>>();
    runtime_constants.sort_unstable_by_key(|constant| constant.name);

    let validation = validate_inventory(
        &catalog_provenance,
        &implementation_provenance,
        &runtime_bindings,
        &gpu_specs,
        &fusion_specs,
        &runtime_constants,
    );
    let snapshot = MigrationInventorySnapshot {
        build: build_configuration(),
        declared: DeclaredInventory {
            namespace_scope: "function_callables_and_constants_are_reported_separately",
            catalog_schema_version: BUILTIN_CATALOG_SCHEMA_VERSION,
            catalog_fingerprint: hex(&catalog_fingerprint),
            catalog_entries,
            catalog_provenance,
            constants: declared_constants,
            legacy_functions,
            legacy_documentation,
        },
        observed: ObservedInventory {
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
