use runmat_builtins::{
    BuiltinBindingIdentity, BuiltinCatalogAlias, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCatalogProvenance, BuiltinConstantCatalogEntry, BuiltinDescriptor,
    BuiltinExtensionDescriptor, BuiltinIntegerAuditDescriptor, BuiltinIntegerCapabilityDescriptor,
    BuiltinSemantics, Type,
};
use runmat_types::{CapabilityRequirement, ExecutionStackRequirement};
use serde::Serialize;

pub const MIGRATION_INVENTORY_SCHEMA_VERSION: u32 = 3;

#[derive(Debug, Serialize)]
pub struct MigrationInventory<'a> {
    pub schema_version: u32,
    pub kind: &'static str,
    pub authority: &'static str,
    pub digest: InventoryDigest,
    pub snapshot: MigrationInventorySnapshot<'a>,
}

#[derive(Debug, Serialize)]
pub struct MigrationInventorySnapshot<'a> {
    pub build: BuildConfiguration,
    pub declared: DeclaredInventory<'a>,
    pub observed: ObservedInventory,
    pub validation: InventoryValidation,
}

#[derive(Debug, Serialize)]
pub struct InventoryDigest {
    pub algorithm: &'static str,
    pub value: String,
}

#[derive(Debug, Serialize)]
pub struct BuildConfiguration {
    pub architecture: &'static str,
    pub operating_system: &'static str,
    pub family: &'static str,
    pub pointer_width: u8,
    pub endianness: &'static str,
    pub crate_feature_inventory: CrateFeatureInventory,
}

#[derive(Debug, Serialize)]
pub struct CrateFeatureInventory {
    pub crate_name: &'static str,
    pub schema_version: u32,
    /// Every named `cfg(feature = "...")` exposed by this crate's manifest.
    /// Cargo's special `default` feature group is represented by the named
    /// features that it enables rather than as a rustc cfg value.
    pub known_features: Vec<&'static str>,
    pub enabled_features: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct DeclaredInventory<'a> {
    pub namespace_scope: &'static str,
    pub catalog_schema_version: u32,
    pub catalog_fingerprint: String,
    pub catalog_entries: Vec<&'a BuiltinCatalogEntry>,
    pub catalog_aliases: Vec<&'a BuiltinCatalogAlias>,
    pub catalog_provenance: Vec<CatalogProvenanceRecord>,
    pub constants: Vec<&'a BuiltinConstantCatalogEntry>,
    pub legacy_functions: Vec<LegacyBuiltinRecord<'a>>,
    pub legacy_documentation: Vec<LegacyDocumentationRecord>,
}

#[derive(Debug, Serialize)]
pub struct CatalogProvenanceRecord {
    pub identity: BuiltinBindingIdentity,
    pub provenance: BuiltinCatalogProvenance,
}

#[derive(Debug, Serialize)]
pub struct LegacyBuiltinRecord<'a> {
    pub name: &'static str,
    pub description: &'static str,
    pub category: &'static str,
    pub parameter_types: &'a [Type],
    pub return_type: &'a Type,
    pub resolver: LegacyResolverKind,
    pub semantic_authority: LegacySemanticAuthority,
    pub semantics: BuiltinSemantics,
    pub accelerator_tags: Vec<&'static str>,
    pub is_sink: bool,
    pub suppress_auto_output: bool,
    pub execution_stack: ExecutionStackRequirement,
    pub required_capabilities: &'static [CapabilityRequirement],
    pub descriptor: Option<&'a BuiltinDescriptor>,
    pub extensions: &'static [BuiltinExtensionDescriptor],
    pub integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    pub integer_audit: Option<&'a BuiltinIntegerAuditDescriptor>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacyResolverKind {
    None,
    Simple,
    WithContext,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacySemanticAuthority {
    Catalog,
    NameTable,
    Derived,
}

#[derive(Debug, Serialize)]
pub struct LegacyDocumentationRecord {
    pub name: &'static str,
    pub category: Option<&'static str>,
    pub summary: Option<&'static str>,
    pub keywords: Option<&'static str>,
    pub errors: Option<&'static str>,
    pub related: Option<&'static str>,
    pub introduced: Option<&'static str>,
    pub status: Option<&'static str>,
    pub examples: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ObservedInventory {
    pub registration_manifest: RegistrationManifest,
    pub runtime_constants: Vec<RuntimeConstantRecord>,
    pub runtime_bindings: Vec<RuntimeBindingRecord>,
    pub implementation_provenance: Vec<ImplementationProvenanceRecord>,
    pub gpu_specs: Vec<GpuSpecRecord>,
    pub fusion_specs: Vec<FusionSpecRecord>,
}

#[derive(Debug, Serialize)]
pub struct RegistrationManifest {
    pub schema_version: u32,
    pub digest: String,
    pub counts: RegistrationManifestCounts,
    pub entries: Vec<RegistrationManifestRecord>,
}

#[derive(Debug, Serialize)]
pub struct RegistrationManifestCounts {
    pub builtin: usize,
    pub constant: usize,
    pub gpu_spec: usize,
    pub fusion_spec: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct RegistrationManifestRecord {
    pub kind: RegistrationKindRecord,
    pub declaration: &'static str,
    pub variant: Option<&'static str>,
    pub builtin_path: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistrationKindRecord {
    Builtin,
    Constant,
    FusionSpec,
    GpuSpec,
}

#[derive(Debug, Serialize)]
pub struct RuntimeConstantRecord {
    pub name: &'static str,
    pub source_file: String,
    pub module_path: &'static str,
    pub builtin_path: &'static str,
}

#[derive(Debug, Serialize)]
pub struct RuntimeBindingRecord {
    pub name: &'static str,
    pub variant: &'static str,
    pub native_symbol: String,
}

#[derive(Debug, Serialize)]
pub struct ImplementationProvenanceRecord {
    pub name: &'static str,
    pub binding_variant: Option<&'static str>,
    pub source_file: String,
    pub module_path: &'static str,
    pub function: &'static str,
    pub builtin_path: &'static str,
    pub authority: &'static str,
}

#[derive(Debug, Serialize)]
pub struct GpuSpecRecord {
    pub key: &'static str,
    pub declaration: &'static str,
    pub builtin_path: &'static str,
    pub source_file: String,
    pub module_path: &'static str,
    pub owner: SpecOwnerRecord,
    pub operation: String,
    pub supported_precisions: Vec<&'static str>,
    pub broadcast: &'static str,
    pub provider_hooks: Vec<ProviderHookRecord>,
    pub constant_strategy: &'static str,
    pub residency: &'static str,
    pub nan_mode: &'static str,
    pub two_pass_threshold: Option<usize>,
    pub workgroup_size: Option<u32>,
    pub accepts_nan_mode: bool,
    pub notes: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ProviderHookRecord {
    pub kind: &'static str,
    pub name: &'static str,
    pub commutative: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct FusionSpecRecord {
    pub key: &'static str,
    pub declaration: &'static str,
    pub builtin_path: &'static str,
    pub source_file: String,
    pub module_path: &'static str,
    pub owner: SpecOwnerRecord,
    pub shape: FusionShapeRecord,
    pub constant_strategy: &'static str,
    pub elementwise: Option<FusionTemplateRecord>,
    pub reduction: Option<FusionTemplateRecord>,
    pub emits_nan: bool,
    pub notes: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpecOwnerRecord {
    ExactBuiltin {
        identity: BuiltinCatalogIdentity,
    },
    LegacyGroup {
        raw: &'static str,
        affected_identities: Vec<BuiltinCatalogIdentity>,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", content = "dimensions", rename_all = "snake_case")]
pub enum FusionShapeRecord {
    BroadcastCompatible,
    Exact(Vec<usize>),
    Any,
}

#[derive(Debug, Serialize)]
pub struct FusionTemplateRecord {
    pub scalar_precisions: Vec<&'static str>,
    pub expression_builder_registered: bool,
}

#[derive(Debug, Serialize)]
pub struct InventoryValidation {
    pub status: &'static str,
    pub errors: Vec<InventoryValidationError>,
    pub migration_readiness: MigrationReadiness,
}

#[derive(Debug, Serialize)]
pub struct InventoryValidationError {
    pub source: &'static str,
    pub identity: Option<String>,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct MigrationReadiness {
    pub status: &'static str,
    pub findings: Vec<MigrationFinding>,
}

#[derive(Debug, Serialize)]
pub struct MigrationFinding {
    pub code: MigrationFindingCode,
    pub source: &'static str,
    pub affected: MigrationFindingAffected,
    pub message: String,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MigrationFindingAffected {
    Identity {
        identity: BuiltinCatalogIdentity,
    },
    Binding {
        identity: BuiltinCatalogIdentity,
        variant: &'static str,
    },
    Owner {
        owner: SpecOwnerRecord,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationFindingCode {
    CatalogLegacyAuthorityOverlap,
    MissingRequiredRuntimeBinding,
    LegacySpecGroupRequiresDisposition,
    PlacementContractMismatch,
}
