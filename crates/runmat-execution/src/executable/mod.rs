mod component;
mod dependencies;
mod envelope;
mod identity;
mod linked_native_import;
mod manifest;
mod revision;
mod section;
mod source_map;
mod static_import;

pub use component::{
    ExecutableComponentDescriptor, ExecutableComponentKind, ExecutableComponentPayload,
};
pub use dependencies::{
    ExecutableDependencyDescriptor, ExecutableDependencyProduct, EXECUTABLE_DEPENDENCY_MAX_PRODUCTS,
};
pub use envelope::{
    ExecutableUnitAdmission, ExecutableUnitEnvelope, EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES,
};
pub use identity::{ExecutableEntrypointKind, ExecutableIdentity, ValidatedExecutableEntrypoint};
pub use linked_native_import::{
    decode_linked_native_imports, linked_native_import_section, LinkedNativeStaticImport,
    LINKED_NATIVE_IMPORT_SCHEMA, LINKED_NATIVE_IMPORT_SECTION,
};
pub use manifest::{ExecutableUnitManifest, EXECUTABLE_UNIT_SCHEMA_VERSION};
pub use revision::ExecutableComponentRevisions;
pub use section::{
    ExecutableOptionalSection, ExecutableSectionSupport, KnownExecutableSection, SectionRequirement,
};
pub use source_map::{ExecutableSourceMapV2, ExecutableSourceRecordV2};
pub use static_import::{
    decode_static_imports, selected_unit_content_digest, static_import_section,
    ExecutableStaticCaller, ExecutableStaticImport, ExecutableStaticImportKind,
    SelectedBodySemanticDigest, SelectedUnitContentDigest, STATIC_IMPORT_SCHEMA,
    STATIC_IMPORT_SECTION,
};
