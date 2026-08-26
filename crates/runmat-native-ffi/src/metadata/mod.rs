mod artifact;
#[cfg(not(target_family = "wasm"))]
mod frontend;
mod normalize;
mod validate;

use serde::{Deserialize, Serialize};

use crate::model::{
    EnumerationDefinition, NativeLibrary, StructureDefinition, TypeAliasDefinition,
};

pub use artifact::{
    artifact_identity, NativeInterfaceArtifactBundle, NativeInterfaceArtifactBundleEntry,
    NativeInterfaceArtifactError, NativeInterfaceArtifactIdentity, NativeInterfaceArtifactManifest,
    NATIVE_FFI_ADAPTER_ID, NATIVE_FFI_ADAPTER_VERSION, NATIVE_FFI_ARTIFACT_SCHEMA_VERSION,
    NATIVE_INTERFACE_BUNDLE_SCHEMA_VERSION, NATIVE_INTERFACE_MANIFEST_MEDIA_TYPE,
    NATIVE_LIBRARY_MEDIA_TYPE,
};
#[cfg(not(target_family = "wasm"))]
pub use frontend::{prepare_header, HeaderPreparation, HeaderPreparationError};
pub use normalize::normalize_metadata;
pub use validate::{validate_metadata, MetadataError};

pub const NATIVE_FFI_METADATA_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeLibraryMetadata {
    pub schema_version: u16,
    pub target_triple: String,
    pub source_digest: String,
    #[serde(default)]
    pub libraries: Vec<NativeLibrary>,
    #[serde(default)]
    pub structures: Vec<StructureDefinition>,
    #[serde(default)]
    pub enumerations: Vec<EnumerationDefinition>,
    #[serde(default)]
    pub aliases: Vec<TypeAliasDefinition>,
}
