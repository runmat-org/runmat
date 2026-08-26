use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};
use std::path::Path;

use runmat_types::{
    CapabilityRequirement, CapabilitySet, ForeignAdapterRequirement, InteropManifest,
    INTEROP_MANIFEST_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use thiserror::Error;

use super::{normalize_metadata, validate_metadata, MetadataError, NativeLibraryMetadata};

pub const NATIVE_FFI_ARTIFACT_SCHEMA_VERSION: u16 = 1;
pub const NATIVE_FFI_ADAPTER_ID: &str = "native-ffi";
pub const NATIVE_FFI_ADAPTER_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NativeInterfaceArtifactIdentity(String);

impl NativeInterfaceArtifactIdentity {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for NativeInterfaceArtifactIdentity {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeInterfaceArtifactManifest {
    pub schema_version: u16,
    pub identity: NativeInterfaceArtifactIdentity,
    pub interface_name: String,
    pub target_triple: String,
    pub metadata: NativeLibraryMetadata,
    pub library_digest: String,
    pub library_bytes: u64,
}

#[derive(Serialize)]
struct IdentityInput<'a> {
    schema_version: u16,
    interface_name: &'a str,
    target_triple: &'a str,
    metadata: &'a NativeLibraryMetadata,
    library_digest: &'a str,
    library_bytes: u64,
}

#[derive(Debug, Error)]
pub enum NativeInterfaceArtifactError {
    #[error(transparent)]
    Metadata(#[from] MetadataError),
    #[error("could not encode native-interface artifact: {0}")]
    Encoding(#[from] serde_json::Error),
    #[error("could not read native-interface artifact {path}: {source}")]
    Read {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not write native-interface artifact {path}: {source}")]
    Write {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid native-interface artifact: {0}")]
    Invalid(String),
}

impl NativeInterfaceArtifactManifest {
    pub fn path_for_library(library: &Path) -> std::path::PathBuf {
        library.with_extension(format!(
            "{}.runmat.json",
            library
                .extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or("native")
        ))
    }

    pub fn from_library(
        interface_name: impl Into<String>,
        metadata: NativeLibraryMetadata,
        library: &[u8],
    ) -> Result<Self, NativeInterfaceArtifactError> {
        let interface_name = interface_name.into();
        let metadata = canonical_artifact_metadata(metadata)?;
        let target_triple = metadata.target_triple.clone();
        let library_digest = digest(library);
        let library_bytes = library.len() as u64;
        let identity = identity(
            NATIVE_FFI_ARTIFACT_SCHEMA_VERSION,
            &interface_name,
            &target_triple,
            &metadata,
            &library_digest,
            library_bytes,
        )?;
        let manifest = Self {
            schema_version: NATIVE_FFI_ARTIFACT_SCHEMA_VERSION,
            identity,
            interface_name,
            target_triple,
            metadata,
            library_digest,
            library_bytes,
        };
        manifest.validate_library(library)?;
        Ok(manifest)
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn from_library_path(
        interface_name: impl Into<String>,
        metadata: NativeLibraryMetadata,
        library_path: &Path,
    ) -> Result<Self, NativeInterfaceArtifactError> {
        let library =
            std::fs::read(library_path).map_err(|source| NativeInterfaceArtifactError::Read {
                path: library_path.to_path_buf(),
                source,
            })?;
        Self::from_library(interface_name, metadata, &library)
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn read(path: &Path) -> Result<Self, NativeInterfaceArtifactError> {
        let bytes = std::fs::read(path).map_err(|source| NativeInterfaceArtifactError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_canonical_bytes(&bytes)
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn publish(&self, path: &Path) -> Result<(), NativeInterfaceArtifactError> {
        use std::io::Write as _;

        let bytes = self.canonical_bytes()?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|source| {
            NativeInterfaceArtifactError::Write {
                path: path.to_path_buf(),
                source,
            }
        })?;
        temporary
            .write_all(&bytes)
            .and_then(|_| temporary.as_file_mut().sync_all())
            .map_err(|source| NativeInterfaceArtifactError::Write {
                path: path.to_path_buf(),
                source,
            })?;
        temporary
            .persist(path)
            .map_err(|error| NativeInterfaceArtifactError::Write {
                path: path.to_path_buf(),
                source: error.error,
            })?;
        Ok(())
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, NativeInterfaceArtifactError> {
        let manifest = serde_json::from_slice::<Self>(bytes)?;
        manifest.validate_metadata()?;
        if manifest.canonical_bytes()? != bytes {
            return Err(NativeInterfaceArtifactError::Invalid(
                "manifest bytes are not canonical".into(),
            ));
        }
        Ok(manifest)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, NativeInterfaceArtifactError> {
        self.validate_metadata()?;
        Ok(serde_json::to_vec(self)?)
    }

    pub fn validate_library(&self, library: &[u8]) -> Result<(), NativeInterfaceArtifactError> {
        self.validate_metadata()?;
        if self.library_digest != digest(library) || self.library_bytes != library.len() as u64 {
            return Err(NativeInterfaceArtifactError::Invalid(
                "library bytes do not match the manifest".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_current_library(
        &self,
        library: &[u8],
    ) -> Result<(), NativeInterfaceArtifactError> {
        self.validate_library(library)?;
        if self.target_triple != target_lexicon::HOST.to_string() {
            return Err(NativeInterfaceArtifactError::Invalid(format!(
                "artifact target {} does not match host {}",
                self.target_triple,
                target_lexicon::HOST
            )));
        }
        Ok(())
    }

    pub fn materialized_metadata(
        &self,
        library_path: &Path,
    ) -> Result<NativeLibraryMetadata, NativeInterfaceArtifactError> {
        self.validate_metadata()?;
        let mut metadata = self.metadata.clone();
        metadata.libraries[0].path = library_path.to_string_lossy().into_owned();
        Ok(metadata)
    }

    pub fn interop_manifest(&self) -> InteropManifest {
        InteropManifest {
            schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: Vec::new(),
            adapters: vec![ForeignAdapterRequirement {
                adapter: NATIVE_FFI_ADAPTER_ID.into(),
                minimum_version: NATIVE_FFI_ADAPTER_VERSION,
                capabilities: CapabilitySet(BTreeSet::from([
                    CapabilityRequirement::NativeCode,
                    CapabilityRequirement::ForeignRuntime,
                ])),
                artifact_identities: vec![self.identity.to_string()],
            }],
        }
    }

    fn validate_metadata(&self) -> Result<(), NativeInterfaceArtifactError> {
        if self.schema_version != NATIVE_FFI_ARTIFACT_SCHEMA_VERSION {
            return Err(NativeInterfaceArtifactError::Invalid(format!(
                "unsupported schema version {}",
                self.schema_version
            )));
        }
        token("interface name", &self.interface_name, 256)?;
        token("target triple", &self.target_triple, 128)?;
        if self.target_triple != self.metadata.target_triple {
            return Err(NativeInterfaceArtifactError::Invalid(
                "manifest and metadata targets differ".into(),
            ));
        }
        if self.library_bytes == 0 || !valid_digest(&self.library_digest) {
            return Err(NativeInterfaceArtifactError::Invalid(
                "library digest or length is invalid".into(),
            ));
        }
        validate_metadata(&self.metadata)?;
        if self.metadata.libraries.len() != 1 {
            return Err(NativeInterfaceArtifactError::Invalid(
                "one prepared interface must bind exactly one library".into(),
            ));
        }
        let expected_path = logical_library_path(&self.metadata.libraries[0].name);
        if self.metadata.libraries[0].path != expected_path {
            return Err(NativeInterfaceArtifactError::Invalid(
                "prepared metadata contains a physical library path".into(),
            ));
        }
        let expected = identity(
            self.schema_version,
            &self.interface_name,
            &self.target_triple,
            &self.metadata,
            &self.library_digest,
            self.library_bytes,
        )?;
        if self.identity != expected {
            return Err(NativeInterfaceArtifactError::Invalid(
                "artifact identity does not match its content".into(),
            ));
        }
        Ok(())
    }
}

/// Deterministic identity for prototype metadata alone. Prepared executable
/// artifacts should use [`NativeInterfaceArtifactManifest`] so the identity
/// also binds exact library bytes and target admission.
pub fn artifact_identity(metadata: &NativeLibraryMetadata) -> Result<String, MetadataError> {
    let normalized = normalize_metadata(metadata.clone());
    validate_metadata(&normalized)?;
    let bytes = serde_json::to_vec(&normalized).map_err(MetadataError::Serialize)?;
    let digest = Sha256::digest(bytes);
    Ok(format!("native-ffi-metadata-v1:{digest:x}"))
}

fn canonical_artifact_metadata(
    mut metadata: NativeLibraryMetadata,
) -> Result<NativeLibraryMetadata, NativeInterfaceArtifactError> {
    metadata = normalize_metadata(metadata);
    if metadata.libraries.len() != 1 {
        return Err(NativeInterfaceArtifactError::Invalid(
            "one prepared interface must bind exactly one library".into(),
        ));
    }
    metadata.libraries[0].path = logical_library_path(&metadata.libraries[0].name);
    validate_metadata(&metadata)?;
    Ok(metadata)
}

fn logical_library_path(name: &str) -> String {
    format!("artifact://{name}")
}

fn identity(
    schema_version: u16,
    interface_name: &str,
    target_triple: &str,
    metadata: &NativeLibraryMetadata,
    library_digest: &str,
    library_bytes: u64,
) -> Result<NativeInterfaceArtifactIdentity, NativeInterfaceArtifactError> {
    let bytes = serde_json::to_vec(&IdentityInput {
        schema_version,
        interface_name,
        target_triple,
        metadata,
        library_digest,
        library_bytes,
    })?;
    Ok(NativeInterfaceArtifactIdentity(format!(
        "native-ffi:v1:{}",
        digest_hex(&bytes)
    )))
}

fn digest(bytes: &[u8]) -> String {
    format!("sha256:{}", digest_hex(bytes))
}

fn digest_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to a string cannot fail");
    }
    encoded
}

fn valid_digest(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    })
}

fn token(field: &str, value: &str, maximum: usize) -> Result<(), NativeInterfaceArtifactError> {
    if value.is_empty()
        || value.len() > maximum
        || !value.is_ascii()
        || value.chars().any(char::is_control)
    {
        return Err(NativeInterfaceArtifactError::Invalid(format!(
            "{field} is invalid"
        )));
    }
    Ok(())
}
