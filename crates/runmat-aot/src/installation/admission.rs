//! Canonical transport for an independently intended native installation.
//!
//! Decoding either record below does not grant installation authority. The
//! selected-program boundary explicitly adopts a validated admission record,
//! while issuance requires both an operator intent and a private completed
//! selected-image receipt retained by [`LinkedProgram`].

use crate::link::{LinkedProgram, NativeLinkBuildIdentity, NativeProgramLinkIdentity};
use crate::tree::{self, physical, NativeStorageGrant};
use crate::{AotError, AotResult};
use runmat_execution::Digest;

use super::{NativeInstallationLayout, NativeLoaderImage};

pub const NATIVE_INSTALLATION_SCHEMA_VERSION: u16 = 2;
const MAX_INSTALLATION_ADMISSION_BYTES: u64 = 1024 * 1024;

/// Independently supplied deployment intent. This is operator authority, not
/// evidence inferred from a linked image or a persisted product record.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalNativeInstallationIntent {
    pub schema_version: u16,
    pub expected_product: Digest,
    pub expected_image: Digest,
    pub layout: NativeInstallationLayout,
    pub storage: NativeStorageGrant,
    pub loader_images: Vec<NativeLoaderImage>,
    pub deployment_qualification: Digest,
}

/// Canonical completed-image admission record. It remains inert transport
/// until a trusted selected-program boundary adopts it for one installation.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeInstallationAdmissionRecord {
    pub schema_version: u16,
    pub product: Digest,
    pub image: Digest,
    pub inputs: NativeProgramLinkIdentity,
    pub build: NativeLinkBuildIdentity,
    pub layout: NativeInstallationLayout,
    pub storage: NativeStorageGrant,
    pub loader_images: Vec<NativeLoaderImage>,
    pub deployment_qualification: Digest,
}

/// Trusted deployment expectations, not decoded from an artifact. Matching
/// this input admits physical installation only, not loader/provider execution.
pub struct ExternalNativeInstallationAdmission {
    pub product: Digest,
    pub image: Digest,
    pub inputs: NativeProgramLinkIdentity,
    pub build: NativeLinkBuildIdentity,
    pub layout: NativeInstallationLayout,
    /// Independent caller storage authority, NOT the manifest's byte counts.
    pub storage: NativeStorageGrant,
    /// Exact identifiers, not resolved installed paths. Expected build-local
    /// identifiers remain unqualified loader obligations, not runtime rights.
    pub loader_images: Vec<NativeLoaderImage>,
    pub deployment_qualification: Digest,
}

impl ExternalNativeInstallationIntent {
    /// Parse explicitly supplied operator configuration, not an artifact.
    /// Whitespace and field ordering are presentation; typed fields and
    /// limits are identical to canonical intent validation.
    pub fn from_operator_bytes(bytes: &[u8]) -> AotResult<Self> {
        decode(bytes, "installation intent", Self::validate)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AotResult<Self> {
        decode_canonical(bytes, "installation intent", Self::validate)
    }

    pub fn to_canonical_bytes(&self) -> AotResult<Vec<u8>> {
        self.validate()?;
        tree::canonical_json(self, MAX_INSTALLATION_ADMISSION_BYTES)
    }

    fn validate(&self) -> AotResult<()> {
        validate_common(
            self.schema_version,
            self.expected_product,
            self.expected_image,
            &self.layout,
            &self.storage,
            &self.loader_images,
            self.deployment_qualification,
        )
    }
}

impl NativeInstallationAdmissionRecord {
    pub fn from_canonical_bytes(bytes: &[u8]) -> AotResult<Self> {
        decode_canonical(bytes, "installation admission", Self::validate)
    }

    pub fn to_canonical_bytes(&self) -> AotResult<Vec<u8>> {
        self.validate()?;
        tree::canonical_json(self, MAX_INSTALLATION_ADMISSION_BYTES)
    }

    fn validate(&self) -> AotResult<()> {
        validate_common(
            self.schema_version,
            self.product,
            self.image,
            &self.layout,
            &self.storage,
            &self.loader_images,
            self.deployment_qualification,
        )?;
        if self.build.product != self.product {
            return Err(invalid(
                "installation build identity differs from its selected product",
            ));
        }
        Ok(())
    }

    pub(crate) fn into_external(self) -> ExternalNativeInstallationAdmission {
        ExternalNativeInstallationAdmission {
            product: self.product,
            image: self.image,
            inputs: self.inputs,
            build: self.build,
            layout: self.layout,
            storage: self.storage,
            loader_images: self.loader_images,
            deployment_qualification: self.deployment_qualification,
        }
    }
}

/// Issue inert canonical evidence for one completed selected image. The
/// operator supplied every deployment-specific expectation independently;
/// only the link inputs and build receipt are copied from the private
/// completed-image authority.
pub fn issue_completed_installation_admission(
    linked: &LinkedProgram,
    intent: &ExternalNativeInstallationIntent,
) -> AotResult<NativeInstallationAdmissionRecord> {
    intent.validate()?;
    let completed = linked
        .verified_selected_image()?
        .ok_or_else(|| invalid("installation admission requires a completed selected image"))?;
    let product = completed.product().fingerprint().map_err(invalid)?;
    if product != intent.expected_product
        || completed.output_digest() != intent.expected_image
        || completed.selected_build().product != product
    {
        return Err(invalid(
            "completed selected image differs from independent installation intent",
        ));
    }

    // Retain and hash the exact opened regular file. The digest in the intent
    // is permission, not something minted from these untrusted path bytes.
    let (image, length) =
        physical::open_bounded_regular(&linked.output, intent.storage.archive.max_file_bytes)?;
    intent.storage.file(length)?;
    if physical::hash_open_file(&image, length, &linked.output)? != intent.expected_image {
        return Err(invalid(
            "completed selected image changed before admission issuance",
        ));
    }

    let record = NativeInstallationAdmissionRecord {
        schema_version: NATIVE_INSTALLATION_SCHEMA_VERSION,
        product,
        image: completed.output_digest(),
        inputs: completed.inputs().clone(),
        build: completed.selected_build().clone(),
        layout: intent.layout.clone(),
        storage: intent.storage,
        loader_images: intent.loader_images.clone(),
        deployment_qualification: intent.deployment_qualification,
    };
    record.validate()?;
    Ok(record)
}

fn decode_canonical<T>(
    bytes: &[u8],
    kind: &'static str,
    validate: impl FnOnce(&T) -> AotResult<()>,
) -> AotResult<T>
where
    T: serde::de::DeserializeOwned + serde::Serialize,
{
    let value = decode(bytes, kind, validate)?;
    let canonical = tree::canonical_json(&value, MAX_INSTALLATION_ADMISSION_BYTES)?;
    if canonical != bytes {
        return Err(invalid(format!("native {kind} is not canonical")));
    }
    Ok(value)
}

fn decode<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    kind: &'static str,
    validate: impl FnOnce(&T) -> AotResult<()>,
) -> AotResult<T> {
    if bytes.len() as u64 > MAX_INSTALLATION_ADMISSION_BYTES {
        return Err(AotError::ResourceUnavailable {
            message: format!("native {kind} exceeds its canonical byte bound"),
        });
    }
    let value = serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("decode native {kind}: {error}")))?;
    validate(&value)?;
    Ok(value)
}

fn validate_common(
    schema_version: u16,
    product: Digest,
    image: Digest,
    layout: &NativeInstallationLayout,
    storage: &NativeStorageGrant,
    loader_images: &[NativeLoaderImage],
    deployment_qualification: Digest,
) -> AotResult<()> {
    if schema_version != NATIVE_INSTALLATION_SCHEMA_VERSION
        || [product, image, deployment_qualification]
            .iter()
            .any(|digest| digest.bytes().iter().all(|byte| *byte == 0))
    {
        return Err(invalid("invalid independent installation identity"));
    }
    storage.archive.validate().map_err(invalid)?;
    if storage.metadata_bytes == 0 {
        return Err(invalid("installation metadata grant must be nonzero"));
    }
    tree::admit_json(&(layout, loader_images), storage.metadata_bytes)?;
    Ok(())
}

fn invalid(message: impl std::fmt::Display) -> AotError {
    AotError::contract("aot.installation.admission", message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link::{NativeLinkAdapter, NativeLinkExecutionHost};
    use runmat_package::NormalizedRelativePath;
    use runmat_package_cache::ArchiveLimits;

    fn intent() -> ExternalNativeInstallationIntent {
        ExternalNativeInstallationIntent {
            schema_version: NATIVE_INSTALLATION_SCHEMA_VERSION,
            expected_product: Digest::sha256(b"product"),
            expected_image: Digest::sha256(b"image"),
            layout: NativeInstallationLayout {
                executable: NormalizedRelativePath::new("bin/runmat").unwrap(),
                sysroot: NormalizedRelativePath::new("sysroot").unwrap(),
                components: Vec::new(),
            },
            storage: NativeStorageGrant {
                archive: ArchiveLimits {
                    max_entries: 8,
                    max_files: 4,
                    max_expanded_bytes: 4096,
                    max_file_bytes: 2048,
                    max_path_bytes: 256,
                    max_component_bytes: 128,
                    max_compression_ratio: 1,
                },
                metadata_bytes: 4096,
            },
            loader_images: Vec::new(),
            deployment_qualification: Digest::sha256(b"qualification"),
        }
    }

    #[test]
    fn intent_codec_is_canonical_and_bounded() {
        let intent = intent();
        let bytes = intent.to_canonical_bytes().unwrap();
        assert_eq!(
            ExternalNativeInstallationIntent::from_canonical_bytes(&bytes).unwrap(),
            intent
        );
        let pretty = serde_json::to_vec_pretty(&intent).unwrap();
        assert_eq!(
            ExternalNativeInstallationIntent::from_operator_bytes(&pretty).unwrap(),
            intent
        );
        assert!(ExternalNativeInstallationIntent::from_canonical_bytes(&pretty).is_err());
        let mut noncanonical = bytes;
        noncanonical.push(b'\n');
        assert!(ExternalNativeInstallationIntent::from_canonical_bytes(&noncanonical).is_err());
        assert!(
            ExternalNativeInstallationIntent::from_canonical_bytes(&vec![
                b' ';
                MAX_INSTALLATION_ADMISSION_BYTES
                    as usize
                    + 1
            ])
            .is_err()
        );
    }

    #[test]
    fn completed_record_codec_preserves_exact_link_identities() {
        let intent = intent();
        let record = NativeInstallationAdmissionRecord {
            schema_version: NATIVE_INSTALLATION_SCHEMA_VERSION,
            product: intent.expected_product,
            image: intent.expected_image,
            inputs: NativeProgramLinkIdentity {
                object: Digest::sha256(b"object"),
                runtime_archive: Digest::sha256(b"runtime"),
                executable_cache_key: Digest::sha256(b"executable"),
                native_cache_key: Digest::sha256(b"native"),
            },
            build: NativeLinkBuildIdentity {
                product: intent.expected_product,
                execution_host: NativeLinkExecutionHost::current().unwrap(),
                adapter: NativeLinkAdapter::MsvcLinker,
                entrypoint: NormalizedRelativePath::new("bin/linker").unwrap(),
                toolchain: Digest::sha256(b"toolchain"),
                sysroot: Digest::sha256(b"sysroot"),
                qualification: Digest::sha256(b"link qualification"),
                storage: intent.storage,
                hosted_dependencies: Vec::new(),
            },
            layout: intent.layout,
            storage: intent.storage,
            loader_images: intent.loader_images,
            deployment_qualification: intent.deployment_qualification,
        };
        let bytes = record.to_canonical_bytes().unwrap();
        assert_eq!(
            NativeInstallationAdmissionRecord::from_canonical_bytes(&bytes).unwrap(),
            record
        );
        let mut inconsistent = record;
        inconsistent.build.product = Digest::sha256(b"different product");
        assert!(inconsistent.to_canonical_bytes().is_err());
    }
}
