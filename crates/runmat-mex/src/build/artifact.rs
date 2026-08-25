use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

use runmat_types::{
    CapabilityRequirement, CapabilitySet, ForeignAdapterRequirement, InteropManifest,
    INTEROP_MANIFEST_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use super::{sdk, CCompilerFamily, MexApi, MexBuildError, MexTarget};
use crate::MEX_HOST_ABI_VERSION;

pub const MEX_ARTIFACT_SCHEMA_VERSION: u16 = 1;
pub const MEX_ADAPTER_ID: &str = "mex-c";
pub const MEX_ADAPTER_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MexArtifactIdentity(String);

impl MexArtifactIdentity {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for MexArtifactIdentity {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MexArtifactManifest {
    pub schema_version: u16,
    pub identity: MexArtifactIdentity,
    pub module_name: String,
    pub target: MexTarget,
    pub api: MexApi,
    pub compiler_family: String,
    pub host_abi_version: u32,
    pub sdk_digest: String,
    pub module_digest: String,
    pub module_bytes: u64,
}

#[derive(Serialize)]
struct IdentityInput<'a> {
    schema_version: u16,
    module_name: &'a str,
    target: &'a MexTarget,
    api: MexApi,
    compiler_family: &'a str,
    host_abi_version: u32,
    sdk_digest: &'a str,
    module_digest: &'a str,
    module_bytes: u64,
}

impl MexArtifactManifest {
    pub fn path_for_module(module: &Path) -> PathBuf {
        module.with_extension(format!(
            "{}.runmat.json",
            module
                .extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or("mex")
        ))
    }

    pub(super) fn from_module(
        module_name: &str,
        target: MexTarget,
        api: MexApi,
        compiler_family: CCompilerFamily,
        module: &[u8],
    ) -> Result<Self, MexBuildError> {
        let module_digest = digest(module);
        let sdk_digest = sdk::content_digest();
        let compiler_family = compiler_family.as_str();
        let input = IdentityInput {
            schema_version: MEX_ARTIFACT_SCHEMA_VERSION,
            module_name,
            target: &target,
            api,
            compiler_family,
            host_abi_version: MEX_HOST_ABI_VERSION,
            sdk_digest: &sdk_digest,
            module_digest: &module_digest,
            module_bytes: module.len() as u64,
        };
        let identity_bytes = serde_json::to_vec(&input)
            .map_err(|source| MexBuildError::ArtifactEncoding { source })?;
        let identity = MexArtifactIdentity(format!("mex:v1:{}", digest(&identity_bytes)));
        let manifest = Self {
            schema_version: MEX_ARTIFACT_SCHEMA_VERSION,
            identity,
            module_name: module_name.to_string(),
            target,
            api,
            compiler_family: compiler_family.to_string(),
            host_abi_version: MEX_HOST_ABI_VERSION,
            sdk_digest,
            module_digest,
            module_bytes: module.len() as u64,
        };
        manifest.validate_module(module)?;
        Ok(manifest)
    }

    pub fn validate_module(&self, module: &[u8]) -> Result<(), MexBuildError> {
        self.validate_metadata()?;
        if self.module_digest != digest(module) || self.module_bytes != module.len() as u64 {
            return Err(MexBuildError::InvalidArtifactManifest);
        }
        Ok(())
    }

    pub fn validate_current_module(&self, module: &[u8]) -> Result<(), MexBuildError> {
        self.validate_module(module)?;
        if self.target != MexTarget::current()? {
            return Err(MexBuildError::InvalidArtifactManifest);
        }
        Ok(())
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, MexBuildError> {
        let manifest = serde_json::from_slice::<Self>(bytes)
            .map_err(|source| MexBuildError::ArtifactEncoding { source })?;
        manifest.validate_metadata()?;
        if manifest.canonical_bytes()? != bytes {
            return Err(MexBuildError::InvalidArtifactManifest);
        }
        Ok(manifest)
    }

    fn validate_metadata(&self) -> Result<(), MexBuildError> {
        self.target.validate()?;
        let compiler_family = match self.compiler_family.as_str() {
            "gnu-like" => CCompilerFamily::GnuLike,
            "msvc" => CCompilerFamily::Msvc,
            _ => return Err(MexBuildError::InvalidArtifactManifest),
        };
        self.target.validate_compiler(compiler_family)?;
        if self.schema_version != MEX_ARTIFACT_SCHEMA_VERSION
            || self.module_name.is_empty()
            || self.module_name.len() > 256
            || !self.module_name.is_ascii()
            || self.module_name.chars().any(char::is_control)
            || self.compiler_family.is_empty()
            || self.host_abi_version != MEX_HOST_ABI_VERSION
            || !valid_digest(&self.sdk_digest)
            || !valid_digest(&self.module_digest)
            || self.module_bytes == 0
        {
            return Err(MexBuildError::InvalidArtifactManifest);
        }
        let input = IdentityInput {
            schema_version: self.schema_version,
            module_name: &self.module_name,
            target: &self.target,
            api: self.api,
            compiler_family: &self.compiler_family,
            host_abi_version: self.host_abi_version,
            sdk_digest: &self.sdk_digest,
            module_digest: &self.module_digest,
            module_bytes: self.module_bytes,
        };
        let identity_bytes = serde_json::to_vec(&input)
            .map_err(|source| MexBuildError::ArtifactEncoding { source })?;
        let expected = format!("mex:v1:{}", digest(&identity_bytes));
        if self.identity.as_str() != expected {
            return Err(MexBuildError::InvalidArtifactManifest);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, MexBuildError> {
        self.validate_metadata()?;
        serde_json::to_vec(self).map_err(|source| MexBuildError::ArtifactEncoding { source })
    }

    pub fn interop_manifest(&self) -> InteropManifest {
        InteropManifest {
            schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: Vec::new(),
            adapters: vec![ForeignAdapterRequirement {
                adapter: MEX_ADAPTER_ID.to_string(),
                minimum_version: MEX_ADAPTER_VERSION,
                capabilities: CapabilitySet(BTreeSet::from([
                    CapabilityRequirement::NativeCode,
                    CapabilityRequirement::ForeignRuntime,
                ])),
                artifact_identities: vec![self.identity.to_string()],
            }],
        }
    }
}

fn digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(71);
    encoded.push_str("sha256:");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_identity_is_exact_deterministic_and_admissible() {
        let target = MexTarget::current().unwrap();
        let compiler_family = crate::build::compiler_family(&crate::build::default_c_compiler());
        let first = MexArtifactManifest::from_module(
            "fixture",
            target.clone(),
            MexApi::R2017b,
            compiler_family,
            b"module bytes",
        )
        .unwrap();
        let second = MexArtifactManifest::from_module(
            "fixture",
            target,
            MexApi::R2017b,
            compiler_family,
            b"module bytes",
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.canonical_bytes().unwrap(),
            second.canonical_bytes().unwrap()
        );
        assert_eq!(
            MexArtifactManifest::from_canonical_bytes(&first.canonical_bytes().unwrap()).unwrap(),
            first
        );
        first.validate_module(b"module bytes").unwrap();
        assert!(first.validate_module(b"different").is_err());
        first.interop_manifest().validate().unwrap();
    }
}
