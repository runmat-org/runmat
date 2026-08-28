use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::MexArtifactManifest;

pub const MEX_ARTIFACT_BUNDLE_SCHEMA_VERSION: u16 = 1;
pub const MEX_ARTIFACT_MANIFEST_MEDIA_TYPE: &str = "application/vnd.runmat.mex-manifest+json";
pub const MEX_MODULE_MEDIA_TYPE: &str = "application/vnd.runmat.mex-module";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MexArtifactBundleEntry {
    pub manifest: Vec<u8>,
    pub module: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MexArtifactBundle {
    pub schema_version: u16,
    pub artifacts: Vec<MexArtifactBundleEntry>,
}

#[derive(Debug, Error)]
pub enum MexArtifactBundleError {
    #[error("MEX artifact bundle could not be encoded: {0}")]
    Encoding(#[from] serde_json::Error),
    #[error("MEX artifact bundle is not canonical")]
    Noncanonical,
    #[error("MEX artifact bundle uses unsupported schema version {0}")]
    UnsupportedSchema(u16),
    #[error("MEX artifact bundle entry is invalid: {0}")]
    InvalidArtifact(String),
    #[error("MEX module name `{0}` appears more than once in the artifact bundle")]
    DuplicateModule(String),
    #[error("MEX artifact identity `{0}` appears more than once in the artifact bundle")]
    DuplicateIdentity(String),
}

impl MexArtifactBundle {
    pub fn empty() -> Self {
        Self {
            schema_version: MEX_ARTIFACT_BUNDLE_SCHEMA_VERSION,
            artifacts: Vec::new(),
        }
    }

    pub fn new(artifacts: Vec<MexArtifactBundleEntry>) -> Result<Self, MexArtifactBundleError> {
        let mut decoded = artifacts
            .into_iter()
            .map(|entry| {
                let manifest = MexArtifactManifest::from_canonical_bytes(&entry.manifest)
                    .map_err(|error| MexArtifactBundleError::InvalidArtifact(error.to_string()))?;
                manifest
                    .validate_module(&entry.module)
                    .map_err(|error| MexArtifactBundleError::InvalidArtifact(error.to_string()))?;
                Ok((manifest, entry))
            })
            .collect::<Result<Vec<_>, MexArtifactBundleError>>()?;
        decoded.sort_by(|(left, _), (right, _)| left.identity.cmp(&right.identity));
        let mut names = BTreeSet::new();
        let mut identities = BTreeSet::new();
        for (manifest, _) in &decoded {
            if !names.insert(manifest.module_name.clone()) {
                return Err(MexArtifactBundleError::DuplicateModule(
                    manifest.module_name.clone(),
                ));
            }
            if !identities.insert(manifest.identity.clone()) {
                return Err(MexArtifactBundleError::DuplicateIdentity(
                    manifest.identity.to_string(),
                ));
            }
        }
        Ok(Self {
            schema_version: MEX_ARTIFACT_BUNDLE_SCHEMA_VERSION,
            artifacts: decoded.into_iter().map(|(_, entry)| entry).collect(),
        })
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, MexArtifactBundleError> {
        self.validate()?;
        Ok(serde_json::to_vec(self)?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, MexArtifactBundleError> {
        let decoded: Self = serde_json::from_slice(bytes)?;
        if decoded.schema_version != MEX_ARTIFACT_BUNDLE_SCHEMA_VERSION {
            return Err(MexArtifactBundleError::UnsupportedSchema(
                decoded.schema_version,
            ));
        }
        let normalized = Self::new(decoded.artifacts.clone())?;
        if normalized != decoded || serde_json::to_vec(&decoded)? != bytes {
            return Err(MexArtifactBundleError::Noncanonical);
        }
        Ok(decoded)
    }

    pub fn validate(&self) -> Result<(), MexArtifactBundleError> {
        if self.schema_version != MEX_ARTIFACT_BUNDLE_SCHEMA_VERSION {
            return Err(MexArtifactBundleError::UnsupportedSchema(
                self.schema_version,
            ));
        }
        if Self::new(self.artifacts.clone())? != *self {
            return Err(MexArtifactBundleError::Noncanonical);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::CCompilerFamily;
    use crate::{MexApi, MexSourceLanguage, MexTarget};

    fn entry(name: &str, module: &[u8]) -> MexArtifactBundleEntry {
        let target = MexTarget::current().unwrap();
        let compiler_family = if target.operating_system == "windows" {
            CCompilerFamily::Msvc
        } else {
            CCompilerFamily::GnuLike
        };
        let manifest = MexArtifactManifest::from_module(
            name,
            target.clone(),
            MexApi::R2018a,
            MexSourceLanguage::Cuda,
            compiler_family,
            module,
        )
        .unwrap();
        MexArtifactBundleEntry {
            manifest: manifest.canonical_bytes().unwrap(),
            module: module.to_vec(),
        }
    }

    #[test]
    fn bundle_is_canonical_exact_and_rejects_duplicate_module_names() {
        let bundle =
            MexArtifactBundle::new(vec![entry("zeta", b"z"), entry("alpha", b"a")]).unwrap();
        let bytes = bundle.canonical_bytes().unwrap();
        assert_eq!(
            MexArtifactBundle::from_canonical_bytes(&bytes).unwrap(),
            bundle
        );
        assert!(matches!(
            MexArtifactBundle::new(vec![entry("same", b"a"), entry("same", b"b")]),
            Err(MexArtifactBundleError::DuplicateModule(name)) if name == "same"
        ));
    }
}
