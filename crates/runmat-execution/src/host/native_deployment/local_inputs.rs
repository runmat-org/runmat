//! Selected-invocation inputs for locally owned native effects.
//!
//! Program and request schemas share these inputs. Linked implementations and
//! independently admitted issuers supply execution authority separately.

use super::{
    NativeDeploymentProfile, NativeDeploymentServiceBehavior, NativeDeploymentServiceKind,
    NativeSelectedProductBinding,
};
use crate::{ContractError, Digest, EnvironmentSnapshot};
use runmat_types::EffectKind;
use serde::{Deserialize, Serialize};

pub const NATIVE_LOCAL_SERVICE_INPUTS_SCHEMA_VERSION: u16 = 1;
const MAX_FILESYSTEM_ENTRIES: usize = 16_384;
// JSON's byte-array encoding can expand each raw byte to four characters.
// Leave bounded room in the existing 64 MiB manifest for the program, the
// environment snapshot, and path metadata without changing its codec.
const MAX_FILESYSTEM_BYTES: usize = 8 * 1024 * 1024;

/// Only invocation data. A decoded value does not issue a service or prove a
/// linked implementation. Workspace frames are derived from the selected
/// program and arguments, so they have no parallel input authority here.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeLocalServiceInputs {
    pub schema_version: u16,
    pub environment: Option<NativeEnvironmentInputs>,
    pub random: Option<NativeRandomInputs>,
    pub filesystem: Option<NativeFilesystemInputs>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeRandomInputs {
    /// Existing language MT policy and existing GPU Threefry policy consume
    /// these independent initial seeds at position zero. Later `rng` changes
    /// remain owned by the same invocation's live stream state.
    pub language_seed: u64,
    pub gpu_seed: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeEnvironmentInputs {
    pub variables: EnvironmentSnapshot,
    /// Language working directory is session environment, not a grant to read
    /// the filesystem. `pwd` is EnvironmentRead; `cd` additionally requires a
    /// selected FilesystemRead issuer for target-directory inspection.
    pub current_directory: String,
}

/// Virtual-rooted, UTF-8 input tree. `MemoryFsProvider` materializes this
/// under an invocation owner, never the process-global filesystem registry.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeFilesystemInputs {
    pub entries: Vec<NativeFilesystemEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeFilesystemEntry {
    Directory {
        path: String,
    },
    File {
        path: String,
        bytes: Vec<u8>,
        readonly: bool,
    },
}

impl NativeFilesystemEntry {
    fn path(&self) -> &str {
        match self {
            Self::Directory { path } | Self::File { path, .. } => path,
        }
    }
}

impl NativeFilesystemInputs {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.entries.len() > MAX_FILESYSTEM_ENTRIES
            || self
                .entries
                .windows(2)
                .any(|pair| pair[0].path() >= pair[1].path())
            || self
                .entries
                .iter()
                .any(|entry| !virtual_path(entry.path()) || entry.path() == "/")
        {
            return Err(invalid(
                "noncanonical virtual filesystem paths or inventory",
            ));
        }
        let total = self.entries.iter().try_fold(0usize, |size, entry| {
            let size = size.checked_add(entry.path().len())?;
            match entry {
                NativeFilesystemEntry::File { bytes, .. } => size.checked_add(bytes.len()),
                NativeFilesystemEntry::Directory { .. } => Some(size),
            }
        });
        if total.is_none_or(|size| size > MAX_FILESYSTEM_BYTES) {
            return Err(invalid("virtual filesystem inputs exceed the bounded size"));
        }
        Ok(())
    }
}

impl NativeLocalServiceInputs {
    pub fn validate_for(
        &self,
        product: &NativeSelectedProductBinding,
    ) -> Result<(), ContractError> {
        product.validate()?;
        if self.schema_version != NATIVE_LOCAL_SERVICE_INPUTS_SCHEMA_VERSION {
            return Err(invalid("unsupported local service input schema"));
        }
        if let Some(environment) = &self.environment {
            environment.variables.validate()?;
            if !virtual_path(&environment.current_directory) {
                return Err(invalid("noncanonical selected working directory"));
            }
        }
        if let Some(filesystem) = &self.filesystem {
            filesystem.validate()?;
        }
        let deterministic = product.profile == NativeDeploymentProfile::DeterministicDeployed;
        if !deterministic {
            return Err(invalid("hosted local effects do not take virtual inputs"));
        }
        for (effects, supplied) in [
            (
                &[EffectKind::EnvironmentRead, EffectKind::EnvironmentWrite][..],
                self.environment.is_some(),
            ),
            (&[EffectKind::Randomness][..], self.random.is_some()),
            (
                &[EffectKind::FilesystemRead, EffectKind::FilesystemWrite][..],
                self.filesystem.is_some(),
            ),
        ] {
            let selected = product.services.iter().any(|service| {
                effects
                    .iter()
                    .copied()
                    .any(|effect| service.kind == NativeDeploymentServiceKind::Effect(effect))
            });
            if supplied != (deterministic && selected) {
                return Err(invalid(
                    "local service inputs differ from the selected behavior/effect closure",
                ));
            }
            if selected
                && product.services.iter().any(|service| {
                    effects.iter().copied().any(|effect| {
                        service.kind == NativeDeploymentServiceKind::Effect(effect)
                            && (service.behavior
                                == NativeDeploymentServiceBehavior::DeterministicDeclared)
                                != deterministic
                    })
                })
            {
                return Err(invalid(
                    "selected local service behavior differs from profile",
                ));
            }
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> Result<Digest, ContractError> {
        if self.schema_version != NATIVE_LOCAL_SERVICE_INPUTS_SCHEMA_VERSION {
            return Err(invalid("unsupported local service input schema"));
        }
        if let Some(environment) = &self.environment {
            environment.variables.validate()?;
            if !virtual_path(&environment.current_directory) {
                return Err(invalid("noncanonical selected working directory"));
            }
        }
        if let Some(filesystem) = &self.filesystem {
            filesystem.validate()?;
        }
        Digest::sha256_serialized(&("runmat-native-local-service-inputs-v1", self))
            .map_err(|error| invalid(&error.to_string()))
    }
}

/// Structural declaration only: selected linked issuers and installed live
/// ports remain separate authority from decoded invocation input bytes.
pub fn validate_native_local_inputs(
    product: Option<&NativeSelectedProductBinding>,
    inputs: Option<&NativeLocalServiceInputs>,
) -> Result<(), ContractError> {
    match (product, inputs) {
        (Some(product), Some(inputs)) => {
            if !product
                .services
                .iter()
                .any(|service| service.kind.requires_local_input_payload())
            {
                return Err(invalid("selected product has no local service inputs"));
            }
            inputs.validate_for(product)
        }
        (Some(product), None) => {
            product.validate()?;
            if product.profile == NativeDeploymentProfile::DeterministicDeployed
                && product
                    .services
                    .iter()
                    .any(|service| service.kind.requires_local_input_payload())
            {
                Err(invalid(
                    "selected deterministic local service inputs are missing",
                ))
            } else {
                Ok(())
            }
        }
        (None, None) => Ok(()),
        (None, Some(_)) => Err(invalid(
            "unselected invocation supplied local service inputs",
        )),
    }
}

fn virtual_path(path: &str) -> bool {
    path == "/"
        || (path.starts_with('/')
            && !path.ends_with('/')
            && path.split('/').skip(1).all(|part| {
                !part.is_empty() && part != "." && part != ".." && !part.contains('\0')
            }))
}

fn invalid(message: &str) -> ContractError {
    ContractError::invalid("native.local_service_inputs", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_tree_rejects_parent_escape_duplicate_and_unbounded_bytes() {
        let valid = NativeFilesystemInputs {
            entries: vec![
                NativeFilesystemEntry::Directory {
                    path: "/data".into(),
                },
                NativeFilesystemEntry::File {
                    path: "/data/input.mat".into(),
                    bytes: vec![0, 1, 255],
                    readonly: true,
                },
            ],
        };
        assert!(valid.validate().is_ok());
        let mut escape = valid.clone();
        escape.entries[1] = NativeFilesystemEntry::File {
            path: "/data/../secret".into(),
            bytes: vec![1],
            readonly: false,
        };
        assert!(escape.validate().is_err());
        let mut duplicate = valid.clone();
        duplicate.entries.push(valid.entries[1].clone());
        assert!(duplicate.validate().is_err());
        let too_large = NativeFilesystemInputs {
            entries: vec![NativeFilesystemEntry::File {
                path: "/blob".into(),
                bytes: vec![0; MAX_FILESYSTEM_BYTES],
                readonly: false,
            }],
        };
        assert!(too_large.validate().is_err());
    }

    #[test]
    fn invocation_input_fingerprint_includes_each_original_stream_seed() {
        let one = NativeLocalServiceInputs {
            schema_version: NATIVE_LOCAL_SERVICE_INPUTS_SCHEMA_VERSION,
            environment: None,
            random: Some(NativeRandomInputs {
                language_seed: 0,
                gpu_seed: 7,
            }),
            filesystem: None,
        };
        let mut two = one.clone();
        two.random.as_mut().unwrap().gpu_seed = 8;
        assert_ne!(one.fingerprint().unwrap(), two.fingerprint().unwrap());
    }
}
