//! Exact invocation data, distinct from compiler/runtime compatibility identity.
use crate::{ContractError, Digest};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
mod storage;
pub use storage::EnvironmentStorageProfile;

pub const MAX_ENVIRONMENT_ENTRIES: usize = 4096;
pub const MAX_ENVIRONMENT_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentSnapshot {
    entries: Vec<(String, String)>,
}

impl std::fmt::Debug for EnvironmentSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvironmentSnapshot")
            .field("entry_count", &self.entries.len())
            .finish_non_exhaustive()
    }
}

impl EnvironmentSnapshot {
    pub fn new(entries: Vec<(String, String)>) -> Result<Self, ContractError> {
        let snapshot = Self { entries };
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn entries(&self) -> &[(String, String)] {
        &self.entries
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        Self::validate_entries(&self.entries)
    }

    pub fn validate_entries(entries: &[(String, String)]) -> Result<(), ContractError> {
        let bytes = entries.iter().try_fold(0_usize, |n, (key, value)| {
            n.checked_add(key.len())?.checked_add(value.len())
        });
        if entries.len() > MAX_ENVIRONMENT_ENTRIES
            || bytes.is_none_or(|bytes| bytes > MAX_ENVIRONMENT_BYTES)
            || entries.windows(2).any(|pair| pair[0].0 >= pair[1].0)
            || entries.iter().any(|(key, value)| {
                key.is_empty() || key.contains(['=', '\0']) || value.contains('\0')
            })
        {
            return Err(ContractError::invalid(
                "environment snapshot",
                "invalid shape or exceeded bound",
            ));
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<Digest, ContractError> {
        self.validate()?;
        let mut hash = Sha256::new();
        hash.update(b"runmat-invocation-environment/v1\0");
        for (name, value) in &self.entries {
            hash.update((name.len() as u64).to_be_bytes());
            hash.update(name.as_bytes());
            hash.update((value.len() as u64).to_be_bytes());
            hash.update(value.as_bytes());
        }
        Ok(Digest::from_bytes(hash.finalize().into()))
    }
}

#[cfg(test)]
mod tests;
