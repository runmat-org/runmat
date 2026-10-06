use serde::{Deserialize, Serialize};

use crate::{ContractError, Digest};
use runmat_types::ProgramSourceId;

/// Exact source text and deterministic ordinal in one immutable program.
/// Session-local `SourceId` values are intentionally absent from this wire
/// contract; every executor must issue its own runtime source leases.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableSourceRecordV2 {
    pub source_id: ProgramSourceId,
    pub owner_identity: String,
    pub relative_path: String,
    pub content_revision: Digest,
    pub display_name: String,
    pub text: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableSourceMapV2 {
    pub entries: Vec<ExecutableSourceRecordV2>,
}

impl ExecutableSourceMapV2 {
    pub fn new(entries: Vec<ExecutableSourceRecordV2>) -> Result<Self, ContractError> {
        let map = Self { entries };
        map.validate()?;
        Ok(map)
    }

    /// Decode only after a caller admits source-map schema 2. A matching
    /// component digest does not by itself authorize forged text provenance.
    pub fn decode(bytes: &[u8]) -> Result<Self, ContractError> {
        let map: Self = serde_json::from_slice(bytes)
            .map_err(|error| ContractError::invalid("executable.source_map", error.to_string()))?;
        map.validate()?;
        Ok(map)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.entries.is_empty() {
            return Err(ContractError::invalid(
                "executable.source_map",
                "source map has no primary source",
            ));
        }
        for (ordinal, entry) in self.entries.iter().enumerate() {
            if u32::try_from(ordinal).ok().map(ProgramSourceId) != Some(entry.source_id) {
                return Err(ContractError::invalid(
                    "executable.source_map.source_id",
                    "source ordinals must be dense, unique, and ordered from zero",
                ));
            }
            if entry.content_revision != Digest::sha256(entry.text.as_bytes()) {
                return Err(ContractError::invalid(
                    "executable.source_map.content_revision",
                    "source content revision does not match exact text",
                ));
            }
        }
        Ok(())
    }
}
