use serde::{Deserialize, Serialize};

use super::{ExecutableComponentKind, ExecutableComponentPayload, ExecutableUnitManifest};
use crate::{ContractError, Digest};

pub const EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableUnitEnvelope {
    pub schema_version: u16,
    pub manifest: ExecutableUnitManifest,
    pub payloads: Vec<ExecutableComponentPayload>,
}

impl ExecutableUnitEnvelope {
    pub fn new(
        manifest: ExecutableUnitManifest,
        payloads: Vec<ExecutableComponentPayload>,
    ) -> Result<Self, ContractError> {
        let envelope = Self {
            schema_version: super::EXECUTABLE_UNIT_SCHEMA_VERSION,
            manifest,
            payloads,
        };
        envelope.validate()?;
        Ok(envelope)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != super::EXECUTABLE_UNIT_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                actual: self.schema_version,
                supported: super::EXECUTABLE_UNIT_SCHEMA_VERSION,
            });
        }
        self.manifest.validate()?;
        if self.payloads.len() != self.manifest.components.len() {
            return Err(ContractError::invalid(
                "executable.payloads",
                "payload count does not match the component manifest",
            ));
        }
        for (payload, descriptor) in self.payloads.iter().zip(&self.manifest.components) {
            if payload.kind != descriptor.kind
                || payload.bytes.len() as u64 != descriptor.encoded_length
                || Digest::sha256(&payload.bytes) != descriptor.digest
            {
                return Err(ContractError::invalid(
                    "executable.payloads",
                    "payload identity does not match the component manifest",
                ));
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ContractError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| ContractError::invalid("executable.envelope", error.to_string()))?;
        if bytes.len() > EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES {
            return Err(ContractError::Limit {
                field: "executable.envelope",
                limit: EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES as u64,
            });
        }
        Ok(bytes)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ContractError> {
        if bytes.len() > EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES {
            return Err(ContractError::Limit {
                field: "executable.envelope",
                limit: EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES as u64,
            });
        }
        let envelope = serde_json::from_slice::<Self>(bytes)
            .map_err(|error| ContractError::invalid("executable.envelope", error.to_string()))?;
        envelope.validate()?;
        if envelope.canonical_bytes()? != bytes {
            return Err(ContractError::invalid(
                "executable.envelope",
                "encoding is valid JSON but not canonical RunMat JSON",
            ));
        }
        Ok(envelope)
    }

    pub fn cache_key(&self) -> Result<Digest, ContractError> {
        self.canonical_bytes().map(Digest::sha256)
    }

    pub fn component(&self, kind: ExecutableComponentKind) -> Option<&ExecutableComponentPayload> {
        self.payloads.iter().find(|payload| payload.kind == kind)
    }
}
