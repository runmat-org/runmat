use serde::{Deserialize, Serialize};

use super::ExecutableComponentRevisions;
use crate::{ContractError, Digest};

pub const EXECUTABLE_COMPONENT_MAX_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutableComponentKind {
    Mir,
    Analysis,
    Bytecode,
    VmLayout,
    FunctionRegistry,
    SourceMap,
}

impl ExecutableComponentKind {
    pub const REQUIRED: [Self; 6] = [
        Self::Mir,
        Self::Analysis,
        Self::Bytecode,
        Self::VmLayout,
        Self::FunctionRegistry,
        Self::SourceMap,
    ];

    pub(crate) const fn schema(self, revisions: &ExecutableComponentRevisions) -> u16 {
        match self {
            Self::Mir => revisions.mir_schema,
            Self::Analysis => revisions.analysis_schema,
            Self::Bytecode => revisions.bytecode_schema,
            Self::VmLayout => revisions.vm_layout_schema,
            Self::FunctionRegistry => revisions.function_registry_schema,
            Self::SourceMap => revisions.source_map_schema,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableComponentDescriptor {
    pub kind: ExecutableComponentKind,
    pub schema_version: u16,
    pub digest: Digest,
    pub encoded_length: u64,
}

impl ExecutableComponentDescriptor {
    pub fn from_payload(
        kind: ExecutableComponentKind,
        schema_version: u16,
        bytes: &[u8],
    ) -> Result<Self, ContractError> {
        let descriptor = Self {
            kind,
            schema_version,
            digest: Digest::sha256(bytes),
            encoded_length: bytes.len() as u64,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version == 0 {
            return Err(ContractError::invalid(
                "executable.components.schema_version",
                "version must be non-zero",
            ));
        }
        if self.encoded_length == 0 {
            return Err(ContractError::invalid(
                "executable.components.encoded_length",
                "required component payloads must not be empty",
            ));
        }
        if self.encoded_length > EXECUTABLE_COMPONENT_MAX_BYTES {
            return Err(ContractError::Limit {
                field: "executable.components.encoded_length",
                limit: EXECUTABLE_COMPONENT_MAX_BYTES,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableComponentPayload {
    pub kind: ExecutableComponentKind,
    pub bytes: Vec<u8>,
}

impl ExecutableComponentPayload {
    pub fn new(kind: ExecutableComponentKind, bytes: Vec<u8>) -> Result<Self, ContractError> {
        if bytes.is_empty() {
            return Err(ContractError::invalid(
                "executable.payloads.bytes",
                "required component payloads must not be empty",
            ));
        }
        if bytes.len() as u64 > EXECUTABLE_COMPONENT_MAX_BYTES {
            return Err(ContractError::Limit {
                field: "executable.payloads.bytes",
                limit: EXECUTABLE_COMPONENT_MAX_BYTES,
            });
        }
        Ok(Self { kind, bytes })
    }
}

pub(crate) fn validate_component_descriptors(
    components: &[ExecutableComponentDescriptor],
    revisions: &ExecutableComponentRevisions,
) -> Result<(), ContractError> {
    if components.len() != ExecutableComponentKind::REQUIRED.len() {
        return Err(ContractError::invalid(
            "executable.components",
            "all required executable components must be present exactly once",
        ));
    }
    for (descriptor, required) in components.iter().zip(ExecutableComponentKind::REQUIRED) {
        descriptor.validate()?;
        if descriptor.kind != required {
            return Err(ContractError::invalid(
                "executable.components",
                "components must be complete, unique, and sorted by kind",
            ));
        }
        if descriptor.schema_version != descriptor.kind.schema(revisions) {
            return Err(ContractError::invalid(
                "executable.components.schema_version",
                "component schema does not match executable revisions",
            ));
        }
    }
    Ok(())
}
