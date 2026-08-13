use serde::{Deserialize, Serialize};

use crate::{ContractError, Digest, ProgramRevision};
use runmat_types::ProgramFunctionId;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableIdentity {
    pub program: ProgramRevision,
    pub root_package: String,
    pub entrypoint: String,
    pub entrypoint_function: ProgramFunctionId,
    pub source_digest: Digest,
}

impl ExecutableIdentity {
    pub(crate) fn validate(&self) -> Result<(), ContractError> {
        validate_identity("executable.root_package", &self.root_package, 256)?;
        validate_identity("executable.entrypoint", &self.entrypoint, 512)
    }
}

pub(crate) fn validate_identity(
    field: &'static str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), ContractError> {
    if value.is_empty() || value.len() > maximum_bytes || value.chars().any(char::is_control) {
        return Err(ContractError::invalid(
            field,
            format!("must be 1..={maximum_bytes} bytes without control characters"),
        ));
    }
    Ok(())
}
