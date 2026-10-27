use serde::{Deserialize, Serialize};

use crate::{ContractError, Digest, ProgramRevision};
use runmat_types::ClassIdentity;
use runmat_types::ProgramFunctionId;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutableEntrypointKind {
    Function,
    Script,
    DeclarationOnly,
}

/// Validated entrypoint identity. A declaration product has no callable
/// function ordinal or synthetic script body.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValidatedExecutableEntrypoint {
    Function(ProgramFunctionId),
    Script(ProgramFunctionId),
    DeclarationOnly(ClassIdentity),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableIdentity {
    pub program: ProgramRevision,
    pub root_package: String,
    pub entrypoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entrypoint_function: Option<ProgramFunctionId>,
    pub entrypoint_kind: ExecutableEntrypointKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub declaration_class: Option<ClassIdentity>,
    pub source_digest: Digest,
}

impl ExecutableIdentity {
    /// Validate the canonical identity fields without hashing executable payloads.
    pub fn validate(&self) -> Result<(), ContractError> {
        validate_identity("executable.root_package", &self.root_package, 256)?;
        validate_identity("executable.entrypoint", &self.entrypoint, 512)?;
        self.validated_entrypoint()?;
        Ok(())
    }

    pub fn validated_entrypoint(&self) -> Result<ValidatedExecutableEntrypoint, ContractError> {
        match (
            self.entrypoint_kind,
            self.entrypoint_function,
            self.declaration_class.as_ref(),
        ) {
            (ExecutableEntrypointKind::Function, Some(function), None) => {
                Ok(ValidatedExecutableEntrypoint::Function(function))
            }
            (ExecutableEntrypointKind::Script, Some(function), None) => {
                Ok(ValidatedExecutableEntrypoint::Script(function))
            }
            (ExecutableEntrypointKind::DeclarationOnly, None, Some(class))
                if self.entrypoint == class.display_name() =>
            {
                ClassIdentity::new(class.display_name()).map_err(|error| {
                    ContractError::invalid("executable.declaration_class", error.to_string())
                })?;
                Ok(ValidatedExecutableEntrypoint::DeclarationOnly(
                    class.clone(),
                ))
            }
            _ => Err(ContractError::invalid(
                "executable.entrypoint",
                "entrypoint kind, function ordinal, and declaration class disagree",
            )),
        }
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
