//! Execution-owner metadata travels beside an unchanged original program request.
use super::ProgramExecutionRequest;
use crate::{ArtifactError, ArtifactResult};
use runmat_execution::resource::ResourceRequest;
use serde::{Deserialize, Serialize};

pub const RESERVED_PROGRAM_REQUEST_SCHEMA_VERSION: u16 = 1;

/// A portable process handoff after the scheduler has reserved this task's resources.
/// Decoding validates the handoff; it does not independently prove scheduler ownership.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReservedProgramExecutionRequest {
    schema_version: u16,
    request: ProgramExecutionRequest,
    resources: ResourceRequest,
}

impl ReservedProgramExecutionRequest {
    pub fn new(
        request: ProgramExecutionRequest,
        resources: ResourceRequest,
    ) -> ArtifactResult<Self> {
        let handoff = Self {
            schema_version: RESERVED_PROGRAM_REQUEST_SCHEMA_VERSION,
            request,
            resources,
        };
        handoff.validate()?;
        Ok(handoff)
    }
    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != RESERVED_PROGRAM_REQUEST_SCHEMA_VERSION {
            return Err(ArtifactError::Invalid(
                "unsupported reserved program handoff schema".into(),
            ));
        }
        self.request.validate_for_portable_host()?;
        self.resources
            .validate()
            .map_err(|error| ArtifactError::Invalid(error.to_string()))
    }
    pub fn request(&self) -> &ProgramExecutionRequest {
        &self.request
    }
    pub fn resources(&self) -> &ResourceRequest {
        &self.resources
    }
    pub fn into_parts(self) -> (ProgramExecutionRequest, ResourceRequest) {
        (self.request, self.resources)
    }
}
