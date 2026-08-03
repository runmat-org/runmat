use runmat_execution::value::{ValueLimits, ValuePayload};
use serde::{Deserialize, Serialize};

use super::{ProgramArtifact, ProgramBuildRecipe};
use crate::{ArtifactError, ArtifactResult};

pub const PROGRAM_EXECUTION_REQUEST_SCHEMA_V1: u16 = 1;
pub const MAX_PROGRAM_EXECUTION_ARGUMENTS: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionRequest {
    pub schema_version: u16,
    pub recipe: ProgramBuildRecipe,
    pub artifact: ProgramArtifact,
    pub function: usize,
    pub arguments: Vec<ValuePayload>,
    pub requested_outputs: u16,
}

impl ProgramExecutionRequest {
    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_V1 {
            return Err(ArtifactError::Invalid(
                "unsupported program execution request schema".into(),
            ));
        }
        self.artifact.validate_against(&self.recipe)?;
        if self.function.to_string() != self.recipe.entrypoint
            || self.requested_outputs != self.recipe.outputs.requested_outputs
            || self.arguments.len() > MAX_PROGRAM_EXECUTION_ARGUMENTS
        {
            return Err(ArtifactError::Invalid(
                "program execution request has an inconsistent callable, output contract, or argument count".into(),
            ));
        }
        for argument in &self.arguments {
            argument
                .validate(ValueLimits::default())
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProgramExecutionResponse {
    Success { value: ValuePayload },
    Failure { message: String },
}

#[cfg(test)]
mod tests {
    use runmat_execution::{Digest, OutputContract, ProgramEnvironment, ProgramRevision};

    use super::*;
    use crate::ExecutableForm;

    fn request() -> ProgramExecutionRequest {
        let revision = ProgramRevision::new(
            Digest::sha256(b"graph"),
            Digest::sha256(b"source"),
            ProgramEnvironment::new(
                1,
                1,
                Digest::sha256(b"runtime"),
                Digest::sha256(b"catalog"),
                "matlab",
            )
            .unwrap(),
        )
        .unwrap();
        let recipe = ProgramBuildRecipe {
            schema_version: 1,
            program_revision: revision,
            entrypoint: "7".into(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            execution_mode: "interpreter".into(),
            target_profile: "portable-test".into(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::InterpreterBytecodeV1,
            b"program".to_vec(),
        )
        .unwrap();
        ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
            recipe,
            artifact,
            function: 7,
            arguments: Vec::new(),
            requested_outputs: 1,
        }
    }

    #[test]
    fn exact_program_request_validates_every_identity_boundary() {
        request().validate().unwrap();
        let mut mismatched = request();
        mismatched.function = 8;
        assert!(mismatched.validate().is_err());
        let mut tampered = request();
        tampered.artifact.executable_bytes.push(0);
        assert!(tampered.validate().is_err());
    }

    #[test]
    fn exact_program_request_rejects_unknown_schemas_and_output_drift() {
        let mut unknown = request();
        unknown.schema_version += 1;
        assert!(unknown.validate().is_err());
        let mut outputs = request();
        outputs.requested_outputs = 2;
        assert!(outputs.validate().is_err());
    }
}
