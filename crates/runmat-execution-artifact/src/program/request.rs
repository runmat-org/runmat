use runmat_execution::value::{ValueLimits, ValuePayload};
use serde::{Deserialize, Serialize};

use super::{ExecutableForm, ProgramArtifact, ProgramBuildRecipe};
use crate::{ArtifactError, ArtifactResult};

pub const PROGRAM_EXECUTION_REQUEST_SCHEMA_V1: u16 = 1;
pub const MAX_PROGRAM_EXECUTION_ARGUMENTS: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionDescriptor {
    pub schema_version: u16,
    pub recipe: ProgramBuildRecipe,
    pub artifact: ProgramArtifact,
    pub function: usize,
    pub requested_outputs: u16,
}

impl ProgramExecutionDescriptor {
    pub fn validate(&self) -> ArtifactResult<()> {
        self.artifact.validate_against(&self.recipe)?;
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_V1
            || !entrypoint_matches(self.artifact.form, self.function, &self.recipe.entrypoint)
            || self.requested_outputs != self.recipe.outputs.requested_outputs
        {
            return Err(ArtifactError::Invalid(
                "program descriptor has an inconsistent callable or output contract".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_for_portable_host(&self) -> ArtifactResult<()> {
        self.validate()?;
        self.artifact.target.validate_for_portable_host()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionInputs {
    pub schema_version: u16,
    pub arguments: Vec<ValuePayload>,
}

impl ProgramExecutionInputs {
    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_V1
            || self.arguments.len() > MAX_PROGRAM_EXECUTION_ARGUMENTS
        {
            return Err(ArtifactError::Invalid(
                "program inputs use an unsupported schema or exceed their bound".into(),
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
    pub fn from_parts(
        descriptor: ProgramExecutionDescriptor,
        inputs: ProgramExecutionInputs,
    ) -> ArtifactResult<Self> {
        descriptor.validate()?;
        inputs.validate()?;
        let request = Self {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
            recipe: descriptor.recipe,
            artifact: descriptor.artifact,
            function: descriptor.function,
            arguments: inputs.arguments,
            requested_outputs: descriptor.requested_outputs,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_V1 {
            return Err(ArtifactError::Invalid(
                "unsupported program execution request schema".into(),
            ));
        }
        self.artifact.validate_against(&self.recipe)?;
        if !entrypoint_matches(self.artifact.form, self.function, &self.recipe.entrypoint)
            || self.requested_outputs != self.recipe.outputs.requested_outputs
            || self.arguments.len() > MAX_PROGRAM_EXECUTION_ARGUMENTS
            || (matches!(
                self.artifact.form,
                ExecutableForm::InterpreterScriptV1 | ExecutableForm::TestAttemptV1
            ) && !self.arguments.is_empty())
        {
            return Err(ArtifactError::Invalid(
                "program execution request has an inconsistent callable, output contract, or argument count".into(),
            ));
        }
        if self.artifact.form == ExecutableForm::ExecutableUnitV3 {
            let envelope = self
                .artifact
                .executable_unit()?
                .expect("executable-unit form returns its validated envelope");
            if usize::try_from(envelope.manifest.identity.entrypoint_function.0).ok()
                != Some(self.function)
                || (envelope.manifest.identity.entrypoint_kind
                    == runmat_execution::ExecutableEntrypointKind::Script
                    && !self.arguments.is_empty())
            {
                return Err(ArtifactError::Invalid(
                    "executable unit request does not match its declared entrypoint".into(),
                ));
            }
        }
        for argument in &self.arguments {
            argument
                .validate(ValueLimits::default())
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        }
        Ok(())
    }

    pub fn validate_for_portable_host(&self) -> ArtifactResult<()> {
        self.validate()?;
        self.artifact.target.validate_for_portable_host()
    }
}

fn entrypoint_matches(form: ExecutableForm, function: usize, entrypoint: &str) -> bool {
    match form {
        ExecutableForm::InterpreterBytecodeV1 => function.to_string() == entrypoint,
        ExecutableForm::InterpreterScriptV1 => function == 0 && entrypoint == "script",
        ExecutableForm::TestAttemptV1 => function == 0 && entrypoint == "test_attempt",
        ExecutableForm::MeshingWorkloadV2 => function == 0 && entrypoint == "meshing_workload",
        ExecutableForm::ExecutableUnitV3 => function.to_string() == entrypoint,
        ExecutableForm::NativeObjectV1 => function.to_string() == entrypoint,
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
            schema_version: crate::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision,
            entrypoint: "7".into(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            execution_mode: "interpreter".into(),
            target: crate::ProgramTarget::portable("portable-test"),
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

    #[test]
    fn script_form_has_an_explicit_argument_free_entrypoint() {
        let mut script = request();
        script.recipe.entrypoint = "script".into();
        script.function = 0;
        script.artifact = ProgramArtifact::materialize(
            &script.recipe,
            ExecutableForm::InterpreterScriptV1,
            b"script-bytecode".to_vec(),
        )
        .unwrap();
        script.validate().unwrap();
        script
            .arguments
            .push(runmat_execution::value::ValuePayload::Inline(Box::new(
                runmat_execution::value::InlineValue::String("unexpected".into()),
            )));
        assert!(script.validate().is_err());
    }
}
