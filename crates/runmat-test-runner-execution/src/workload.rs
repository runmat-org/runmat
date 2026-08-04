use runmat_execution::value::{DenseValue, ElementType, InlineValue, ValuePayload};
use runmat_execution::OutputContract;
use runmat_execution_artifact::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_test::identity::TestId;
use runmat_test_runner::worker::{RunSubmission, WorkerExecution};
use serde::{Deserialize, Serialize};

pub const TEST_ATTEMPT_EXECUTION_MODE: &str = "test";
pub const TEST_ATTEMPT_TARGET_PROFILE: &str = "portable-test-attempt-v1";
const TEST_ATTEMPT_SCHEMA_V1: u16 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestAttemptWorkload {
    pub schema_version: u16,
    pub submission: RunSubmission,
    pub test_id: TestId,
    pub attempt: u32,
}

impl TestAttemptWorkload {
    pub fn new(submission: RunSubmission, test_id: TestId, attempt: u32) -> Result<Self, String> {
        let workload = Self {
            schema_version: TEST_ATTEMPT_SCHEMA_V1,
            submission,
            test_id,
            attempt,
        };
        workload.validate()?;
        Ok(workload)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.submission
            .snapshot
            .validate()
            .map_err(|error| error.to_string())?;
        if self.schema_version != TEST_ATTEMPT_SCHEMA_V1
            || self.attempt == 0
            || self.submission.plan.program_revision != self.submission.snapshot.program_revision
            || !self
                .submission
                .plan
                .tests()
                .any(|test| test.id == self.test_id)
        {
            return Err("test attempt does not belong to its exact frozen submission".into());
        }
        Ok(())
    }

    pub fn program_request(&self) -> Result<ProgramExecutionRequest, String> {
        self.validate()?;
        let recipe = ProgramBuildRecipe {
            schema_version: 1,
            program_revision: self.submission.plan.program_revision.clone(),
            entrypoint: "test_attempt".into(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            execution_mode: TEST_ATTEMPT_EXECUTION_MODE.into(),
            target_profile: TEST_ATTEMPT_TARGET_PROFILE.into(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let bytes = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        let artifact = ProgramArtifact::materialize(&recipe, ExecutableForm::TestAttemptV1, bytes)
            .map_err(|error| error.to_string())?;
        let request = ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
            recipe,
            artifact,
            function: 0,
            arguments: Vec::new(),
            requested_outputs: 1,
        };
        request.validate().map_err(|error| error.to_string())?;
        Ok(request)
    }

    pub fn from_program_request(request: &ProgramExecutionRequest) -> Result<Self, String> {
        request.validate().map_err(|error| error.to_string())?;
        if request.artifact.form != ExecutableForm::TestAttemptV1
            || request.recipe.execution_mode != TEST_ATTEMPT_EXECUTION_MODE
            || request.recipe.target_profile != TEST_ATTEMPT_TARGET_PROFILE
        {
            return Err("program request is not a test attempt".into());
        }
        let workload: Self = serde_json::from_slice(&request.artifact.executable_bytes)
            .map_err(|error| error.to_string())?;
        workload.validate()?;
        if workload.submission.plan.program_revision != request.recipe.program_revision {
            return Err("test workload and program recipe revisions differ".into());
        }
        Ok(workload)
    }
}

pub fn encode_execution(execution: &WorkerExecution) -> Result<ValuePayload, String> {
    let bytes = serde_json::to_vec(execution).map_err(|error| error.to_string())?;
    Ok(ValuePayload::Inline(Box::new(InlineValue::Dense(
        DenseValue {
            element_type: ElementType::U8,
            shape: vec![bytes.len() as u64],
            little_endian_data: bytes,
        },
    ))))
}

pub fn decode_execution(value: &ValuePayload) -> Result<WorkerExecution, String> {
    let ValuePayload::Inline(value) = value else {
        return Err("test execution result must be inline".into());
    };
    let InlineValue::Dense(value) = value.as_ref() else {
        return Err("test execution result must be an encoded byte vector".into());
    };
    if value.element_type != ElementType::U8
        || value.shape != [value.little_endian_data.len() as u64]
    {
        return Err("test execution result byte-vector shape is invalid".into());
    }
    serde_json::from_slice(&value.little_endian_data).map_err(|error| error.to_string())
}
