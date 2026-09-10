use std::collections::BTreeSet;

use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{
    JobId, ProgramCallable, ProgramExecutionAssignment, ProgramInvocationContext,
    ProgramRuntimeFailure,
};
use serde::{Deserialize, Serialize};

use super::{ExecutableForm, ProgramArtifact, ProgramBuildRecipe};
use crate::{ArtifactError, ArtifactResult};

pub const PROGRAM_EXECUTION_REQUEST_SCHEMA_V5: u16 = 5;
pub const MAX_PROGRAM_EXECUTION_ARGUMENTS: usize = 4096;
pub const MAX_PROGRAM_EXECUTION_RESULT_OBJECTS: usize = 65_538;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionDescriptor {
    pub schema_version: u16,
    pub recipe: ProgramBuildRecipe,
    pub artifact: ProgramArtifact,
    pub callable: ProgramCallable,
    pub requested_outputs: u16,
}

impl ProgramExecutionDescriptor {
    pub fn validate(&self) -> ArtifactResult<()> {
        self.artifact.validate_against(&self.recipe)?;
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_V5
            || self.callable.validate().is_err()
            || !entrypoint_matches(self.artifact.form, &self.callable, &self.recipe.entrypoint)
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
    pub context: ProgramInvocationContext,
    pub arguments: Vec<ValuePayload>,
}

impl ProgramExecutionInputs {
    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_V5
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
    pub callable: ProgramCallable,
    pub context: ProgramInvocationContext,
    /// Scheduler-owned placement identity for this invocation. Direct calls
    /// have no assignment; execution hosts attach one after placement.
    pub assignment: Option<ProgramExecutionAssignment>,
    /// Durable job identity, independent of whether this invocation is also a
    /// scheduler task. A batch driver has a job without a task assignment.
    pub job_id: Option<JobId>,
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
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
            recipe: descriptor.recipe,
            artifact: descriptor.artifact,
            callable: descriptor.callable,
            context: inputs.context,
            assignment: None,
            job_id: None,
            arguments: inputs.arguments,
            requested_outputs: descriptor.requested_outputs,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_V5 {
            return Err(ArtifactError::Invalid(
                "unsupported program execution request schema".into(),
            ));
        }
        self.artifact.validate_against(&self.recipe)?;
        if self.callable.validate().is_err()
            || self
                .assignment
                .as_ref()
                .is_some_and(|assignment| assignment.validate().is_err())
            || self.context.validate_for(&self.callable).is_err()
            || !entrypoint_matches(self.artifact.form, &self.callable, &self.recipe.entrypoint)
            || self.requested_outputs != self.recipe.outputs.requested_outputs
            || self.arguments.len() > MAX_PROGRAM_EXECUTION_ARGUMENTS
            || (matches!(
                self.artifact.form,
                ExecutableForm::InterpreterScriptV1
                    | ExecutableForm::InterpreterScriptV2
                    | ExecutableForm::TestAttemptV1
            ) && !self.arguments.is_empty())
        {
            return Err(ArtifactError::Invalid(
                "program execution request has an inconsistent callable, output contract, or argument count".into(),
            ));
        }
        if self.artifact.form == ExecutableForm::ExecutableUnitV3 {
            let admission = self
                .artifact
                .executable_unit_admission()?
                .expect("executable-unit form returns its validated admission header");
            if self.callable.semantic_function() != Some(admission.identity.entrypoint_function)
                || (admission.identity.entrypoint_kind
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

fn entrypoint_matches(form: ExecutableForm, callable: &ProgramCallable, entrypoint: &str) -> bool {
    match form {
        ExecutableForm::InterpreterBytecodeV1 | ExecutableForm::InterpreterBytecodeV2 => {
            callable.recipe_entrypoint() == entrypoint
        }
        ExecutableForm::InterpreterScriptV1 | ExecutableForm::InterpreterScriptV2 => {
            callable
                .semantic_function()
                .is_some_and(|function| function.0 == 0)
                && entrypoint == "script"
        }
        ExecutableForm::TestAttemptV1 => {
            callable
                .semantic_function()
                .is_some_and(|function| function.0 == 0)
                && entrypoint == "test_attempt"
        }
        ExecutableForm::MeshingWorkload => {
            callable
                .semantic_function()
                .is_some_and(|function| function.0 == 0)
                && entrypoint == "meshing_workload"
        }
        ExecutableForm::ExecutableUnitV3 | ExecutableForm::NativeObjectV1 => {
            callable.recipe_entrypoint() == entrypoint
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProgramExecutionResponse {
    Success {
        value: ValuePayload,
    },
    ExternalizedSuccess {
        outputs: Vec<ValuePayload>,
        result_objects: Vec<ValueRef>,
    },
    /// Per-rank outputs from one compiler-bound SPMD region invocation.
    /// `None` means the corresponding output was not assigned on this rank;
    /// it is distinct from every representable language value.
    SpmdSuccess {
        outputs: Vec<Option<runmat_execution::SpmdOutputValue>>,
    },
    Failure {
        message: String,
    },
    RuntimeFailure {
        failure: ProgramRuntimeFailure,
    },
}

impl ProgramExecutionResponse {
    pub fn validate_against(&self, request: &ProgramExecutionRequest) -> ArtifactResult<()> {
        request.validate()?;
        match self {
            Self::Success { value } => {
                if request.requested_outputs != 1 {
                    return Err(ArtifactError::Invalid(
                        "single-value response differs from its output contract".into(),
                    ));
                }
                value
                    .validate(ValueLimits::default())
                    .map_err(|error| ArtifactError::Invalid(error.to_string()))
            }
            Self::ExternalizedSuccess {
                outputs,
                result_objects,
            } => validate_externalized_success(request, outputs, result_objects),
            Self::SpmdSuccess { outputs } => {
                if !matches!(request.callable, ProgramCallable::SpmdRegion { .. })
                    || outputs.len() != usize::from(request.requested_outputs)
                {
                    return Err(ArtifactError::Invalid(
                        "SPMD response differs from its callable or output contract".into(),
                    ));
                }
                for output in outputs.iter().flatten() {
                    output
                        .validate()
                        .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
                }
                Ok(())
            }
            Self::Failure { message } => {
                if message.is_empty() || message.len() > 1024 * 1024 {
                    return Err(ArtifactError::Limit(
                        "program failure message is empty or exceeds its byte bound".into(),
                    ));
                }
                Ok(())
            }
            Self::RuntimeFailure { failure } => failure
                .validate()
                .map_err(|error| ArtifactError::Limit(error.to_string())),
        }
    }
}

fn validate_externalized_success(
    request: &ProgramExecutionRequest,
    outputs: &[ValuePayload],
    result_objects: &[ValueRef],
) -> ArtifactResult<()> {
    if outputs.len() != usize::from(request.requested_outputs)
        || result_objects.is_empty()
        || result_objects.len() > MAX_PROGRAM_EXECUTION_RESULT_OBJECTS
    {
        return Err(ArtifactError::Limit(
            "externalized program response exceeds its output or object inventory contract".into(),
        ));
    }
    for output in outputs {
        output
            .validate(ValueLimits::default())
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
    }
    let mut value_ids = BTreeSet::new();
    let mut logical_digests = BTreeSet::new();
    for object in result_objects {
        ValuePayload::Object(Box::new(object.clone()))
            .validate(ValueLimits::default())
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        if object.kind != ValueRefKind::ResultObject
            || object.resident_fence.is_some()
            || !value_ids.insert(object.id)
            || !logical_digests.insert(object.logical_digest)
        {
            return Err(ArtifactError::Invalid(
                "externalized program response inventory is invalid or duplicated".into(),
            ));
        }
    }
    if outputs.iter().any(|output| match output {
        ValuePayload::Object(reference) if reference.kind == ValueRefKind::ResultObject => {
            !result_objects.contains(reference.as_ref())
        }
        _ => false,
    }) {
        return Err(ArtifactError::Invalid(
            "externalized program response inventory omits a result root".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use runmat_execution::identity::{AttemptId, ValueId, WorkerId};
    use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1;
    use runmat_execution::{
        Digest, ExecutionScopeId, JobId, OutputContract, PoolBackend, PoolId, ProgramEnvironment,
        ProgramExecutionAssignment, ProgramFunctionId, ProgramRevision, TaskId,
    };

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
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::InterpreterBytecodeV2,
            b"program".to_vec(),
        )
        .unwrap();
        ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
            recipe,
            artifact,
            callable: ProgramCallable::semantic(ProgramFunctionId(7), Some("test_function".into())),
            context: ProgramInvocationContext::Direct,
            assignment: None,
            job_id: None,
            arguments: Vec::new(),
            requested_outputs: 1,
        }
    }

    fn result_reference(bytes: &[u8]) -> ValueRef {
        let logical_digest = Digest::sha256(bytes);
        ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_V1,
            id: ValueId::derive(&[b"program-response-test", logical_digest.bytes()]),
            logical_digest,
            encoded_length: bytes.len() as u64,
            media_type: "application/vnd.runmat.test-object".into(),
            value_schema: "runmat.test-object.v1".into(),
            encryption_context: Digest::sha256(b"test-encryption-context"),
            kind: ValueRefKind::ResultObject,
            authorization_scope: "test-scope".into(),
            resident_fence: None,
        }
    }

    #[test]
    fn exact_program_request_validates_every_identity_boundary() {
        request().validate().unwrap();
        let mut mismatched = request();
        mismatched.callable = ProgramCallable::semantic(ProgramFunctionId(8), None);
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
    fn execution_identity_round_trips_without_becoming_artifact_identity() {
        let mut request = request();
        request.assignment = Some(ProgramExecutionAssignment {
            scope_id: ExecutionScopeId::derive(&[b"scope"]),
            pool_id: PoolId::derive(&[b"pool"]),
            task_id: TaskId::derive(&[b"task"]),
            attempt_id: AttemptId::derive(&[b"attempt"]),
            worker_id: WorkerId::derive(&[b"worker"]),
            backend: PoolBackend::Remote,
            resources: Default::default(),
        });
        request.job_id = Some(JobId::derive(&[b"job"]));

        let encoded = serde_json::to_vec(&request).unwrap();
        let decoded: ProgramExecutionRequest = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, request);
        decoded.validate().unwrap();

        let mut with_unknown = serde_json::to_value(&request).unwrap();
        with_unknown
            .as_object_mut()
            .unwrap()
            .insert("worker_hint".into(), serde_json::json!("untyped"));
        assert!(serde_json::from_value::<ProgramExecutionRequest>(with_unknown).is_err());
    }

    #[test]
    fn script_form_has_an_explicit_argument_free_entrypoint() {
        let mut script = request();
        script.recipe.entrypoint = "script".into();
        script.callable = ProgramCallable::semantic(ProgramFunctionId(0), None);
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

    #[test]
    fn externalized_response_requires_a_complete_unique_result_inventory() {
        let request = request();
        let root = result_reference(b"root");
        let response = ProgramExecutionResponse::ExternalizedSuccess {
            outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
            result_objects: vec![root.clone(), result_reference(b"chunk")],
        };
        response.validate_against(&request).unwrap();

        let missing_root = ProgramExecutionResponse::ExternalizedSuccess {
            outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
            result_objects: vec![result_reference(b"chunk")],
        };
        assert!(missing_root.validate_against(&request).is_err());

        let duplicate = ProgramExecutionResponse::ExternalizedSuccess {
            outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
            result_objects: vec![root.clone(), root],
        };
        assert!(duplicate.validate_against(&request).is_err());
    }

    #[test]
    fn spmd_response_preserves_unassigned_outputs_as_protocol_state() {
        let ordinary_request = request();
        let mut spmd_request = ordinary_request.clone();
        let region = runmat_types::ParallelRegionId(runmat_types::RegionId {
            function: runmat_types::ProgramFunctionId(7),
            ordinal: 3,
        });
        let scope_id = ExecutionScopeId::derive(&[b"spmd-response-scope"]);
        let pool = runmat_execution::PoolHandle {
            id: PoolId::derive(&[b"spmd-response-pool"]),
            scope_id,
            generation: 1,
        };
        let gang = runmat_execution::GangHandle {
            id: runmat_execution::GangId::derive(&[b"spmd-response-gang"]),
            scope_id,
            generation: 1,
            pool,
            labs: runmat_types::LabCount(2),
        };
        spmd_request.callable = ProgramCallable::spmd_region(region);
        spmd_request.context = ProgramInvocationContext::SpmdTask {
            task: runmat_execution::SpmdTaskContext {
                gang,
                region,
                rank: runmat_types::LabRank(1),
            },
        };
        spmd_request.recipe.entrypoint = spmd_request.callable.recipe_entrypoint();
        spmd_request.recipe.outputs.requested_outputs = 2;
        spmd_request.requested_outputs = 2;
        spmd_request.artifact = ProgramArtifact::materialize(
            &spmd_request.recipe,
            ExecutableForm::InterpreterBytecodeV2,
            b"spmd-program".to_vec(),
        )
        .unwrap();

        let response = ProgramExecutionResponse::SpmdSuccess {
            outputs: vec![
                Some(runmat_execution::SpmdOutputValue::Value(
                    ValuePayload::Inline(Box::new(runmat_execution::value::InlineValue::U64(
                        9_007_199_254_740_993,
                    ))),
                )),
                None,
            ],
        };
        response.validate_against(&spmd_request).unwrap();
        let encoded = serde_json::to_vec(&response).unwrap();
        let decoded: ProgramExecutionResponse = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, response);

        let wrong_callable = ProgramExecutionResponse::SpmdSuccess {
            outputs: vec![None],
        };
        assert!(wrong_callable.validate_against(&ordinary_request).is_err());
    }
}
