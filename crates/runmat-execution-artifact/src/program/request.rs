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
pub const PROGRAM_EXECUTION_REQUEST_SCHEMA_V6: u16 = 6;
pub const PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION: u16 = PROGRAM_EXECUTION_REQUEST_SCHEMA_V6;
pub const MAX_PROGRAM_EXECUTION_ARGUMENTS: usize = 4096;
pub const MAX_PROGRAM_EXECUTION_RESULT_OBJECTS: usize = 65_538;

#[derive(Deserialize)]
struct ProgramExecutionSchemaHeader {
    schema_version: u16,
}

pub fn admit_program_execution_request_bytes(bytes: &[u8]) -> ArtifactResult<()> {
    let header: ProgramExecutionSchemaHeader = serde_json::from_slice(bytes).map_err(|error| {
        ArtifactError::Invalid(format!("invalid program request envelope: {error}"))
    })?;
    if header.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION {
        return Err(ArtifactError::Invalid(format!(
            "unsupported program execution request schema {}; expected {}; rebuild the request with the current value-payload codec",
            header.schema_version, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION
        )));
    }
    Ok(())
}

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
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION
            || self.callable.validate().is_err()
            || !entrypoint_matches(self.artifact.form, &self.callable, &self.recipe.entrypoint)
            || self.requested_outputs != self.recipe.outputs.requested_outputs
        {
            return Err(ArtifactError::Invalid(
                "program descriptor has an inconsistent callable or output contract".into(),
            ));
        }
        if let Some(admission) = self.artifact.executable_unit_admission()? {
            if self.callable.semantic_function() != Some(admission.identity.entrypoint_function) {
                return Err(ArtifactError::Invalid(
                    "program descriptor callable does not match its executable-unit entrypoint"
                        .into(),
                ));
            }
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
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION
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
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
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
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION {
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
            || (artifact_form_requires_argument_free_entrypoint(self.artifact.form, &self.callable)
                && !self.arguments.is_empty())
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
        ExecutableForm::InterpreterScriptV2
            if matches!(
                callable,
                ProgramCallable::ParallelRegion { .. } | ProgramCallable::SpmdRegion { .. }
            ) =>
        {
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
        ExecutableForm::ExecutableUnitV3 => true,
        ExecutableForm::NativeObjectV1 => callable.recipe_entrypoint() == entrypoint,
    }
}

fn artifact_form_requires_argument_free_entrypoint(
    form: ExecutableForm,
    callable: &ProgramCallable,
) -> bool {
    match form {
        ExecutableForm::InterpreterScriptV1 | ExecutableForm::TestAttemptV1 => true,
        ExecutableForm::InterpreterScriptV2 => !matches!(
            callable,
            ProgramCallable::ParallelRegion { .. } | ProgramCallable::SpmdRegion { .. }
        ),
        ExecutableForm::InterpreterBytecodeV1
        | ExecutableForm::InterpreterBytecodeV2
        | ExecutableForm::MeshingWorkload
        | ExecutableForm::ExecutableUnitV3
        | ExecutableForm::NativeObjectV1 => false,
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProgramExecutionResponse {
    Success {
        outputs: Vec<ValuePayload>,
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
            Self::Success { outputs } => {
                if outputs.len() != usize::from(request.requested_outputs) {
                    return Err(ArtifactError::Invalid(
                        "program response differs from its output contract".into(),
                    ));
                }
                validate_output_payloads(outputs)
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

fn validate_output_payloads(outputs: &[ValuePayload]) -> ArtifactResult<()> {
    for output in outputs {
        output
            .validate(ValueLimits::default())
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
    }
    Ok(())
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
    validate_output_payloads(outputs)?;
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
    use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION;
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
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
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

    fn executable_unit_request() -> ProgramExecutionRequest {
        let bytes = include_bytes!(
            "../../../runmat-execution/tests/fixtures/executable-unit-stale-components.json"
        )
        .to_vec();
        let envelope = runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(&bytes)
            .expect("frozen executable-unit fixture remains canonical");
        let identity = &envelope.manifest.identity;
        let recipe = ProgramBuildRecipe {
            schema_version: crate::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: identity.program.clone(),
            entrypoint: identity.entrypoint.clone(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            execution_mode: "interpreter".into(),
            target: crate::ProgramTarget::portable("portable-executable-unit-test"),
            interop: envelope.manifest.interop.clone(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let artifact =
            ProgramArtifact::materialize(&recipe, ExecutableForm::ExecutableUnitV3, bytes).unwrap();
        ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            recipe,
            artifact,
            callable: ProgramCallable::semantic(identity.entrypoint_function, Some("main".into())),
            context: ProgramInvocationContext::Direct,
            assignment: None,
            job_id: None,
            arguments: Vec::new(),
            requested_outputs: 1,
        }
    }

    fn inline_argument(value: &str) -> ValuePayload {
        ValuePayload::Inline(Box::new(runmat_execution::value::InlineValue::String(
            value.into(),
        )))
    }

    fn result_reference(bytes: &[u8]) -> ValueRef {
        let logical_digest = Digest::sha256(bytes);
        ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_VERSION,
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
        let mut stale = request();
        stale.schema_version = PROGRAM_EXECUTION_REQUEST_SCHEMA_V5;
        assert!(stale.validate().is_err());
        let mut unknown = request();
        unknown.schema_version += 1;
        assert!(unknown.validate().is_err());
        let mut outputs = request();
        outputs.requested_outputs = 2;
        assert!(outputs.validate().is_err());
    }

    #[test]
    fn frozen_v5_request_is_rejected_before_payload_deserialization() {
        let frozen = br#"{"schema_version":5,"arguments":[{"form":"inline","value":{"type":"output_list","value":[]}}]}"#;
        let error = admit_program_execution_request_bytes(frozen).unwrap_err();
        assert!(error
            .to_string()
            .contains("unsupported program execution request schema 5; expected 6"));
        assert!(serde_json::from_slice::<ProgramExecutionRequest>(frozen).is_err());
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
    fn named_executable_unit_binds_recipe_name_and_typed_function_identity() {
        let request = executable_unit_request();
        assert_eq!(request.recipe.entrypoint, "main");
        request.validate().unwrap();

        let mut wrong_function = request.clone();
        wrong_function.callable =
            ProgramCallable::semantic(ProgramFunctionId(1), Some("main".into()));
        let error = wrong_function.validate().unwrap_err();
        assert!(error
            .to_string()
            .contains("executable unit request does not match its declared entrypoint"));

        let mut wrong_name = request.recipe.clone();
        wrong_name.entrypoint = "different_entrypoint".into();
        let error = ProgramArtifact::materialize(
            &wrong_name,
            ExecutableForm::ExecutableUnitV3,
            request.artifact.executable_bytes.clone(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("entrypoint"));
    }

    #[test]
    fn compiler_bound_script_regions_accept_arguments_but_top_level_scripts_do_not() {
        let region = runmat_types::ParallelRegionId(runmat_types::RegionId {
            function: runmat_types::ProgramFunctionId(7),
            ordinal: 3,
        });
        let mut parallel = request();
        parallel.callable = ProgramCallable::parallel_region(region);
        parallel.context = ProgramInvocationContext::ParallelTask {
            task: runmat_execution::ParallelTaskContext {
                region,
                chunk: runmat_execution::ParallelChunk {
                    ordinal: 0,
                    start: 0,
                    len: 1,
                },
                randomness: runmat_execution::ParallelRandomnessContext::Inherit,
            },
        };
        parallel.recipe.entrypoint = parallel.callable.recipe_entrypoint();
        parallel.artifact = ProgramArtifact::materialize(
            &parallel.recipe,
            ExecutableForm::InterpreterScriptV2,
            b"compiler-bound-parallel-region".to_vec(),
        )
        .unwrap();
        parallel.arguments = vec![inline_argument("parallel capture")];
        parallel.validate().unwrap();

        let scope_id = ExecutionScopeId::derive(&[b"script-region-scope"]);
        let pool = runmat_execution::PoolHandle {
            id: PoolId::derive(&[b"script-region-pool"]),
            scope_id,
            generation: 1,
        };
        let gang = runmat_execution::GangHandle {
            id: runmat_execution::GangId::derive(&[b"script-region-gang"]),
            scope_id,
            generation: 1,
            pool,
            labs: runmat_types::LabCount(2),
        };
        let mut spmd = request();
        spmd.callable = ProgramCallable::spmd_region(region);
        spmd.context = ProgramInvocationContext::SpmdTask {
            task: runmat_execution::SpmdTaskContext {
                gang,
                region,
                rank: runmat_types::LabRank(1),
            },
        };
        spmd.recipe.entrypoint = spmd.callable.recipe_entrypoint();
        spmd.artifact = ProgramArtifact::materialize(
            &spmd.recipe,
            ExecutableForm::InterpreterScriptV2,
            b"compiler-bound-spmd-region".to_vec(),
        )
        .unwrap();
        spmd.arguments = vec![inline_argument("spmd capture")];
        spmd.validate().unwrap();

        let mut top_level = request();
        top_level.recipe.entrypoint = "script".into();
        top_level.callable = ProgramCallable::semantic(ProgramFunctionId(0), None);
        top_level.artifact = ProgramArtifact::materialize(
            &top_level.recipe,
            ExecutableForm::InterpreterScriptV2,
            b"top-level-script".to_vec(),
        )
        .unwrap();
        top_level.validate().unwrap();
        top_level.arguments = vec![inline_argument("unexpected")];
        assert!(top_level.validate().is_err());
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
    fn inline_response_uses_a_bounded_output_vector() {
        let request = request();
        let response = ProgramExecutionResponse::Success {
            outputs: vec![ValuePayload::Inline(Box::new(
                runmat_execution::value::InlineValue::F64Bits(3.0_f64.to_bits()),
            ))],
        };
        response.validate_against(&request).unwrap();

        let wrong_count = ProgramExecutionResponse::Success {
            outputs: Vec::new(),
        };
        assert!(wrong_count.validate_against(&request).is_err());

        let frozen_v5 = r#"{"outcome":"success","value":{"form":"inline","value":{"type":"logical","value":true}}}"#;
        assert!(serde_json::from_str::<ProgramExecutionResponse>(frozen_v5).is_err());
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
