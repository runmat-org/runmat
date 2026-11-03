use std::borrow::Borrow;
use std::collections::BTreeSet;

use minicbor::Encoder;
use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{
    Digest, EnvironmentSnapshot, JobId, ProgramCallable, ProgramExecutionAssignment,
    ProgramInvocationContext, ProgramRuntimeFailure,
};
use serde::{Deserialize, Serialize};

use super::{ExecutableForm, ProgramArtifact, ProgramBuildRecipe};
use crate::{error::contract_error, ArtifactError, ArtifactResult};

pub const PROGRAM_EXECUTION_REQUEST_SCHEMA_V5: u16 = 5;
pub const PROGRAM_EXECUTION_REQUEST_SCHEMA_V6: u16 = 6;
/// V7 makes failure-frame source IDs program ordinals, never worker-local IDs.
pub const PROGRAM_EXECUTION_REQUEST_SCHEMA_V7: u16 = 7;
/// V8 binds the frozen invocation environment and console controls to every
/// direct request and remote descriptor.
pub const PROGRAM_EXECUTION_REQUEST_SCHEMA_V8: u16 = 8;
/// V10 binds one optional canonical bundle archive to every selected V3 unit.
pub use runmat_execution::value::{
    GRAPH_REQUEST_SCHEMA_V10 as GRAPH_EXECUTION_REQUEST_SCHEMA_V10,
    GRAPH_REQUEST_SCHEMA_V11 as GRAPH_EXECUTION_REQUEST_SCHEMA_V11,
    GRAPH_REQUEST_SCHEMA_V12 as GRAPH_EXECUTION_REQUEST_SCHEMA_V12,
    GRAPH_REQUEST_SCHEMA_V13 as GRAPH_EXECUTION_REQUEST_SCHEMA_V13,
    GRAPH_REQUEST_SCHEMA_V14 as GRAPH_EXECUTION_REQUEST_SCHEMA_V14,
};
pub const PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION: u16 = PROGRAM_EXECUTION_REQUEST_SCHEMA_V8;
pub const MAX_PROGRAM_EXECUTION_ARGUMENTS: usize = 4096;
pub const MAX_PROGRAM_EXECUTION_RESULT_OBJECTS: usize = 65_539;

#[path = "request/graph.rs"]
mod graph;
pub use graph::{
    AdmittedGraphUnitBundle, ArchiveAdmittedCapturedGraphCode,
    ArchiveAdmittedGraphExecutionRequest, CapturedGraphCode, GraphExecutionRequest,
    GraphInvocationControls, GraphInvocationRoots, GraphResponseAuthority,
    ValidatedCapturedGraphCode, ValidatedGraphExecutionRequest,
};

/// Direct V8 requests keep their exact public wire shape. Graph V10 is a
/// separate typed worker route and cannot reinterpret a direct manifest call.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WorkerProgramRequest {
    Direct(Box<ProgramExecutionRequest>),
    Graph(Box<GraphExecutionRequest>),
}

/// The encrypted remote `program` object. Direct V7 keeps its historical
/// descriptor bytes and separate input object; Graph V10 carries its complete
/// archive-bearing request in that same encrypted object.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RemoteProgramObject {
    Direct(Box<ProgramExecutionDescriptor>),
    Graph(Box<GraphExecutionRequest>),
}

impl RemoteProgramObject {
    pub fn validate(&self) -> ArtifactResult<()> {
        match self {
            Self::Direct(descriptor) => descriptor.validate_for_portable_host(),
            Self::Graph(request) => request.validate(),
        }
    }
}

pub fn admit_remote_program_object_bytes(bytes: &[u8]) -> ArtifactResult<()> {
    let header: ProgramExecutionSchemaHeader = serde_json::from_slice(bytes).map_err(|error| {
        ArtifactError::Invalid(format!("invalid remote program object envelope: {error}"))
    })?;
    match header.schema_version {
        PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION => {
            let descriptor: ProgramExecutionDescriptor = serde_json::from_slice(bytes)
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
            descriptor.validate_for_portable_host()
        }
        GRAPH_EXECUTION_REQUEST_SCHEMA_V10
        | GRAPH_EXECUTION_REQUEST_SCHEMA_V11
        | GRAPH_EXECUTION_REQUEST_SCHEMA_V12
        | GRAPH_EXECUTION_REQUEST_SCHEMA_V13
        | GRAPH_EXECUTION_REQUEST_SCHEMA_V14 => {
            let request: GraphExecutionRequest = serde_json::from_slice(bytes)
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
            request.validate()
        }
        version => Err(ArtifactError::Invalid(format!(
            "unsupported remote program object schema {version}"
        ))),
    }
}

impl WorkerProgramRequest {
    pub fn direct(request: ProgramExecutionRequest) -> Self {
        Self::Direct(Box::new(request))
    }

    pub fn graph(request: GraphExecutionRequest) -> Self {
        Self::Graph(Box::new(request))
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        match self {
            Self::Direct(request) => request.validate(),
            Self::Graph(request) => request.validate(),
        }
    }
}

pub fn admit_worker_program_request_bytes(bytes: &[u8]) -> ArtifactResult<()> {
    let header: ProgramExecutionSchemaHeader = serde_json::from_slice(bytes).map_err(|error| {
        ArtifactError::Invalid(format!("invalid worker program request envelope: {error}"))
    })?;
    match header.schema_version {
        PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION => admit_program_execution_request_bytes(bytes),
        GRAPH_EXECUTION_REQUEST_SCHEMA_V10
        | GRAPH_EXECUTION_REQUEST_SCHEMA_V11
        | GRAPH_EXECUTION_REQUEST_SCHEMA_V12
        | GRAPH_EXECUTION_REQUEST_SCHEMA_V13
        | GRAPH_EXECUTION_REQUEST_SCHEMA_V14 => {
            let request: GraphExecutionRequest =
                serde_json::from_slice(bytes).map_err(|error| {
                    ArtifactError::Invalid(format!("invalid graph worker request: {error}"))
                })?;
            request.validate()
        }
        version => Err(ArtifactError::Invalid(format!(
            "unsupported worker program request schema {version}"
        ))),
    }
}

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
            "unsupported program execution request schema {}; expected {}; rebuild the request with the current execution contract",
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
    pub environment: EnvironmentSnapshot,
    #[serde(deserialize_with = "runmat_execution::console::deserialize_controls")]
    pub console: Option<runmat_execution::console::ConsoleLimits>,
}

impl ProgramExecutionDescriptor {
    pub fn validate(&self) -> ArtifactResult<()> {
        validate_console(self.console, self.artifact.form)?;
        self.environment.validate().map_err(contract_error)?;
        let artifact = self.artifact.validate_context(&self.recipe)?;
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION
            || self.callable.validate().is_err()
            || matches!(self.callable, ProgramCallable::GraphTarget { .. })
            || !entrypoint_matches(self.artifact.form, &self.callable, &self.recipe.entrypoint)
            || self.requested_outputs != self.recipe.outputs.requested_outputs
        {
            return Err(ArtifactError::Invalid(
                "program descriptor has an inconsistent callable or output contract".into(),
            ));
        }
        if let Some(admission) = artifact.unit_admission.as_ref() {
            let entrypoint = direct_executable_entrypoint(&admission.identity)?;
            if self.callable.semantic_function() != Some(entrypoint) {
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
        crate::environment::validate(self.environment.entries())?;
        self.artifact.target.validate_for_portable_host()
    }
}

/// Per-invocation inputs. External arguments must be listed in `objects`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionInputs {
    pub schema_version: u16,
    pub context: ProgramInvocationContext,
    pub arguments: Vec<ValuePayload>,
    pub objects: Vec<super::ProgramExecutionObjectLocation>,
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
                .map_err(contract_error)?;
        }
        super::transport::validate_input_objects(&self.arguments, &self.objects)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramExecutionRequest<
    R = ProgramBuildRecipe,
    A = ProgramArtifact,
    V = Vec<ValuePayload>,
    E = EnvironmentSnapshot,
> {
    pub schema_version: u16,
    pub recipe: R,
    pub artifact: A,
    pub callable: ProgramCallable,
    pub context: ProgramInvocationContext,
    /// Scheduler-owned placement identity for this invocation. Direct calls
    /// have no assignment; execution hosts attach one after placement.
    pub assignment: Option<ProgramExecutionAssignment>,
    /// Durable job identity, independent of whether this invocation is also a
    /// scheduler task. A batch driver has a job without a task assignment.
    pub job_id: Option<JobId>,
    pub arguments: V,
    pub requested_outputs: u16,
    pub environment: E,
    #[serde(deserialize_with = "runmat_execution::console::deserialize_controls")]
    pub console: Option<runmat_execution::console::ConsoleLimits>,
}

/// Original request fields borrowed for validation, identity and transport encoding.
/// Owned deserialization retains the same default request type and wire schema.
pub type BorrowedProgramExecutionRequest<'a> = ProgramExecutionRequest<
    &'a ProgramBuildRecipe,
    &'a ProgramArtifact,
    &'a [ValuePayload],
    &'a EnvironmentSnapshot,
>;

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
            environment: descriptor.environment,
            console: descriptor.console,
        };
        request.validate()?;
        Ok(request)
    }
}

impl<
        R: Borrow<ProgramBuildRecipe>,
        A: Borrow<ProgramArtifact>,
        V: AsRef<[ValuePayload]>,
        E: Borrow<EnvironmentSnapshot>,
    > ProgramExecutionRequest<R, A, V, E>
{
    pub fn as_borrowed(&self) -> BorrowedProgramExecutionRequest<'_> {
        ProgramExecutionRequest {
            schema_version: self.schema_version,
            recipe: self.recipe.borrow(),
            artifact: self.artifact.borrow(),
            callable: self.callable.clone(),
            context: self.context.clone(),
            assignment: self.assignment.clone(),
            job_id: self.job_id,
            arguments: self.arguments.as_ref(),
            requested_outputs: self.requested_outputs,
            environment: self.environment.borrow(),
            console: self.console,
        }
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        self.validate_context().map(|_| ())
    }

    fn validate_context(&self) -> ArtifactResult<super::artifact::ValidatedArtifactContext> {
        if self.schema_version != PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION {
            return Err(ArtifactError::Invalid(
                "unsupported program execution request schema".into(),
            ));
        }
        let recipe = self.recipe.borrow();
        let form = self.artifact.borrow().form;
        let arguments = self.arguments.as_ref();
        validate_console(self.console, form)?;
        self.environment
            .borrow()
            .validate()
            .map_err(contract_error)?;
        let artifact = self.artifact.borrow().validate_context(recipe)?;
        if self.callable.validate().is_err()
            || matches!(self.callable, ProgramCallable::GraphTarget { .. })
            || self
                .assignment
                .as_ref()
                .is_some_and(|assignment| assignment.validate().is_err())
            || self.context.validate_for(&self.callable).is_err()
            || !entrypoint_matches(form, &self.callable, &recipe.entrypoint)
            || self.requested_outputs != recipe.outputs.requested_outputs
            || arguments.len() > MAX_PROGRAM_EXECUTION_ARGUMENTS
            || (artifact_form_requires_argument_free_entrypoint(form, &self.callable)
                && !arguments.is_empty())
        {
            return Err(ArtifactError::Invalid(
                "program execution request has an inconsistent callable, output contract, or argument count".into(),
            ));
        }
        if form == ExecutableForm::ExecutableUnitV4 {
            let admission = artifact
                .unit_admission
                .as_ref()
                .expect("executable-unit form returns its validated admission header");
            let entrypoint = direct_executable_entrypoint(&admission.identity)?;
            if self.callable.semantic_function() != Some(entrypoint)
                || (admission.identity.entrypoint_kind
                    == runmat_execution::ExecutableEntrypointKind::Script
                    && !arguments.is_empty())
            {
                return Err(ArtifactError::Invalid(
                    "executable unit request does not match its declared entrypoint".into(),
                ));
            }
        }
        for argument in arguments {
            argument
                .validate_for_program_sources(ValueLimits::default(), artifact.source_map.as_ref())
                .map_err(contract_error)?;
        }
        Ok(artifact)
    }

    pub fn validate_for_portable_host(&self) -> ArtifactResult<()> {
        self.validate()?;
        crate::environment::validate(self.environment.borrow().entries())?;
        self.artifact.borrow().target.validate_for_portable_host()
    }

    /// Exact request identity over every invocation-bound field, without a
    /// second serialized copy of the program artifact.
    pub fn canonical_digest(&self) -> ArtifactResult<Digest> {
        self.validate()?;
        let recipe_id = self.recipe.borrow().id()?;
        let arguments = self.arguments.as_ref();
        let argument_count = u64::try_from(arguments.len())
            .map_err(|_| ArtifactError::Limit("program argument count exceeds u64".into()))?;
        let mut bytes = b"runmat-program-execution-request\0".to_vec();
        let mut encoder = Encoder::new(&mut bytes);
        encoder
            .array(11)
            .and_then(|encoder| encoder.u16(self.schema_version))
            .and_then(|encoder| encoder.bytes(recipe_id.0.bytes()))
            .and_then(|encoder| encoder.bytes(self.artifact.borrow().id.0.bytes()))
            .map_err(encoding)?;
        for digest in [
            json_digest(&self.callable)?,
            json_digest(&self.context)?,
            json_digest(&self.assignment)?,
            json_digest(&self.job_id)?,
        ] {
            encoder.bytes(digest.bytes()).map_err(encoding)?;
        }
        encoder
            .u16(self.requested_outputs)
            .and_then(|encoder| encoder.array(argument_count))
            .map_err(encoding)?;
        for argument in arguments {
            encoder
                .bytes(json_digest(argument)?.bytes())
                .map_err(encoding)?;
        }
        encoder
            .bytes(
                self.environment
                    .borrow()
                    .identity()
                    .map_err(contract_error)?
                    .bytes(),
            )
            .map_err(encoding)?;
        match self.console {
            None => {
                encoder.null().map_err(encoding)?;
            }
            Some(limits) => {
                encoder
                    .array(2)
                    .and_then(|encoder| encoder.u32(limits.max_records))
                    .and_then(|encoder| encoder.u64(limits.max_text_bytes))
                    .map_err(encoding)?;
            }
        }
        Ok(Digest::sha256(bytes))
    }
}

fn json_digest<T: Serialize>(value: &T) -> ArtifactResult<Digest> {
    Digest::sha256_serialized(value).map_err(|error| ArtifactError::Encoding(error.to_string()))
}

fn validate_console(
    console: Option<runmat_execution::console::ConsoleLimits>,
    form: ExecutableForm,
) -> ArtifactResult<()> {
    if let Some(limits) = console {
        limits
            .validate_transport(64 * 1024 * 1024)
            .map_err(contract_error)?;
        if !matches!(
            form,
            ExecutableForm::ExecutableUnitV4 | ExecutableForm::InterpreterRegionSourcesV1
        ) {
            return Err(ArtifactError::Invalid(
                "console capture requires a session-capable complete executable unit or typed region".into(),
            ));
        }
    }
    Ok(())
}

fn encoding(error: minicbor::encode::Error<std::convert::Infallible>) -> ArtifactError {
    ArtifactError::Encoding(error.to_string())
}

fn direct_executable_entrypoint(
    identity: &runmat_execution::ExecutableIdentity,
) -> ArtifactResult<runmat_execution::ProgramFunctionId> {
    match identity.validated_entrypoint().map_err(contract_error)? {
        runmat_execution::ValidatedExecutableEntrypoint::Function(function)
        | runmat_execution::ValidatedExecutableEntrypoint::Script(function) => Ok(function),
        runmat_execution::ValidatedExecutableEntrypoint::DeclarationOnly(_) => {
            Err(ArtifactError::Invalid(
                "declaration-only product cannot be executed as a Direct callable".into(),
            ))
        }
    }
}

fn entrypoint_matches(form: ExecutableForm, callable: &ProgramCallable, entrypoint: &str) -> bool {
    match form {
        ExecutableForm::InterpreterBytecodeV1 | ExecutableForm::InterpreterBytecodeV2 => {
            callable.recipe_entrypoint() == entrypoint
        }
        ExecutableForm::InterpreterScriptV2 | ExecutableForm::InterpreterRegionSourcesV1
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
        ExecutableForm::InterpreterRegionSourcesV1 => false,
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
        ExecutableForm::AnalysisWorkload => {
            callable
                .semantic_function()
                .is_some_and(|function| function.0 == 0)
                && entrypoint == "analysis_workload"
        }
        ExecutableForm::GeometryWorkload => {
            callable
                .semantic_function()
                .is_some_and(|function| function.0 == 0)
                && entrypoint == "geometry_workload"
        }
        ExecutableForm::ExecutableUnitV4 => true,
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
        ExecutableForm::InterpreterRegionSourcesV1 => false,
        ExecutableForm::InterpreterBytecodeV1
        | ExecutableForm::InterpreterBytecodeV2
        | ExecutableForm::MeshingWorkload
        | ExecutableForm::AnalysisWorkload
        | ExecutableForm::GeometryWorkload
        | ExecutableForm::ExecutableUnitV4
        | ExecutableForm::NativeObjectV1 => false,
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProgramExecutionResponse {
    Success {
        outputs: Vec<ValuePayload>,
    },
    /// One rooted output graph for a V8 invocation; Direct V7 rejects it.
    GraphSuccess {
        graph: runmat_execution::value::PayloadGraph,
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
    /// Preserve inline versus object-backed delivery when projecting admitted task outputs.
    /// The resulting response still requires validation against its original request.
    pub fn from_success_outputs(outputs: Vec<ValuePayload>, result_objects: Vec<ValueRef>) -> Self {
        if result_objects.is_empty() {
            Self::Success { outputs }
        } else {
            Self::ExternalizedSuccess {
                outputs,
                result_objects,
            }
        }
    }

    pub fn validate_owned_builtin(
        &self,
        request: &runmat_execution::OwnedBuiltinWorkRequest,
    ) -> ArtifactResult<()> {
        request.validate_wire().map_err(contract_error)?;
        request
            .response
            .validate_json_wire(self)
            .map_err(contract_error)?;
        match self {
            Self::Success { outputs } => request
                .validate_output_values(outputs)
                .map_err(contract_error),
            Self::RuntimeFailure { failure } => {
                failure.validate().map_err(contract_error)?;
                if failure.span.is_some()
                    || failure.call_frames.iter().any(|frame| {
                        frame.program_source_id.is_some()
                            || frame.span.is_some()
                            || frame.detached_diagnostic.is_some()
                    })
                {
                    return Err(ArtifactError::Invalid(
                        "owned builtin failure has an unselected source".into(),
                    ));
                }
                Ok(())
            }
            Self::Failure { message } if !message.is_empty() && message.len() <= 1024 * 1024 => {
                Ok(())
            }
            _ => Err(ArtifactError::Invalid(
                "response form is not admitted for owned builtin work".into(),
            )),
        }
    }

    pub fn validate_against<
        R: Borrow<ProgramBuildRecipe>,
        A: Borrow<ProgramArtifact>,
        V: AsRef<[ValuePayload]>,
        E: Borrow<EnvironmentSnapshot>,
    >(
        &self,
        request: &ProgramExecutionRequest<R, A, V, E>,
    ) -> ArtifactResult<()> {
        let artifact = request.validate_context()?;
        let source_map = artifact.source_map.as_ref();
        match self {
            Self::Success { outputs } => {
                if outputs.len() != usize::from(request.requested_outputs) {
                    return Err(ArtifactError::Invalid(
                        "program response differs from its output contract".into(),
                    ));
                }
                validate_output_payloads(outputs, source_map)
            }
            Self::GraphSuccess { .. } => Err(ArtifactError::Invalid(
                "direct program request cannot receive a graph response".into(),
            )),
            Self::ExternalizedSuccess {
                outputs,
                result_objects,
            } => validate_externalized_inventory(
                outputs,
                result_objects,
                request.requested_outputs,
                source_map,
            ),
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
                        .validate_for_program_sources(source_map)
                        .map_err(contract_error)?;
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
            Self::RuntimeFailure { failure } => {
                if failure
                    .call_frames
                    .iter()
                    .any(|frame| frame.detached_diagnostic.is_some())
                {
                    return Err(ArtifactError::Invalid(
                        "direct program failure cannot carry a graph detached diagnostic".into(),
                    ));
                }
                failure
                    .validate_for_program_sources(source_map)
                    .map_err(contract_error)
            }
        }
    }
}

fn validate_output_payloads(
    outputs: &[ValuePayload],
    source_map: Option<&runmat_execution::ExecutableSourceMapV2>,
) -> ArtifactResult<()> {
    for output in outputs {
        output
            .validate_for_program_sources(ValueLimits::default(), source_map)
            .map_err(contract_error)?;
    }
    Ok(())
}

pub(super) fn validate_externalized_inventory(
    outputs: &[ValuePayload],
    result_objects: &[ValueRef],
    requested_outputs: u16,
    source_map: Option<&runmat_execution::ExecutableSourceMapV2>,
) -> ArtifactResult<()> {
    if outputs.len() != usize::from(requested_outputs)
        || result_objects.is_empty()
        || result_objects.len() > MAX_PROGRAM_EXECUTION_RESULT_OBJECTS
    {
        return Err(ArtifactError::Limit(
            "externalized program response exceeds its output or object inventory contract".into(),
        ));
    }
    validate_output_payloads(outputs, source_map)?;
    let mut value_ids = BTreeSet::new();
    let mut logical_digests = BTreeSet::new();
    for object in result_objects {
        ValuePayload::Object(Box::new(object.clone()))
            .validate(ValueLimits::default())
            .map_err(contract_error)?;
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
#[path = "request/environment_tests.rs"]
mod environment_tests;
#[cfg(test)]
#[path = "request/reservation_tests.rs"]
mod reservation_tests;
#[cfg(test)]
#[path = "request/tests.rs"]
mod tests;
