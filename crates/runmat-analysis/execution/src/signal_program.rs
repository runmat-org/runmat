use std::collections::{BTreeMap, BTreeSet};

use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::ResourceRequest;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::{
    Digest, ExecutableEntrypointKind, ExecutableUnitEnvelope, ExecutionScopeId, OutputContract,
    PoolId, TaskId,
};
use runmat_execution_artifact::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionDescriptor,
    ProgramExecutionInputs, ProgramExecutionObject, ProgramExecutionObjectLocation,
    ProgramExecutionRequest, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_execution_runner::TaskSubmission;

use crate::signal_input::validate_signal_input_payload;
use crate::{
    validate_signal_generator_chunks, AdmittedGeneratorInvocation, AdmittedNumericalGenerator,
    AdmittedSignalGenerator, AnalysisBridgeError, AnalysisBridgeResult, GeneratorChunk,
    SignalInputAdmission,
};

pub const GENERATOR_EXECUTION_MODE: &str = "analysis-generator";
pub const GENERATOR_TARGET_PROFILE: &str = "portable-executable-unit-v3";

pub trait GeneratorProgramAdmission {
    fn invocation_admission(&self) -> GeneratorProgramAdmissionRef<'_>;
    fn materialization_digest(&self) -> Digest;
}

#[derive(Clone, Copy)]
pub struct GeneratorProgramAdmissionRef<'a> {
    pub program_revision: &'a runmat_execution::ProgramRevision,
    pub callable: &'a runmat_execution_artifact::BundleCallable,
    pub executable_source_digest: Digest,
    pub function: runmat_types::ProgramFunctionId,
    pub generator_identity: runmat_analysis_fea::execution::GeneratorIdentity,
    pub required_capabilities: &'a BTreeSet<runmat_types::CapabilityRequirement>,
}

impl GeneratorProgramAdmission for AdmittedSignalGenerator {
    fn invocation_admission(&self) -> GeneratorProgramAdmissionRef<'_> {
        GeneratorProgramAdmissionRef {
            program_revision: &self.program_revision,
            callable: &self.callable,
            executable_source_digest: self.executable_source_digest,
            function: self.function,
            generator_identity: self.generator_identity,
            required_capabilities: &self.required_capabilities,
        }
    }

    fn materialization_digest(&self) -> Digest {
        self.materialization_identity.digest()
    }
}

impl GeneratorProgramAdmission for AdmittedNumericalGenerator {
    fn invocation_admission(&self) -> GeneratorProgramAdmissionRef<'_> {
        invocation_admission_ref(&self.invocation)
    }

    fn materialization_digest(&self) -> Digest {
        self.materialization_identity.digest()
    }
}

fn invocation_admission_ref(
    admitted: &AdmittedGeneratorInvocation,
) -> GeneratorProgramAdmissionRef<'_> {
    GeneratorProgramAdmissionRef {
        program_revision: &admitted.program_revision,
        callable: &admitted.callable,
        executable_source_digest: admitted.executable_source_digest,
        function: admitted.function,
        generator_identity: admitted.generator_identity,
        required_capabilities: &admitted.required_capabilities,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedGeneratorProgram {
    pub descriptor: ProgramExecutionDescriptor,
    pub chunks: Vec<PreparedGeneratorChunk>,
    pub input_references: Vec<runmat_execution::value::ValueRef>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedGeneratorChunk {
    pub plan: GeneratorChunk,
    pub request: ProgramExecutionRequest,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedSignalMaterialization {
    pub authored: runmat_analysis_core::PhysicalSignal,
    pub admitted: AdmittedSignalGenerator,
    pub program: PreparedGeneratorProgram,
    pub input_objects: Vec<ProgramExecutionObject>,
    pub input_admissions: Vec<SignalInputAdmission>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalMaterializationExpectation {
    pub authored: runmat_analysis_core::PhysicalSignal,
    pub admitted: AdmittedSignalGenerator,
}

impl SignalMaterializationExpectation {
    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        self.admitted.program_revision.validate()?;
        let generator = self.authored.generator().ok_or_else(|| {
            AnalysisBridgeError::Invalid(
                "signal materialization expectation is not callable-backed".into(),
            )
        })?;
        if generator.invocation.callable.display_name() != self.admitted.callable.qualified_name
            || self.admitted.sample_points.len() < 2
            || self
                .admitted
                .sample_points
                .iter()
                .any(|point| !point.is_finite())
            || self
                .admitted
                .sample_points
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(AnalysisBridgeError::Identity(
                "signal materialization expectation differs from its admission",
            ));
        }
        Ok(())
    }
}

impl PreparedSignalMaterialization {
    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        self.program.descriptor.validate_for_portable_host()?;
        validate_signal_generator_chunks(
            &self.authored,
            &self.admitted,
            &self
                .program
                .chunks
                .iter()
                .map(|chunk| chunk.plan.clone())
                .collect::<Vec<_>>(),
        )?;
        validate_prepared_generator_program(&self.admitted, &self.program)?;
        self.validate_input_admissions()?;
        Ok(())
    }

    fn validate_input_admissions(&self) -> AnalysisBridgeResult<()> {
        let declared_admissions = self
            .authored
            .generator()
            .into_iter()
            .flat_map(|generator| generator.invocation.arguments.iter())
            .filter_map(|argument| match argument {
                runmat_analysis_core::GeneratorArgument::Artifact(artifact) => Some(
                    artifact
                        .content_digest
                        .parse::<Digest>()
                        .map(|digest| (artifact.artifact_id.clone(), digest))
                        .map_err(AnalysisBridgeError::Platform),
                ),
                _ => None,
            })
            .collect::<AnalysisBridgeResult<BTreeSet<_>>>()?;
        let admitted_inputs = self
            .input_admissions
            .iter()
            .map(|admission| {
                admission.validate()?;
                Ok((admission.artifact_id.clone(), admission.logical_digest))
            })
            .collect::<AnalysisBridgeResult<BTreeSet<_>>>()?;
        if declared_admissions.len() != self.input_admissions.len()
            || declared_admissions != admitted_inputs
        {
            return Err(AnalysisBridgeError::Identity(
                "signal input admissions differ from authored artifact arguments",
            ));
        }
        Ok(())
    }

    pub fn validate_closed(&self) -> AnalysisBridgeResult<()> {
        self.validate()?;
        validate_closed_generator_inputs(&self.program, &self.input_objects, &self.input_admissions)
    }

    pub fn rebind(
        mut self,
        access: &runmat_execution_artifact::ArtifactAccess,
    ) -> AnalysisBridgeResult<Self> {
        rebind_generator_inputs(&mut self.program, &mut self.input_objects, access)?;
        self.validate_closed()?;
        Ok(self)
    }

    pub fn expectation(&self) -> AnalysisBridgeResult<SignalMaterializationExpectation> {
        self.validate()?;
        Ok(SignalMaterializationExpectation {
            authored: self.authored.clone(),
            admitted: self.admitted.clone(),
        })
    }
}

pub(crate) fn validate_prepared_generator_program(
    admitted: &impl GeneratorProgramAdmission,
    program: &PreparedGeneratorProgram,
) -> AnalysisBridgeResult<()> {
    let admission = admitted.invocation_admission();
    if &program.descriptor.recipe.program_revision != admission.program_revision
        || program.descriptor.function
            != usize::try_from(admission.function.0).unwrap_or(usize::MAX)
        || program.descriptor.requested_outputs != 1
        || program.descriptor.recipe.execution_mode != GENERATOR_EXECUTION_MODE
        || program.descriptor.artifact.form != ExecutableForm::ExecutableUnitV3
    {
        return Err(AnalysisBridgeError::Identity(
            "prepared generator program differs from its admitted callable",
        ));
    }
    for chunk in &program.chunks {
        chunk.request.validate_for_portable_host()?;
        ProgramExecutionInputs {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
            arguments: chunk.request.arguments.clone(),
            objects: input_locations(&program.input_references),
        }
        .validate()?;
        if chunk.request.recipe != program.descriptor.recipe
            || chunk.request.artifact != program.descriptor.artifact
            || chunk.request.function != program.descriptor.function
            || chunk.request.requested_outputs != program.descriptor.requested_outputs
            || chunk.request.arguments != chunk.plan.arguments
        {
            return Err(AnalysisBridgeError::Identity(
                "prepared generator chunk differs from its exact program descriptor",
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_closed_generator_inputs(
    program: &PreparedGeneratorProgram,
    input_objects: &[ProgramExecutionObject],
    input_admissions: &[SignalInputAdmission],
) -> AnalysisBridgeResult<()> {
    if input_objects.len() != program.input_references.len() {
        return Err(AnalysisBridgeError::Identity(
            "generator input object closure differs from its reference inventory",
        ));
    }
    for (object, reference) in input_objects.iter().zip(&program.input_references) {
        object.validate()?;
        if &object.reference != reference || !object.is_externalized_value() {
            return Err(AnalysisBridgeError::Identity(
                "generator input object differs from its immutable value reference",
            ));
        }
    }
    let objects = input_objects
        .iter()
        .map(|object| (object.reference.logical_digest, object))
        .collect::<BTreeMap<_, _>>();
    for admission in input_admissions {
        let object =
            objects
                .get(&admission.logical_digest)
                .ok_or(AnalysisBridgeError::Identity(
                    "generator input admission has no immutable value object",
                ))?;
        let value = object
            .decode_externalized_value()?
            .ok_or(AnalysisBridgeError::Identity(
                "generator input admission refers to a non-value object",
            ))?;
        validate_signal_input_payload(admission, &value)?;
    }
    Ok(())
}

pub(crate) fn rebind_generator_inputs(
    program: &mut PreparedGeneratorProgram,
    input_objects: &mut Vec<ProgramExecutionObject>,
    access: &runmat_execution_artifact::ArtifactAccess,
) -> AnalysisBridgeResult<()> {
    let mut replacements = BTreeMap::new();
    for object in input_objects.iter() {
        let rebound = object.rebind_externalized_value(access)?;
        replacements.insert(object.reference.logical_digest, rebound);
    }
    for chunk in &mut program.chunks {
        for argument in &mut chunk.plan.arguments {
            rebind_value(argument, &replacements)?;
        }
        chunk.request.arguments = chunk.plan.arguments.clone();
    }
    *input_objects = replacements.into_values().collect();
    program.input_references = input_objects
        .iter()
        .map(|object| object.reference.clone())
        .collect();
    Ok(())
}

#[derive(Clone, Debug)]
pub struct GeneratorExecutionContext {
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

/// Bind admitted callable facts to the exact portable executable and produce one request per
/// deterministic materialization chunk.
pub fn prepare_generator_program(
    admitted: &impl GeneratorProgramAdmission,
    executable: ExecutableUnitEnvelope,
    chunks: Vec<GeneratorChunk>,
    input_objects: &[ProgramExecutionObject],
) -> AnalysisBridgeResult<PreparedGeneratorProgram> {
    let admission = admitted.invocation_admission();
    executable.validate()?;
    let identity = &executable.manifest.identity;
    if &identity.program != admission.program_revision
        || identity.root_package != admission.callable.owner_identity
        || identity.entrypoint_kind != ExecutableEntrypointKind::Function
        || identity.entrypoint_function != admission.function
        || identity.source_digest != admission.executable_source_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "portable executable differs from the admitted frozen callable",
        ));
    }
    if !admission
        .required_capabilities
        .is_subset(&executable.manifest.capabilities.0)
    {
        return Err(AnalysisBridgeError::Identity(
            "portable executable omits admitted callable capabilities",
        ));
    }
    if chunks.is_empty() {
        return Err(AnalysisBridgeError::Invalid(
            "generator program requires at least one chunk".into(),
        ));
    }
    let executable_bytes = executable.canonical_bytes()?;
    let entrypoint = usize::try_from(admission.function.0).map_err(|_| {
        AnalysisBridgeError::Invalid("signal function identity exceeds this host".into())
    })?;
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: admission.program_revision.clone(),
        entrypoint: entrypoint.to_string(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: GENERATOR_EXECUTION_MODE.into(),
        target: ProgramTarget::portable(GENERATOR_TARGET_PROFILE),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let artifact =
        ProgramArtifact::materialize(&recipe, ExecutableForm::ExecutableUnitV3, executable_bytes)?;
    let descriptor = ProgramExecutionDescriptor {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe,
        artifact,
        function: entrypoint,
        requested_outputs: 1,
    };
    descriptor.validate_for_portable_host()?;

    let mut input_references = input_objects
        .iter()
        .map(|object| {
            object.validate()?;
            if object.reference.kind != runmat_execution::value::ValueRefKind::DriverObject
                || !object.is_externalized_value()
            {
                return Err(AnalysisBridgeError::Invalid(
                    "generator inputs must be externalized immutable values".into(),
                ));
            }
            Ok(object.reference.clone())
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    input_references.sort_by_key(|reference| reference.logical_digest);
    input_references.dedup();
    if input_references.len() != input_objects.len() {
        return Err(AnalysisBridgeError::Invalid(
            "generator input object inventory is duplicated".into(),
        ));
    }

    let chunks = chunks
        .into_iter()
        .map(|plan| {
            let inputs = ProgramExecutionInputs {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
                arguments: plan.arguments.clone(),
                objects: input_locations(&input_references),
            };
            let request = ProgramExecutionRequest::from_parts(descriptor.clone(), inputs)?;
            request.validate_for_portable_host()?;
            Ok(PreparedGeneratorChunk { plan, request })
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    let referenced_inputs = chunks
        .iter()
        .flat_map(|chunk| chunk.plan.arguments.iter())
        .fold(BTreeSet::new(), |mut references, argument| {
            collect_value_reference_ids(argument, &mut references);
            references
        });
    let supplied_inputs = input_references
        .iter()
        .map(|reference| reference.id)
        .collect::<BTreeSet<_>>();
    if referenced_inputs != supplied_inputs {
        return Err(AnalysisBridgeError::Invalid(
            "generator input object inventory is not exact".into(),
        ));
    }
    Ok(PreparedGeneratorProgram {
        descriptor,
        chunks,
        input_references,
    })
}

pub fn prepare_signal_materialization(
    authored: runmat_analysis_core::PhysicalSignal,
    admitted: AdmittedSignalGenerator,
    executable: ExecutableUnitEnvelope,
    chunks: Vec<GeneratorChunk>,
    mut input_objects: Vec<ProgramExecutionObject>,
    mut input_admissions: Vec<SignalInputAdmission>,
) -> AnalysisBridgeResult<PreparedSignalMaterialization> {
    input_objects.sort_by_key(|object| object.reference.logical_digest);
    input_admissions.sort_by(|left, right| {
        (&left.artifact_id, left.logical_digest).cmp(&(&right.artifact_id, right.logical_digest))
    });
    let program = prepare_generator_program(&admitted, executable, chunks, &input_objects)?;
    let prepared = PreparedSignalMaterialization {
        authored,
        admitted,
        program,
        input_objects,
        input_admissions,
    };
    prepared.validate_closed()?;
    Ok(prepared)
}

fn input_locations(
    references: &[runmat_execution::value::ValueRef],
) -> Vec<ProgramExecutionObjectLocation> {
    references
        .iter()
        .enumerate()
        .map(|(index, reference)| ProgramExecutionObjectLocation {
            reference: reference.clone(),
            artifact_id: format!("signal-input-{index}"),
        })
        .collect()
}

fn collect_value_reference_ids(
    value: &runmat_execution::value::ValuePayload,
    references: &mut BTreeSet<runmat_execution::identity::ValueId>,
) {
    match value {
        runmat_execution::value::ValuePayload::Object(reference) => {
            references.insert(reference.id);
        }
        runmat_execution::value::ValuePayload::Inline(inline) => match inline.as_ref() {
            runmat_execution::value::InlineValue::Cell { values, .. }
            | runmat_execution::value::InlineValue::OutputList(values) => {
                for value in values {
                    collect_value_reference_ids(value, references);
                }
            }
            runmat_execution::value::InlineValue::Struct(fields) => {
                for field in fields {
                    collect_value_reference_ids(&field.value, references);
                }
            }
            runmat_execution::value::InlineValue::Symbolic(data)
            | runmat_execution::value::InlineValue::ImmutableValueClass(data) => {
                for field in &data.fields {
                    collect_value_reference_ids(&field.value, references);
                }
            }
            runmat_execution::value::InlineValue::Callable(callable) => {
                for capture in &callable.captures {
                    collect_value_reference_ids(capture, references);
                }
            }
            _ => {}
        },
    }
}

fn rebind_value(
    value: &mut runmat_execution::value::ValuePayload,
    replacements: &BTreeMap<Digest, ProgramExecutionObject>,
) -> AnalysisBridgeResult<()> {
    match value {
        runmat_execution::value::ValuePayload::Object(reference) => {
            let replacement = replacements.get(&reference.logical_digest).ok_or(
                AnalysisBridgeError::Identity(
                    "signal argument references an absent immutable value object",
                ),
            )?;
            **reference = replacement.reference.clone();
        }
        runmat_execution::value::ValuePayload::Inline(inline) => match inline.as_mut() {
            runmat_execution::value::InlineValue::Cell { values, .. }
            | runmat_execution::value::InlineValue::OutputList(values) => {
                for value in values {
                    rebind_value(value, replacements)?;
                }
            }
            runmat_execution::value::InlineValue::Struct(fields) => {
                for field in fields {
                    rebind_value(&mut field.value, replacements)?;
                }
            }
            runmat_execution::value::InlineValue::Symbolic(data)
            | runmat_execution::value::InlineValue::ImmutableValueClass(data) => {
                for field in &mut data.fields {
                    rebind_value(&mut field.value, replacements)?;
                }
            }
            runmat_execution::value::InlineValue::Callable(callable) => {
                for capture in &mut callable.captures {
                    rebind_value(capture, replacements)?;
                }
            }
            _ => {}
        },
    }
    Ok(())
}

pub fn build_generator_task_submission(
    admitted: &impl GeneratorProgramAdmission,
    chunk: &PreparedGeneratorChunk,
    dependencies: BTreeSet<TaskId>,
    context: GeneratorExecutionContext,
) -> AnalysisBridgeResult<TaskSubmission> {
    let admission = admitted.invocation_admission();
    chunk.request.validate_for_portable_host()?;
    context.resources.validate()?;
    let expected_artifact = ArtifactId::derive(&[chunk.request.artifact.id.0.bytes()]);
    if &chunk.request.recipe.program_revision != admission.program_revision
        || chunk.request.function != usize::try_from(admission.function.0).unwrap_or(usize::MAX)
        || !context
            .resources
            .required_capabilities
            .contains(&runmat_execution::resource::Capability::ProcessIsolation)
    {
        return Err(AnalysisBridgeError::Invalid(
            "signal task context differs from its admitted portable program".into(),
        ));
    }
    let task_id = TaskId::derive(&[
        admitted.materialization_digest().bytes(),
        chunk.plan.identity.bytes(),
    ]);
    Ok(TaskSubmission {
        request: TaskRequest {
            id: task_id,
            scope_id: context.scope_id,
            pool_id: context.pool_id,
            program_artifact_id: expected_artifact,
            callable: Callable {
                owner_identity: admission.callable.owner_identity.clone(),
                qualified_name: admission.callable.qualified_name.clone(),
                entrypoint_digest: Digest::sha256(
                    [
                        admission.generator_identity.digest().bytes().as_slice(),
                        chunk.plan.identity.bytes().as_slice(),
                    ]
                    .concat(),
                ),
            },
            inputs: chunk.request.arguments.clone(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            resources: context.resources,
            retry: context.retry,
            deadline_unix_millis: context.deadline_unix_millis,
        },
        dependencies,
        priority: context.priority,
    })
}

#[cfg(test)]
#[path = "signal_program_tests.rs"]
mod tests;
