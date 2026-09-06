use std::collections::BTreeSet;

use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::ResourceRequest;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::{
    Digest, ExecutableEntrypointKind, ExecutableUnitEnvelope, ExecutionScopeId, OutputContract,
    PoolId, TaskId,
};
use runmat_execution_artifact::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionDescriptor,
    ProgramExecutionInputs, ProgramExecutionObjectLocation, ProgramExecutionRequest, ProgramTarget,
    PROGRAM_BUILD_RECIPE_SCHEMA_VERSION, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_execution_runner::TaskSubmission;

use crate::{
    validate_signal_generator_chunks, AdmittedSignalGenerator, AnalysisBridgeError,
    AnalysisBridgeResult, SignalGeneratorChunk,
};

pub const SIGNAL_GENERATOR_EXECUTION_MODE: &str = "analysis-signal-generator";
pub const SIGNAL_GENERATOR_TARGET_PROFILE: &str = "portable-executable-unit-v3";

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedSignalGeneratorProgram {
    pub descriptor: ProgramExecutionDescriptor,
    pub chunks: Vec<PreparedSignalGeneratorChunk>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedSignalGeneratorChunk {
    pub plan: SignalGeneratorChunk,
    pub request: ProgramExecutionRequest,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedSignalMaterialization {
    pub authored: runmat_analysis_core::PhysicalSignal,
    pub admitted: AdmittedSignalGenerator,
    pub program: PreparedSignalGeneratorProgram,
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
        if generator.qualified_function != self.admitted.callable.qualified_name
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
        if self.program.descriptor.recipe.program_revision != self.admitted.program_revision
            || self.program.descriptor.function
                != usize::try_from(self.admitted.function.0).unwrap_or(usize::MAX)
            || self.program.descriptor.requested_outputs != 1
            || self.program.descriptor.recipe.execution_mode != SIGNAL_GENERATOR_EXECUTION_MODE
            || self.program.descriptor.artifact.form != ExecutableForm::ExecutableUnitV3
        {
            return Err(AnalysisBridgeError::Identity(
                "prepared signal program differs from its admitted callable",
            ));
        }
        for chunk in &self.program.chunks {
            chunk.request.validate_for_portable_host()?;
            if chunk.request.recipe != self.program.descriptor.recipe
                || chunk.request.artifact != self.program.descriptor.artifact
                || chunk.request.function != self.program.descriptor.function
                || chunk.request.requested_outputs != self.program.descriptor.requested_outputs
                || chunk.request.arguments != chunk.plan.arguments
            {
                return Err(AnalysisBridgeError::Identity(
                    "prepared signal chunk differs from its exact program descriptor",
                ));
            }
        }
        Ok(())
    }

    pub fn expectation(&self) -> AnalysisBridgeResult<SignalMaterializationExpectation> {
        self.validate()?;
        Ok(SignalMaterializationExpectation {
            authored: self.authored.clone(),
            admitted: self.admitted.clone(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct SignalGeneratorExecutionContext {
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

/// Bind admitted callable facts to the exact portable executable and produce one request per
/// deterministic materialization chunk.
pub fn prepare_signal_generator_program(
    admitted: &AdmittedSignalGenerator,
    executable: ExecutableUnitEnvelope,
    chunks: Vec<SignalGeneratorChunk>,
    input_objects: Vec<ProgramExecutionObjectLocation>,
) -> AnalysisBridgeResult<PreparedSignalGeneratorProgram> {
    executable.validate()?;
    let identity = &executable.manifest.identity;
    if identity.program != admitted.program_revision
        || identity.root_package != admitted.callable.owner_identity
        || identity.entrypoint_kind != ExecutableEntrypointKind::Function
        || identity.entrypoint_function != admitted.function
        || identity.source_digest != admitted.executable_source_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "portable executable differs from the admitted frozen callable",
        ));
    }
    if !admitted
        .required_capabilities
        .is_subset(&executable.manifest.capabilities.0)
    {
        return Err(AnalysisBridgeError::Identity(
            "portable executable omits admitted callable capabilities",
        ));
    }
    if chunks.is_empty() {
        return Err(AnalysisBridgeError::Invalid(
            "signal generator program requires at least one chunk".into(),
        ));
    }
    let executable_bytes = executable.canonical_bytes()?;
    let entrypoint = usize::try_from(admitted.function.0).map_err(|_| {
        AnalysisBridgeError::Invalid("signal function identity exceeds this host".into())
    })?;
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: admitted.program_revision.clone(),
        entrypoint: entrypoint.to_string(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: SIGNAL_GENERATOR_EXECUTION_MODE.into(),
        target: ProgramTarget::portable(SIGNAL_GENERATOR_TARGET_PROFILE),
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

    let chunks = chunks
        .into_iter()
        .map(|plan| {
            let inputs = ProgramExecutionInputs {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
                arguments: plan.arguments.clone(),
                objects: input_objects.clone(),
            };
            let request = ProgramExecutionRequest::from_parts(descriptor.clone(), inputs)?;
            request.validate_for_portable_host()?;
            Ok(PreparedSignalGeneratorChunk { plan, request })
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    Ok(PreparedSignalGeneratorProgram { descriptor, chunks })
}

pub fn prepare_signal_materialization(
    authored: runmat_analysis_core::PhysicalSignal,
    admitted: AdmittedSignalGenerator,
    executable: ExecutableUnitEnvelope,
    chunks: Vec<SignalGeneratorChunk>,
    input_objects: Vec<ProgramExecutionObjectLocation>,
) -> AnalysisBridgeResult<PreparedSignalMaterialization> {
    let program = prepare_signal_generator_program(&admitted, executable, chunks, input_objects)?;
    let prepared = PreparedSignalMaterialization {
        authored,
        admitted,
        program,
    };
    prepared.validate()?;
    Ok(prepared)
}

pub fn build_signal_generator_task_submission(
    admitted: &AdmittedSignalGenerator,
    chunk: &PreparedSignalGeneratorChunk,
    dependencies: BTreeSet<TaskId>,
    context: SignalGeneratorExecutionContext,
) -> AnalysisBridgeResult<TaskSubmission> {
    chunk.request.validate_for_portable_host()?;
    context.resources.validate()?;
    let expected_artifact = ArtifactId::derive(&[chunk.request.artifact.id.0.bytes()]);
    if chunk.request.recipe.program_revision != admitted.program_revision
        || chunk.request.function != usize::try_from(admitted.function.0).unwrap_or(usize::MAX)
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
        admitted.materialization_identity.digest().bytes(),
        chunk.plan.identity.bytes(),
    ]);
    Ok(TaskSubmission {
        request: TaskRequest {
            id: task_id,
            scope_id: context.scope_id,
            pool_id: context.pool_id,
            program_artifact_id: expected_artifact,
            callable: Callable {
                owner_identity: admitted.callable.owner_identity.clone(),
                qualified_name: admitted.callable.qualified_name.clone(),
                entrypoint_digest: Digest::sha256(
                    [
                        admitted.generator_identity.digest().bytes().as_slice(),
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
