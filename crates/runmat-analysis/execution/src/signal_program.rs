use std::collections::{BTreeMap, BTreeSet};

use runmat_execution::identity::ArtifactId;
use runmat_execution::task::TaskRequest;
use runmat_execution::{Digest, ExecutableUnitEnvelope, OutputContract, TaskId};
use runmat_execution_artifact::{
    ExecutableForm, ProgramExecutionDescriptor, ProgramExecutionInputs, ProgramExecutionObject,
    ProgramExecutionObjectLocation, ProgramExecutionRequest, WorkerProgramRequest,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_execution_runner::TaskSubmission;

use crate::signal_input::validate_signal_input_payload;
use crate::{
    validate_signal_generator_chunks, AdmittedGeneratorInvocation, AdmittedNumericalGenerator,
    AdmittedSignalGenerator, AnalysisBridgeError, AnalysisBridgeResult, GeneratorChunk,
    SignalInputAdmission,
};

pub const GENERATOR_EXECUTION_MODE: &str = "analysis-generator";
pub const GENERATOR_TARGET_PROFILE: &str = crate::named_callable::ANALYSIS_CALLABLE_TARGET_PROFILE;
mod graph;
mod output;
pub use output::generator_chunk_output_from_success;

pub trait GeneratorProgramAdmission {
    fn invocation_admission(&self) -> GeneratorProgramAdmissionRef<'_>;
    fn materialization_digest(&self) -> Digest;
}

#[derive(Clone, Copy)]
pub struct GeneratorProgramAdmissionRef<'a> {
    pub program_revision: &'a runmat_execution::ProgramRevision,
    pub callable: &'a runmat_execution_artifact::BundleCallable,
    pub executable_source_digest: Digest,
    pub compiled_binding: &'a crate::named_callable::CompiledCallableBinding,
    pub function: runmat_types::ProgramFunctionId,
    pub generator_identity: runmat_analysis_fea::execution::GeneratorIdentity,
    pub required_capabilities: &'a BTreeSet<runmat_types::CapabilityRequirement>,
    pub frozen_inputs: &'a crate::FrozenGeneratorInputs,
    pub provenance: &'a runmat_analysis_core::GeneratorProvenance,
}

impl GeneratorProgramAdmission for AdmittedSignalGenerator {
    fn invocation_admission(&self) -> GeneratorProgramAdmissionRef<'_> {
        GeneratorProgramAdmissionRef {
            program_revision: &self.program_revision,
            callable: &self.callable,
            executable_source_digest: self.executable_source_digest,
            compiled_binding: &self.compiled_binding,
            function: self.function,
            generator_identity: self.generator_identity,
            required_capabilities: &self.required_capabilities,
            frozen_inputs: &self.frozen_inputs,
            provenance: &self.provenance,
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
        compiled_binding: &admitted.compiled_binding,
        function: admitted.function,
        generator_identity: admitted.generator_identity,
        required_capabilities: &admitted.required_capabilities,
        frozen_inputs: &admitted.frozen_inputs,
        provenance: &admitted.provenance,
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
    pub request: WorkerProgramRequest,
}
impl PreparedGeneratorChunk {
    pub fn execution_host_requirement(
        &self,
        trust: BTreeSet<runmat_execution::security::ExecutionTrustTier>,
    ) -> AnalysisBridgeResult<runmat_execution::host::ExecutionHostRequirement> {
        Ok(match &self.request {
            WorkerProgramRequest::Direct(request) => request
                .artifact
                .execution_host_requirement(&request.recipe, trust)?,
            WorkerProgramRequest::Graph(request) => request.execution_host_requirement(trust)?,
        })
    }
    pub fn graph_capability(
        &self,
    ) -> AnalysisBridgeResult<Option<runmat_execution::resource::Capability>> {
        match &self.request {
            WorkerProgramRequest::Direct(_) => Ok(None),
            WorkerProgramRequest::Graph(request) => Ok(
                runmat_execution::value::GraphRequiredSchema::from_version(request.schema_version)
                    .ok_or(AnalysisBridgeError::Identity(
                        "unsupported generator Graph schema",
                    ))?
                    .required_capability(),
            ),
        }
    }
    /// Adapters whose domain admits named/by-value functions retain an explicit
    /// direct route. They must not silently discard a selected graph closure.
    pub fn direct_request(&self) -> AnalysisBridgeResult<&ProgramExecutionRequest> {
        match &self.request {
            WorkerProgramRequest::Direct(request) => Ok(request),
            WorkerProgramRequest::Graph(_) => Err(AnalysisBridgeError::Invalid(
                "selected shared-closure generator requires the graph invocation route".into(),
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedSignalMaterialization {
    pub authored: runmat_analysis_core::PhysicalSignal,
    pub admitted: AdmittedSignalGenerator,
    pub topology_binding: Option<crate::SignalTopologyBinding>,
    pub program: PreparedGeneratorProgram,
    pub input_objects: Vec<ProgramExecutionObject>,
    pub input_admissions: Vec<SignalInputAdmission>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalMaterializationExpectation {
    pub authored: runmat_analysis_core::PhysicalSignal,
    pub admitted: AdmittedSignalGenerator,
    pub topology_binding: Option<crate::SignalTopologyBinding>,
}

impl SignalMaterializationExpectation {
    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        runmat_analysis_core::validate_physical_signals(std::slice::from_ref(&self.authored))
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        validate_topology_binding(&self.authored, self.topology_binding.as_ref())?;
        self.admitted.program_revision.validate()?;
        let generator = self.authored.generator().ok_or_else(|| {
            AnalysisBridgeError::Invalid(
                "signal materialization expectation is not callable-backed".into(),
            )
        })?;
        if self.admitted.schema_version != 2
            || self.admitted.sample_points.descriptor()
                != generator
                    .domain
                    .axis_descriptor(None)
                    .map_err(AnalysisBridgeError::Invalid)?
            || generator.invocation.callable.display_name() != self.admitted.callable.qualified_name
            || self.admitted.sample_points.scalar_count() < 2
            || self
                .admitted
                .sample_points
                .canonical_values()
                .iter()
                .any(|point| !point.is_finite())
            || self
                .admitted
                .sample_points
                .canonical_values()
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(AnalysisBridgeError::Identity(
                "signal materialization expectation differs from its admission",
            ));
        }
        Ok(())
    }

    pub fn validate_for_mesh(
        &self,
        mesh: &runmat_meshing_core::SolverMeshArtifact,
    ) -> AnalysisBridgeResult<()> {
        self.validate()?;
        if let Some(binding) = &self.topology_binding {
            binding.validate_for(&self.authored, mesh)?;
        }
        Ok(())
    }

    pub fn validate_realization(
        &self,
        realization: &runmat_analysis_fea::execution::CanonicalSignalRealization,
    ) -> AnalysisBridgeResult<()> {
        self.validate()?;
        let count = self
            .topology_binding
            .as_ref()
            .map(|binding| {
                u64::try_from(binding.entity_count()).map_err(|_| {
                    AnalysisBridgeError::Invalid("signal entity count exceeds u64".into())
                })
            })
            .transpose()?;
        if realization.topology_entity_count != count {
            return Err(AnalysisBridgeError::Identity(
                "signal realization rows differ from the prepared mesh binding",
            ));
        }
        Ok(())
    }
}

impl PreparedSignalMaterialization {
    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        runmat_analysis_core::validate_physical_signals(std::slice::from_ref(&self.authored))
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        validate_topology_binding(&self.authored, self.topology_binding.as_ref())?;
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

    pub fn validate_for_mesh(
        &self,
        mesh: &runmat_meshing_core::SolverMeshArtifact,
    ) -> AnalysisBridgeResult<()> {
        self.validate()?;
        if let Some(binding) = &self.topology_binding {
            binding.validate_for(&self.authored, mesh)?;
        }
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
            topology_binding: self.topology_binding.clone(),
        })
    }
}

fn validate_topology_binding(
    authored: &runmat_analysis_core::PhysicalSignal,
    binding: Option<&crate::SignalTopologyBinding>,
) -> AnalysisBridgeResult<()> {
    match (authored.shape(), binding) {
        (runmat_analysis_core::SignalShape::DeferredTopologyField { .. }, _) => {
            Err(AnalysisBridgeError::Invalid(
                "deferred topology requires resolution before materialization preparation".into(),
            ))
        }
        (runmat_analysis_core::SignalShape::TopologyField { .. }, Some(binding)) => {
            binding.validate_shape(authored)
        }
        (runmat_analysis_core::SignalShape::TopologyField { .. }, None) => {
            Err(AnalysisBridgeError::Invalid(
                "topology signal materialization requires an authoritative mesh binding".into(),
            ))
        }
        (_, Some(_)) => Err(AnalysisBridgeError::Invalid(
            "non-topology signal materialization carries a mesh binding".into(),
        )),
        (_, None) => Ok(()),
    }
}

pub(crate) fn validate_prepared_generator_program(
    admitted: &impl GeneratorProgramAdmission,
    program: &PreparedGeneratorProgram,
) -> AnalysisBridgeResult<()> {
    let admission = admitted.invocation_admission();
    if let Some(graph) = admission.frozen_inputs.graph() {
        crate::callable::graph::validate_identity(admission, graph)?;
    }
    if &program.descriptor.recipe.program_revision != admission.program_revision
        || program.descriptor.callable.semantic_function() != Some(admission.function)
        || program.descriptor.requested_outputs != 1
        || program.descriptor.recipe.execution_mode != GENERATOR_EXECUTION_MODE
        || program.descriptor.artifact.form != ExecutableForm::ExecutableUnitV4
    {
        return Err(AnalysisBridgeError::Identity(
            "prepared generator program differs from its admitted callable",
        ));
    }
    for chunk in &program.chunks {
        if let WorkerProgramRequest::Graph(request) = &chunk.request {
            graph::validate(admission, &chunk.plan, request)?;
            if request.recipe != program.descriptor.recipe
                || request.artifact != program.descriptor.artifact
            {
                return Err(AnalysisBridgeError::Identity(
                    "graph generator changed its original compiled program",
                ));
            }
            continue;
        }
        if admission.frozen_inputs.graph().is_some() {
            return Err(AnalysisBridgeError::Identity(
                "graph generator was demoted to flat capture inputs",
            ));
        }
        let request = chunk.direct_request()?;
        request.validate_for_portable_host()?;
        ProgramExecutionInputs {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            context: runmat_execution::ProgramInvocationContext::Direct,
            arguments: request.arguments.clone(),
            objects: input_locations(&program.input_references),
        }
        .validate()?;
        if request.recipe != program.descriptor.recipe
            || request.artifact != program.descriptor.artifact
            || request.callable != program.descriptor.callable
            || request.console != program.descriptor.console
            || request.environment != program.descriptor.environment
            || request.assignment.is_some()
            || request.job_id.is_some()
            || request.context != runmat_execution::ProgramInvocationContext::Direct
            || request.requested_outputs != program.descriptor.requested_outputs
            || request.arguments != chunk.plan.arguments
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
        if let WorkerProgramRequest::Direct(request) = &mut chunk.request {
            request.arguments = chunk.plan.arguments.clone();
        }
    }
    *input_objects = replacements.into_values().collect();
    program.input_references = input_objects
        .iter()
        .map(|object| object.reference.clone())
        .collect();
    Ok(())
}

pub type GeneratorExecutionContext = crate::named_callable::NamedCallableExecutionContext;

/// Bind admitted callable facts to the exact portable executable without planning materialization
/// chunks. This descriptor preserves executable authority while topology-dependent cardinality is
/// still deferred; it is not a prepared or executable materialization by itself.
pub fn prepare_generator_descriptor(
    admitted: &impl GeneratorProgramAdmission,
    executable: ExecutableUnitEnvelope,
) -> AnalysisBridgeResult<ProgramExecutionDescriptor> {
    let admission = admitted.invocation_admission();
    crate::named_callable::prepare_named_callable_descriptor(
        crate::named_callable::NamedCallableAdmissionRef {
            program_revision: admission.program_revision,
            callable: admission.callable,
            executable_source_digest: admission.executable_source_digest,
            compiled_binding: admission.compiled_binding,
            function: admission.function,
            required_capabilities: admission.required_capabilities,
        },
        executable,
        crate::named_callable::AnalysisCallablePurpose::Generator,
    )
}

/// Bind admitted callable facts to the exact portable executable and produce one request per
/// deterministic materialization chunk.
pub fn prepare_generator_program(
    admitted: &impl GeneratorProgramAdmission,
    executable: ExecutableUnitEnvelope,
    chunks: Vec<GeneratorChunk>,
    input_objects: &[ProgramExecutionObject],
) -> AnalysisBridgeResult<PreparedGeneratorProgram> {
    let descriptor = prepare_generator_descriptor(admitted, executable)?;
    let admission = admitted.invocation_admission();
    if let Some(graph) = admission.frozen_inputs.graph() {
        crate::callable::graph::validate_identity(admission, graph)?;
    }
    if chunks.is_empty() {
        return Err(AnalysisBridgeError::Invalid(
            "generator program requires at least one chunk".into(),
        ));
    }
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
            if admitted
                .invocation_admission()
                .frozen_inputs
                .graph()
                .is_some()
            {
                let request =
                    graph::prepare(admitted.invocation_admission(), &plan, input_objects)?;
                if request.recipe != descriptor.recipe || request.artifact != descriptor.artifact {
                    return Err(AnalysisBridgeError::Identity(
                        "graph generator differs from its exact compiled descriptor",
                    ));
                }
                return Ok(PreparedGeneratorChunk {
                    plan,
                    request: WorkerProgramRequest::graph(request),
                });
            }
            let inputs = ProgramExecutionInputs {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
                context: runmat_execution::ProgramInvocationContext::Direct,
                arguments: plan.arguments.clone(),
                objects: input_locations(&input_references),
            };
            let request = ProgramExecutionRequest::from_parts(descriptor.clone(), inputs)?;
            request.validate_for_portable_host()?;
            Ok(PreparedGeneratorChunk {
                plan,
                request: WorkerProgramRequest::direct(request),
            })
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
    mesh: Option<&runmat_meshing_core::SolverMeshArtifact>,
) -> AnalysisBridgeResult<PreparedSignalMaterialization> {
    let topology_binding = match mesh {
        Some(mesh) => crate::SignalTopologyBinding::derive(&authored, mesh)?,
        None => None,
    };
    validate_topology_binding(&authored, topology_binding.as_ref())?;
    input_objects.sort_by_key(|object| object.reference.logical_digest);
    input_admissions.sort_by(|left, right| {
        (&left.artifact_id, left.logical_digest).cmp(&(&right.artifact_id, right.logical_digest))
    });
    let program = prepare_generator_program(&admitted, executable, chunks, &input_objects)?;
    let prepared = PreparedSignalMaterialization {
        authored,
        admitted,
        topology_binding,
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
        // Pool-resident partitions are addressed by their pool handle.
        runmat_execution::value::ValuePayload::Distributed(_)
        | runmat_execution::value::ValuePayload::Composite(_) => {}
        runmat_execution::value::ValuePayload::Inline(inline) => match inline.as_ref() {
            runmat_execution::value::InlineValue::Cell { values, .. } => {
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
        runmat_execution::value::ValuePayload::Distributed(_)
        | runmat_execution::value::ValuePayload::Composite(_) => {
            return Err(AnalysisBridgeError::Invalid(
                "signal generator arguments cannot carry pool-resident values".into(),
            ));
        }
        runmat_execution::value::ValuePayload::Inline(inline) => match inline.as_mut() {
            runmat_execution::value::InlineValue::Cell { values, .. } => {
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
    chunk.request.validate()?;
    context.resources.validate()?;
    let (artifact, callable, invocation_context, inputs) = match &chunk.request {
        WorkerProgramRequest::Direct(request) => {
            if admission.frozen_inputs.graph().is_some()
                || request.callable.semantic_function() != Some(admission.function)
                || &request.recipe.program_revision != admission.program_revision
            {
                return Err(AnalysisBridgeError::Identity(
                    "direct generator task changed its original callable",
                ));
            }
            request.validate_for_portable_host()?;
            crate::named_callable::validate_program_host(request, &context.host)?;
            (
                request.artifact.id,
                request.callable.clone(),
                request.context.clone(),
                request.arguments.clone(),
            )
        }
        WorkerProgramRequest::Graph(request) => {
            crate::callable::graph::validate_identity(
                admission,
                admission
                    .frozen_inputs
                    .graph()
                    .ok_or(AnalysisBridgeError::Identity(
                        "graph generator has no frozen callable",
                    ))?,
            )?;
            graph::validate(admission, &chunk.plan, request)?;
            let derived =
                request.execution_host_requirement(context.host.permitted_trust_tiers.clone())?;
            if derived != context.host {
                return Err(AnalysisBridgeError::Identity(
                    "graph generator task changed its derived host requirement",
                ));
            }
            let schema =
                runmat_execution::value::GraphRequiredSchema::from_version(request.schema_version)
                    .ok_or(AnalysisBridgeError::Identity(
                        "generator graph schema is unsupported",
                    ))?;
            if schema.required_capability().is_some_and(|capability| {
                !context
                    .resources
                    .required_capabilities
                    .contains(&capability)
            }) {
                return Err(AnalysisBridgeError::Identity(
                    "graph generator task omits its typed schema capability",
                ));
            }
            (
                request.artifact.id,
                request.scheduler_callable()?,
                request.context.clone(),
                Vec::new(),
            )
        }
    };
    let expected_artifact = ArtifactId::derive(&[artifact.0.bytes()]);
    if !context
        .resources
        .required_capabilities
        .contains(&runmat_execution::resource::Capability::ProcessIsolation)
    {
        return Err(AnalysisBridgeError::Invalid(
            "signal task context differs from its admitted portable program".into(),
        ));
    }
    let task_id = generator_task_id(admitted.materialization_digest(), chunk.plan.identity);
    Ok(TaskSubmission {
        request: TaskRequest {
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            id: task_id,
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            scope_id: context.scope_id,
            pool_id: context.pool_id,
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                expected_artifact,
            ),
            callable: runmat_execution::task::Callable::for_program(
                &admission.callable.owner_identity,
                &callable,
            ),
            host: context.host,
            invocation_context,
            inputs,
            outputs: OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            resources: context.resources,
            retry: context.retry,
            deadline_unix_millis: context.deadline_unix_millis,
        },
        dependencies,
        priority: context.priority,
    })
}

/// Exact generator task identity, available before lowering a prepared chunk.
pub fn generator_task_id(materialization_digest: Digest, chunk_identity: Digest) -> TaskId {
    TaskId::derive(&[materialization_digest.bytes(), chunk_identity.bytes()])
}

#[cfg(test)]
#[path = "signal_program_tests.rs"]
mod tests;
