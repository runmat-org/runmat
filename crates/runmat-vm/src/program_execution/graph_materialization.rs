use std::rc::Rc;

use runmat_execution::{value::ValueLimits, OutputContract, ProgramCallable};
use runmat_execution_artifact::{
    ExecutableForm, GraphExecutionRequest, ProgramArtifact, ProgramBuildRecipe, ProgramTarget,
    PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
};
use runmat_runtime::execution::{
    DeferredCall, DeferredProgram, ExecutionServiceError, GraphResultReceipt,
};

/// Materialize one accepted selected-graph task. The deferred authority owns
/// the target and values until this exact spawn boundary; neither a name nor
/// the caller's current program revision can select a worker product here.
pub fn materialize_deferred_graph_call(
    call: &DeferredCall,
    outputs: OutputContract,
    target: ProgramTarget,
    limits: ValueLimits,
) -> Result<
    (
        ProgramCallable,
        GraphExecutionRequest,
        Rc<dyn GraphResultReceipt>,
    ),
    ExecutionServiceError,
> {
    let DeferredProgram::SelectedGraph(selected) = &call.program else {
        return Err(ExecutionServiceError::Failed(
            "graph materialization requires one selected V3 authority".into(),
        ));
    };
    let requested_outputs = u16::try_from(call.invocation.requested_outputs())
        .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
    if requested_outputs != outputs.requested_outputs {
        return Err(ExecutionServiceError::InvalidOutputContract);
    }
    let captured = selected
        .snapshot(&call.invocation, limits)
        .map_err(|error| match error {
            runmat_runtime::execution::value_codec::ValueCodecError::Allocation {
                source, ..
            } => ExecutionServiceError::HostAllocation(source),
            error => ExecutionServiceError::Failed(error.to_string()),
        })?;
    // Preserve the capture's canonical encoding. The shared request validator
    // checks empty pure policy sets and exact ordered foreign-policy sets;
    // materialization must not invent entries or erase malformed claims.
    let primary_policy = captured
        .selected_foreign_policies
        .first()
        .copied()
        .flatten();
    let primary =
        runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(&captured.primary_unit)
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    let revision = primary.manifest.identity.program.clone();
    let selected_entrypoint = primary.manifest.identity.entrypoint.clone();
    let selected_interop = primary.manifest.interop.clone();
    if call
        .program_revision
        .as_ref()
        .is_some_and(|claimed| claimed != &revision)
    {
        return Err(ExecutionServiceError::Failed(
            "deferred graph revision differs from its admitted V3 product".into(),
        ));
    }
    let invocation_context = match &call.invocation {
        runmat_runtime::execution::DeferredInvocation::Callable(_) => {
            runmat_execution::ProgramInvocationContext::Direct
        }
        runmat_runtime::execution::DeferredInvocation::Program { context, .. } => context.clone(),
    };
    let mut selected_capabilities = call.capabilities.clone();
    selected_capabilities
        .0
        .extend(primary.manifest.capabilities.0.iter().cloned());
    for unit in &captured.retained_units {
        let admitted = runmat_execution::ExecutableUnitEnvelope::admission(unit)
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        selected_capabilities
            .0
            .extend(admitted.capabilities.0.into_iter());
    }
    let accelerators = runmat_execution::resource::accelerator_requirements_for_capabilities(
        &selected_capabilities,
    )
    .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    let callable = match &captured.roots {
        runmat_execution_artifact::GraphInvocationRoots::Callable { target, .. } => {
            let Some(runmat_execution::value::GraphNode::Callable(node)) =
                captured.graph.nodes.get(target.0 as usize)
            else {
                return Err(ExecutionServiceError::Failed(
                    "selected graph target is not a callable node".into(),
                ));
            };
            ProgramCallable::graph_target(node.callable_digest)
        }
        runmat_execution_artifact::GraphInvocationRoots::ParallelRegion { region, .. } => {
            ProgramCallable::parallel_region(*region)
        }
        runmat_execution_artifact::GraphInvocationRoots::Builtin { name, .. } => {
            ProgramCallable::Builtin { name: name.clone() }
        }
    };
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision,
        // GraphTarget/Builtin/ParallelRegion is only the scheduler label.
        // Artifact identity remains the exact admitted primary V3 entrypoint.
        entrypoint: selected_entrypoint,
        outputs,
        execution_mode: "interpreter".into(),
        target,
        interop: selected_interop,
        accelerators,
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        foreign_policy: primary_policy
            .map(runmat_execution_artifact::foreign_policy_descriptor)
            .transpose()
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?,
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::ExecutableUnitV4,
        captured.primary_unit.to_vec(),
    )
    .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    let controls = runmat_execution_artifact::GraphInvocationControls {
        environment: call.environment.clone(),
        console: call.console,
    };
    controls
        .validate()
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    let invocation_controls = (!controls.is_empty()).then_some(controls);
    let required_schema = if invocation_controls.is_some() {
        runmat_execution::value::GraphRequiredSchema::InvocationControlsV14
    } else {
        captured.required_schema
    };
    let request = GraphExecutionRequest {
        invocation_controls,
        project_runtime_policy: None,
        schema_version: required_schema.version(),
        recipe,
        artifact,
        class_dependencies: captured.class_dependencies,
        class_method_dependencies: captured.class_method_dependencies,
        static_dependencies: captured.static_dependencies,
        selected_bundle_archives: captured
            .selected_bundle_archives
            .into_iter()
            .map(|archive| archive.map(|bytes| bytes.as_ref().to_vec()))
            .collect(),
        selected_foreign_policies: captured.selected_foreign_policies,
        retained_units: captured
            .retained_units
            .iter()
            .map(|unit| unit.to_vec())
            .collect(),
        graph: captured.graph,
        roots: captured.roots,
        context: invocation_context,
        assignment: None,
        job_id: None,
        requested_outputs,
    };
    request
        .validate()
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    Ok((callable, request, captured.receipt))
}
