//! Bind an original caller grant to the existing authenticated Graph projection.
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution::{identity::ArtifactId, resource::Capability};
use runmat_execution_artifact::GraphExecutionRequest;
use runmat_execution_runner::TaskSubmission;

/// Backend requirements added while constructing a submission, before admission.
/// Original admitted grants are validated below, never expanded in place.
pub(crate) fn required_capabilities(
    schema_version: u16,
) -> NativeExecutionResult<std::collections::BTreeSet<Capability>> {
    let schema = runmat_execution::value::GraphRequiredSchema::from_version(schema_version)
        .ok_or_else(|| invalid("unsupported Graph schema"))?;
    let mut capabilities = std::collections::BTreeSet::from([Capability::ProcessIsolation]);
    if let Some(capability) = schema.required_capability() {
        capabilities.insert(capability);
    }
    Ok(capabilities)
}

pub(crate) fn primary_artifact(request: &GraphExecutionRequest) -> ArtifactId {
    ArtifactId::derive(&[request.artifact.id.0.bytes()])
}
pub(crate) fn validate_original(
    request: &GraphExecutionRequest,
    task: &TaskSubmission,
) -> NativeExecutionResult<()> {
    let original = &task.request;
    original.validate_executable().map_err(invalid)?;
    original
        .validate_invocation_reservation()
        .map_err(invalid)?;
    request
        .recipe
        .validate_resource_request(&original.resources)
        .map_err(invalid)?;
    let callable = request.scheduler_callable().map_err(invalid)?;
    let host = request
        .execution_host_requirement(original.host.permitted_trust_tiers.clone())
        .map_err(invalid)?;
    if original.program_artifact_id() != Some(primary_artifact(request))
        || original.callable.program != callable
        || original.host != host
        || original.invocation_context != request.context
        || original.outputs != request.recipe.outputs
        || !original.inputs.is_empty()
        || !runmat_execution_artifact::GraphInvocationControls::matches_original(
            request.invocation_controls.as_ref(),
            &original.environment,
            original.console,
        )
    {
        return Err(invalid("original task differs from its exact Graph source, callable, context, input or output authority"));
    }
    if !original
        .resources
        .required_capabilities
        .contains(&Capability::ProcessIsolation)
    {
        return Err(invalid(
            "Graph task requires process isolation in its original grant",
        ));
    }
    let schema = runmat_execution::value::GraphRequiredSchema::from_version(request.schema_version)
        .ok_or_else(|| invalid("unsupported Graph schema"))?;
    if schema
        .required_capability()
        .is_some_and(|cap| !original.resources.required_capabilities.contains(&cap))
    {
        return Err(invalid(
            "original task omits its required Graph decoder capability",
        ));
    }
    Ok(())
}
pub(crate) fn validate_floor(
    request: &GraphExecutionRequest,
    task: &TaskSubmission,
    wire_length: u64,
) -> NativeExecutionResult<()> {
    let wire = usize::try_from(wire_length).map_err(invalid)?;
    let memory = request
        .input_allocation_floor_bytes(wire)
        .map_err(invalid)?;
    if task.request.resources.memory_bytes < memory {
        return Err(invalid(
            "original task memory grant is below the authenticated Graph input-allocation floor",
        ));
    }
    // CPU/wall defaults are not proven input minima. Scratch is checked from
    // authenticated materialized objects separately, not from wire length.
    Ok(())
}
pub(crate) fn validate_scratch(
    admitted: &runmat_execution_artifact::ValidatedGraphExecutionRequest,
    task: &TaskSubmission,
) -> NativeExecutionResult<()> {
    validate_scratch_bundles(admitted.selected_bundles(), task)
}
pub(crate) fn validate_scratch_bundles(
    bundles: &[Option<runmat_execution_artifact::ExecutionBundle>],
    task: &TaskSubmission,
) -> NativeExecutionResult<()> {
    // SelectedForeignMaterialization writes exactly these objects, once per
    // selected bundle, into parent-owned attempt scratch before executing.
    // This is an input floor, not an OS filesystem quota or a prediction of
    // arbitrary foreign-code writes.
    let bytes = crate::attempt_scratch::materialized_bytes(bundles)?;
    crate::attempt_scratch::remaining_bytes(
        task.request.resources.scratch_bytes,
        bytes,
        task.request.console,
    )?;
    Ok(())
}
fn invalid(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
