//! Revalidate staged Graph work against its original scheduler authority.
use runmat_execution::identity::ArtifactId;
use runmat_execution::{PoolBackend, ProgramExecutionAssignment};
use runmat_execution_artifact::GraphExecutionRequest;
use runmat_execution_runner::{AttemptRequest, WorkerSpec};

pub(super) fn validate(
    scheduling: &AttemptRequest,
    graph: &GraphExecutionRequest,
    wire_bytes: Option<usize>,
    worker: &WorkerSpec,
) -> Result<(), &'static str> {
    graph
        .validate()
        .map_err(|_| "attempt graph or its selected archives are invalid")?;
    scheduling
        .validate()
        .map_err(|_| "graph scheduler attempt is invalid")?;
    let expected = ProgramExecutionAssignment {
        scope_id: scheduling.scope_id,
        pool_id: scheduling.task.pool_id,
        task_id: scheduling.task_id,
        attempt_id: scheduling.id,
        worker_id: scheduling.worker_id,
        backend: PoolBackend::Remote,
        resources: scheduling.resource_assignment.clone(),
    };
    if graph.assignment.as_ref() != Some(&expected) || graph.job_id.is_some() {
        return Err("graph assignment differs from its original scheduler attempt");
    }
    let callable = graph
        .scheduler_callable()
        .map_err(|_| "graph scheduler callable is invalid")?;
    // The caller's owner label is retained metadata, not a worker-chosen
    // program identity. The typed callable binds the selected source/target.
    if !scheduling.task.callable.identifies_program(&callable)
        || scheduling.task.program_artifact_id()
            != Some(ArtifactId::derive(&[graph.artifact.id.0.bytes()]))
    {
        return Err("graph executable differs from its original scheduler identity");
    }
    let host = graph
        .execution_host_requirement(scheduling.task.host.permitted_trust_tiers.clone())
        .map_err(|_| "graph selected host requirement is invalid")?;
    if host != scheduling.task.host || host.is_satisfied_by(&worker.host).is_err() {
        return Err("graph selected host policy is not satisfied by this worker");
    }
    if !scheduling.task.inputs.is_empty()
        || scheduling.task.invocation_context != graph.context
        || scheduling.task.outputs != graph.recipe.outputs
        || !runmat_execution_artifact::GraphInvocationControls::matches_original(
            graph.invocation_controls.as_ref(),
            &scheduling.task.environment,
            scheduling.task.console,
        )
    {
        return Err("graph invocation differs from its original scheduler contract");
    }
    let demand = wire_bytes
        .and_then(|bytes| graph.input_allocation_floor_bytes(bytes).ok())
        .ok_or("graph input allocation bound is invalid")?;
    if scheduling.task.resources.memory_bytes < demand {
        return Err("graph input allocation exceeds its original memory grant");
    }
    graph
        .recipe
        .validate_resource_request(&scheduling.task.resources)
        .map_err(|_| "graph resources differ from its selected recipe")
}
