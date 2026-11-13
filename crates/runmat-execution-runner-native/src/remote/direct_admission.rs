//! The same Direct invocation contract at original pool and worker admission.
//! Bundle membership is separate from exact scheduler/invocation authority.
use runmat_execution::identity::ArtifactId;
use runmat_execution::task::TaskRequest;
use runmat_execution::{ExecutionScopeId, PoolBackend, PoolId, ProgramExecutionAssignment};
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_execution_runner::{AttemptRequest, WorkerSpec};

pub(super) enum Admission<'a> {
    Pool {
        task: &'a TaskRequest,
        scope_id: ExecutionScopeId,
        pool_id: PoolId,
    },
    Worker {
        scheduling: &'a AttemptRequest,
        worker: &'a WorkerSpec,
    },
}

pub(super) fn validate(
    program: &ProgramExecutionRequest,
    admission: Admission<'_>,
) -> Result<(), &'static str> {
    let task = match admission {
        Admission::Pool {
            task,
            scope_id,
            pool_id,
        } => {
            if task.scope_id != scope_id || task.pool_id != pool_id || program.assignment.is_some()
            {
                return Err("Direct pool request has foreign scope or a preselected assignment");
            }
            task
        }
        Admission::Worker { scheduling, worker } => {
            scheduling
                .validate()
                .map_err(|_| "Direct scheduler attempt is invalid")?;
            let expected = ProgramExecutionAssignment {
                scope_id: scheduling.scope_id,
                pool_id: scheduling.task.pool_id,
                task_id: scheduling.task_id,
                attempt_id: scheduling.id,
                worker_id: scheduling.worker_id,
                backend: PoolBackend::Remote,
                resources: scheduling.resource_assignment.clone(),
            };
            if program.assignment.as_ref() != Some(&expected)
                || scheduling.worker_id != worker.id
                || scheduling.task.pool_id != worker.pool_id
            {
                return Err("Direct assignment differs from its original scheduler attempt");
            }
            scheduling
                .task
                .host
                .is_satisfied_by(&worker.host)
                .map_err(|_| "Direct selected host policy is not satisfied by this worker")?;
            &scheduling.task
        }
    };
    program
        .validate_for_portable_host()
        .map_err(|_| "Direct program request is invalid for a portable host")?;
    task.validate_executable()
        .map_err(|_| "Direct scheduler executable is invalid")?;
    task.validate_invocation_reservation()
        .map_err(|_| "Direct invocation exceeds its original reservation")?;
    program
        .recipe
        .validate_resource_request(&task.resources)
        .map_err(|_| "Direct resources differ from its selected recipe")?;
    if task.program_artifact_id() != Some(ArtifactId::derive(&[program.artifact.id.0.bytes()]))
        || !task.callable.identifies_program(&program.callable)
        || task.invocation_context != program.context
        || task.environment != program.environment
        || task.console != program.console
        || task.inputs != program.arguments
        || task.outputs != program.recipe.outputs
    {
        return Err("Direct invocation differs from its original scheduler contract");
    }
    // Preserve the caller's trust policy; derive all executable host facts from
    // the admitted artifact rather than a worker-issued display/owner label.
    let host = program
        .artifact
        .execution_host_requirement(&program.recipe, task.host.permitted_trust_tiers.clone())
        .map_err(|_| "Direct selected host requirement is invalid")?;
    if task.host != host {
        return Err("Direct host requirement differs from its selected artifact");
    }
    // job_id is retained request metadata, not a scheduler placement grant.
    // This boundary neither fabricates nor replaces the original job identity.
    Ok(())
}
