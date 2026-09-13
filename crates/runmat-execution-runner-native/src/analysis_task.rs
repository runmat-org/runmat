use std::collections::BTreeSet;

use runmat_analysis_execution::{
    build_analysis_task_submission, build_structural_task_submission, AnalysisExecutionContext,
    AnalysisHostWorkload, PreparedAnalysisProgram,
};
use runmat_analysis_fea::execution::AnalysisTaskIdentity;
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::ValuePayload;
use runmat_execution::{ExecutionScopeId, PoolId, TaskId};
use runmat_execution_runner::TaskSubmission;

use crate::{NativeExecutionError, NativeExecutionResult};

pub(crate) struct AnalysisTaskPolicy<'a> {
    pub resources: ResourceRequest,
    pub resource_plan: Option<&'a FeaResourcePlan>,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

pub(crate) fn prepare_analysis_task(
    scope_id: ExecutionScopeId,
    pool_id: PoolId,
    host: &AnalysisHostWorkload,
    program: &PreparedAnalysisProgram,
    dependency_identities: Vec<AnalysisTaskIdentity>,
    dependency_tasks: BTreeSet<TaskId>,
    policy: AnalysisTaskPolicy<'_>,
) -> NativeExecutionResult<(TaskSubmission, AnalysisTaskIdentity)> {
    let roots = program
        .request
        .arguments
        .iter()
        .map(|argument| match argument {
            ValuePayload::Object(root) => Ok((**root).clone()),
            ValuePayload::Inline(_) => Err(NativeExecutionError::Protocol(
                "analysis input was unexpectedly inline".into(),
            )),
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let mut resources = policy.resources;
    resources
        .required_capabilities
        .extend(host.workload.generic_capabilities().map_err(protocol)?);
    resources
        .required_capabilities
        .insert(Capability::ProcessIsolation);
    let context = AnalysisExecutionContext {
        scope_id,
        pool_id,
        program_artifact_id: ArtifactId::derive(&[program.request.artifact.id.0.bytes()]),
        resources,
        retry: policy.retry,
        deadline_unix_millis: policy.deadline_unix_millis,
        priority: policy.priority,
    };
    let submission = match policy.resource_plan {
        Some(resource_plan) => build_structural_task_submission(
            host,
            resource_plan,
            roots,
            dependency_identities.clone(),
            dependency_tasks,
            context,
        ),
        None => build_analysis_task_submission(
            host,
            roots,
            dependency_identities.clone(),
            dependency_tasks,
            context,
        ),
    }
    .map_err(protocol)?;
    let identity = AnalysisTaskIdentity::new(host.workload.identity, dependency_identities)
        .map_err(protocol)?;
    if submission.request.id != TaskId::derive(&[identity.digest().bytes()]) {
        return Err(NativeExecutionError::Protocol(
            "analysis task identity does not converge".into(),
        ));
    }
    Ok((submission, identity))
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
