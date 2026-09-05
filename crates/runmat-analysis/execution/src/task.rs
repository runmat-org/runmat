use std::collections::BTreeSet;

use runmat_analysis_fea::execution::AnalysisTaskIdentity;
use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::ResourceRequest;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::value::ValueRef;
use runmat_execution::{ExecutionScopeId, OutputContract, PoolId, TaskId};
use runmat_execution_runner::TaskSubmission;

use crate::{AnalysisBridgeResult, AnalysisWorkloadRequest};

pub const ANALYSIS_EXECUTION_CALLABLE_OWNER: &str = "runmat.analysis.host";

#[derive(Clone, Debug)]
pub struct AnalysisExecutionContext {
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
    pub program_artifact_id: ArtifactId,
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

pub fn build_analysis_task_submission(
    workload: &AnalysisWorkloadRequest,
    input_roots: Vec<ValueRef>,
    dependency_identities: Vec<AnalysisTaskIdentity>,
    dependency_tasks: BTreeSet<TaskId>,
    context: AnalysisExecutionContext,
) -> AnalysisBridgeResult<TaskSubmission> {
    workload.validate()?;
    context.resources.validate()?;
    if input_roots.len() != workload.inputs.len()
        || dependency_identities.len() != dependency_tasks.len()
        || !workload
            .generic_capabilities()?
            .is_subset(&context.resources.required_capabilities)
    {
        return Err(crate::AnalysisBridgeError::Invalid(
            "task inputs or worker capabilities do not satisfy the analysis workload".into(),
        ));
    }
    let identity = AnalysisTaskIdentity::new(workload.identity, dependency_identities)?;
    let task_id = TaskId::derive(&[identity.digest().bytes()]);
    let request = TaskRequest {
        id: task_id,
        scope_id: context.scope_id,
        pool_id: context.pool_id,
        program_artifact_id: context.program_artifact_id,
        callable: Callable {
            owner_identity: ANALYSIS_EXECUTION_CALLABLE_OWNER.into(),
            qualified_name: "analysis_workload".into(),
            entrypoint_digest: workload.identity.digest(),
        },
        inputs: input_roots
            .into_iter()
            .map(|root| runmat_execution::value::ValuePayload::Object(Box::new(root)))
            .collect(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        resources: context.resources,
        retry: context.retry,
        deadline_unix_millis: context.deadline_unix_millis,
    };
    Ok(TaskSubmission {
        request,
        dependencies: dependency_tasks,
        priority: context.priority,
    })
}
