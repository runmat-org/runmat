use std::collections::BTreeSet;

use runmat_analysis_fea::execution::AnalysisTaskIdentity;
use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::ResourceRequest;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution::{ExecutionScopeId, OutputContract, PoolId, TaskId};
use runmat_execution_runner::TaskSubmission;

use crate::{
    AnalysisBridgeResult, AnalysisHostWorkload, ANALYSIS_HOST_INVOCATION_MEDIA_TYPE,
    ANALYSIS_HOST_INVOCATION_VALUE_SCHEMA,
};

pub const ANALYSIS_EXECUTION_CALLABLE_OWNER: &str = "runmat.analysis.host";
pub const DEFAULT_ANALYSIS_MAXIMUM_ATTEMPTS: u16 = 3;

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
    host: &AnalysisHostWorkload,
    program_roots: Vec<ValueRef>,
    dependency_identities: Vec<AnalysisTaskIdentity>,
    dependency_tasks: BTreeSet<TaskId>,
    context: AnalysisExecutionContext,
) -> AnalysisBridgeResult<TaskSubmission> {
    let workload = &host.workload;
    workload.validate()?;
    context.resources.validate()?;
    let Some((invocation_root, input_roots)) = program_roots.split_first() else {
        return Err(crate::AnalysisBridgeError::Invalid(
            "analysis task is missing its invocation descriptor".into(),
        ));
    };
    if host
        .artifact_access
        .validate_reference(
            invocation_root,
            ValueRefKind::DriverObject,
            ANALYSIS_HOST_INVOCATION_MEDIA_TYPE,
            ANALYSIS_HOST_INVOCATION_VALUE_SCHEMA,
        )
        .is_err()
        || host.validate_inputs(input_roots).is_err()
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
        inputs: program_roots
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
