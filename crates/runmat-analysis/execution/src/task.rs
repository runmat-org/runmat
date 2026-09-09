use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{AnalysisNumericalObjectKind, AnalysisTaskIdentity};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
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

pub fn build_structural_task_submission(
    host: &AnalysisHostWorkload,
    resource_plan: &FeaResourcePlan,
    program_roots: Vec<ValueRef>,
    dependency_identities: Vec<AnalysisTaskIdentity>,
    dependency_tasks: BTreeSet<TaskId>,
    context: AnalysisExecutionContext,
) -> AnalysisBridgeResult<TaskSubmission> {
    resource_plan.validate_identity().map_err(|error| {
        crate::AnalysisBridgeError::Invalid(format!("invalid structural resource plan: {error}"))
    })?;
    let matching_plan_inputs = host
        .workload
        .inputs
        .iter()
        .filter(|input| input.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .collect::<Vec<_>>();
    if !matches!(matching_plan_inputs.as_slice(), [input] if input.logical_identity == resource_plan.identity.digest())
    {
        return Err(crate::AnalysisBridgeError::Identity(
            "structural workload does not bind the canonical resource plan",
        ));
    }
    let planned = &resource_plan.request;
    if context.resources.cpu_millicores != planned.cpu_millicores
        || context.resources.memory_bytes != planned.memory_bytes
        || context.resources.scratch_bytes != planned.scratch_bytes
        || context.resources.max_wall_millis != planned.max_wall_millis
        || context.resources.max_artifact_bytes != planned.max_artifact_bytes
        || context.resources.max_egress_bytes != planned.max_egress_bytes
        || context.resources.max_relay_bytes != planned.max_relay_bytes
        || context.resources.accelerators != planned.accelerators
        || !planned
            .required_capabilities
            .is_subset(&context.resources.required_capabilities)
    {
        return Err(crate::AnalysisBridgeError::Identity(
            "structural task resources differ from the canonical post-projection plan",
        ));
    }
    build_analysis_task_submission(
        host,
        program_roots,
        dependency_identities,
        dependency_tasks,
        context,
    )
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
