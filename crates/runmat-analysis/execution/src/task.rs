use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{AnalysisNumericalObjectKind, AnalysisTaskIdentity};
use runmat_analysis_fea::planning::resource::{allocation_matches_plan, FeaResourcePlan};
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

/// Borrowed immutable inputs provide source authority unavailable to metadata-only submission.
pub struct PreparedAnalysisTaskClosure<'a> {
    pub inputs: &'a [crate::PreparedAnalysisObjects],
    pub output_policy: crate::AnalysisOutputPolicy,
}

pub fn build_resource_planned_task_submission_with_closure(
    host: &AnalysisHostWorkload,
    resource_plan: &FeaResourcePlan,
    closure: PreparedAnalysisTaskClosure<'_>,
    program_roots: Vec<ValueRef>,
    dependency_identities: Vec<AnalysisTaskIdentity>,
    dependency_tasks: BTreeSet<TaskId>,
    context: AnalysisExecutionContext,
) -> AnalysisBridgeResult<TaskSubmission> {
    resource_plan.validate_identity()?;
    host.validate()?;
    let Some((_, input_roots)) = program_roots.split_first() else {
        return Err(crate::AnalysisBridgeError::Identity(
            "analysis closure has no invocation",
        ));
    };
    host.validate_inputs(input_roots)?;
    crate::transport::validate_prepared_analysis_inputs(host, closure.inputs)?;
    let plans = closure
        .inputs
        .iter()
        .filter(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .collect::<Vec<_>>();
    let current = plans
        .iter()
        .filter(|input| input.manifest.logical_identity == resource_plan.identity.digest())
        .collect::<Vec<_>>();
    if current.len() != 1 {
        return Err(crate::AnalysisBridgeError::Identity(
            "analysis current resource authority absent or repeated",
        ));
    }
    if crate::decode_admitted_resource_plan(current[0], closure.output_policy.inventory_limits())?
        != *resource_plan
    {
        return Err(crate::AnalysisBridgeError::Identity(
            "analysis current resource payload differs",
        ));
    }
    if plans.len() == 1 {
        return build_resource_planned_task_submission(
            host,
            resource_plan,
            program_roots,
            dependency_identities,
            dependency_tasks,
            context,
        );
    }
    require_original_source_authority(host, resource_plan, closure)?;
    if !allocation_matches_plan(&context.resources, &resource_plan.request) {
        return Err(crate::AnalysisBridgeError::Identity(
            "analysis task resources differ from the canonical post-projection plan",
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

fn require_original_source_authority(
    host: &AnalysisHostWorkload,
    plan: &FeaResourcePlan,
    closure: PreparedAnalysisTaskClosure<'_>,
) -> AnalysisBridgeResult<()> {
    use runmat_analysis_fea::{
        execution::{
            AnalysisWorkKind, ExecutionPlanIdentity, NumericalPlanIdentity,
            SemanticStatePublicationRequest, SemanticStatePublicationSource,
        },
        planning::resource::FeaResourceDemand,
    };
    let FeaResourceDemand::SemanticStatePublication(demand) = &plan.input.demand else {
        return Err(crate::AnalysisBridgeError::Identity(
            "analysis phase does not admit original resource authority",
        ));
    };
    if host.workload.kind != AnalysisWorkKind::PublishState
        || host.workload.outputs != [AnalysisNumericalObjectKind::SemanticState]
        || demand.source != SemanticStatePublicationSource::FluidPrimaryFields
        || closure.inputs.len() != 6
        || host.workload.inputs.iter().any(|input| {
            input.reference_kind
                != if input.kind == AnalysisNumericalObjectKind::SolverResult {
                    ValueRefKind::ResultObject
                } else {
                    ValueRefKind::DriverObject
                }
        })
    {
        return Err(crate::AnalysisBridgeError::Identity(
            "analysis source publication authority differs",
        ));
    }
    // The phase validator authenticates raw immutable wires and exact decode facts before joins.
    crate::require_fluid_state_source_decode_budget(
        closure.inputs,
        plan,
        closure.output_policy.inventory_limits(),
    )?;
    let original = crate::decode_admitted_resource_plan(
        &closure.inputs[2],
        closure.output_policy.inventory_limits(),
    )?;
    let FeaResourceDemand::FluidSteadySolve(original) = &original.input.demand else {
        return Err(crate::AnalysisBridgeError::Identity(
            "analysis publication original phase differs",
        ));
    };
    let numerical = NumericalPlanIdentity::from_fluid_steady_workload(
        original.projection.workload,
        original.projection.topology_digest,
    )?;
    if host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, closure.inputs[2].manifest.logical_identity)?
        || host.workload.step != original.projection.producer
    {
        return Err(crate::AnalysisBridgeError::Identity(
            "analysis publication original host lineage differs",
        ));
    }
    let request =
        SemanticStatePublicationRequest::from_canonical_bytes(&closure.inputs[4].payload())?;
    if crate::build_prepared_fluid_state_publication_resource_plan(
        &request,
        closure.inputs,
        closure.output_policy,
    )? != *plan
    {
        return Err(crate::AnalysisBridgeError::Identity(
            "analysis publication resource authority differs from original source",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct AnalysisExecutionContext {
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
    pub program_artifact_id: ArtifactId,
    pub host: runmat_execution::host::ExecutionHostRequirement,
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

pub fn build_resource_planned_task_submission(
    host: &AnalysisHostWorkload,
    resource_plan: &FeaResourcePlan,
    program_roots: Vec<ValueRef>,
    dependency_identities: Vec<AnalysisTaskIdentity>,
    dependency_tasks: BTreeSet<TaskId>,
    context: AnalysisExecutionContext,
) -> AnalysisBridgeResult<TaskSubmission> {
    resource_plan.validate_identity().map_err(|error| {
        crate::AnalysisBridgeError::Invalid(format!("invalid analysis resource plan: {error}"))
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
            "analysis workload does not bind the canonical resource plan",
        ));
    }
    let planned = &resource_plan.request;
    if !allocation_matches_plan(&context.resources, planned) {
        return Err(crate::AnalysisBridgeError::Identity(
            "analysis task resources differ from the canonical post-projection plan",
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
        || dependency_tasks.len() > dependency_identities.len()
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
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        id: task_id,
        schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
        scope_id: context.scope_id,
        pool_id: context.pool_id,
        executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
            context.program_artifact_id,
        ),
        callable: Callable::for_program(
            ANALYSIS_EXECUTION_CALLABLE_OWNER,
            &crate::host::analysis_host_callable(),
        ),
        host: context.host,
        invocation_context: runmat_execution::ProgramInvocationContext::Direct,
        inputs: program_roots
            .into_iter()
            .map(|root| runmat_execution::value::ValuePayload::Object(Box::new(root)))
            .collect(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
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
