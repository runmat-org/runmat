use std::collections::BTreeSet;

use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::{Digest, ExecutionScopeId, OutputContract, PoolId, TaskId};
use runmat_execution_runner::TaskSubmission;

use crate::{
    AnalysisBridgeError, AnalysisBridgeResult, GeometryMaterializationRequest,
    PreparedGeometryMaterializationProgram,
};

pub const GEOMETRY_EXECUTION_CALLABLE_OWNER: &str = "runmat.geometry.host";
pub const DEFAULT_GEOMETRY_MAXIMUM_ATTEMPTS: u16 = 3;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeometryExecutionContext {
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
    pub program_artifact_id: ArtifactId,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
    pub maximum_attempts: u16,
}

pub fn geometry_provider_capability(profile: &crate::GeometryProviderProfile) -> Capability {
    Capability::Custom(format!(
        "runmat.geometry.provider:{}:{}:{}",
        profile.provider_id, profile.provider_version, profile.kernel_abi
    ))
}

pub fn build_geometry_task_submission(
    prepared: &PreparedGeometryMaterializationProgram,
    dependencies: BTreeSet<TaskId>,
    context: GeometryExecutionContext,
) -> AnalysisBridgeResult<TaskSubmission> {
    prepared.request.validate()?;
    prepared.program.validate_for_portable_host()?;
    let expected_artifact_id = ArtifactId::derive(&[prepared.program.artifact.id.0.bytes()]);
    if expected_artifact_id != context.program_artifact_id
        || context.maximum_attempts == 0
        || context.maximum_attempts > 16
    {
        return Err(AnalysisBridgeError::Invalid(
            "geometry task context differs from its program or retry bound".into(),
        ));
    }
    let identity = prepared.request.identity()?;
    let task_id = TaskId::derive(&[
        b"runmat-geometry-materialization-task",
        context.scope_id.bytes(),
        identity.bytes(),
    ]);
    if dependencies.contains(&task_id) {
        return Err(AnalysisBridgeError::Invalid(
            "geometry task cannot depend on itself".into(),
        ));
    }
    let resources = task_resources(&prepared.request)?;
    Ok(TaskSubmission {
        request: TaskRequest {
            id: task_id,
            scope_id: context.scope_id,
            pool_id: context.pool_id,
            program_artifact_id: context.program_artifact_id,
            callable: Callable {
                owner_identity: GEOMETRY_EXECUTION_CALLABLE_OWNER.into(),
                qualified_name: "geometry_workload".into(),
                entrypoint_digest: Digest::sha256(
                    [
                        b"runmat-geometry-materialization-entrypoint\0".as_slice(),
                        identity.bytes(),
                    ]
                    .concat(),
                ),
            },
            inputs: prepared.program.arguments.clone(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            resources,
            retry: RetryPolicy::ExplicitlyIdempotent {
                max_attempts: context.maximum_attempts,
            },
            deadline_unix_millis: context.deadline_unix_millis,
        },
        dependencies,
        priority: context.priority,
    })
}

fn task_resources(
    request: &GeometryMaterializationRequest,
) -> AnalysisBridgeResult<ResourceRequest> {
    request.resources.validate(&request.import_options)?;
    let resources = ResourceRequest {
        cpu_millicores: request.resources.cpu_millicores,
        memory_bytes: request.resources.maximum_memory_bytes,
        scratch_bytes: request.resources.maximum_scratch_bytes,
        max_wall_millis: request.resources.maximum_wall_millis,
        max_artifact_bytes: request.resources.maximum_artifact_bytes,
        max_egress_bytes: request.resources.maximum_egress_bytes,
        max_relay_bytes: request.resources.maximum_relay_bytes,
        accelerators: Vec::new(),
        required_capabilities: BTreeSet::from([geometry_provider_capability(&request.provider)]),
    };
    resources.validate()?;
    Ok(resources)
}
