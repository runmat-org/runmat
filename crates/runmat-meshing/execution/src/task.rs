//! Projection of meshing-owned workload identity onto generic execution tasks.
//!
//! The resolved meshing request remains the hard algorithm budget authority. Scope, pool,
//! transfer allowances, deadline, and priority are execution context and never enter meshing
//! identity. Retrying requires an explicit replay proof; final publication is always an
//! unknown-effect operation.

use std::collections::BTreeSet;

use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{Digest, ExecutionScopeId, OutputContract, PoolId, TaskId};
use runmat_execution_runner::TaskSubmission;
use runmat_meshing_core::{
    CanonicalMeshingContract, MeshElementOrderV2, MeshingCapabilityRequirementV2,
    MeshingPartitionIdentityV2, MeshingRequestV2, MeshingStageIdentityV2, MeshingStageV2,
    MeshingWorkloadRequestV2, StableDigest, MESHING_IDENTITY_SCHEMA_VERSION,
};

use crate::{
    MeshingArtifactAccess, MeshingExecutionError, MeshingExecutionResult,
    MESHING_STAGE_MANIFEST_MEDIA_TYPE,
};

pub const MESHING_EXECUTION_CALLABLE_OWNER: &str = "runmat.meshing.v2";
const STAGE_MANIFEST_SCHEMA: &str = "runmat.meshing.stage-manifest.v2";
const MAX_REVIEWED_ATTEMPTS: u16 = 16;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshingExecutionContext {
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
    pub program_artifact_id: ArtifactId,
    pub artifact_access: MeshingArtifactAccess,
    pub cpu_millicores: u32,
    pub maximum_egress_bytes: u64,
    pub maximum_relay_bytes: u64,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshingTaskEffectPolicy {
    UnknownEffect,
    ContentAddressedPure {
        maximum_attempts: u16,
        replay_proof_digest: StableDigest,
    },
}

pub fn build_task_submission(
    workload: &MeshingWorkloadRequestV2,
    stage_identity: &MeshingStageIdentityV2,
    resolved_request: &MeshingRequestV2,
    input_roots: &[ValueRef],
    dependencies: BTreeSet<TaskId>,
    context: &MeshingExecutionContext,
    effect: MeshingTaskEffectPolicy,
) -> MeshingExecutionResult<TaskSubmission> {
    workload.validate()?;
    stage_identity.validate()?;
    resolved_request.validate()?;
    context.artifact_access.validate()?;
    validate_identity_binding(workload, stage_identity, resolved_request)?;
    validate_inputs(workload, input_roots, &context.artifact_access)?;
    let capabilities = map_capabilities(workload, stage_identity, resolved_request)?;
    let retry = map_retry(workload.stage, effect)?;
    let partition_identity = MeshingPartitionIdentityV2 {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage_identity_digest: workload.stage_identity_digest,
        partition: workload.partition.clone(),
    };
    partition_identity.validate()?;
    let partition_digest = partition_identity.canonical_digest()?;
    let task_id = TaskId::derive(&[
        b"runmat-meshing-task-v2",
        context.scope_id.bytes(),
        partition_digest.bytes(),
    ]);
    if dependencies.contains(&task_id) {
        return Err(MeshingExecutionError::Invalid(
            "meshing task cannot depend on itself".into(),
        ));
    }
    let resources = ResourceRequest {
        cpu_millicores: context.cpu_millicores,
        memory_bytes: resolved_request.resources.maximum_memory_bytes,
        scratch_bytes: resolved_request.resources.maximum_scratch_bytes,
        max_wall_millis: resolved_request.resources.maximum_wall_time_ms,
        max_artifact_bytes: resolved_request.resources.maximum_artifact_bytes,
        max_egress_bytes: context.maximum_egress_bytes,
        max_relay_bytes: context.maximum_relay_bytes,
        accelerators: Vec::new(),
        required_capabilities: capabilities,
    };
    resources.validate()?;
    let qualified_name = stage_callable(workload.stage).to_string();
    Ok(TaskSubmission {
        request: TaskRequest {
            id: task_id,
            scope_id: context.scope_id,
            pool_id: context.pool_id,
            program_artifact_id: context.program_artifact_id,
            callable: Callable {
                owner_identity: MESHING_EXECUTION_CALLABLE_OWNER.into(),
                qualified_name: qualified_name.clone(),
                entrypoint_digest: Digest::sha256(format!(
                    "runmat-meshing-host-entrypoint-v2\0{qualified_name}"
                )),
            },
            inputs: input_roots
                .iter()
                .cloned()
                .map(|reference| ValuePayload::Object(Box::new(reference)))
                .collect(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            resources,
            retry,
            deadline_unix_millis: context.deadline_unix_millis,
        },
        dependencies,
        priority: context.priority,
    })
}

fn validate_identity_binding(
    workload: &MeshingWorkloadRequestV2,
    stage_identity: &MeshingStageIdentityV2,
    resolved_request: &MeshingRequestV2,
) -> MeshingExecutionResult<()> {
    if workload.stage != stage_identity.stage
        || workload.stage_identity_digest != stage_identity.canonical_digest()?
        || stage_identity.resolved_request_digest != resolved_request.canonical_digest()?
        || workload.input_manifest_digests != stage_identity.prerequisite_artifact_digests
    {
        return Err(MeshingExecutionError::Invalid(
            "workload, stage identity, resolved request, and prerequisites do not converge".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_inputs(
    workload: &MeshingWorkloadRequestV2,
    roots: &[ValueRef],
    access: &MeshingArtifactAccess,
) -> MeshingExecutionResult<()> {
    if roots.len() != workload.input_manifest_digests.len() {
        return Err(MeshingExecutionError::Invalid(
            "externalized input root count differs from workload manifests".into(),
        ));
    }
    for (root, digest) in roots.iter().zip(&workload.input_manifest_digests) {
        ValuePayload::Object(Box::new(root.clone())).validate(ValueLimits::default())?;
        if root.logical_digest.bytes() != digest.bytes()
            || root.id != access.value_id(root.logical_digest)
            || root.encoded_length == 0
            || root.kind != ValueRefKind::ResultObject
            || root.media_type != MESHING_STAGE_MANIFEST_MEDIA_TYPE
            || root.value_schema != STAGE_MANIFEST_SCHEMA
            || root.authorization_scope != access.authorization_scope
            || root.encryption_context != access.encryption_context
        {
            return Err(MeshingExecutionError::Invalid(
                "externalized input root is outside workload identity or artifact authority".into(),
            ));
        }
    }
    Ok(())
}

fn map_capabilities(
    workload: &MeshingWorkloadRequestV2,
    stage_identity: &MeshingStageIdentityV2,
    request: &MeshingRequestV2,
) -> MeshingExecutionResult<BTreeSet<Capability>> {
    let mut mapped = BTreeSet::new();
    let mut host_count = 0;
    let mut algorithm_count = 0;
    let mut order_count = 0;
    let mut cohort = None;
    for requirement in &workload.required_capabilities {
        let custom = match requirement {
            MeshingCapabilityRequirementV2::HostWorkload { abi } => {
                host_count += 1;
                format!("runmat.meshing.host:{abi}")
            }
            MeshingCapabilityRequirementV2::ExactCadKernel { abi } => {
                mapped.insert(Capability::ProcessIsolation);
                format!("runmat.meshing.exact-cad:{abi}")
            }
            MeshingCapabilityRequirementV2::MeshingAlgorithm { version } => {
                algorithm_count += 1;
                if version != algorithm_for_stage(workload.stage, request) {
                    return Err(MeshingExecutionError::Invalid(
                        "meshing algorithm capability differs from resolved request".into(),
                    ));
                }
                format!("runmat.meshing.algorithm:{version}")
            }
            MeshingCapabilityRequirementV2::ElementOrder { order } => {
                order_count += 1;
                if *order != request.element_order {
                    return Err(MeshingExecutionError::Invalid(
                        "element-order capability differs from resolved request".into(),
                    ));
                }
                format!("runmat.meshing.element-order:{}", order_name(*order))
            }
            MeshingCapabilityRequirementV2::DeterministicPlatformCohort { cohort: required } => {
                cohort = Some(required.as_str());
                format!("runmat.meshing.cohort:{required}")
            }
        };
        mapped.insert(Capability::Custom(custom));
    }
    if host_count != 1
        || algorithm_count != 1
        || order_count != 1
        || cohort != stage_identity.capability_cohort.as_deref()
    {
        return Err(MeshingExecutionError::Invalid(
            "workload must declare one host, algorithm, and element-order capability and its exact identity cohort".into(),
        ));
    }
    Ok(mapped)
}

fn map_retry(
    stage: MeshingStageV2,
    effect: MeshingTaskEffectPolicy,
) -> MeshingExecutionResult<RetryPolicy> {
    if stage == MeshingStageV2::Publication {
        return match effect {
            MeshingTaskEffectPolicy::UnknownEffect => Ok(RetryPolicy::Never),
            MeshingTaskEffectPolicy::ContentAddressedPure { .. } => {
                Err(MeshingExecutionError::Invalid(
                    "final meshing publication must declare unknown effects and never retry".into(),
                ))
            }
        };
    }
    match effect {
        MeshingTaskEffectPolicy::UnknownEffect => Ok(RetryPolicy::Never),
        MeshingTaskEffectPolicy::ContentAddressedPure {
            maximum_attempts,
            replay_proof_digest,
        } => {
            replay_proof_digest.validate_nonzero("meshing retry replay proof")?;
            if !(2..=MAX_REVIEWED_ATTEMPTS).contains(&maximum_attempts) {
                return Err(MeshingExecutionError::Invalid(
                    "reviewed meshing retry count must be within 2..=16".into(),
                ));
            }
            Ok(RetryPolicy::ExplicitlyIdempotent {
                max_attempts: maximum_attempts,
            })
        }
    }
}

fn algorithm_for_stage(stage: MeshingStageV2, request: &MeshingRequestV2) -> &str {
    // Stages share only the algorithm families frozen in `AlgorithmVersionSet`; this mapping is
    // explicit so adding a new stage or version family cannot silently inherit another domain.
    match stage {
        MeshingStageV2::GeometryAdmission | MeshingStageV2::Healing | MeshingStageV2::Sizing => {
            &request.algorithms.geometry
        }
        MeshingStageV2::CurveMesh => &request.algorithms.curve,
        MeshingStageV2::SurfaceMesh => &request.algorithms.surface,
        MeshingStageV2::ProtectedBoundaryComplex => &request.algorithms.plc,
        MeshingStageV2::Tetrahedralization
        | MeshingStageV2::ConstraintRecovery
        | MeshingStageV2::Refinement => &request.algorithms.tetrahedron,
        MeshingStageV2::Optimization | MeshingStageV2::OrderElevation => {
            &request.algorithms.optimization
        }
        MeshingStageV2::Validation
        | MeshingStageV2::Serialization
        | MeshingStageV2::Publication => &request.algorithms.validation,
    }
}

const fn order_name(order: MeshElementOrderV2) -> &'static str {
    match order {
        MeshElementOrderV2::Tet4 => "tet4",
        MeshElementOrderV2::Tet10 => "tet10",
    }
}

const fn stage_callable(stage: MeshingStageV2) -> &'static str {
    match stage {
        MeshingStageV2::GeometryAdmission => "geometry-admission",
        MeshingStageV2::Healing => "healing",
        MeshingStageV2::Sizing => "sizing",
        MeshingStageV2::CurveMesh => "curve-mesh",
        MeshingStageV2::SurfaceMesh => "surface-mesh",
        MeshingStageV2::ProtectedBoundaryComplex => "protected-boundary-complex",
        MeshingStageV2::Tetrahedralization => "tetrahedralization",
        MeshingStageV2::ConstraintRecovery => "constraint-recovery",
        MeshingStageV2::Refinement => "refinement",
        MeshingStageV2::Optimization => "optimization",
        MeshingStageV2::OrderElevation => "order-elevation",
        MeshingStageV2::Validation => "validation",
        MeshingStageV2::Serialization => "serialization",
        MeshingStageV2::Publication => "publication",
    }
}
