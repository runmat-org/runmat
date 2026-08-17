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
    CanonicalMeshingContract, ElementOrder, MeshingCapabilityRequirement, MeshingInputKind,
    MeshingPartitionIdentity, MeshingRequest, MeshingStageIdentity, MeshingStageKind,
    MeshingWorkloadRequest, StableDigest, MESHING_IDENTITY_SCHEMA_VERSION,
};

use crate::{
    MeshingArtifactAccess, MeshingExecutionError, MeshingExecutionResult,
    MESHING_STAGE_MANIFEST_MEDIA_TYPE,
};

pub const MESHING_EXECUTION_CALLABLE_OWNER: &str = "runmat.meshing.v2";
const STAGE_MANIFEST_SCHEMA: &str = "runmat.meshing.stage-manifest.v2";
const EXACT_GEOMETRY_SCHEMA: &str = "runmat.geometry.exact-manifest.v2";
const FACETED_GEOMETRY_SCHEMA: &str = "runmat.geometry.faceted-solid.v2";
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
    workload: &MeshingWorkloadRequest,
    stage_identity: &MeshingStageIdentity,
    resolved_request: &MeshingRequest,
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
    let partition_identity = MeshingPartitionIdentity {
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
    workload: &MeshingWorkloadRequest,
    stage_identity: &MeshingStageIdentity,
    resolved_request: &MeshingRequest,
) -> MeshingExecutionResult<()> {
    if workload.stage != stage_identity.stage
        || workload.stage_identity_digest != stage_identity.canonical_digest()?
        || stage_identity.resolved_request_digest != resolved_request.canonical_digest()?
        || workload.inputs != stage_identity.prerequisites
    {
        return Err(MeshingExecutionError::Invalid(
            "workload, stage identity, resolved request, and prerequisites do not converge".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_inputs(
    workload: &MeshingWorkloadRequest,
    roots: &[ValueRef],
    access: &MeshingArtifactAccess,
) -> MeshingExecutionResult<()> {
    if roots.len() != workload.inputs.len() {
        return Err(MeshingExecutionError::Invalid(
            "externalized input root count differs from workload inputs".into(),
        ));
    }
    for (root, input) in roots.iter().zip(&workload.inputs) {
        ValuePayload::Object(Box::new(root.clone())).validate(ValueLimits::default())?;
        let expected_shape = match input.kind {
            MeshingInputKind::StageArtifact => {
                root.kind == ValueRefKind::ResultObject
                    && root.media_type == MESHING_STAGE_MANIFEST_MEDIA_TYPE
                    && root.value_schema == STAGE_MANIFEST_SCHEMA
            }
            MeshingInputKind::ExactGeometry => {
                root.kind == ValueRefKind::DriverObject
                    && root.media_type == runmat_geometry_core::EXACT_BREP_MEDIA_TYPE
                    && root.value_schema == EXACT_GEOMETRY_SCHEMA
            }
            MeshingInputKind::FacetedGeometry => {
                root.kind == ValueRefKind::DriverObject
                    && root.media_type == runmat_geometry_core::FACETED_SOLID_MEDIA_TYPE
                    && root.value_schema == FACETED_GEOMETRY_SCHEMA
            }
        };
        if root.logical_digest.bytes() != input.digest.bytes()
            || root.id != access.value_id(root.logical_digest)
            || root.encoded_length == 0
            || !expected_shape
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
    workload: &MeshingWorkloadRequest,
    stage_identity: &MeshingStageIdentity,
    request: &MeshingRequest,
) -> MeshingExecutionResult<BTreeSet<Capability>> {
    let mut mapped = BTreeSet::new();
    let mut host_count = 0;
    let mut algorithm_count = 0;
    let mut order_count = 0;
    let mut cohort = None;
    for requirement in &workload.required_capabilities {
        let custom = match requirement {
            MeshingCapabilityRequirement::HostWorkload { abi } => {
                host_count += 1;
                format!("runmat.meshing.host:{abi}")
            }
            MeshingCapabilityRequirement::ExactCadKernel { abi } => {
                mapped.insert(Capability::ProcessIsolation);
                format!("runmat.meshing.exact-cad:{abi}")
            }
            MeshingCapabilityRequirement::MeshingAlgorithm { version } => {
                algorithm_count += 1;
                if version != algorithm_for_stage(workload.stage, request) {
                    return Err(MeshingExecutionError::Invalid(
                        "meshing algorithm capability differs from resolved request".into(),
                    ));
                }
                format!("runmat.meshing.algorithm:{version}")
            }
            MeshingCapabilityRequirement::ElementOrder { order } => {
                order_count += 1;
                if *order != request.element_order {
                    return Err(MeshingExecutionError::Invalid(
                        "element-order capability differs from resolved request".into(),
                    ));
                }
                format!("runmat.meshing.element-order:{}", order_name(*order))
            }
            MeshingCapabilityRequirement::DeterministicPlatformCohort { cohort: required } => {
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
    stage: MeshingStageKind,
    effect: MeshingTaskEffectPolicy,
) -> MeshingExecutionResult<RetryPolicy> {
    if stage == MeshingStageKind::Publication {
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

fn algorithm_for_stage(stage: MeshingStageKind, request: &MeshingRequest) -> &str {
    // Stages share only the algorithm families frozen in `AlgorithmVersionSet`; this mapping is
    // explicit so adding a new stage or version family cannot silently inherit another domain.
    match stage {
        MeshingStageKind::GeometryAdmission
        | MeshingStageKind::Healing
        | MeshingStageKind::Sizing => &request.algorithms.geometry,
        MeshingStageKind::CurveMesh => &request.algorithms.curve,
        MeshingStageKind::SurfaceMesh => &request.algorithms.surface,
        MeshingStageKind::ProtectedBoundaryComplex => &request.algorithms.plc,
        MeshingStageKind::Tetrahedralization
        | MeshingStageKind::ConstraintRecovery
        | MeshingStageKind::Refinement => &request.algorithms.tetrahedron,
        MeshingStageKind::Optimization | MeshingStageKind::OrderElevation => {
            &request.algorithms.optimization
        }
        MeshingStageKind::Validation
        | MeshingStageKind::Serialization
        | MeshingStageKind::Publication => &request.algorithms.validation,
    }
}

const fn order_name(order: ElementOrder) -> &'static str {
    match order {
        ElementOrder::Tet4 => "tet4",
        ElementOrder::Tet10 => "tet10",
    }
}

const fn stage_callable(stage: MeshingStageKind) -> &'static str {
    match stage {
        MeshingStageKind::GeometryAdmission => "geometry-admission",
        MeshingStageKind::Healing => "healing",
        MeshingStageKind::Sizing => "sizing",
        MeshingStageKind::CurveMesh => "curve-mesh",
        MeshingStageKind::SurfaceMesh => "surface-mesh",
        MeshingStageKind::ProtectedBoundaryComplex => "protected-boundary-complex",
        MeshingStageKind::Tetrahedralization => "tetrahedralization",
        MeshingStageKind::ConstraintRecovery => "constraint-recovery",
        MeshingStageKind::Refinement => "refinement",
        MeshingStageKind::Optimization => "optimization",
        MeshingStageKind::OrderElevation => "order-elevation",
        MeshingStageKind::Validation => "validation",
        MeshingStageKind::Serialization => "serialization",
        MeshingStageKind::Publication => "publication",
    }
}
