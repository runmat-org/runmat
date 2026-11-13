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
use runmat_execution::{ExecutionScopeId, OutputContract, PoolId, TaskId};
use runmat_execution_runner::TaskSubmission;
use runmat_meshing_core::{
    CanonicalMeshingContract, ElementOrder, MeshingCapabilityRequirement, MeshingInputKind,
    MeshingPartitionIdentity, MeshingRequest, MeshingStageIdentity, MeshingStageKind,
    MeshingWorkloadRequest, StableDigest, MESHING_IDENTITY_SCHEMA_VERSION,
};

use crate::{
    ArtifactAccess, MeshingExecutionError, MeshingExecutionResult, MeshingHostWorkload,
    MESHING_HOST_INVOCATION_MEDIA_TYPE, MESHING_HOST_INVOCATION_VALUE_SCHEMA,
    MESHING_STAGE_MANIFEST_MEDIA_TYPE,
};

pub const MESHING_EXECUTION_CALLABLE_OWNER: &str = "runmat.meshing.v2";
pub const EXACT_CAD_ADMISSION_CAPABILITY: &str = "runmat.meshing.exact-cad-admission";
const STAGE_MANIFEST_SCHEMA: &str = "runmat.meshing.stage-manifest.v2";
const EXACT_GEOMETRY_SCHEMA: &str = "runmat.geometry.exact-manifest.v2";
const FACETED_GEOMETRY_SCHEMA: &str = "runmat.geometry.faceted-solid.v2";
const MAX_REVIEWED_ATTEMPTS: u16 = 16;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshingExecutionContext {
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
    /// Trust tiers this placement admits. The executable, callable, inputs and host
    /// requirement all come from the program request itself.
    pub permitted_trust_tiers: BTreeSet<runmat_execution::security::ExecutionTrustTier>,
    pub artifact_access: ArtifactAccess,
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
    host: &MeshingHostWorkload,
    program: &runmat_execution_artifact::ProgramExecutionRequest,
    dependencies: BTreeSet<TaskId>,
    context: &MeshingExecutionContext,
    effect: MeshingTaskEffectPolicy,
) -> MeshingExecutionResult<TaskSubmission> {
    host.validate()?;
    let workload = &host.workload;
    let stage_identity = &host.stage_identity;
    let resolved_request = &host.resolved_request;
    workload.validate()?;
    stage_identity.validate()?;
    resolved_request.validate()?;
    context.artifact_access.validate()?;
    validate_identity_binding(workload, stage_identity, resolved_request)?;
    let program_inputs = program.arguments.as_slice();
    let (invocation, semantic_inputs) = program_inputs.split_first().ok_or_else(|| {
        MeshingExecutionError::Invalid("meshing task is missing its invocation descriptor".into())
    })?;
    let ValuePayload::Object(invocation_root) = invocation else {
        return Err(MeshingExecutionError::Invalid(
            "meshing task invocation must remain externalized".into(),
        ));
    };
    context.artifact_access.validate_reference(
        invocation_root,
        ValueRefKind::DriverObject,
        MESHING_HOST_INVOCATION_MEDIA_TYPE,
        MESHING_HOST_INVOCATION_VALUE_SCHEMA,
    )?;
    if invocation_root.logical_digest != runmat_execution::Digest::sha256(host.canonical_bytes()?) {
        return Err(MeshingExecutionError::Identity(
            "meshing invocation descriptor differs from its task workload",
        ));
    }
    let input_roots = semantic_inputs
        .iter()
        .map(|input| match input {
            ValuePayload::Object(root) => Ok((**root).clone()),
            _ => Err(MeshingExecutionError::Invalid(
                "meshing semantic inputs must remain externalized".into(),
            )),
        })
        .collect::<MeshingExecutionResult<Vec<_>>>()?;
    validate_inputs(workload, &input_roots, &context.artifact_access)?;
    let capabilities = map_capabilities(workload, stage_identity, resolved_request)?;
    let retry = map_retry(workload.stage, effect)?;
    let task_id = meshing_task_id(context.scope_id, host)?;
    if dependencies.contains(&task_id) {
        return Err(MeshingExecutionError::Invalid(
            "meshing task cannot depend on itself".into(),
        ));
    }
    let resources = ResourceRequest {
        cpu_millicores: context.cpu_millicores,
        memory_bytes: host.ceilings.maximum_memory_bytes,
        scratch_bytes: host.ceilings.maximum_scratch_bytes,
        max_wall_millis: host.ceilings.maximum_wall_time_ms,
        max_artifact_bytes: host.ceilings.maximum_artifact_bytes,
        max_egress_bytes: context.maximum_egress_bytes,
        max_relay_bytes: context.maximum_relay_bytes,
        accelerators: Vec::new(),
        required_capabilities: capabilities,
    };
    resources.validate()?;
    let host_requirement = program
        .artifact
        .execution_host_requirement(&program.recipe, context.permitted_trust_tiers.clone())
        .map_err(|error| MeshingExecutionError::Invalid(error.to_string()))?;
    Ok(TaskSubmission {
        request: TaskRequest {
            console: program.console,
            environment: program.environment.clone(),
            id: task_id,
            scope_id: context.scope_id,
            pool_id: context.pool_id,
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                ArtifactId::derive(&[program.artifact.id.0.bytes()]),
            ),
            callable: Callable::for_program(MESHING_EXECUTION_CALLABLE_OWNER, &program.callable),
            host: host_requirement,
            invocation_context: program.context.clone(),
            inputs: program_inputs.to_vec(),
            outputs: OutputContract {
                requested_outputs: program.requested_outputs,
                owned_work: None,
            },
            resources,
            retry,
            deadline_unix_millis: context.deadline_unix_millis,
        },
        dependencies,
        priority: context.priority,
    })
}

/// Canonical stage identity before program lowering or scheduler admission.
pub fn meshing_task_id(
    scope_id: runmat_execution::ExecutionScopeId,
    host: &MeshingHostWorkload,
) -> MeshingExecutionResult<TaskId> {
    let partition_identity = MeshingPartitionIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage_identity_digest: host.workload.stage_identity_digest,
        partition: host.workload.partition.clone(),
    };
    partition_identity.validate()?;
    let partition_digest = partition_identity.canonical_digest()?;
    Ok(TaskId::derive(&[
        b"runmat-meshing-task-v2",
        scope_id.bytes(),
        partition_digest.bytes(),
    ]))
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
    access: &ArtifactAccess,
) -> MeshingExecutionResult<()> {
    if roots.len() != workload.inputs.len() {
        return Err(MeshingExecutionError::Invalid(
            "externalized input root count differs from workload inputs".into(),
        ));
    }
    for (root, input) in roots.iter().zip(&workload.inputs) {
        validate_input(root, input, access)?;
    }
    Ok(())
}

pub(crate) fn validate_input(
    root: &ValueRef,
    input: &runmat_meshing_core::MeshingInputRef,
    access: &ArtifactAccess,
) -> MeshingExecutionResult<()> {
    root.validate(ValueLimits::default())?;
    let expected_shape = match input.kind {
        MeshingInputKind::StageArtifact => {
            root.kind == ValueRefKind::ResultObject
                && root.media_type == MESHING_STAGE_MANIFEST_MEDIA_TYPE
                && root.value_schema == STAGE_MANIFEST_SCHEMA
        }
        MeshingInputKind::ExactGeometry => {
            matches!(
                root.kind,
                ValueRefKind::DriverObject | ValueRefKind::ResultObject
            ) && root.media_type == runmat_geometry_core::EXACT_BREP_MEDIA_TYPE
                && root.value_schema == EXACT_GEOMETRY_SCHEMA
        }
        MeshingInputKind::FacetedGeometry => {
            root.kind == ValueRefKind::DriverObject
                && root.media_type == runmat_geometry_core::FACETED_SOLID_MEDIA_TYPE
                && root.value_schema == FACETED_GEOMETRY_SCHEMA
        }
        MeshingInputKind::Evidence => {
            root.kind == ValueRefKind::DriverObject
                && root.media_type
                    == runmat_meshing_core::MeshingChunkMediaType::MeshingEvidence.media_type()
                && root.value_schema == crate::evidence_objects::EVIDENCE_VALUE_SCHEMA
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

/// Returns the complete generic worker capability inventory for an exact-meshing DAG.
/// This is the single product-side authority for configuring a pool before any stage is submitted.
pub fn exact_meshing_worker_capabilities(
    document: &runmat_geometry_core::GeometryDocument,
    request: &MeshingRequest,
    capability_cohort: Option<&str>,
) -> MeshingExecutionResult<BTreeSet<Capability>> {
    document.validate()?;
    request.validate()?;
    let runmat_geometry_core::GeometryModel::ExactBRep { model } = &document.model else {
        return Err(MeshingExecutionError::Invalid(
            "exact meshing capabilities require an exact B-rep document".into(),
        ));
    };
    exact_meshing_worker_capability_inventory(
        std::slice::from_ref(&model.kernel_abi),
        &request.algorithms,
        std::slice::from_ref(&request.element_order),
        capability_cohort,
    )
}

/// Returns the complete capability inventory implemented by a native meshing worker.
///
/// A worker advertises all exact-kernel ABIs, algorithm versions, element orders, and deterministic
/// cohorts it can actually execute. Individual workloads continue to request one exact member of
/// each category through [`exact_meshing_worker_capabilities`].
pub fn exact_meshing_worker_capability_inventory<'a>(
    exact_kernel_abis: &[String],
    algorithms: &runmat_meshing_core::AlgorithmVersionSet,
    element_orders: &[runmat_meshing_core::ElementOrder],
    capability_cohorts: impl IntoIterator<Item = &'a str>,
) -> MeshingExecutionResult<BTreeSet<Capability>> {
    algorithms.validate()?;
    if exact_kernel_abis.is_empty() || element_orders.is_empty() {
        return Err(MeshingExecutionError::Invalid(
            "meshing worker capability inventory is incomplete".into(),
        ));
    }
    let mut capabilities = BTreeSet::from([Capability::ProcessIsolation]);
    capabilities.insert(Capability::Custom(EXACT_CAD_ADMISSION_CAPABILITY.into()));
    capabilities.insert(Capability::Custom(format!(
        "runmat.meshing.host:{}",
        crate::MESHING_HOST_ABI
    )));
    for abi in exact_kernel_abis {
        if abi.is_empty() || abi.len() > 256 || abi.chars().any(char::is_control) {
            return Err(MeshingExecutionError::Invalid(
                "meshing exact-kernel capability is invalid".into(),
            ));
        }
        capabilities.insert(Capability::Custom(format!(
            "runmat.meshing.exact-cad:{abi}"
        )));
    }
    for version in [
        &algorithms.geometry,
        &algorithms.curve,
        &algorithms.surface,
        &algorithms.plc,
        &algorithms.tetrahedron,
        &algorithms.optimization,
        &algorithms.validation,
    ] {
        capabilities.insert(Capability::Custom(format!(
            "runmat.meshing.algorithm:{version}"
        )));
    }
    for order in element_orders {
        capabilities.insert(Capability::Custom(format!(
            "runmat.meshing.element-order:{}",
            order_name(*order)
        )));
    }
    for cohort in capability_cohorts {
        if cohort.is_empty() || cohort.len() > 256 || cohort.chars().any(char::is_control) {
            return Err(MeshingExecutionError::Invalid(
                "meshing capability cohort is invalid".into(),
            ));
        }
        capabilities.insert(Capability::Custom(format!(
            "runmat.meshing.cohort:{cohort}"
        )));
    }
    Ok(capabilities)
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
