use std::collections::BTreeSet;

use runmat_execution::identity::{ArtifactId, WorkerId};
use runmat_execution::resource::{Capability, ResourceInventory};
use runmat_execution::state::PoolState;
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution::{Digest, ExecutionScopeId, PoolId};
use runmat_execution_runner::driver::{DriverAction, DriverCommand, DriverConfig};
use runmat_execution_runner::{Driver, PoolSpec, WorkerSpec};
use runmat_meshing_core::{
    AlgorithmVersionSet, CancellationPolicy, CanonicalMeshingContract, ElementOrder,
    GeometryRevisionRef, GeometryTolerancePolicy, MeshingCapabilityRequirement, MeshingInputKind,
    MeshingInputRef, MeshingPartitionDescriptor, MeshingPartitionKind, MeshingQualityTargets,
    MeshingRequest, MeshingResourceBudget, MeshingStageIdentity, MeshingStageKind,
    MeshingWorkloadRequest, MetricCombinationRule, MetricFieldRequest, MetricTensor3, StableDigest,
    SurfaceQualityTargets, VolumeQualityTargets, MESHING_IDENTITY_SCHEMA_VERSION,
    MESHING_REQUEST_SCHEMA_VERSION, MESHING_WORKLOAD_SCHEMA_VERSION,
};

use crate::{
    build_task_submission, MeshingArtifactAccess, MeshingExecutionContext, MeshingTaskEffectPolicy,
    MESHING_STAGE_MANIFEST_MEDIA_TYPE,
};

#[test]
fn task_projection_binds_identity_inputs_resources_and_capabilities() {
    let fixture = Fixture::new(MeshingStageKind::SurfaceMesh);
    let submission = fixture.submit(MeshingTaskEffectPolicy::ContentAddressedPure {
        maximum_attempts: 3,
        replay_proof_digest: stable(31),
    });

    assert_eq!(submission.request.inputs.len(), 1);
    assert_eq!(
        submission.request.resources.memory_bytes,
        fixture.request.resources.maximum_memory_bytes
    );
    assert_eq!(
        submission.request.resources.scratch_bytes,
        fixture.request.resources.maximum_scratch_bytes
    );
    assert_eq!(
        submission.request.resources.max_wall_millis,
        fixture.request.resources.maximum_wall_time_ms
    );
    assert_eq!(
        submission.request.resources.max_artifact_bytes,
        fixture.request.resources.maximum_artifact_bytes
    );
    assert_eq!(
        submission.request.retry,
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 3 }
    );
    assert!(submission
        .request
        .resources
        .required_capabilities
        .contains(&Capability::ProcessIsolation));
    for capability in [
        "runmat.meshing.host:host-v2",
        "runmat.meshing.exact-cad:cad-abi-v1",
        "runmat.meshing.algorithm:surface/v2",
        "runmat.meshing.element-order:tet10",
        "runmat.meshing.cohort:native-cohort-v1",
    ] {
        assert!(submission
            .request
            .resources
            .required_capabilities
            .contains(&Capability::Custom(capability.into())));
    }
}

#[test]
fn logical_partition_task_is_placement_independent_within_a_scope() {
    let fixture = Fixture::new(MeshingStageKind::SurfaceMesh);
    let first = fixture.submit(MeshingTaskEffectPolicy::UnknownEffect);
    let mut other = fixture.context.clone();
    other.pool_id = PoolId::derive(&[b"other-pool"]);
    other.cpu_millicores = 4_000;
    let second = build_task_submission(
        &fixture.workload,
        &fixture.identity,
        &fixture.request,
        std::slice::from_ref(&fixture.input),
        BTreeSet::new(),
        &other,
        MeshingTaskEffectPolicy::UnknownEffect,
    )
    .unwrap();

    assert_eq!(first.request.id, second.request.id);
    assert_ne!(first.request.pool_id, second.request.pool_id);
    assert_ne!(
        first.request.resources.cpu_millicores,
        second.request.resources.cpu_millicores
    );
}

#[test]
fn mismatched_request_input_or_authority_fails_closed() {
    let fixture = Fixture::new(MeshingStageKind::SurfaceMesh);
    let mut wrong_request = fixture.request.clone();
    wrong_request.resources.maximum_memory_bytes += 1;
    assert!(build_task_submission(
        &fixture.workload,
        &fixture.identity,
        &wrong_request,
        std::slice::from_ref(&fixture.input),
        BTreeSet::new(),
        &fixture.context,
        MeshingTaskEffectPolicy::UnknownEffect,
    )
    .is_err());

    let mut wrong_input = fixture.input.clone();
    wrong_input.authorization_scope = "another-run".into();
    assert!(build_task_submission(
        &fixture.workload,
        &fixture.identity,
        &fixture.request,
        &[wrong_input],
        BTreeSet::new(),
        &fixture.context,
        MeshingTaskEffectPolicy::UnknownEffect,
    )
    .is_err());
}

#[test]
fn exact_geometry_inputs_use_the_authoritative_geometry_object_shape() {
    let mut fixture = Fixture::new(MeshingStageKind::SurfaceMesh);
    fixture.identity.prerequisites[0].kind = MeshingInputKind::ExactGeometry;
    fixture.workload.inputs = fixture.identity.prerequisites.clone();
    fixture.workload.stage_identity_digest = fixture.identity.canonical_digest().unwrap();
    fixture.input.kind = ValueRefKind::DriverObject;
    fixture.input.media_type = runmat_geometry_core::EXACT_BREP_MEDIA_TYPE.into();
    fixture.input.value_schema = "runmat.geometry.exact-manifest.v2".into();

    fixture.submit(MeshingTaskEffectPolicy::UnknownEffect);

    fixture.input.kind = ValueRefKind::ResultObject;
    assert!(build_task_submission(
        &fixture.workload,
        &fixture.identity,
        &fixture.request,
        std::slice::from_ref(&fixture.input),
        BTreeSet::new(),
        &fixture.context,
        MeshingTaskEffectPolicy::UnknownEffect,
    )
    .is_err());
}

#[test]
fn incomplete_or_inconsistent_capability_declarations_are_rejected() {
    let fixture = Fixture::new(MeshingStageKind::SurfaceMesh);
    let mut missing_host = fixture.workload.clone();
    missing_host.required_capabilities.remove(0);
    assert!(build_task_submission(
        &missing_host,
        &fixture.identity,
        &fixture.request,
        std::slice::from_ref(&fixture.input),
        BTreeSet::new(),
        &fixture.context,
        MeshingTaskEffectPolicy::UnknownEffect,
    )
    .is_err());

    let mut wrong_order = fixture.workload.clone();
    wrong_order.required_capabilities[3] = MeshingCapabilityRequirement::ElementOrder {
        order: ElementOrder::Tet4,
    };
    assert!(build_task_submission(
        &wrong_order,
        &fixture.identity,
        &fixture.request,
        std::slice::from_ref(&fixture.input),
        BTreeSet::new(),
        &fixture.context,
        MeshingTaskEffectPolicy::UnknownEffect,
    )
    .is_err());
}

#[test]
fn final_publication_and_unknown_effects_never_retry() {
    let ordinary = Fixture::new(MeshingStageKind::SurfaceMesh);
    assert_eq!(
        ordinary
            .submit(MeshingTaskEffectPolicy::UnknownEffect)
            .request
            .retry,
        RetryPolicy::Never
    );

    let final_stage = Fixture::new(MeshingStageKind::Publication);
    assert!(build_task_submission(
        &final_stage.workload,
        &final_stage.identity,
        &final_stage.request,
        std::slice::from_ref(&final_stage.input),
        BTreeSet::new(),
        &final_stage.context,
        MeshingTaskEffectPolicy::ContentAddressedPure {
            maximum_attempts: 3,
            replay_proof_digest: stable(31),
        },
    )
    .is_err());
    assert_eq!(
        final_stage
            .submit(MeshingTaskEffectPolicy::UnknownEffect)
            .request
            .retry,
        RetryPolicy::Never
    );
}

#[test]
fn unchanged_scheduler_admits_only_a_capable_worker() {
    let fixture = Fixture::new(MeshingStageKind::SurfaceMesh);
    let submission = fixture.submit(MeshingTaskEffectPolicy::UnknownEffect);
    let scope = submission.request.scope_id;
    let pool = submission.request.pool_id;
    let mut driver = Driver::new(DriverConfig::default(), 3).unwrap();
    driver
        .handle(DriverCommand::RegisterScope {
            scope_id: scope,
            parent: None,
        })
        .unwrap();
    driver
        .handle(DriverCommand::CreatePool(PoolSpec {
            id: pool,
            min_workers: 1,
            max_workers: 2,
            max_in_flight: 2,
            resource_limit: ResourceInventory {
                cpu_millicores: 8_000,
                memory_bytes: 16_000_000,
                scratch_bytes: 16_000_000,
                accelerators: Vec::new(),
                capabilities: submission.request.resources.required_capabilities.clone(),
            },
        }))
        .unwrap();
    driver
        .handle(DriverCommand::SetPoolState {
            pool_id: pool,
            state: PoolState::Ready,
        })
        .unwrap();
    driver
        .handle(DriverCommand::RegisterWorker(WorkerSpec {
            id: WorkerId::derive(&[b"incapable"]),
            pool_id: pool,
            resources: ResourceInventory {
                cpu_millicores: 4_000,
                memory_bytes: 8_000_000,
                scratch_bytes: 8_000_000,
                accelerators: Vec::new(),
                capabilities: BTreeSet::new(),
            },
        }))
        .unwrap();
    assert!(driver
        .handle(DriverCommand::Submit(Box::new(submission.clone())))
        .unwrap()
        .iter()
        .all(|action| !matches!(action, DriverAction::Launch(_))));

    let actions = driver
        .handle(DriverCommand::RegisterWorker(WorkerSpec {
            id: WorkerId::derive(&[b"capable"]),
            pool_id: pool,
            resources: ResourceInventory {
                cpu_millicores: 4_000,
                memory_bytes: 8_000_000,
                scratch_bytes: 8_000_000,
                accelerators: Vec::new(),
                capabilities: submission.request.resources.required_capabilities,
            },
        }))
        .unwrap();
    assert!(actions
        .iter()
        .any(|action| matches!(action, DriverAction::Launch(_))));
}

pub(crate) struct Fixture {
    pub(crate) request: MeshingRequest,
    pub(crate) identity: MeshingStageIdentity,
    pub(crate) workload: MeshingWorkloadRequest,
    pub(crate) input: ValueRef,
    pub(crate) context: MeshingExecutionContext,
}

impl Fixture {
    pub(crate) fn new(stage: MeshingStageKind) -> Self {
        let request = request();
        let input_digest = stable(20);
        let cohort = "native-cohort-v1".to_string();
        let identity = MeshingStageIdentity {
            schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
            stage,
            geometry: GeometryRevisionRef {
                source_digest: stable(1),
                geometry_revision: 2,
                persistent_mapping_version: 1,
            },
            resolved_request_digest: request.canonical_digest().unwrap(),
            tolerance_policy_digest: request.tolerance.canonical_digest().unwrap(),
            metric_policy_digest: request.metric.canonical_digest().unwrap(),
            algorithm_set_digest: request.algorithms.canonical_digest().unwrap(),
            deterministic_seed: request.deterministic_seed,
            prerequisites: vec![stage_input(input_digest)],
            capability_cohort: Some(cohort.clone()),
        };
        let workload = MeshingWorkloadRequest {
            schema_version: MESHING_WORKLOAD_SCHEMA_VERSION,
            stage,
            stage_identity_digest: identity.canonical_digest().unwrap(),
            partition: MeshingPartitionDescriptor {
                kind: MeshingPartitionKind::WholeStage,
                partition_index: 0,
                partition_count: 1,
                entity_range: None,
            },
            inputs: vec![stage_input(input_digest)],
            required_capabilities: vec![
                MeshingCapabilityRequirement::HostWorkload {
                    abi: "host-v2".into(),
                },
                MeshingCapabilityRequirement::ExactCadKernel {
                    abi: "cad-abi-v1".into(),
                },
                MeshingCapabilityRequirement::MeshingAlgorithm {
                    version: algorithm(stage, &request).into(),
                },
                MeshingCapabilityRequirement::ElementOrder {
                    order: request.element_order,
                },
                MeshingCapabilityRequirement::DeterministicPlatformCohort { cohort },
            ],
        };
        let access = MeshingArtifactAccess {
            authorization_scope: "run-scope".into(),
            encryption_context: Digest::sha256(b"encryption-context"),
        };
        let input = ValueRef {
            schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1,
            id: access.value_id(Digest::from_bytes(*input_digest.bytes())),
            logical_digest: Digest::from_bytes(*input_digest.bytes()),
            encoded_length: 512,
            media_type: MESHING_STAGE_MANIFEST_MEDIA_TYPE.into(),
            value_schema: "runmat.meshing.stage-manifest.v2".into(),
            encryption_context: access.encryption_context,
            kind: ValueRefKind::ResultObject,
            authorization_scope: access.authorization_scope.clone(),
            resident_fence: None,
        };
        let context = MeshingExecutionContext {
            scope_id: ExecutionScopeId::derive(&[b"scope"]),
            pool_id: PoolId::derive(&[b"pool"]),
            program_artifact_id: ArtifactId::derive(&[b"meshing-host-v2"]),
            artifact_access: access,
            cpu_millicores: 2_000,
            maximum_egress_bytes: 4_000_000,
            maximum_relay_bytes: 4_000_000,
            deadline_unix_millis: Some(100_000),
            priority: 2,
        };
        Self {
            request,
            identity,
            workload,
            input,
            context,
        }
    }

    fn submit(&self, effect: MeshingTaskEffectPolicy) -> runmat_execution_runner::TaskSubmission {
        build_task_submission(
            &self.workload,
            &self.identity,
            &self.request,
            std::slice::from_ref(&self.input),
            BTreeSet::new(),
            &self.context,
            effect,
        )
        .unwrap()
    }
}

fn request() -> MeshingRequest {
    MeshingRequest {
        schema_version: MESHING_REQUEST_SCHEMA_VERSION,
        element_order: ElementOrder::Tet10,
        deterministic_seed: 7,
        algorithms: AlgorithmVersionSet {
            geometry: "geometry/v2".into(),
            curve: "curve/v2".into(),
            surface: "surface/v2".into(),
            plc: "plc/v2".into(),
            tetrahedron: "tetrahedron/v2".into(),
            optimization: "optimization/v2".into(),
            validation: "validation/v2".into(),
        },
        tolerance: GeometryTolerancePolicy {
            source_tolerance_m: 1.0e-8,
            absolute_floor_m: 1.0e-10,
            model_relative_term: 1.0e-9,
            requested_deviation_m: 1.0e-5,
            maximum_healing_displacement_m: 1.0e-6,
        },
        metric: MetricFieldRequest {
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(0.5).unwrap(),
            maximum_grading_ratio: 1.3,
            contributions: Vec::new(),
        },
        quality: MeshingQualityTargets {
            surface: SurfaceQualityTargets {
                minimum_metric_angle_degrees: 20.0,
                maximum_physical_aspect_ratio: 10.0,
                maximum_chordal_deviation_m: 1.0e-5,
                maximum_normal_deviation_degrees: 5.0,
            },
            volume: VolumeQualityTargets {
                maximum_radius_edge_ratio: 2.0,
                minimum_scaled_jacobian: 0.05,
                maximum_metric_edge_length: 1.5,
            },
        },
        resources: MeshingResourceBudget {
            maximum_nodes: 100,
            maximum_elements: 100,
            maximum_memory_bytes: 4_000_000,
            maximum_scratch_bytes: 4_000_000,
            maximum_wall_time_ms: 10_000,
            maximum_artifact_bytes: 2_000_000,
            maximum_search_work: 10_000,
            maximum_recursion_depth: 32,
            maximum_iterations: 10_000,
        },
        cancellation: CancellationPolicy {
            maximum_checkpoint_latency_ms: 100,
            maximum_work_units_between_checks: 100,
        },
    }
}

fn algorithm(stage: MeshingStageKind, request: &MeshingRequest) -> &str {
    match stage {
        MeshingStageKind::SurfaceMesh => &request.algorithms.surface,
        MeshingStageKind::Publication => &request.algorithms.validation,
        _ => panic!("fixture only supports exercised stages"),
    }
}

fn stable(seed: u8) -> StableDigest {
    StableDigest::from_bytes([seed; 32])
}

fn stage_input(digest: StableDigest) -> MeshingInputRef {
    MeshingInputRef {
        kind: MeshingInputKind::StageArtifact,
        digest,
    }
}
