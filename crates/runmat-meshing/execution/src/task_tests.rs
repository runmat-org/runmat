use std::collections::BTreeSet;

use runmat_execution::identity::WorkerId;
use runmat_execution::resource::{Capability, ResourceInventory};
use runmat_execution::state::PoolState;
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{Digest, ExecutionScopeId, PoolId};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_runner::driver::{DriverAction, DriverCommand, DriverConfig};
use runmat_execution_runner::{Driver, PoolSpec, WorkerSpec};
use runmat_meshing_core::{
    AlgorithmVersionSet, CancellationPolicy, CanonicalMeshingContract, CurveQualityTargets,
    ElementOrder, GeometryRevisionRef, GeometryTolerancePolicy, MeshingCapabilityRequirement,
    MeshingInputKind, MeshingInputRef, MeshingPartitionDescriptor, MeshingPartitionKind,
    MeshingQualityTargets, MeshingRequest, MeshingSizeLimits, MeshingStageIdentity,
    MeshingStageKind, MeshingWorkloadRequest, MetricCombinationRule, MetricFieldRequest,
    MetricTensor3, StableDigest, SurfaceQualityTargets, VolumeQualityTargets,
    MESHING_IDENTITY_SCHEMA_VERSION, MESHING_REQUEST_SCHEMA_VERSION,
    MESHING_WORKLOAD_SCHEMA_VERSION,
};

use crate::{
    build_task_submission, exact_meshing_worker_capabilities, prepare_exact_geometry_input,
    prepare_exact_geometry_objects, prepare_faceted_geometry_input,
    prepare_faceted_geometry_objects, ArtifactAccess, MeshingExecutionContext,
    MeshingTaskEffectPolicy, MESHING_STAGE_MANIFEST_MEDIA_TYPE,
};

/// The host the meshing program actually requires, derived as production derives it.
pub(crate) fn host_requirement() -> runmat_execution::host::ExecutionHostRequirement {
    let crate::MeshingHostProgram { recipe, artifact } =
        crate::prepare_meshing_host_program(crate::serial_tests::revision()).unwrap();
    artifact
        .execution_host_requirement(
            &recipe,
            BTreeSet::from([runmat_execution::security::ExecutionTrustTier::CustomerTrusted]),
        )
        .unwrap()
}

pub(crate) fn host_inventory() -> runmat_execution::host::ExecutionHostInventory {
    let requirement = host_requirement();
    runmat_execution::host::ExecutionHostInventory {
        schema_version: runmat_execution::host::EXECUTION_HOST_SCHEMA_VERSION,
        semantic_schema: requirement.environment.semantic_schema,
        compiler_schema: requirement.environment.compiler_schema,
        runtime_fingerprint: requirement.environment.runtime_fingerprint,
        catalog_fingerprint: requirement.environment.catalog_fingerprint,
        compatibility_modes: BTreeSet::from([requirement.environment.compatibility_mode]),
        target: runmat_execution::host::ExecutionHostTarget::BrowserWasm,
        capabilities: requirement.capabilities,
        adapters: Vec::new(),
        process_stack_available: false,
        host_bridge_available: false,
        trust_tier: runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
    }
}

#[test]
fn exact_dag_worker_inventory_covers_every_stage_family() {
    let (document, _, _) = runmat_geometry_fixtures::exact_tetrahedron();
    let mut request = Fixture::new(MeshingStageKind::SurfaceMesh).request;
    request.tolerance = document.tolerance;
    let capabilities =
        exact_meshing_worker_capabilities(&document, &request, Some("native-cohort-v1")).unwrap();
    for capability in [
        "runmat.meshing.host:host-v2",
        "runmat.meshing.exact-cad:occt-v1",
        "runmat.meshing.algorithm:geometry/v2",
        "runmat.meshing.algorithm:curve/v2",
        "runmat.meshing.algorithm:surface/v2",
        "runmat.meshing.algorithm:plc/v2",
        "runmat.meshing.algorithm:tetrahedron/v2",
        "runmat.meshing.algorithm:optimization/v2",
        "runmat.meshing.algorithm:validation/v2",
        "runmat.meshing.element-order:tet10",
        "runmat.meshing.cohort:native-cohort-v1",
    ] {
        assert!(capabilities.contains(&Capability::Custom(capability.into())));
    }
    assert!(capabilities.contains(&Capability::ProcessIsolation));
}

#[test]
fn task_projection_binds_identity_inputs_resources_and_capabilities() {
    let ceilings = ceilings();
    let fixture = Fixture::new(MeshingStageKind::SurfaceMesh);
    let submission = fixture.submit(MeshingTaskEffectPolicy::ContentAddressedPure {
        maximum_attempts: 3,
        replay_proof_digest: stable(31),
    });

    assert_eq!(submission.request.inputs.len(), 2);
    assert_eq!(
        submission.request.resources.memory_bytes,
        ceilings.maximum_memory_bytes
    );
    assert_eq!(
        submission.request.resources.scratch_bytes,
        ceilings.maximum_scratch_bytes
    );
    assert_eq!(
        submission.request.resources.max_wall_millis,
        ceilings.maximum_wall_time_ms
    );
    assert_eq!(
        submission.request.resources.max_artifact_bytes,
        ceilings.maximum_artifact_bytes
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
        &fixture.host(),
        &fixture.program(fixture.program_inputs()),
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
    wrong_request.limits.maximum_nodes += 1;
    assert!(build_task_submission(
        &fixture.host_with_request(wrong_request),
        &fixture.program(fixture.program_inputs()),
        BTreeSet::new(),
        &fixture.context,
        MeshingTaskEffectPolicy::UnknownEffect,
    )
    .is_err());

    let mut wrong_input = fixture.input.clone();
    wrong_input.authorization_scope = "another-run".into();
    assert!(build_task_submission(
        &fixture.host(),
        &fixture.program(fixture.program_inputs_with(wrong_input)),
        BTreeSet::new(),
        &fixture.context,
        MeshingTaskEffectPolicy::UnknownEffect,
    )
    .is_err());
}

#[test]
fn task_rejects_an_invocation_for_another_stage() {
    let expected = Fixture::new(MeshingStageKind::SurfaceMesh);
    let other = Fixture::new(MeshingStageKind::Publication);

    assert!(build_task_submission(
        &expected.host(),
        &other.program(other.program_inputs()),
        BTreeSet::new(),
        &expected.context,
        MeshingTaskEffectPolicy::UnknownEffect,
    )
    .is_err());
}

#[test]
fn exact_geometry_inputs_use_the_authoritative_geometry_object_shape() {
    let mut fixture = Fixture::new(MeshingStageKind::SurfaceMesh);
    let (document, topology, evaluators) = runmat_geometry_fixtures::exact_circle();
    let objects = prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
        None,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let document = objects.document.clone();
    let input = prepare_exact_geometry_input(
        objects,
        fixture.context.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    fixture.bind_exact_geometry(&document, input.root_input().clone());

    fixture.submit(MeshingTaskEffectPolicy::UnknownEffect);

    fixture.input.kind = ValueRefKind::WorkerObject;
    assert!(build_task_submission(
        &fixture.host(),
        &fixture.program(fixture.program_inputs()),
        BTreeSet::new(),
        &fixture.context,
        MeshingTaskEffectPolicy::UnknownEffect,
    )
    .is_err());
}

#[test]
fn faceted_geometry_inputs_use_the_authoritative_geometry_object_shape() {
    let mut fixture = Fixture::new(MeshingStageKind::SurfaceMesh);
    let (document, solid) = runmat_geometry_fixtures::faceted_tetrahedron();
    let objects =
        prepare_faceted_geometry_objects(document, solid, ObjectInventoryLimits::default())
            .unwrap();
    let document = objects.document.clone();
    let input = prepare_faceted_geometry_input(
        objects,
        fixture.context.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    fixture.bind_faceted_geometry(&document, input.root_input().clone());

    fixture.submit(MeshingTaskEffectPolicy::UnknownEffect);

    fixture.input.media_type = runmat_geometry_core::EXACT_BREP_MEDIA_TYPE.into();
    assert!(build_task_submission(
        &fixture.host(),
        &fixture.program(fixture.program_inputs()),
        BTreeSet::new(),
        &fixture.context,
        MeshingTaskEffectPolicy::UnknownEffect,
    )
    .is_err());
}

#[test]
fn evidence_inputs_use_the_authoritative_driver_object_shape() {
    let mut fixture = Fixture::new(MeshingStageKind::Publication);
    fixture.identity.prerequisites[0].kind = MeshingInputKind::Evidence;
    fixture.workload.inputs = fixture.identity.prerequisites.clone();
    fixture.workload.stage_identity_digest = fixture.identity.canonical_digest().unwrap();
    fixture.input.kind = ValueRefKind::DriverObject;
    fixture.input.media_type = runmat_meshing_core::MeshingChunkMediaType::MeshingEvidence
        .media_type()
        .into();
    fixture.input.value_schema = crate::evidence_objects::EVIDENCE_VALUE_SCHEMA.into();

    fixture.submit(MeshingTaskEffectPolicy::UnknownEffect);

    fixture.input.kind = ValueRefKind::ResultObject;
    assert!(build_task_submission(
        &fixture.host(),
        &fixture.program(fixture.program_inputs()),
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
        &fixture.host_with_workload(missing_host),
        &fixture.program(fixture.program_inputs()),
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
        &fixture.host_with_workload(wrong_order),
        &fixture.program(fixture.program_inputs()),
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
        &final_stage.host(),
        &final_stage.program(final_stage.program_inputs()),
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
    // The driver refuses work that no registered worker can serve, so queue the
    // task first; an incapable worker joining later must still not receive it.
    assert!(driver
        .handle(DriverCommand::Submit(Box::new(submission.clone())))
        .unwrap()
        .iter()
        .all(|action| !matches!(action, DriverAction::Launch(_))));
    assert!(driver
        .handle(DriverCommand::RegisterWorker(WorkerSpec {
            id: WorkerId::derive(&[b"incapable"]),
            pool_id: pool,
            host: host_inventory(),
            resources: ResourceInventory {
                cpu_millicores: 4_000,
                memory_bytes: 8_000_000,
                scratch_bytes: 8_000_000,
                accelerators: Vec::new(),
                capabilities: BTreeSet::new(),
            },
        }))
        .unwrap()
        .iter()
        .all(|action| !matches!(action, DriverAction::Launch(_))));

    let actions = driver
        .handle(DriverCommand::RegisterWorker(WorkerSpec {
            id: WorkerId::derive(&[b"capable"]),
            pool_id: pool,
            host: host_inventory(),
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
    geometry_document: Option<runmat_geometry_core::GeometryDocument>,
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
                source_unit_evidence:
                    runmat_geometry_core::physical::LengthUnitEvidence::explicit_fallback(
                        runmat_geometry_core::LengthUnit::Meter,
                        "canonical solver fixture",
                    ),
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
        let access = ArtifactAccess {
            authorization_scope: "run-scope".into(),
            encryption_context: Digest::sha256(b"encryption-context"),
        };
        let input = ValueRef {
            schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
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
            permitted_trust_tiers: BTreeSet::from([
                runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
            ]),
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
            geometry_document: None,
        }
    }

    fn submit(&self, effect: MeshingTaskEffectPolicy) -> runmat_execution_runner::TaskSubmission {
        build_task_submission(
            &self.host(),
            &self.program(self.program_inputs()),
            BTreeSet::new(),
            &self.context,
            effect,
        )
        .unwrap()
    }

    fn host(&self) -> crate::MeshingHostWorkload {
        crate::MeshingHostWorkload {
            schema_version: crate::MESHING_HOST_WORKLOAD_SCHEMA_VERSION,
            workload: self.workload.clone(),
            stage_identity: self.identity.clone(),
            resolved_request: self.request.clone(),
            ceilings: ceilings(),
            artifact_access: self.context.artifact_access.clone(),
            geometry_document: self.geometry_document.clone(),
        }
    }

    fn host_with_request(&self, request: MeshingRequest) -> crate::MeshingHostWorkload {
        crate::MeshingHostWorkload {
            resolved_request: request,
            ..self.host()
        }
    }

    fn host_with_workload(&self, workload: MeshingWorkloadRequest) -> crate::MeshingHostWorkload {
        crate::MeshingHostWorkload {
            workload,
            ..self.host()
        }
    }

    /// The meshing host program request carrying `arguments`, built as the library builds it.
    fn program(
        &self,
        arguments: Vec<ValuePayload>,
    ) -> runmat_execution_artifact::ProgramExecutionRequest {
        let crate::MeshingHostProgram { recipe, artifact } =
            crate::prepare_meshing_host_program(crate::serial_tests::revision()).unwrap();
        runmat_execution_artifact::ProgramExecutionRequest {
            schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            recipe,
            artifact,
            callable: runmat_execution::ProgramCallable::semantic(
                runmat_execution::ProgramFunctionId(0),
                None,
            ),
            context: runmat_execution::ProgramInvocationContext::Direct,
            assignment: None,
            job_id: None,
            arguments,
            requested_outputs: 1,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            console: None,
        }
    }

    fn program_inputs(&self) -> Vec<ValuePayload> {
        self.program_inputs_with(self.input.clone())
    }

    fn program_inputs_with(&self, input: ValueRef) -> Vec<ValuePayload> {
        let host_bytes = self.host().canonical_bytes().unwrap();
        let logical_digest = Digest::sha256(&host_bytes);
        let invocation = ValueRef {
            schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
            id: self.context.artifact_access.value_id(logical_digest),
            logical_digest,
            encoded_length: host_bytes.len() as u64,
            media_type: crate::MESHING_HOST_INVOCATION_MEDIA_TYPE.into(),
            value_schema: crate::MESHING_HOST_INVOCATION_VALUE_SCHEMA.into(),
            encryption_context: self.context.artifact_access.encryption_context,
            kind: ValueRefKind::DriverObject,
            authorization_scope: self.context.artifact_access.authorization_scope.clone(),
            resident_fence: None,
        };
        vec![
            ValuePayload::Object(Box::new(invocation)),
            ValuePayload::Object(Box::new(input)),
        ]
    }

    pub(crate) fn bind_exact_geometry(
        &mut self,
        document: &runmat_geometry_core::GeometryDocument,
        input: ValueRef,
    ) {
        let runmat_geometry_core::GeometryModel::ExactBRep { model } = &document.model else {
            panic!("test fixture requires exact geometry")
        };
        self.request.tolerance = document.tolerance;
        self.identity.geometry.source_digest =
            StableDigest::from_bytes(*document.source.content_digest.bytes());
        self.identity.geometry.geometry_revision = document.revision.revision;
        self.identity.geometry.persistent_mapping_version =
            document.revision.persistent_mapping_version;
        self.identity.resolved_request_digest = self.request.canonical_digest().unwrap();
        self.identity.tolerance_policy_digest = self.request.tolerance.canonical_digest().unwrap();
        self.identity.prerequisites = vec![MeshingInputRef {
            kind: MeshingInputKind::ExactGeometry,
            digest: StableDigest::from_bytes(*input.logical_digest.bytes()),
        }];
        self.workload.inputs = self.identity.prerequisites.clone();
        self.workload.stage_identity_digest = self.identity.canonical_digest().unwrap();
        for capability in &mut self.workload.required_capabilities {
            if let MeshingCapabilityRequirement::ExactCadKernel { abi } = capability {
                *abi = model.kernel_abi.clone();
            }
        }
        self.input = input;
        self.geometry_document = Some(document.clone());
    }

    pub(crate) fn bind_faceted_geometry(
        &mut self,
        document: &runmat_geometry_core::GeometryDocument,
        input: ValueRef,
    ) {
        assert!(matches!(
            document.model,
            runmat_geometry_core::GeometryModel::FacetedSolid { .. }
        ));
        self.request.tolerance = document.tolerance;
        self.identity.geometry.source_digest =
            StableDigest::from_bytes(*document.source.content_digest.bytes());
        self.identity.geometry.geometry_revision = document.revision.revision;
        self.identity.geometry.persistent_mapping_version =
            document.revision.persistent_mapping_version;
        self.identity.resolved_request_digest = self.request.canonical_digest().unwrap();
        self.identity.tolerance_policy_digest = self.request.tolerance.canonical_digest().unwrap();
        self.identity.prerequisites = vec![MeshingInputRef {
            kind: MeshingInputKind::FacetedGeometry,
            digest: StableDigest::from_bytes(*input.logical_digest.bytes()),
        }];
        self.workload.inputs = self.identity.prerequisites.clone();
        self.workload.stage_identity_digest = self.identity.canonical_digest().unwrap();
        self.workload.required_capabilities.retain(|capability| {
            !matches!(
                capability,
                MeshingCapabilityRequirement::ExactCadKernel { .. }
            )
        });
        self.input = input;
        self.geometry_document = Some(document.clone());
    }
}

fn request() -> MeshingRequest {
    MeshingRequest {
        schema_version: MESHING_REQUEST_SCHEMA_VERSION,
        element_order: ElementOrder::Tet10,
        dimension: runmat_meshing_core::MeshDimension::Solid,
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
            source_tolerance_m: runmat_types::physical::Length::new(1.0e-8),
            absolute_floor_m: runmat_types::physical::Length::new(1.0e-10),
            model_relative_term: 1.0e-9,
            requested_deviation_m: runmat_types::physical::Length::new(1.0e-5),
            maximum_healing_displacement_m: runmat_types::physical::Length::new(1.0e-6),
        },
        metric: MetricFieldRequest {
            entity_supports: Vec::new(),
            located_sources: Vec::new(),
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(0.5).unwrap(),
            maximum_grading_ratio: 1.3,
            contributions: Vec::new(),
            cells_across_gap: None,
        },
        quality: MeshingQualityTargets {
            curve: CurveQualityTargets {
                maximum_chordal_deviation_m: runmat_types::physical::Length::new(1.0e-5),
                maximum_tangent_change_degrees: 5.0,
                minimum_metric_edge_length: 0.1,
                maximum_metric_edge_length: 1.5,
            },
            surface: SurfaceQualityTargets {
                minimum_metric_edge_length: 0.1,
                minimum_metric_angle_degrees: 20.0,
                maximum_physical_aspect_ratio: 10.0,
                maximum_chordal_deviation_m: runmat_types::physical::Length::new(1.0e-5),
                maximum_normal_deviation_degrees: 5.0,
            },
            volume: VolumeQualityTargets {
                maximum_radius_edge_ratio: 2.0,
                minimum_scaled_jacobian: 0.05,
                maximum_metric_edge_length: 1.5,
            },
        },
        limits: MeshingSizeLimits {
            maximum_nodes: 100,
            maximum_elements: 100,
            maximum_recursion_depth: 32,
        },
        cancellation: CancellationPolicy {
            maximum_checkpoint_latency_ms: 100,
            maximum_work_units_between_checks: 100,
        },
        size_controls: Vec::new(),
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

/// Execution ceilings of the task test fixture.
pub(crate) fn ceilings() -> runmat_meshing_core::MeshingExecutionCeilings {
    runmat_meshing_core::MeshingExecutionCeilings {
        maximum_memory_bytes: 4_000_000,
        maximum_scratch_bytes: 4_000_000,
        maximum_wall_time_ms: 10_000,
        maximum_artifact_bytes: 2_000_000,
        maximum_search_work: 10_000,
        maximum_iterations: 10_000,
    }
}
