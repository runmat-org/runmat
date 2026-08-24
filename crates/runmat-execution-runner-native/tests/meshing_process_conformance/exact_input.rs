use super::*;

pub(super) struct ExactFixture {
    pub(super) host: MeshingHostWorkload,
    pub(super) request: runmat_execution_artifact::ProgramExecutionRequest,
    pub(super) input: PreparedExactGeometryInput,
}

pub(super) async fn native_conformance() {
    let directory = tempfile::tempdir().unwrap();
    let exact = fixture(revision(), "native-exact-meshing-run");
    let session = NativeProgramSession::new(config(directory.path(), "--exact-child")).unwrap();
    let mut store = session.object_store();
    for object in &exact.input.geometry_objects().objects {
        store.write_verified(object).unwrap();
    }
    let task = session
        .submit(
            exact.request.clone(),
            submission(&session, &exact.host, &exact.request),
        )
        .unwrap();
    let success = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Some(result) = task.try_result() {
                break result.unwrap();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let [ValuePayload::Object(root)] = success.outputs.as_slice() else {
        panic!("native exact-input stage returned a non-object root")
    };
    import_result_publication(&store, root, exact.host.artifact_access, limits().inventory)
        .unwrap();
}

pub(super) fn native_dag_conformance() {
    let serial_layout = execute_native_dag(1, 1, 1);
    let parallel_layout = execute_native_dag(2, 3, 32);

    assert_eq!(serial_layout.topology, parallel_layout.topology);
    assert_eq!(
        serial_layout.canonical_digest,
        parallel_layout.canonical_digest
    );
}

fn execute_native_dag(
    max_workers: u32,
    preferred_edges_per_partition: u32,
    preferred_faces_per_partition: u32,
) -> runmat_meshing_core::SolverMeshArtifact {
    let directory = tempfile::tempdir().unwrap();
    let mut config = config(directory.path(), "--exact-dag-child");
    config.max_workers = max_workers;
    let session = NativeProgramSession::new(config).unwrap();
    let access = MeshingArtifactAccess {
        authorization_scope: format!("native-exact-dag-{max_workers}"),
        encryption_context: Digest::sha256(
            [
                b"native-exact-dag-context".as_slice(),
                &max_workers.to_be_bytes(),
            ]
            .concat(),
        ),
    };
    let (document, topology, evaluators) = runmat_geometry_fixtures::exact_tetrahedron();
    let geometry_objects = prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
        None,
        limits().inventory,
    )
    .unwrap();
    let document = geometry_objects.document.clone();
    let geometry =
        prepare_exact_geometry_input(geometry_objects, access.clone(), limits().inventory).unwrap();
    let mut store = session.object_store();
    for object in &geometry.geometry_objects().objects {
        store.write_verified(object).unwrap();
    }
    let mut request = request();
    request.tolerance = document.tolerance;
    request.metric.global_metric = MetricTensor3::isotropic_length_m(10.0).unwrap();
    request.quality.curve.maximum_chordal_deviation_m = 0.1;
    request.quality.curve.maximum_tangent_change_degrees = 180.0;
    request.quality.curve.minimum_metric_edge_length = 0.01;
    request.quality.curve.maximum_metric_edge_length = 10.0;
    request.quality.surface.minimum_metric_angle_degrees = 0.1;
    request.quality.surface.maximum_physical_aspect_ratio = 1_000.0;
    request.quality.surface.maximum_chordal_deviation_m = 0.1;
    request.quality.surface.maximum_normal_deviation_degrees = 180.0;
    request.quality.volume.maximum_metric_edge_length = 2.0;
    request.quality.volume.maximum_radius_edge_ratio = 10.0;
    request.quality.volume.minimum_scaled_jacobian = 0.01;
    request.resources.maximum_search_work = 1_000_000;
    request.resources.maximum_iterations = 1_000_000;

    let mut executor =
        NativeExactMeshingExecutor::new(&session, NativeMeshingExecutionPolicy::default()).unwrap();
    let result = runmat_meshing_execution::execute_exact_meshing_dag(
        runmat_meshing_execution::ExactMeshingDagRun {
            geometry: &geometry,
            request,
            artifact_access: access,
            capability_cohort: Some("native-cohort-v1".into()),
            program_revision: revision(),
            preferred_edges_per_partition,
            preferred_faces_per_partition,
            inventory_limits: limits().inventory,
            evidence: runmat_meshing_execution::MeshingRunEvidenceContext {
                platform: runmat_meshing_core::PlatformBuildIdentity {
                    capability_cohort: "native-cohort-v1".into(),
                    target_triple: "native-process-test".into(),
                    build_digest: stable(71),
                    exact_kernel_abi: Some("occt-v1".into()),
                },
                sizing: Vec::new(),
                cache_admission: runmat_meshing_core::CacheAdmissionDecision::Admitted,
            },
        },
        &mut executor,
    )
    .unwrap();
    result.evidence.validate(&result.artifact).unwrap();
    assert_eq!(result.artifact.topology.volume_elements.len(), 1);
    assert!(!executor.drain_progress().is_empty());
    result.artifact
}

pub(super) fn fixture(revision: ProgramRevision, authorization_scope: &str) -> ExactFixture {
    let access = MeshingArtifactAccess {
        authorization_scope: authorization_scope.into(),
        encryption_context: Digest::sha256(b"native-meshing-encryption-context"),
    };
    let (document, topology, evaluators) = runmat_geometry_fixtures::exact_circle();
    let geometry_objects = prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
        None,
        limits().inventory,
    )
    .unwrap();
    let document = geometry_objects.document.clone();
    let input =
        prepare_exact_geometry_input(geometry_objects, access.clone(), limits().inventory).unwrap();
    let mut request = request();
    request.tolerance = document.tolerance;
    let input_ref = MeshingInputRef {
        kind: MeshingInputKind::ExactGeometry,
        digest: StableDigest::from_bytes(*input.root_input().logical_digest.bytes()),
    };
    let identity = MeshingStageIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage: MeshingStageKind::SurfaceMesh,
        geometry: GeometryRevisionRef {
            source_digest: StableDigest::from_bytes(*document.source.content_digest.bytes()),
            geometry_revision: document.revision.revision,
            persistent_mapping_version: document.revision.persistent_mapping_version,
        },
        resolved_request_digest: request.canonical_digest().unwrap(),
        tolerance_policy_digest: request.tolerance.canonical_digest().unwrap(),
        metric_policy_digest: request.metric.canonical_digest().unwrap(),
        algorithm_set_digest: request.algorithms.canonical_digest().unwrap(),
        deterministic_seed: request.deterministic_seed,
        prerequisites: vec![input_ref.clone()],
        capability_cohort: Some("native-cohort-v1".into()),
    };
    let workload = MeshingWorkloadRequest {
        schema_version: MESHING_WORKLOAD_SCHEMA_VERSION,
        stage: MeshingStageKind::SurfaceMesh,
        stage_identity_digest: identity.canonical_digest().unwrap(),
        partition: MeshingPartitionDescriptor {
            kind: MeshingPartitionKind::WholeStage,
            partition_index: 0,
            partition_count: 1,
            entity_range: None,
        },
        inputs: vec![input_ref],
        required_capabilities: vec![
            MeshingCapabilityRequirement::HostWorkload {
                abi: "host-v2".into(),
            },
            MeshingCapabilityRequirement::ExactCadKernel {
                abi: "occt-v1".into(),
            },
            MeshingCapabilityRequirement::MeshingAlgorithm {
                version: "surface/v2".into(),
            },
            MeshingCapabilityRequirement::ElementOrder {
                order: ElementOrder::Tet4,
            },
            MeshingCapabilityRequirement::DeterministicPlatformCohort {
                cohort: "native-cohort-v1".into(),
            },
        ],
    };
    let host =
        MeshingHostWorkload::new(workload, identity, request, access, Some(document)).unwrap();
    let request = host
        .program_request(revision, std::slice::from_ref(input.root_input()))
        .unwrap();
    ExactFixture {
        host,
        request,
        input,
    }
}
