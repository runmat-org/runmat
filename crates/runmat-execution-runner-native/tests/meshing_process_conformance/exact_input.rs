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
