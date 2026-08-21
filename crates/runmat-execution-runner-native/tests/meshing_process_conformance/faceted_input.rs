use super::*;

pub(super) async fn native_conformance() {
    let directory = tempfile::tempdir().unwrap();
    let access = MeshingArtifactAccess {
        authorization_scope: "native-faceted-meshing-run".into(),
        encryption_context: Digest::sha256(b"native-faceted-meshing-context"),
    };
    let (document, solid) = runmat_geometry_fixtures::faceted_tetrahedron();
    let objects = prepare_faceted_geometry_objects(document, solid, limits().inventory).unwrap();
    let document = objects.document.clone();
    let input =
        prepare_faceted_geometry_input(objects, access.clone(), limits().inventory).unwrap();
    let mut meshing_request = request();
    meshing_request.tolerance = document.tolerance;
    let input_ref = MeshingInputRef {
        kind: MeshingInputKind::FacetedGeometry,
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
        resolved_request_digest: meshing_request.canonical_digest().unwrap(),
        tolerance_policy_digest: meshing_request.tolerance.canonical_digest().unwrap(),
        metric_policy_digest: meshing_request.metric.canonical_digest().unwrap(),
        algorithm_set_digest: meshing_request.algorithms.canonical_digest().unwrap(),
        deterministic_seed: meshing_request.deterministic_seed,
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
        MeshingHostWorkload::new(workload, identity, meshing_request, access, Some(document))
            .unwrap();
    let request = host
        .program_request(revision(), std::slice::from_ref(input.root_input()))
        .unwrap();
    let session = NativeProgramSession::new(config(directory.path(), "--faceted-child")).unwrap();
    let mut store = session.object_store();
    for object in &input.geometry_objects().objects {
        store.write_verified(object).unwrap();
    }
    let task = session
        .submit(request.clone(), submission(&session, &host, &request))
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
        panic!("native faceted-input stage returned a non-object root")
    };
    import_result_publication(&store, root, host.artifact_access, limits().inventory).unwrap();
}
