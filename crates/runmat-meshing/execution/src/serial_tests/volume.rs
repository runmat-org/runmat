use super::*;

#[test]
fn serial_dispatcher_publishes_independently_validated_general_cdt_volume() {
    let fixture = Fixture::with_exact_tetrahedron_curve_partition();
    let (mut fixture, surface_host, exact_root, _, _, surface_stage) =
        super::surface_restart::execute_surface_join_pipeline_from_fixture(
            &ExactSurfacePartitionKernel::default(),
            fixture,
        );
    let surface_root = root(surface_stage.publication().root_output());
    let exact_input = surface_host
        .workload
        .inputs
        .iter()
        .find(|input| input.kind == MeshingInputKind::ExactGeometry)
        .unwrap()
        .clone();
    let surface_input = MeshingInputRef {
        kind: MeshingInputKind::StageArtifact,
        digest: StableDigest::from_bytes(*surface_root.logical_digest.bytes()),
    };
    let request = surface_host.resolved_request.clone();
    let mut dependencies = vec![
        (exact_input, exact_root.clone()),
        (surface_input, surface_root),
    ];
    dependencies.sort_by(|left, right| left.0.cmp(&right.0));
    let inputs = dependencies
        .iter()
        .map(|(input, _)| input.clone())
        .collect::<Vec<_>>();
    let roots = dependencies
        .into_iter()
        .map(|(_, root)| root)
        .collect::<Vec<_>>();
    let identity = MeshingStageIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage: MeshingStageKind::Tetrahedralization,
        geometry: surface_host.stage_identity.geometry.clone(),
        resolved_request_digest: request.canonical_digest().unwrap(),
        tolerance_policy_digest: request.tolerance.canonical_digest().unwrap(),
        metric_policy_digest: request.metric.canonical_digest().unwrap(),
        algorithm_set_digest: request.algorithms.canonical_digest().unwrap(),
        deterministic_seed: request.deterministic_seed,
        prerequisites: inputs.clone(),
        capability_cohort: surface_host.stage_identity.capability_cohort.clone(),
    };
    let workload = MeshingWorkloadRequest {
        schema_version: MESHING_WORKLOAD_SCHEMA_VERSION,
        stage: MeshingStageKind::Tetrahedralization,
        stage_identity_digest: identity.canonical_digest().unwrap(),
        partition: MeshingPartitionDescriptor {
            kind: MeshingPartitionKind::WholeStage,
            partition_index: 0,
            partition_count: 1,
            entity_range: None,
        },
        inputs,
        required_capabilities: surface_host
            .workload
            .required_capabilities
            .iter()
            .map(|capability| match capability {
                MeshingCapabilityRequirement::MeshingAlgorithm { .. } => {
                    MeshingCapabilityRequirement::MeshingAlgorithm {
                        version: request.algorithms.tetrahedron.clone(),
                    }
                }
                capability => capability.clone(),
            })
            .collect(),
    };
    let host = MeshingHostWorkload::new(
        workload,
        identity,
        request,
        surface_host.artifact_access.clone(),
        surface_host.geometry_document.clone(),
    )
    .unwrap();
    let program = host.program_request(revision(), &roots).unwrap();
    let completed = execute_serial_stage(
        &program,
        &mut fixture.store,
        &crate::MeshingKernelDispatcher,
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(1_000_000),
        ObjectInventoryLimits::default(),
    )
    .unwrap();

    let exact = import_exact_geometry_input(
        &fixture.store,
        host.geometry_document.clone().unwrap(),
        &exact_root,
        host.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let surface_streams = surface_stage
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap();
    let surface = runmat_meshing_surface::decode_published_exact_surface_mesh(
        &surface_streams[1].records[0],
        &exact.geometry_objects().topology,
        runmat_meshing_surface::ExactSurfaceJoinOptions {
            coordinate_tolerance_m: host.resolved_request.tolerance.absolute_floor_m,
            maximum_nodes: host.resolved_request.resources.maximum_nodes,
            maximum_triangles: host.resolved_request.resources.maximum_elements,
            maximum_boundary_segments: host.resolved_request.resources.maximum_elements,
        },
    )
    .unwrap();
    let streams = completed
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap();
    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].media_type, MeshingChunkMediaType::VolumeTopology);
    let mesh = runmat_meshing_tetrahedron::cdt::decode_delaunay_volume_mesh(
        &streams[0].records[0],
        &exact.geometry_objects().topology,
        &surface,
        &host.resolved_request.metric,
        crate::volume_kernel::volume_options(&host.resolved_request),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(mesh.topology.tetrahedra.len(), 1);
    assert_eq!(mesh.topology.incidence.regions.len(), 1);
    assert_eq!(mesh.provenance.facets.len(), 4);
}
