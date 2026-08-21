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
    let planner =
        crate::ExactMeshingDagPlanner::from_exact_host(&surface_host, exact_root.clone()).unwrap();
    let volume_stage = planner.tetrahedralization(surface_root).unwrap();
    let host = volume_stage.host().clone();
    assert_eq!(host.workload.stage, MeshingStageKind::Tetrahedralization);
    assert_eq!(
        host.workload.partition.kind,
        MeshingPartitionKind::WholeStage
    );
    assert_eq!(host.workload.inputs.len(), 2);
    assert!(host
        .workload
        .required_capabilities
        .iter()
        .any(|capability| {
            matches!(
                capability,
                MeshingCapabilityRequirement::MeshingAlgorithm { version }
                    if version == &host.resolved_request.algorithms.tetrahedron
            )
        }));
    let program = volume_stage.program_request(revision()).unwrap();
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
