use super::*;

#[test]
fn exact_meshing_dag_plans_initial_curve_batches_and_an_order_independent_join() {
    let mut fixture = Fixture::with_exact_tetrahedron_curve_partition();
    let exact_root = root(&fixture.program.arguments[0]);
    let exact = import_exact_geometry_input(
        &fixture.store,
        fixture.host.geometry_document.clone().unwrap(),
        &exact_root,
        fixture.host.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let planner =
        crate::ExactMeshingDagPlanner::from_exact_host(&fixture.host, exact_root.clone()).unwrap();
    let pass = planner
        .initial_curve_pass(&exact.geometry_objects().topology, 3)
        .unwrap();
    assert_eq!(pass.partitions().len(), 2);
    for (index, stage) in pass.partitions().iter().enumerate() {
        assert_eq!(
            stage.host().workload.partition.kind,
            MeshingPartitionKind::CanonicalEntityBatch
        );
        assert_eq!(
            stage.host().workload.partition.partition_index,
            index as u32
        );
        assert_eq!(stage.input_roots(), std::slice::from_ref(&exact_root));
        assert_eq!(
            algorithm_capability(stage.host()),
            fixture.host.resolved_request.algorithms.curve
        );
    }

    let first = execute_planned(
        &mut fixture,
        &pass.partitions()[0],
        &crate::ExactCurveStageKernel::default(),
    );
    let second = execute_planned(
        &mut fixture,
        &pass.partitions()[1],
        &crate::ExactCurveStageKernel::default(),
    );
    let first = root(first.publication().root_output());
    let second = root(second.publication().root_output());
    let forward = planner
        .curve_join(&pass, vec![first.clone(), second.clone()])
        .unwrap();
    let reverse = planner.curve_join(&pass, vec![second, first]).unwrap();
    assert_eq!(forward, reverse);
    assert_eq!(
        forward.host().workload.partition.kind,
        MeshingPartitionKind::DeterministicJoin
    );
    assert!(planner.curve_join(&pass, Vec::new()).is_err());
}

#[test]
fn exact_surface_dag_restarts_on_a_refined_curve_and_rejects_stale_partitions() {
    let (mut fixture, curve_host, exact_root, initial_curve) = execute_curve_pipeline();
    let initial_curve_root = root(initial_curve.publication().root_output());
    let exact = import_exact_geometry_input(
        &fixture.store,
        curve_host.geometry_document.clone().unwrap(),
        &exact_root,
        curve_host.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let planner =
        crate::ExactMeshingDagPlanner::from_exact_host(&curve_host, exact_root.clone()).unwrap();
    let first_pass = planner
        .begin_surface_pass(
            &exact.geometry_objects().topology,
            initial_curve_root.clone(),
            32,
        )
        .unwrap();
    assert_eq!(first_pass.pass_index(), 0);
    assert_eq!(first_pass.partitions().len(), 1);

    let first_partition = execute_planned(
        &mut fixture,
        &first_pass.partitions()[0],
        &SurfaceRestartKernel,
    );
    let first_partition_root = root(first_partition.publication().root_output());
    let first_partition_roots = vec![first_partition_root.clone()];
    let restart_stage = planner
        .surface_join(&first_pass, first_partition_roots.clone())
        .unwrap();
    let restart = execute_planned(&mut fixture, &restart_stage, &crate::ExactSurfaceJoinKernel);
    let restart_streams = restart
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap();
    assert_eq!(restart_streams.len(), 1);
    assert_eq!(
        restart_streams[0].media_type,
        runmat_meshing_core::MeshingChunkMediaType::SurfacePartitions
    );
    let restart_root = root(restart.publication().root_output());

    let refinement_stage = planner
        .curve_refinement(&first_pass, first_partition_roots, restart_root.clone())
        .unwrap();
    assert_eq!(
        algorithm_capability(refinement_stage.host()),
        curve_host.resolved_request.algorithms.curve
    );
    let refined_curve = execute_planned(
        &mut fixture,
        &refinement_stage,
        &crate::ExactCurveRefinementKernel::default(),
    );
    let refined_curve_root = root(refined_curve.publication().root_output());
    assert_ne!(
        refined_curve_root.logical_digest,
        initial_curve_root.logical_digest
    );

    let second_pass = planner
        .next_surface_pass(
            &first_pass,
            &exact.geometry_objects().topology,
            refined_curve_root,
            32,
        )
        .unwrap();
    assert_eq!(second_pass.pass_index(), 1);
    assert_eq!(
        algorithm_capability(second_pass.partitions()[0].host()),
        curve_host.resolved_request.algorithms.surface
    );

    let stale_join = planner
        .surface_join(&second_pass, vec![first_partition_root])
        .unwrap();
    let stale_error =
        execute_planned_result(&mut fixture, &stale_join, &crate::ExactSurfaceJoinKernel)
            .unwrap_err();
    assert_eq!(
        stage_failure(&stale_error).category,
        MeshingFailureCategory::InvalidGeometry
    );

    let second_partition = execute_planned(
        &mut fixture,
        &second_pass.partitions()[0],
        &ExactSurfacePartitionKernel::default(),
    );
    let second_partition_root = root(second_partition.publication().root_output());
    let final_stage = planner
        .surface_join(&second_pass, vec![second_partition_root])
        .unwrap();
    let final_pass = execute_planned(&mut fixture, &final_stage, &crate::ExactSurfaceJoinKernel);
    let final_streams = final_pass
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap();
    assert_eq!(final_streams.len(), 2);
    assert_eq!(
        final_streams[1].media_type,
        runmat_meshing_core::MeshingChunkMediaType::SurfaceMesh
    );

    let refined_curve_bytes = refined_curve
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap()[0]
        .records[0]
        .clone();
    let curves = runmat_meshing_curve::decode_shared_curve_mesh(
        &refined_curve_bytes,
        &exact.geometry_objects().topology,
    )
    .unwrap();
    let partition_bytes = second_partition
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap()[0]
        .records[0]
        .clone();
    let partition = runmat_meshing_surface::decode_exact_face_partition_result(
        &partition_bytes,
        &exact.geometry_objects().topology,
        &curves,
    )
    .unwrap();
    let pass_bytes = final_pass
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap()[0]
        .records[0]
        .clone();
    let pass = runmat_meshing_surface::decode_exact_surface_pass_result(
        &pass_bytes,
        &exact.geometry_objects().topology,
        &curves,
        &[partition],
        runmat_meshing_surface::ExactSurfaceJoinOptions {
            coordinate_tolerance_m: curve_host.resolved_request.tolerance.absolute_floor_m,
            maximum_nodes: curve_host.resolved_request.resources.maximum_nodes,
            maximum_triangles: curve_host.resolved_request.resources.maximum_elements,
            maximum_boundary_segments: curve_host.resolved_request.resources.maximum_elements,
        },
    )
    .unwrap();
    assert!(matches!(
        pass.outcome,
        runmat_meshing_surface::ExactSurfacePassOutcome::Converged { .. }
    ));
}

#[test]
fn exact_surface_dag_enforces_its_convergence_budget_and_context() {
    let (fixture, curve_host, exact_root, initial_curve) = execute_curve_pipeline();
    let exact = import_exact_geometry_input(
        &fixture.store,
        curve_host.geometry_document.clone().unwrap(),
        &exact_root,
        curve_host.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let planner =
        crate::ExactMeshingDagPlanner::from_exact_host(&curve_host, exact_root.clone()).unwrap();
    let initial_curve_root = root(initial_curve.publication().root_output());
    let pass = planner
        .begin_surface_pass(
            &exact.geometry_objects().topology,
            initial_curve_root.clone(),
            32,
        )
        .unwrap();
    assert!(planner
        .next_surface_pass(
            &pass,
            &exact.geometry_objects().topology,
            initial_curve_root,
            32,
        )
        .unwrap_err()
        .to_string()
        .contains("newly refined"));

    let mut shallow_host = curve_host.clone();
    shallow_host
        .resolved_request
        .resources
        .maximum_recursion_depth = 1;
    shallow_host.stage_identity.resolved_request_digest =
        shallow_host.resolved_request.canonical_digest().unwrap();
    shallow_host.workload.stage_identity_digest =
        shallow_host.stage_identity.canonical_digest().unwrap();
    shallow_host.validate().unwrap();
    let shallow =
        crate::ExactMeshingDagPlanner::from_exact_host(&shallow_host, exact_root).unwrap();
    let shallow_pass = shallow
        .begin_surface_pass(
            &exact.geometry_objects().topology,
            pass.curve_root().clone(),
            32,
        )
        .unwrap();
    assert!(shallow
        .surface_join(&pass, vec![pass.curve_root().clone()])
        .unwrap_err()
        .to_string()
        .contains("different geometry, request, or artifact authority"));
    let mut different_curve = pass.curve_root().clone();
    different_curve.logical_digest = Digest::sha256(b"different-refined-curve");
    different_curve.id = shallow_host
        .artifact_access
        .value_id(different_curve.logical_digest);
    assert!(shallow
        .next_surface_pass(
            &shallow_pass,
            &exact.geometry_objects().topology,
            different_curve,
            32,
        )
        .unwrap_err()
        .to_string()
        .contains("recursion budget"));
}

#[test]
fn exact_surface_dag_identity_is_independent_of_partition_completion_order() {
    let (fixture, curve_host, exact_root, initial_curve) = execute_curve_pipeline();
    let exact = import_exact_geometry_input(
        &fixture.store,
        curve_host.geometry_document.clone().unwrap(),
        &exact_root,
        curve_host.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let planner = crate::ExactMeshingDagPlanner::from_exact_host(&curve_host, exact_root).unwrap();
    let mut topology = exact.geometry_objects().topology.clone();
    let mut second_face = topology.faces[0].clone();
    second_face.id.source_topology_id = "face-second".into();
    topology.faces.push(second_face);
    topology.faces.sort_by(|left, right| left.id.cmp(&right.id));
    let curve_root = root(initial_curve.publication().root_output());
    let pass = planner
        .begin_surface_pass(&topology, curve_root.clone(), 1)
        .unwrap();
    assert_eq!(pass.partitions().len(), 2);

    let partition_root = |label: &[u8]| {
        let mut result = curve_root.clone();
        result.logical_digest = Digest::sha256(label);
        result.id = curve_host.artifact_access.value_id(result.logical_digest);
        result
    };
    let first = partition_root(b"first-finished-partition");
    let second = partition_root(b"second-finished-partition");
    let forward = planner
        .surface_join(&pass, vec![first.clone(), second.clone()])
        .unwrap();
    let reverse = planner.surface_join(&pass, vec![second, first]).unwrap();

    assert_eq!(forward, reverse);
    assert!(forward
        .host()
        .workload
        .inputs
        .windows(2)
        .all(|pair| pair[0] < pair[1]));
}

fn execute_planned(
    fixture: &mut Fixture,
    stage: &crate::PlannedMeshingStage,
    kernel: &dyn MeshingStageKernel,
) -> CompletedMeshingStage {
    execute_planned_result(fixture, stage, kernel).unwrap()
}

fn execute_planned_result(
    fixture: &mut Fixture,
    stage: &crate::PlannedMeshingStage,
    kernel: &dyn MeshingStageKernel,
) -> Result<CompletedMeshingStage, MeshingSerialExecutionError> {
    let program = stage.program_request(revision()).unwrap();
    execute_serial_stage(
        &program,
        &mut fixture.store,
        kernel,
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(1_000_000),
        ObjectInventoryLimits::default(),
    )
}

fn algorithm_capability(host: &MeshingHostWorkload) -> String {
    host.workload
        .required_capabilities
        .iter()
        .find_map(|capability| match capability {
            MeshingCapabilityRequirement::MeshingAlgorithm { version } => Some(version.clone()),
            _ => None,
        })
        .unwrap()
}
