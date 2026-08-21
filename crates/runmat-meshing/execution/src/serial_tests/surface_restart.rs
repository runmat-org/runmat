use super::*;

mod dag;

struct SurfaceRestartKernel;

impl MeshingStageKernel for SurfaceRestartKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        let geometry = invocation
            .inputs
            .iter()
            .find_map(|input| match input {
                PreparedMeshingInput::ExactGeometry(input) => Some(input.geometry_objects()),
                _ => None,
            })
            .expect("restart fixture has exact geometry");
        let curve_publication = invocation
            .inputs
            .iter()
            .find_map(PreparedMeshingInput::stage_artifact)
            .expect("restart fixture has shared curves");
        let curve_bytes = crate::surface_kernel::curve_record(geometry, curve_publication)?;
        let curves =
            runmat_meshing_curve::decode_shared_curve_mesh(&curve_bytes, &geometry.topology)
                .map_err(crate::curve_kernel::map_curve_error)?;
        let edge = &curves.edges[0];
        let split = runmat_meshing_curve::SharedCurveSegmentSplit {
            source_edge_id: edge.source_edge_id.clone(),
            endpoint_node_ids: [edge.nodes[0].node_id, edge.nodes[1].node_id],
            edge_parameters: [edge.nodes[0].parameter, edge.nodes[1].parameter],
            split_parameter: edge.nodes[0].parameter * 0.5 + edge.nodes[1].parameter * 0.5,
        };
        let result = runmat_meshing_surface::build_exact_face_partition_result(
            &geometry.topology,
            &curves,
            invocation.host.workload.partition.clone(),
            runmat_meshing_surface::ExactFacePartitionOutcome::RequiresCurveSplits {
                splits: vec![split],
            },
        )
        .unwrap();
        let encoded = runmat_meshing_surface::encode_exact_face_partition_result(
            &result,
            &geometry.topology,
            &curves,
        )
        .unwrap();
        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: stable(91),
            streams: vec![MeshingChunkStream {
                media_type: MeshingChunkMediaType::SurfacePartitions,
                schema_version: runmat_meshing_surface::EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION,
                records: vec![encoded],
            }],
            final_checkpoint: MeshingStageCheckpoint {
                completed_work: 1,
                estimated_work: 1,
                ..MeshingStageCheckpoint::default()
            },
        })
    }
}

#[test]
fn serial_surface_partition_can_publish_a_curve_restart() {
    let (fixture, host, exact_root, joined, completed, _) =
        execute_surface_pipeline_with(&SurfaceRestartKernel);
    let exact = import_exact_geometry_input(
        &fixture.store,
        host.geometry_document.clone().unwrap(),
        &exact_root,
        host.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let curve_streams = joined
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap();
    let curves = runmat_meshing_curve::decode_shared_curve_mesh(
        &curve_streams[0].records[0],
        &exact.geometry_objects().topology,
    )
    .unwrap();
    let streams = completed
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap();
    let result = runmat_meshing_surface::decode_exact_face_partition_result(
        &streams[0].records[0],
        &exact.geometry_objects().topology,
        &curves,
    )
    .unwrap();
    assert!(matches!(
        result.outcome,
        runmat_meshing_surface::ExactFacePartitionOutcome::RequiresCurveSplits { .. }
    ));
}

#[test]
fn serial_surface_join_publishes_the_reconstructed_pass_decision() {
    let (fixture, host, exact_root, joined, partition, completed) =
        execute_surface_join_pipeline(&ExactSurfacePartitionKernel::default());
    let exact = import_exact_geometry_input(
        &fixture.store,
        host.geometry_document.clone().unwrap(),
        &exact_root,
        host.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let curve_streams = joined
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap();
    let curves = runmat_meshing_curve::decode_shared_curve_mesh(
        &curve_streams[0].records[0],
        &exact.geometry_objects().topology,
    )
    .unwrap();
    let partition_streams = partition
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap();
    let partition_result = runmat_meshing_surface::decode_exact_face_partition_result(
        &partition_streams[0].records[0],
        &exact.geometry_objects().topology,
        &curves,
    )
    .unwrap();
    let streams = completed
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap();
    assert_eq!(streams.len(), 2);
    assert_eq!(
        streams[0].media_type,
        runmat_meshing_core::MeshingChunkMediaType::SurfacePartitions
    );
    assert_eq!(
        streams[1].media_type,
        runmat_meshing_core::MeshingChunkMediaType::SurfaceMesh
    );
    let options = runmat_meshing_surface::ExactSurfaceJoinOptions {
        coordinate_tolerance_m: host.resolved_request.tolerance.absolute_floor_m,
        maximum_nodes: host.resolved_request.resources.maximum_nodes,
        maximum_triangles: host.resolved_request.resources.maximum_elements,
        maximum_boundary_segments: host.resolved_request.resources.maximum_elements,
    };
    let partition_results = [partition_result];
    let pass = runmat_meshing_surface::decode_exact_surface_pass_result(
        &streams[0].records[0],
        &exact.geometry_objects().topology,
        &curves,
        &partition_results,
        options,
    )
    .unwrap();
    let surface = runmat_meshing_surface::decode_exact_surface_mesh_from_pass(
        &streams[1].records[0],
        &pass,
        &exact.geometry_objects().topology,
        &curves,
        &partition_results,
        options,
    )
    .unwrap();
    let runmat_meshing_surface::ExactSurfacePassOutcome::Converged { surface: expected } =
        pass.outcome
    else {
        panic!("surface publication requires a converged pass")
    };
    assert_eq!(surface, expected);
}

#[test]
fn serial_curve_refinement_consumes_the_surface_restart_pass() {
    let (mut fixture, pass_host, exact_root, joined, partition, pass) =
        execute_surface_join_pipeline(&SurfaceRestartKernel);
    let curve_root = root(joined.publication().root_output());
    let partition_root = root(partition.publication().root_output());
    let pass_root = root(pass.publication().root_output());
    let exact_input = pass_host
        .workload
        .inputs
        .iter()
        .find(|input| input.kind == MeshingInputKind::ExactGeometry)
        .unwrap()
        .clone();
    let artifact_input = |root: &ValueRef| MeshingInputRef {
        kind: MeshingInputKind::StageArtifact,
        digest: StableDigest::from_bytes(*root.logical_digest.bytes()),
    };
    let request = pass_host.resolved_request.clone();
    let mut dependencies = vec![
        (exact_input, exact_root.clone()),
        (artifact_input(&curve_root), curve_root),
        (artifact_input(&partition_root), partition_root),
        (artifact_input(&pass_root), pass_root),
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
        stage: MeshingStageKind::CurveMesh,
        geometry: pass_host.stage_identity.geometry.clone(),
        resolved_request_digest: request.canonical_digest().unwrap(),
        tolerance_policy_digest: request.tolerance.canonical_digest().unwrap(),
        metric_policy_digest: request.metric.canonical_digest().unwrap(),
        algorithm_set_digest: request.algorithms.canonical_digest().unwrap(),
        deterministic_seed: request.deterministic_seed,
        prerequisites: inputs.clone(),
        capability_cohort: pass_host.stage_identity.capability_cohort.clone(),
    };
    let workload = MeshingWorkloadRequest {
        schema_version: MESHING_WORKLOAD_SCHEMA_VERSION,
        stage: MeshingStageKind::CurveMesh,
        stage_identity_digest: identity.canonical_digest().unwrap(),
        partition: MeshingPartitionDescriptor {
            kind: MeshingPartitionKind::WholeStage,
            partition_index: 0,
            partition_count: 1,
            entity_range: None,
        },
        inputs,
        required_capabilities: pass_host
            .workload
            .required_capabilities
            .iter()
            .map(|capability| match capability {
                MeshingCapabilityRequirement::MeshingAlgorithm { .. } => {
                    MeshingCapabilityRequirement::MeshingAlgorithm {
                        version: request.algorithms.curve.clone(),
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
        pass_host.artifact_access.clone(),
        pass_host.geometry_document.clone(),
    )
    .unwrap();
    let program = host.program_request(revision(), &roots).unwrap();
    let refined = execute_serial_stage(
        &program,
        &mut fixture.store,
        &crate::ExactCurveRefinementKernel::default(),
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
    let current_streams = joined
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap();
    let current = runmat_meshing_curve::decode_shared_curve_mesh(
        &current_streams[0].records[0],
        &exact.geometry_objects().topology,
    )
    .unwrap();
    let refined_streams = refined
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap();
    let refined = runmat_meshing_curve::decode_shared_curve_mesh(
        &refined_streams[0].records[0],
        &exact.geometry_objects().topology,
    )
    .unwrap();
    assert_eq!(
        refined.edges[0].nodes.len(),
        current.edges[0].nodes.len() + 1
    );
}

fn execute_surface_join_pipeline(
    partition_kernel: &dyn MeshingStageKernel,
) -> (
    Fixture,
    MeshingHostWorkload,
    ValueRef,
    CompletedMeshingStage,
    CompletedMeshingStage,
    CompletedMeshingStage,
) {
    execute_surface_join_pipeline_from_fixture(
        partition_kernel,
        Fixture::with_exact_curve_partition(),
    )
}

pub(super) fn execute_surface_join_pipeline_from_fixture(
    partition_kernel: &dyn MeshingStageKernel,
    fixture: Fixture,
) -> (
    Fixture,
    MeshingHostWorkload,
    ValueRef,
    CompletedMeshingStage,
    CompletedMeshingStage,
    CompletedMeshingStage,
) {
    let (mut fixture, partition_host, exact_root, joined, partition, _) =
        execute_surface_pipeline_with_fixture(partition_kernel, fixture);
    let curve_root = root(joined.publication().root_output());
    let partition_root = root(partition.publication().root_output());
    let exact = import_exact_geometry_input(
        &fixture.store,
        partition_host.geometry_document.clone().unwrap(),
        &exact_root,
        partition_host.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let planner =
        crate::ExactMeshingDagPlanner::from_exact_host(&partition_host, exact_root.clone())
            .unwrap();
    let pass = planner
        .begin_surface_pass(&exact.geometry_objects().topology, curve_root, 32)
        .unwrap();
    let join = planner.surface_join(&pass, vec![partition_root]).unwrap();
    let host = join.host().clone();
    let program = join.program_request(revision()).unwrap();
    let completed = execute_serial_stage(
        &program,
        &mut fixture.store,
        &crate::ExactSurfaceJoinKernel,
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(1_000_000),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    (fixture, host, exact_root, joined, partition, completed)
}
