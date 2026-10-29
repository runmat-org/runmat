mod diagnostic;
mod invariants;
use super::*;
use crate::{ExactMeshingEvaluatorProvider, PortableMeshingEvaluatorProvider};
use std::sync::Arc;

#[test]
fn production_dag_runner_executes_the_complete_serial_exact_pipeline() {
    let mut fixture = Fixture::with_exact_tetrahedron_curve_partition_order(ElementOrder::Tet10);
    let policy = workflow_chunk_policy(&fixture.host.resolved_request);
    let exact_root = root(&fixture.program.arguments[1]);
    let exact = import_exact_geometry_input(
        &fixture.store,
        fixture.host.geometry_document.clone().unwrap(),
        &exact_root,
        fixture.host.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let request = fixture.host.resolved_request.clone();
    let access = fixture.host.artifact_access.clone();
    let cohort = fixture.host.stage_identity.capability_cohort.clone();
    let mut progress = Progress::default();
    let mut executor = crate::SerialExactMeshingExecutor {
        store: &mut fixture.store,
        kernel: &crate::MeshingKernelDispatcher::default(),
        cancellation: &NeverCancelled,
        progress: &mut progress,
        chunk_policy: policy,
        inventory_limits: ObjectInventoryLimits::default(),
    };
    let result = crate::execute_exact_meshing_dag(
        crate::ExactMeshingDagRun {
            geometry: &exact,
            ceilings: runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT,
            request,
            artifact_access: access,
            capability_cohort: cohort,
            program_revision: revision(),
            preferred_edges_per_partition: 3,
            preferred_faces_per_partition: 32,
            inventory_limits: ObjectInventoryLimits::default(),
            evidence: crate::MeshingRunEvidenceContext {
                platform: runmat_meshing_core::PlatformBuildIdentity {
                    capability_cohort: "serial-exact-test".into(),
                    target_triple: "portable-test-host".into(),
                    build_digest: StableDigest::from_bytes(
                        *Digest::sha256(b"meshing-test-build").bytes(),
                    ),
                    exact_kernel_abi: Some("portable-exact-test".into()),
                },
                sizing: Vec::new(),
                cache_admission: runmat_meshing_core::CacheAdmissionDecision::Admitted,
            },
            named_regions: Default::default(),
        },
        &mut executor,
    )
    .unwrap();

    result.evidence.validate(&result.artifact).unwrap();
    invariants::assert_unit_tetrahedron(
        &result.artifact.topology,
        &result.artifact.resolved_request,
    );
    assert_eq!(
        result.evidence.stages.last().unwrap().stage,
        MeshingStageKind::Serialization
    );
    assert!(result
        .result_objects
        .iter()
        .any(|object| object.media_type == crate::STAGE_EVIDENCE_MEDIA_TYPE));
    assert!(!progress.0.is_empty());
}

#[test]
fn serial_dispatcher_publishes_volume_and_canonical_solver_projection() {
    let ceilings = runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT;
    let fixture = Fixture::with_exact_tetrahedron_curve_partition_order(ElementOrder::Tet10);
    let policy = workflow_chunk_policy(&fixture.host.resolved_request);
    let (mut fixture, surface_host, exact_root, curve_stage, surface_partition, surface_stage) =
        super::surface_restart::convergence::execute_from_fixture(fixture);
    let surface_root = root(surface_stage.publication().root_output());
    let planner =
        crate::ExactMeshingDagPlanner::from_exact_host(&surface_host, exact_root.clone()).unwrap();
    let volume_stage = planner.tetrahedralization(surface_root.clone()).unwrap();
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
    let program = stage_program(&volume_stage, &mut fixture.store);
    let diagnostic_store = Arc::new(crate::subtask_cache::tests::Store::default());
    let kernel =
        crate::MeshingKernelDispatcher::default().with_subtask_cache(diagnostic_store.clone());
    let outcome = execute_serial_stage(
        &program,
        &mut fixture.store,
        &kernel,
        &NeverCancelled,
        &mut Progress::default(),
        policy,
        ObjectInventoryLimits::default(),
    );
    if outcome.is_err() {
        diagnostic::retain_failure(&diagnostic_store, &host.resolved_request);
    }
    let completed = outcome.unwrap();

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
        &surface_streams[0].records[0],
        &exact.geometry_objects().topology,
        runmat_meshing_surface::ExactSurfaceJoinOptions {
            coordinate_tolerance_m: runmat_types::physical::Length::new(
                host.resolved_request
                    .tolerance
                    .absolute_floor_m
                    .canonical_value(),
            ),
            maximum_nodes: host.resolved_request.limits.maximum_nodes,
            maximum_triangles: host.resolved_request.limits.maximum_elements,
            maximum_boundary_segments: host.resolved_request.limits.maximum_elements,
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
    let volume_metric = {
        let mut metric_progress = Progress::default();
        let stage_control = MeshingStageControl::new(
            MeshingStageKind::OrderElevation,
            0,
            &host.resolved_request,
            ceilings,
            &NeverCancelled,
            &mut metric_progress,
        )
        .unwrap();
        let control = stage_control.geometry_evaluation_control();
        let provider = PortableMeshingEvaluatorProvider;
        let evaluator = provider
            .evaluator(exact.geometry_objects(), &control)
            .unwrap();
        provider
            .geometry_metric(
                exact.geometry_objects(),
                evaluator.as_ref(),
                &control,
                &host.resolved_request,
            )
            .unwrap()
    };
    let mesh = runmat_meshing_tetrahedron::cdt::decode_delaunay_volume_mesh(
        &streams[0].records[0],
        &exact.geometry_objects().topology,
        &surface,
        &volume_metric,
        crate::volume_kernel::volume_options(&host.resolved_request),
        &NeverCancelled,
    )
    .unwrap();
    assert!(!mesh.topology.tetrahedra.is_empty());
    assert!(mesh.topology.tetrahedra.len() as u64 <= host.resolved_request.limits.maximum_elements);
    assert_eq!(mesh.topology.incidence.regions.len(), 1);
    invariants::assert_boundary_conservation(&mesh, &surface);

    // Read through the same bounded opaque persistence port used by the native
    // adapter. Only its manifest can yield a complete diagnostic; other cache
    // record kinds must not be mistaken for one.
    let keys = diagnostic_store
        .0
        .lock()
        .unwrap()
        .keys()
        .copied()
        .collect::<Vec<_>>();
    let mut diagnostic_progress = Progress::default();
    let diagnostic_control = MeshingStageControl::new(
        MeshingStageKind::Tetrahedralization,
        0,
        &host.resolved_request,
        ceilings,
        &NeverCancelled,
        &mut diagnostic_progress,
    )
    .unwrap();
    let control = diagnostic_control.geometry_evaluation_control();
    let records = keys
        .into_iter()
        .filter_map(|key| {
            crate::read_volume_diagnostic(diagnostic_store.as_ref(), key, &control)
                .ok()
                .flatten()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        records.len(),
        1,
        "production dispatcher did not retain exactly one complete quality input"
    );
    let codec = crate::codec_control::CodecControl::new(&control);
    let replayed = runmat_meshing_tetrahedron::cdt::replay_volume_quality(
        records[0].bytes(),
        runmat_meshing_tetrahedron::cdt::VolumeQualityBinding {
            request: host.resolved_request.canonical_digest().unwrap(),
            geometry: StableDigest::from_bytes(*exact.geometry_objects().root.digest.bytes()),
            implementation: crate::MeshingSubtaskCacheStore::volume_implementation_identity(
                diagnostic_store.as_ref(),
            )
            .unwrap(),
        },
        crate::volume_kernel::volume_options(&host.resolved_request),
        runmat_canonical_codec::CanonicalLimits::new(1024 * 1024, 100_000, 1_048_576, 64),
        &codec,
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(replayed.topology, mesh.topology);
    assert_eq!(replayed.quality, mesh.quality);
    drop(records);

    let volume_root = root(completed.publication().root_output());
    let projection_stage = planner
        .solver_projection(surface_root, volume_root)
        .unwrap();
    assert_eq!(
        projection_stage.host().workload.stage,
        MeshingStageKind::OrderElevation
    );
    assert_eq!(projection_stage.host().workload.inputs.len(), 3);
    // The compact projection is small; a narrower chunk keeps publication multi-chunk.
    let projection_policy = MeshingChunkPolicy {
        maximum_chunk_bytes: 64 * 1024,
        ..policy
    };
    let projection = execute_serial_stage(
        &stage_program(&projection_stage, &mut fixture.store),
        &mut fixture.store,
        &crate::MeshingKernelDispatcher::default(),
        &NeverCancelled,
        &mut Progress::default(),
        projection_policy,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert!(
        projection
            .publication()
            .stage_objects()
            .manifest
            .chunks
            .len()
            > 1,
        "the realistic Tet10 fixture must exercise multi-chunk publication"
    );
    let projection_streams = projection
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap();
    assert_eq!(projection_streams.len(), 1);
    assert_eq!(
        projection_streams[0].media_type,
        MeshingChunkMediaType::SolverMeshProjection
    );
    let solver_projection = runmat_meshing_core::SolverMeshProjection::canonical_decode(
        &projection_streams[0].records[0],
    )
    .unwrap();
    solver_projection.validate().unwrap();
    invariants::assert_unit_tetrahedron(&solver_projection.topology, &host.resolved_request);
    assert_eq!(
        solver_projection.topology.volume_elements.len(),
        mesh.topology.tetrahedra.len()
    );

    let projection_root = root(projection.publication().root_output());
    let validation_stage = planner.solver_validation(projection_root.clone()).unwrap();
    let validation = execute_serial_stage(
        &stage_program(&validation_stage, &mut fixture.store),
        &mut fixture.store,
        &crate::MeshingKernelDispatcher::default(),
        &NeverCancelled,
        &mut Progress::default(),
        policy,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let validation_streams = validation
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap();
    let validation_receipt = runmat_meshing_core::SolverMeshValidation::canonical_decode(
        &validation_streams[0].records[0],
    )
    .unwrap();
    validation_receipt
        .validate_against(&solver_projection)
        .unwrap();

    let validation_root = root(validation.publication().root_output());
    let serialization_stage = planner
        .solver_serialization(projection_root, validation_root.clone())
        .unwrap();
    let serialized = execute_serial_stage(
        &stage_program(&serialization_stage, &mut fixture.store),
        &mut fixture.store,
        &crate::MeshingKernelDispatcher::default(),
        &NeverCancelled,
        &mut Progress::default(),
        policy,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let artifact_streams = serialized
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap();
    let artifact =
        runmat_meshing_core::SolverMeshArtifact::canonical_decode(&artifact_streams[0].records[0])
            .unwrap();
    artifact.validate().unwrap();
    assert_eq!(
        artifact.root_stage_manifest_digest.bytes(),
        validation_root.logical_digest.bytes()
    );
    assert_eq!(artifact.topology, solver_projection.topology);

    let stage_evidence = vec![
        curve_stage.stage_evidence().clone(),
        surface_partition.stage_evidence().clone(),
        surface_stage.stage_evidence().clone(),
        completed.stage_evidence().clone(),
        projection.stage_evidence().clone(),
        validation.stage_evidence().clone(),
        serialized.stage_evidence().clone(),
    ];
    let serial_wall_time_ms = stage_evidence
        .iter()
        .map(|stage| stage.elapsed_time_ms)
        .sum();
    let evidence = crate::assemble_meshing_evidence(
        &artifact,
        stage_evidence,
        crate::MeshingEvidenceContext {
            platform: runmat_meshing_core::PlatformBuildIdentity {
                capability_cohort: "serial-exact-test".into(),
                target_triple: "portable-test-host".into(),
                build_digest: StableDigest::from_bytes(
                    *Digest::sha256(b"meshing-test-build").bytes(),
                ),
                exact_kernel_abi: Some("portable-exact-test".into()),
            },
            sizing: Vec::new(),
            cache_admission: runmat_meshing_core::CacheAdmissionDecision::Admitted,
            wall_time_ms: serial_wall_time_ms,
            ceilings: runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT,
        },
    )
    .unwrap();
    assert_eq!(
        evidence.stages.last().unwrap().stage_result_digest.bytes(),
        root(serialized.publication().root_output())
            .logical_digest
            .bytes()
    );
    let mut mismatched_evidence = evidence.clone();
    mismatched_evidence
        .stages
        .last_mut()
        .unwrap()
        .stage_result_digest = StableDigest::from_bytes([91; 32]);
    let mismatched_objects =
        crate::prepare_evidence_objects(mismatched_evidence, ObjectInventoryLimits::default())
            .unwrap();
    let mismatched_input = crate::prepare_evidence_input(
        mismatched_objects,
        host.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    for object in &mismatched_input.evidence_objects().objects {
        fixture.store.write_verified(object).unwrap();
    }
    let serialization_root = root(serialized.publication().root_output());
    let mismatched_publication = planner
        .solver_publication(
            serialization_root.clone(),
            mismatched_input.root_input().clone(),
        )
        .unwrap();
    assert!(execute_serial_stage(
        &stage_program(&mismatched_publication, &mut fixture.store),
        &mut fixture.store,
        &crate::MeshingKernelDispatcher::default(),
        &NeverCancelled,
        &mut Progress::default(),
        policy,
        ObjectInventoryLimits::default(),
    )
    .is_err());

    let evidence_objects =
        crate::prepare_evidence_objects(evidence, ObjectInventoryLimits::default()).unwrap();
    let evidence_input = crate::prepare_evidence_input(
        evidence_objects,
        host.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    for object in &evidence_input.evidence_objects().objects {
        fixture.store.write_verified(object).unwrap();
    }
    let evidence_reads = crate::tests::read_buffers::ReadBuffers::new(&fixture.store);
    let cold_evidence = crate::import_evidence_input(
        &evidence_reads,
        evidence_input.root_input(),
        host.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    evidence_reads.assert_retained(&cold_evidence.evidence_objects().objects);
    assert_eq!(cold_evidence, evidence_input);
    drop(evidence_reads);
    let publication_stage = planner
        .solver_publication(serialization_root, evidence_input.root_input().clone())
        .unwrap();
    assert_eq!(
        publication_stage.host().workload.stage,
        MeshingStageKind::Publication
    );
    let published = execute_serial_stage(
        &stage_program(&publication_stage, &mut fixture.store),
        &mut fixture.store,
        &crate::MeshingKernelDispatcher::default(),
        &NeverCancelled,
        &mut Progress::default(),
        policy,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let final_streams = published
        .publication()
        .stage_objects()
        .decoded_streams()
        .unwrap();
    assert_eq!(final_streams.len(), 2);
    assert_eq!(
        final_streams[0].media_type,
        MeshingChunkMediaType::SolverMeshArtifact
    );
    assert_eq!(
        final_streams[1].media_type,
        MeshingChunkMediaType::MeshingEvidence
    );
    let final_artifact =
        runmat_meshing_core::SolverMeshArtifact::canonical_decode(&final_streams[0].records[0])
            .unwrap();
    let final_evidence =
        runmat_meshing_core::MeshingEvidence::canonical_decode(&final_streams[1].records[0])
            .unwrap();
    final_evidence.validate(&final_artifact).unwrap();
    assert_eq!(final_artifact, artifact);
}

// Total artifact admission belongs to the original resolved workload. A chunk
// size is a transport bound; it must not silently become a one-chunk total cap.
fn workflow_chunk_policy(_request: &runmat_meshing_core::MeshingRequest) -> MeshingChunkPolicy {
    let ceilings = runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT;
    MeshingChunkPolicy {
        maximum_total_encoded_bytes: ceilings.maximum_artifact_bytes,
        ..chunk_policy(1_000_000)
    }
}

// witness: FEA-MESH-EXECUTION-001/budgets_and_cancellation @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Execution ceilings decide only whether a run completes; the same request meshed
//   under 512 MiB and 8 GiB ceilings publishes the same artifact and digest.
#[test]
fn execution_ceilings_do_not_change_the_published_mesh() {
    let run = |ceilings: runmat_meshing_core::MeshingExecutionCeilings| {
        let mut fixture =
            Fixture::with_exact_tetrahedron_curve_partition_order(ElementOrder::Tet10);
        let policy = workflow_chunk_policy(&fixture.host.resolved_request);
        let exact_root = root(&fixture.program.arguments[1]);
        let exact = import_exact_geometry_input(
            &fixture.store,
            fixture.host.geometry_document.clone().unwrap(),
            &exact_root,
            fixture.host.artifact_access.clone(),
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let request = fixture.host.resolved_request.clone();
        let access = fixture.host.artifact_access.clone();
        let cohort = fixture.host.stage_identity.capability_cohort.clone();
        let mut progress = Progress::default();
        let mut executor = crate::SerialExactMeshingExecutor {
            store: &mut fixture.store,
            kernel: &crate::MeshingKernelDispatcher::default(),
            cancellation: &NeverCancelled,
            progress: &mut progress,
            chunk_policy: policy,
            inventory_limits: ObjectInventoryLimits::default(),
        };
        crate::execute_exact_meshing_dag(
            crate::ExactMeshingDagRun {
                geometry: &exact,
                ceilings,
                request,
                artifact_access: access,
                capability_cohort: cohort,
                program_revision: revision(),
                preferred_edges_per_partition: 3,
                preferred_faces_per_partition: 32,
                inventory_limits: ObjectInventoryLimits::default(),
                evidence: crate::MeshingRunEvidenceContext {
                    platform: runmat_meshing_core::PlatformBuildIdentity {
                        capability_cohort: "serial-exact-test".into(),
                        target_triple: "portable-test-host".into(),
                        build_digest: StableDigest::from_bytes(
                            *Digest::sha256(b"meshing-test-build").bytes(),
                        ),
                        exact_kernel_abi: Some("portable-exact-test".into()),
                    },
                    sizing: Vec::new(),
                    cache_admission: runmat_meshing_core::CacheAdmissionDecision::Admitted,
                },
                named_regions: Default::default(),
            },
            &mut executor,
        )
        .unwrap()
    };
    let portable = runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT;
    let large = runmat_meshing_core::MeshingExecutionCeilings {
        maximum_memory_bytes: 8 << 30,
        maximum_scratch_bytes: 8 << 30,
        maximum_wall_time_ms: 7_200_000,
        maximum_artifact_bytes: 4 << 30,
        maximum_search_work: 1 << 40,
        maximum_iterations: 1 << 40,
    };
    let first = run(portable);
    let second = run(large);
    assert_eq!(
        first.artifact.canonical_digest,
        second.artifact.canonical_digest
    );
    assert_eq!(first.artifact, second.artifact);
    assert_eq!(first.evidence.ceilings, portable);
    assert_eq!(second.evidence.ceilings, large);
    second.evidence.validate(&second.artifact).unwrap();
}
