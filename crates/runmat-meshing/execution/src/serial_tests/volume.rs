use super::*;

#[test]
fn production_dag_runner_executes_the_complete_serial_exact_pipeline() {
    let mut fixture = Fixture::with_exact_tetrahedron_curve_partition_order(ElementOrder::Tet10);
    let exact_root = root(&fixture.program.arguments[0]);
    let exact = import_exact_geometry_input(
        &fixture.store,
        fixture.host.geometry_document.clone().unwrap(),
        &exact_root,
        fixture.host.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let domain_objects = prepare_domain_model_objects(
        runmat_meshing_core::MeshingDomainModel {
            schema_version: MESHING_DOMAIN_MODEL_SCHEMA_VERSION,
            region_materials: vec![RegionMaterialAssignment {
                region_id: exact.geometry_objects().topology.regions[0].id.clone(),
                material_id: "steel".into(),
            }],
            contact_ids: Vec::new(),
        },
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let domain = prepare_domain_model_input(
        domain_objects,
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
        chunk_policy: chunk_policy(1_000_000),
        inventory_limits: ObjectInventoryLimits::default(),
    };
    let result = crate::execute_exact_meshing_dag(
        crate::ExactMeshingDagRun {
            geometry: &exact,
            domain_model: &domain,
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
        },
        &mut executor,
    )
    .unwrap();

    result.evidence.validate(&result.artifact).unwrap();
    assert_eq!(result.artifact.topology.nodes.len(), 10);
    assert_eq!(result.artifact.topology.volume_elements.len(), 1);
    assert_eq!(
        result.artifact.topology.volume_elements[0].material_id,
        "steel"
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
    let fixture = Fixture::with_exact_tetrahedron_curve_partition_order(ElementOrder::Tet10);
    let (mut fixture, surface_host, exact_root, curve_stage, surface_partition, surface_stage) =
        super::surface_restart::execute_surface_join_pipeline_from_fixture(
            &ExactSurfacePartitionKernel::default(),
            fixture,
        );
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
    let program = volume_stage.program_request(revision()).unwrap();
    let completed = execute_serial_stage(
        &program,
        &mut fixture.store,
        &crate::MeshingKernelDispatcher::default(),
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

    let domain_objects = prepare_domain_model_objects(
        runmat_meshing_core::MeshingDomainModel {
            schema_version: MESHING_DOMAIN_MODEL_SCHEMA_VERSION,
            region_materials: vec![RegionMaterialAssignment {
                region_id: exact.geometry_objects().topology.regions[0].id.clone(),
                material_id: "steel".into(),
            }],
            contact_ids: Vec::new(),
        },
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let domain_input = prepare_domain_model_input(
        domain_objects,
        host.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    for object in &domain_input.domain_model_objects().objects {
        fixture.store.write_verified(object).unwrap();
    }
    let domain_root = domain_input.root_input().clone();
    let volume_root = root(completed.publication().root_output());
    let projection_stage = planner
        .solver_projection(surface_root, volume_root, domain_root)
        .unwrap();
    assert_eq!(
        projection_stage.host().workload.stage,
        MeshingStageKind::OrderElevation
    );
    assert_eq!(projection_stage.host().workload.inputs.len(), 4);
    let projection = execute_serial_stage(
        &projection_stage.program_request(revision()).unwrap(),
        &mut fixture.store,
        &crate::MeshingKernelDispatcher::default(),
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(1_000_000),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
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
    assert_eq!(solver_projection.topology.nodes.len(), 10);
    assert_eq!(solver_projection.topology.volume_elements.len(), 1);
    assert_eq!(
        solver_projection.topology.volume_elements[0].material_id,
        "steel"
    );

    let projection_root = root(projection.publication().root_output());
    let validation_stage = planner.solver_validation(projection_root.clone()).unwrap();
    let validation = execute_serial_stage(
        &validation_stage.program_request(revision()).unwrap(),
        &mut fixture.store,
        &crate::MeshingKernelDispatcher::default(),
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(1_000_000),
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
        &serialization_stage.program_request(revision()).unwrap(),
        &mut fixture.store,
        &crate::MeshingKernelDispatcher::default(),
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(1_000_000),
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
        &mismatched_publication.program_request(revision()).unwrap(),
        &mut fixture.store,
        &crate::MeshingKernelDispatcher::default(),
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(1_000_000),
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
    let publication_stage = planner
        .solver_publication(serialization_root, evidence_input.root_input().clone())
        .unwrap();
    assert_eq!(
        publication_stage.host().workload.stage,
        MeshingStageKind::Publication
    );
    let published = execute_serial_stage(
        &publication_stage.program_request(revision()).unwrap(),
        &mut fixture.store,
        &crate::MeshingKernelDispatcher::default(),
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(1_000_000),
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
        MeshingChunkMediaType::AnalysisMeshArtifact
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
