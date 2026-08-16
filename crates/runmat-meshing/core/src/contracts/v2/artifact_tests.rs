use std::collections::BTreeMap;

use super::*;

fn digest(byte: u8) -> StableDigest {
    StableDigest::from_bytes([byte; 32])
}

fn entity(kind: PersistentEntityKind, id: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: id.into(),
        assembly_path: vec!["root".into()],
    }
}

fn request() -> MeshingRequestV2 {
    MeshingRequestV2 {
        schema_version: MESHING_REQUEST_SCHEMA_VERSION,
        element_order: MeshElementOrderV2::Tet4,
        deterministic_seed: 17,
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
            source_tolerance_m: 1.0e-8,
            absolute_floor_m: 1.0e-10,
            model_relative_term: 1.0e-9,
            requested_deviation_m: 1.0e-5,
            maximum_healing_displacement_m: 1.0e-6,
        },
        metric: MetricFieldRequestV2 {
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(0.5).unwrap(),
            maximum_grading_ratio: 1.3,
            contributions: Vec::new(),
        },
        quality: MeshingQualityTargetsV2 {
            surface: SurfaceQualityTargetsV2 {
                minimum_metric_angle_degrees: 20.0,
                maximum_physical_aspect_ratio: 10.0,
                maximum_chordal_deviation_m: 1.0e-5,
                maximum_normal_deviation_degrees: 5.0,
            },
            volume: VolumeQualityTargetsV2 {
                maximum_radius_edge_ratio: 2.0,
                minimum_scaled_jacobian: 0.05,
                maximum_metric_edge_length: 1.5,
            },
        },
        resources: MeshingResourceBudgetV2 {
            maximum_nodes: 100,
            maximum_elements: 100,
            maximum_memory_bytes: 1_000_000,
            maximum_scratch_bytes: 1_000_000,
            maximum_wall_time_ms: 10_000,
            maximum_artifact_bytes: 1_000_000,
            maximum_search_work: 10_000,
            maximum_recursion_depth: 32,
            maximum_iterations: 10_000,
        },
        cancellation: CancellationPolicyV2 {
            maximum_checkpoint_latency_ms: 100,
            maximum_work_units_between_checks: 100,
        },
    }
}

fn artifact() -> AnalysisMeshArtifactV2 {
    let solid = entity(PersistentEntityKind::Solid, "solid:1");
    let region = entity(PersistentEntityKind::Region, "region:1");
    let nodes = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    .into_iter()
    .enumerate()
    .map(|(index, coordinates_m)| AnalysisMeshNodeV2 {
        node_id: index as u64 + 1,
        coordinates_m,
        provenance: vec![solid.clone()],
    })
    .collect();
    let face_nodes = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]];
    let boundary_faces = face_nodes
        .into_iter()
        .enumerate()
        .map(|(index, node_ids)| AnalysisBoundaryFaceV2 {
            face_id: index as u64 + 10,
            order: BoundaryTriangleOrderV2::Tri3,
            node_ids: node_ids.into(),
            adjacent_volume_element_ids: vec![1],
            role: BoundaryFaceRoleV2::Exterior,
            provenance: vec![entity(PersistentEntityKind::Face, &format!("face:{index}"))],
        })
        .collect();
    let boundary_edges = [
        ([1, 2], vec![10, 11]),
        ([1, 3], vec![10, 12]),
        ([1, 4], vec![11, 12]),
        ([2, 3], vec![10, 13]),
        ([2, 4], vec![11, 13]),
        ([3, 4], vec![12, 13]),
    ]
    .into_iter()
    .enumerate()
    .map(
        |(index, (node_ids, adjacent_boundary_face_ids))| AnalysisBoundaryEdgeV2 {
            edge_id: index as u64 + 20,
            node_ids,
            adjacent_boundary_face_ids,
            provenance: vec![entity(PersistentEntityKind::Edge, &format!("edge:{index}"))],
        },
    )
    .collect();
    AnalysisMeshArtifactV2 {
        schema_version: ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION,
        canonical_digest: digest(1),
        root_stage_manifest_digest: digest(2),
        geometry: GeometryRevisionRef {
            source_digest: digest(3),
            geometry_revision: 4,
            persistent_mapping_version: 2,
        },
        resolved_request: request(),
        topology: AnalysisMeshTopologyV2 {
            nodes,
            volume_elements: vec![AnalysisVolumeElementV2 {
                element_id: 1,
                order: MeshElementOrderV2::Tet4,
                node_ids: vec![1, 2, 3, 4],
                region_id: region.clone(),
                material_id: "steel".into(),
                provenance: vec![solid],
            }],
            neighbors: (0..4)
                .map(|local_face_index| MeshNeighborV2 {
                    element_id: 1,
                    local_face_index,
                    adjacent_element_id: None,
                })
                .collect(),
            boundary_faces,
            boundary_edges,
            regions: vec![MeshRegionV2 {
                region_id: region,
                material_id: "steel".into(),
                element_ids: vec![1],
            }],
            material_interfaces: Vec::new(),
            contacts: Vec::new(),
            field_topologies: vec![
                FieldTopologyMapV2 {
                    topology_id: "nodes".into(),
                    location: FieldTopologyLocationV2::Node,
                    ordered_entity_ids: vec![1, 2, 3, 4],
                },
                FieldTopologyMapV2 {
                    topology_id: "elements".into(),
                    location: FieldTopologyLocationV2::VolumeElement,
                    ordered_entity_ids: vec![1],
                },
                FieldTopologyMapV2 {
                    topology_id: "boundary_faces".into(),
                    location: FieldTopologyLocationV2::BoundaryFace,
                    ordered_entity_ids: vec![10, 11, 12, 13],
                },
                FieldTopologyMapV2 {
                    topology_id: "boundary_edges".into(),
                    location: FieldTopologyLocationV2::BoundaryEdge,
                    ordered_entity_ids: vec![20, 21, 22, 23, 24, 25],
                },
            ],
        },
    }
}

fn evidence(artifact: &AnalysisMeshArtifactV2) -> MeshingEvidenceV2 {
    MeshingEvidenceV2 {
        schema_version: MESHING_EVIDENCE_SCHEMA_VERSION,
        geometry: artifact.geometry.clone(),
        resolved_request_digest: digest(4),
        artifact_digest: artifact.canonical_digest,
        algorithms: artifact.resolved_request.algorithms.clone(),
        deterministic_seed: artifact.resolved_request.deterministic_seed,
        platform: PlatformBuildIdentityV2 {
            capability_cohort: "native-exact-cad-v1".into(),
            target_triple: "x86_64-unknown-linux-gnu".into(),
            build_digest: digest(5),
            exact_kernel_abi: Some("opencascade-7.9".into()),
        },
        stages: MeshingStageV2::ALL
            .into_iter()
            .enumerate()
            .map(|(index, stage)| StageEvidenceV2 {
                stage,
                stage_result_digest: digest(index as u8 + 10),
                entity_counts: BTreeMap::from([("accepted".into(), 1)]),
                invariants: vec![InvariantEvidenceV2 {
                    invariant_id: "stage_complete".into(),
                    passed: true,
                    measured: None,
                    required: None,
                    unit: None,
                }],
                achieved_error_distributions: BTreeMap::new(),
                completed_work: 1,
                estimated_work: 1,
                peak_memory_bytes: 100,
                elapsed_time_ms: 1,
                cancellation_checkpoints: 1,
            })
            .collect(),
        sizing: vec![SizingResolutionEvidenceV2 {
            scope: entity(PersistentEntityKind::Region, "region:1"),
            requested_size_m: 0.1,
            resolved_size_m: 0.1,
            achieved_maximum_size_m: 0.09,
            clipped_contribution_count: 0,
            rejected_contribution_count: 0,
        }],
        resources: MeshingResourceUsageV2 {
            generated_nodes: 4,
            generated_elements: 1,
            peak_memory_bytes: 100,
            peak_scratch_bytes: 10,
            wall_time_ms: 14,
            artifact_bytes: 1_000,
            search_work: 30,
            maximum_recursion_depth: 2,
            iterations: 8,
        },
        cache_admission: CacheAdmissionDecisionV2::Admitted,
    }
}

#[test]
fn artifact_and_evidence_round_trip_with_complete_canonical_topology() {
    let artifact = artifact();
    artifact.validate().unwrap();
    let evidence = evidence(&artifact);
    evidence.validate(&artifact).unwrap();

    let encoded = serde_json::to_vec(&artifact).unwrap();
    assert_eq!(
        serde_json::from_slice::<AnalysisMeshArtifactV2>(&encoded).unwrap(),
        artifact
    );
    let encoded = serde_json::to_vec(&evidence).unwrap();
    assert_eq!(
        serde_json::from_slice::<MeshingEvidenceV2>(&encoded).unwrap(),
        evidence
    );
}

#[test]
fn artifact_rejects_noncanonical_or_dangling_connectivity() {
    let mut invalid = artifact();
    invalid.topology.volume_elements[0].node_ids[3] = 99;
    assert_eq!(invalid.validate().unwrap_err().field, "volume element");

    let mut invalid = artifact();
    invalid.topology.boundary_edges.swap(0, 1);
    assert_eq!(invalid.validate().unwrap_err().field, "boundary edges");

    let mut invalid = artifact();
    invalid.topology.regions[0].element_ids.clear();
    assert_eq!(invalid.validate().unwrap_err().field, "mesh region");
}

#[test]
fn successful_evidence_requires_every_stage_and_respects_hard_budgets() {
    let artifact = artifact();
    let mut invalid = evidence(&artifact);
    invalid.stages.pop();
    assert_eq!(
        invalid.validate(&artifact).unwrap_err().field,
        "meshing evidence"
    );

    let mut invalid = evidence(&artifact);
    invalid.resources.generated_elements = 101;
    assert_eq!(
        invalid.validate(&artifact).unwrap_err().field,
        "meshing resource usage"
    );

    let mut invalid = evidence(&artifact);
    invalid.artifact_digest = digest(99);
    assert_eq!(
        invalid.validate(&artifact).unwrap_err().field,
        "meshing evidence"
    );
}

#[test]
fn v2_artifact_rejects_unknown_legacy_backend_evidence() {
    let mut value = serde_json::to_value(artifact()).unwrap();
    value["backend"] = serde_json::json!("structured_grid_tetrahedron");
    assert!(serde_json::from_value::<AnalysisMeshArtifactV2>(value).is_err());
}
