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

pub(super) fn request() -> MeshingRequest {
    MeshingRequest {
        schema_version: MESHING_REQUEST_SCHEMA_VERSION,
        element_order: ElementOrder::Tet4,
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
        metric: MetricFieldRequest {
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(0.5).unwrap(),
            maximum_grading_ratio: 1.3,
            contributions: Vec::new(),
        },
        quality: MeshingQualityTargets {
            curve: CurveQualityTargets {
                maximum_chordal_deviation_m: 1.0e-5,
                maximum_tangent_change_degrees: 5.0,
                minimum_metric_edge_length: 0.1,
                maximum_metric_edge_length: 1.5,
            },
            surface: SurfaceQualityTargets {
                minimum_metric_angle_degrees: 20.0,
                maximum_physical_aspect_ratio: 10.0,
                maximum_chordal_deviation_m: 1.0e-5,
                maximum_normal_deviation_degrees: 5.0,
            },
            volume: VolumeQualityTargets {
                maximum_radius_edge_ratio: 2.0,
                minimum_scaled_jacobian: 0.05,
                maximum_metric_edge_length: 1.5,
            },
        },
        resources: MeshingResourceBudget {
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
        cancellation: CancellationPolicy {
            maximum_checkpoint_latency_ms: 100,
            maximum_work_units_between_checks: 100,
        },
    }
}

pub(super) fn artifact() -> SolverMeshArtifact {
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
    .map(|(index, coordinates_m)| SolverMeshNode {
        node_id: index as u64 + 1,
        stable_identity: digest(index as u8 + 1),
        coordinates_m,
        provenance: vec![solid.clone()],
        exact_parameters: Vec::new(),
    })
    .collect();
    let face_nodes = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]];
    let boundary_faces = face_nodes
        .into_iter()
        .enumerate()
        .map(|(index, node_ids)| SolverBoundaryFace {
            face_id: index as u64 + 10,
            stable_identity: solver_boundary_face_identity(node_ids.map(|id| digest(id as u8))),
            order: BoundaryTriangleOrder::Tri3,
            node_ids: node_ids.into(),
            adjacent_volume_element_ids: vec![1],
            role: BoundaryFaceRole::Exterior,
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
        |(index, (node_ids, adjacent_boundary_face_ids))| SolverBoundaryEdge {
            edge_id: index as u64 + 20,
            stable_identity: solver_boundary_edge_identity(node_ids.map(|id| digest(id as u8))),
            order: BoundaryEdgeOrder::Line2,
            node_ids: node_ids.into(),
            adjacent_boundary_face_ids,
            provenance: vec![entity(PersistentEntityKind::Edge, &format!("edge:{index}"))],
        },
    )
    .collect();
    let mut artifact = SolverMeshArtifact {
        schema_version: ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION,
        canonical_digest: digest(1),
        root_stage_manifest_digest: digest(2),
        geometry: GeometryRevisionRef {
            source_digest: digest(3),
            geometry_revision: 4,
            persistent_mapping_version: 2,
        },
        resolved_request: request(),
        topology: SolverMeshTopology {
            nodes,
            volume_elements: vec![SolverVolumeElement {
                element_id: 1,
                stable_identity: solver_volume_element_identity([
                    digest(1),
                    digest(2),
                    digest(3),
                    digest(4),
                ]),
                order: ElementOrder::Tet4,
                node_ids: vec![1, 2, 3, 4],
                region_id: region.clone(),
                material_id: "steel".into(),
                provenance: vec![solid],
            }],
            neighbors: (0..4)
                .map(|local_face_index| MeshNeighbor {
                    element_id: 1,
                    local_face_index,
                    adjacent_element_id: None,
                })
                .collect(),
            boundary_faces,
            boundary_edges,
            regions: vec![MeshRegion {
                region_id: region,
                material_id: "steel".into(),
                element_ids: vec![1],
            }],
            material_interfaces: Vec::new(),
            contacts: Vec::new(),
            field_topologies: vec![
                FieldTopologyMap {
                    topology_id: "nodes".into(),
                    location: FieldTopologyLocation::Node,
                    ordered_entity_ids: vec![1, 2, 3, 4],
                },
                FieldTopologyMap {
                    topology_id: "elements".into(),
                    location: FieldTopologyLocation::VolumeElement,
                    ordered_entity_ids: vec![1],
                },
                FieldTopologyMap {
                    topology_id: "boundary_faces".into(),
                    location: FieldTopologyLocation::BoundaryFace,
                    ordered_entity_ids: vec![10, 11, 12, 13],
                },
                FieldTopologyMap {
                    topology_id: "boundary_edges".into(),
                    location: FieldTopologyLocation::BoundaryEdge,
                    ordered_entity_ids: vec![20, 21, 22, 23, 24, 25],
                },
            ],
        },
    };
    artifact.seal_canonical_digest().unwrap();
    artifact
}

pub(super) fn evidence(artifact: &SolverMeshArtifact) -> MeshingEvidence {
    MeshingEvidence {
        schema_version: MESHING_EVIDENCE_SCHEMA_VERSION,
        geometry: artifact.geometry.clone(),
        resolved_request_digest: artifact.resolved_request.canonical_digest().unwrap(),
        artifact_digest: artifact.canonical_digest,
        algorithms: artifact.resolved_request.algorithms.clone(),
        deterministic_seed: artifact.resolved_request.deterministic_seed,
        platform: PlatformBuildIdentity {
            capability_cohort: "native-exact-cad-v1".into(),
            target_triple: "x86_64-unknown-linux-gnu".into(),
            build_digest: digest(5),
            exact_kernel_abi: Some("opencascade-7.9".into()),
        },
        stages: MeshingStageKind::ALL
            .into_iter()
            .take(MeshingStageKind::ALL.len() - 1)
            .enumerate()
            .map(|(index, stage)| MeshingStageEvidence {
                stage,
                partition: MeshingPartitionDescriptor {
                    kind: MeshingPartitionKind::WholeStage,
                    partition_index: 0,
                    partition_count: 1,
                    entity_range: None,
                },
                stage_result_digest: digest(index as u8 + 10),
                entity_counts: BTreeMap::from([("accepted".into(), 1)]),
                invariants: vec![InvariantEvidence {
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
                peak_scratch_bytes: 10,
                search_work: 1,
                maximum_recursion_depth: 1,
                iterations: 1,
                elapsed_time_ms: 1,
                cancellation_checkpoints: 1,
            })
            .collect(),
        sizing: vec![SizingResolutionEvidence {
            scope: entity(PersistentEntityKind::Region, "region:1"),
            requested_size_m: 0.1,
            resolved_size_m: 0.1,
            achieved_maximum_size_m: 0.09,
            clipped_contribution_count: 0,
            rejected_contribution_count: 0,
        }],
        resources: MeshingResourceUsage {
            generated_nodes: 4,
            generated_elements: 1,
            peak_memory_bytes: 100,
            peak_scratch_bytes: 10,
            wall_time_ms: 14,
            artifact_bytes: artifact.canonical_encode().unwrap().len() as u64,
            search_work: 13,
            maximum_recursion_depth: 1,
            iterations: 13,
        },
        cache_admission: CacheAdmissionDecision::Admitted,
    }
}

#[test]
fn artifact_identity_excludes_the_physical_validation_manifest_layout() {
    let first = artifact();
    let mut second = first.clone();
    second.root_stage_manifest_digest = digest(99);
    second.seal_canonical_digest().unwrap();

    assert_eq!(first.canonical_digest, second.canonical_digest);
    assert_ne!(
        first.canonical_encode().unwrap(),
        second.canonical_encode().unwrap()
    );
    first.validate().unwrap();
    second.validate().unwrap();
}

#[test]
fn artifact_and_evidence_round_trip_with_complete_canonical_topology() {
    let artifact = artifact();
    artifact.validate().unwrap();
    let evidence = evidence(&artifact);
    evidence.validate(&artifact).unwrap();

    let encoded = serde_json::to_vec(&artifact).unwrap();
    assert_eq!(
        serde_json::from_slice::<SolverMeshArtifact>(&encoded).unwrap(),
        artifact
    );
    let encoded = serde_json::to_vec(&evidence).unwrap();
    assert_eq!(
        serde_json::from_slice::<MeshingEvidence>(&encoded).unwrap(),
        evidence
    );
}

#[test]
fn solver_projection_validation_forms_an_acyclic_final_artifact_chain() {
    let expected = artifact();
    let projection = SolverMeshProjection {
        schema_version: SOLVER_MESH_PROJECTION_SCHEMA_VERSION,
        geometry: expected.geometry.clone(),
        resolved_request: expected.resolved_request.clone(),
        topology: expected.topology.clone(),
    };
    let encoded = projection.canonical_encode().unwrap();
    let decoded = SolverMeshProjection::canonical_decode(&encoded).unwrap();
    assert_eq!(decoded, projection);

    let validation = SolverMeshValidation::from_validated_projection(&projection).unwrap();
    validation.validate_against(&projection).unwrap();
    assert_eq!(
        SolverMeshValidation::canonical_decode(&validation.canonical_encode().unwrap()).unwrap(),
        validation
    );

    let validation_manifest_digest = digest(71);
    let artifact = projection
        .clone()
        .into_artifact(validation_manifest_digest)
        .unwrap();
    artifact.validate().unwrap();
    assert_eq!(
        artifact.root_stage_manifest_digest,
        validation_manifest_digest
    );
    assert_eq!(artifact.topology, projection.topology);

    let mut tampered = validation;
    tampered.boundary_face_count += 1;
    assert!(tampered.validate_against(&projection).is_err());
}

#[test]
fn solver_projection_and_validation_reject_incompatible_contracts() {
    let source = artifact();
    let mut projection = SolverMeshProjection {
        schema_version: SOLVER_MESH_PROJECTION_SCHEMA_VERSION,
        geometry: source.geometry,
        resolved_request: source.resolved_request,
        topology: source.topology,
    };
    let validation = SolverMeshValidation::from_validated_projection(&projection).unwrap();

    projection.schema_version += 1;
    assert!(projection.validate().is_err());

    let mut invalid = validation;
    invalid.projection_digest = StableDigest::ZERO;
    assert!(invalid.validate().is_err());
}

#[test]
fn artifact_rejects_noncanonical_or_dangling_connectivity() {
    let mut invalid = artifact();
    invalid.topology.nodes[0].stable_identity = StableDigest::ZERO;
    assert_eq!(invalid.validate().unwrap_err().field, "mesh nodes");

    let mut invalid = artifact();
    invalid.topology.nodes[1].stable_identity = invalid.topology.nodes[0].stable_identity;
    assert_eq!(invalid.validate().unwrap_err().field, "mesh nodes");

    let mut invalid = artifact();
    invalid.topology.volume_elements[0].stable_identity = digest(99);
    assert_eq!(
        invalid.validate().unwrap_err().field,
        "volume element stable identity"
    );

    let mut invalid = artifact();
    invalid.topology.boundary_faces[0].stable_identity = digest(99);
    assert_eq!(
        invalid.validate().unwrap_err().field,
        "boundary face stable identity"
    );

    let mut invalid = artifact();
    invalid.topology.boundary_edges[0].stable_identity = digest(99);
    assert_eq!(
        invalid.validate().unwrap_err().field,
        "boundary edge stable identity"
    );

    let mut invalid = artifact();
    invalid.topology.volume_elements[0].node_ids[3] = 99;
    assert_eq!(invalid.validate().unwrap_err().field, "volume element");

    let mut invalid = artifact();
    invalid.topology.boundary_edges.swap(0, 1);
    assert_eq!(invalid.validate().unwrap_err().field, "boundary edges");

    let mut invalid = artifact();
    invalid.topology.regions[0].element_ids.clear();
    assert_eq!(invalid.validate().unwrap_err().field, "mesh region");

    let mut invalid = artifact();
    invalid.topology.volume_elements[0].node_ids[3] = 1;
    assert_eq!(invalid.validate().unwrap_err().field, "volume element");

    let mut invalid = artifact();
    invalid.topology.boundary_faces[0].role = BoundaryFaceRole::MaterialInterface;
    assert_eq!(invalid.validate().unwrap_err().field, "boundary face");

    let mut invalid = artifact();
    invalid.topology.boundary_edges[0].adjacent_boundary_face_ids = vec![12];
    assert_eq!(invalid.validate().unwrap_err().field, "boundary edge");
}

#[test]
fn solver_node_exact_parameters_are_typed_bounded_and_canonical() {
    let edge = entity(PersistentEntityKind::Edge, "edge:exact");
    let face = entity(PersistentEntityKind::Face, "face:exact");
    let chart = digest(42);
    let mut valid = artifact();
    valid.topology.nodes[0]
        .provenance
        .extend([edge.clone(), face.clone()]);
    valid.topology.nodes[0].provenance.sort();
    valid.topology.nodes[0].exact_parameters = vec![
        SolverNodeExactParameter::Curve {
            source_edge_id: edge.clone(),
            parameter: 0.25,
        },
        SolverNodeExactParameter::Surface {
            source_face_id: face.clone(),
            chart_id: chart,
            evaluator_uv: [0.1, 0.2],
        },
    ];
    let validate = |candidate: &SolverMeshArtifact| {
        validate_solver_mesh_topology(&candidate.topology, &candidate.resolved_request)
    };
    validate(&valid).unwrap();

    let mut invalid = valid.clone();
    invalid.topology.nodes[0].exact_parameters.swap(0, 1);
    assert_eq!(
        validate(&invalid).unwrap_err().field,
        "mesh node exact parameters"
    );

    let mut invalid = valid.clone();
    invalid.topology.nodes[0].exact_parameters[0] = SolverNodeExactParameter::Curve {
        source_edge_id: edge.clone(),
        parameter: f64::NAN,
    };
    assert_eq!(
        validate(&invalid).unwrap_err().field,
        "mesh node exact parameters"
    );

    let mut invalid = valid.clone();
    invalid.topology.nodes[0].exact_parameters[0] = SolverNodeExactParameter::Curve {
        source_edge_id: face.clone(),
        parameter: 0.25,
    };
    assert_eq!(
        validate(&invalid).unwrap_err().field,
        "mesh node exact parameters"
    );

    let mut invalid = valid.clone();
    invalid.topology.nodes[0]
        .provenance
        .retain(|entity| entity != &edge);
    assert_eq!(
        validate(&invalid).unwrap_err().field,
        "mesh node exact parameters"
    );

    let mut invalid = valid.clone();
    invalid.topology.nodes[0].exact_parameters[1] = SolverNodeExactParameter::Surface {
        source_face_id: face.clone(),
        chart_id: StableDigest::ZERO,
        evaluator_uv: [0.1, 0.2],
    };
    assert_eq!(
        validate(&invalid).unwrap_err().field,
        "mesh node exact parameters"
    );

    let mut invalid = valid;
    invalid.topology.nodes[0].exact_parameters = (1..=65)
        .map(|value| SolverNodeExactParameter::Surface {
            source_face_id: face.clone(),
            chart_id: digest(value),
            evaluator_uv: [0.1, 0.2],
        })
        .collect();
    assert_eq!(
        validate(&invalid).unwrap_err().field,
        "mesh node exact parameters"
    );
}

#[test]
fn quadratic_solver_topology_requires_shared_exact_midside_connectivity() {
    let mut quadratic = artifact();
    quadratic.resolved_request.element_order = ElementOrder::Tet10;
    let midpoint_coordinates = [
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let source_faces = (0..4)
        .map(|index| entity(PersistentEntityKind::Face, &format!("face:{index}")))
        .collect::<Vec<_>>();
    let source_edges = (0..6)
        .map(|index| entity(PersistentEntityKind::Edge, &format!("edge:{index}")))
        .collect::<Vec<_>>();
    let midpoint_source_edges = [0, 3, 1, 2, 4, 5];
    for (index, coordinates_m) in midpoint_coordinates.into_iter().enumerate() {
        let source_edge = &source_edges[midpoint_source_edges[index]];
        let mut provenance = quadratic.topology.nodes[0].provenance.clone();
        provenance.extend(source_faces.iter().cloned());
        provenance.push(source_edge.clone());
        provenance.sort();
        let mut exact_parameters = vec![SolverNodeExactParameter::Curve {
            source_edge_id: source_edge.clone(),
            parameter: 0.5,
        }];
        exact_parameters.extend(
            source_faces
                .iter()
                .enumerate()
                .map(|(face, source_face_id)| SolverNodeExactParameter::Surface {
                    source_face_id: source_face_id.clone(),
                    chart_id: digest(face as u8 + 40),
                    evaluator_uv: [0.25, 0.25],
                }),
        );
        sort_solver_node_exact_parameters(&mut exact_parameters);
        quadratic.topology.nodes.push(SolverMeshNode {
            node_id: index as u64 + 5,
            stable_identity: solver_midside_node_identity(
                TETRAHEDRON_MIDSIDE_EDGE_CORNERS[index]
                    .map(|corner| quadratic.topology.nodes[corner].stable_identity),
            ),
            coordinates_m,
            provenance,
            exact_parameters,
        });
    }
    quadratic.topology.volume_elements[0].order = ElementOrder::Tet10;
    quadratic.topology.volume_elements[0].node_ids = (1..=10).collect();
    let face_midpoints = [[5, 6, 7], [5, 9, 8], [7, 10, 8], [6, 10, 9]];
    for (face, midpoints) in quadratic
        .topology
        .boundary_faces
        .iter_mut()
        .zip(face_midpoints)
    {
        face.order = BoundaryTriangleOrder::Tri6;
        face.node_ids.extend(midpoints);
    }
    let edge_midpoints = [5, 7, 8, 6, 9, 10];
    for (edge, midpoint) in quadratic
        .topology
        .boundary_edges
        .iter_mut()
        .zip(edge_midpoints)
    {
        edge.order = BoundaryEdgeOrder::Line3;
        edge.node_ids.push(midpoint);
    }
    quadratic
        .topology
        .field_topologies
        .iter_mut()
        .find(|field| field.location == FieldTopologyLocation::Node)
        .unwrap()
        .ordered_entity_ids = (1..=10).collect();
    validate_solver_mesh_topology(&quadratic.topology, &quadratic.resolved_request).unwrap();

    let mut invalid = quadratic.clone();
    invalid.topology.boundary_faces[0].node_ids.swap(3, 4);
    assert_eq!(
        validate_solver_mesh_topology(&invalid.topology, &invalid.resolved_request)
            .unwrap_err()
            .field,
        "mesh element order topology"
    );

    let mut invalid = quadratic.clone();
    invalid.topology.boundary_edges[0].node_ids[2] = 6;
    assert_eq!(
        validate_solver_mesh_topology(&invalid.topology, &invalid.resolved_request)
            .unwrap_err()
            .field,
        "boundary edge"
    );

    let mut invalid = quadratic.clone();
    invalid.topology.nodes[4].stable_identity = digest(99);
    assert_eq!(
        validate_solver_mesh_topology(&invalid.topology, &invalid.resolved_request)
            .unwrap_err()
            .field,
        "mesh element order topology"
    );

    let mut invalid = quadratic;
    invalid.topology.nodes[4]
        .exact_parameters
        .retain(|parameter| matches!(parameter, SolverNodeExactParameter::Curve { .. }));
    assert_eq!(
        validate_solver_mesh_topology(&invalid.topology, &invalid.resolved_request)
            .unwrap_err()
            .field,
        "mesh element order topology"
    );
}

#[test]
fn successful_evidence_requires_terminal_order_and_respects_hard_budgets() {
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
    invalid.resources.search_work += 1;
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
fn successful_evidence_admits_canonically_ordered_partition_observations() {
    let artifact = artifact();
    let mut evidence = evidence(&artifact);
    evidence.stages[0].partition.kind = MeshingPartitionKind::CanonicalEntityBatch;
    evidence.stages[0].partition.partition_count = 2;
    evidence.stages[0].partition.entity_range = Some(CanonicalEntityRange {
        first: entity(PersistentEntityKind::Edge, "edge:1"),
        last: entity(PersistentEntityKind::Edge, "edge:1"),
        entity_count: 1,
    });
    let mut second_partition = evidence.stages[0].clone();
    second_partition.partition.partition_index = 1;
    second_partition.stage_result_digest = digest(98);
    evidence.stages.insert(1, second_partition);
    evidence.resources.search_work += 1;
    evidence.resources.iterations += 1;
    evidence.validate(&artifact).unwrap();

    evidence.stages.swap(0, 1);
    assert_eq!(
        evidence.validate(&artifact).unwrap_err().field,
        "meshing evidence"
    );
}

#[test]
fn solver_artifact_rejects_unknown_legacy_backend_evidence() {
    let mut value = serde_json::to_value(artifact()).unwrap();
    value["backend"] = serde_json::json!("removed");
    assert!(serde_json::from_value::<SolverMeshArtifact>(value).is_err());
}
