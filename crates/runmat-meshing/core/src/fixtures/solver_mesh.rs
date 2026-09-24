use crate::{
    solver_boundary_edge_identity, solver_boundary_face_identity, solver_midside_node_identity,
    solver_volume_element_identity, sort_solver_node_exact_parameters, AlgorithmVersionSet,
    BoundaryEdgeOrder, BoundaryFaceRole, BoundaryTriangleOrder, CacheAdmissionDecision,
    CancellationPolicy, CanonicalMeshingContract, CurveQualityTargets, ElementOrder,
    FieldTopologyLocation, FieldTopologyMap, GeometryRevisionRef, GeometryTolerancePolicy,
    InvariantEvidence, MeshNeighbor, MeshRegion, MeshingEvidence, MeshingPartitionDescriptor,
    MeshingPartitionKind, MeshingQualityTargets, MeshingRequest, MeshingResourceBudget,
    MeshingResourceUsage, MeshingStageEvidence, MeshingStageKind, MetricCombinationRule,
    MetricFieldRequest, MetricTensor3, PersistentEntityId, PersistentEntityKind,
    PlatformBuildIdentity, SolverBoundaryEdge, SolverBoundaryFace, SolverMeshArtifact,
    SolverMeshNode, SolverMeshTopology, SolverNodeExactParameter, SolverVolumeElement,
    StableDigest, SurfaceQualityTargets, VolumeQualityTargets, MESHING_EVIDENCE_SCHEMA_VERSION,
    MESHING_REQUEST_SCHEMA_VERSION, SOLVER_MESH_ARTIFACT_SCHEMA_VERSION,
};

pub fn canonical_tetrahedron_solver_mesh(order: ElementOrder) -> SolverMeshArtifact {
    let solid = entity(PersistentEntityKind::Solid, "solid");
    let region = entity(PersistentEntityKind::Region, "region");
    let faces = (0..4)
        .map(|index| entity(PersistentEntityKind::Face, &format!("face:{index}")))
        .collect::<Vec<_>>();
    let mut nodes = [
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
    .collect::<Vec<_>>();
    if order == ElementOrder::Tet10 {
        let coordinates = [
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ];
        for (index, coordinates_m) in coordinates.into_iter().enumerate() {
            let mut exact_parameters = faces
                .iter()
                .enumerate()
                .map(|(face, source_face_id)| SolverNodeExactParameter::Surface {
                    source_face_id: source_face_id.clone(),
                    chart_id: digest(face as u8 + 10),
                    evaluator_uv: [0.25, 0.25],
                })
                .collect::<Vec<_>>();
            sort_solver_node_exact_parameters(&mut exact_parameters);
            let mut provenance = faces.clone();
            provenance.push(solid.clone());
            provenance.sort();
            nodes.push(SolverMeshNode {
                node_id: index as u64 + 5,
                stable_identity: solver_midside_node_identity(
                    runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS[index]
                        .map(|corner| nodes[corner].stable_identity),
                ),
                coordinates_m,
                provenance,
                exact_parameters,
            });
        }
    }
    let face_corners = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]];
    let face_midpoints = [[5, 6, 7], [5, 9, 8], [7, 10, 8], [6, 10, 9]];
    let boundary_faces = face_corners
        .into_iter()
        .enumerate()
        .map(|(index, corners)| {
            let mut node_ids = corners.to_vec();
            if order == ElementOrder::Tet10 {
                node_ids.extend(face_midpoints[index]);
            }
            SolverBoundaryFace {
                face_id: index as u64 + 10,
                stable_identity: solver_boundary_face_identity(
                    corners.map(|node_id| digest(node_id as u8)),
                ),
                order: if order == ElementOrder::Tet4 {
                    BoundaryTriangleOrder::Tri3
                } else {
                    BoundaryTriangleOrder::Tri6
                },
                node_ids,
                adjacent_volume_element_ids: vec![1],
                role: BoundaryFaceRole::Exterior,
                provenance: vec![faces[index].clone()],
            }
        })
        .collect();
    let edge_corners = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]];
    let edge_midpoints = [5, 7, 8, 6, 9, 10];
    let edge_faces = [[10, 11], [10, 12], [11, 12], [10, 13], [11, 13], [12, 13]];
    let boundary_edges = edge_corners
        .into_iter()
        .enumerate()
        .map(|(index, corners)| {
            let mut node_ids = corners.to_vec();
            if order == ElementOrder::Tet10 {
                node_ids.push(edge_midpoints[index]);
            }
            SolverBoundaryEdge {
                edge_id: index as u64 + 20,
                stable_identity: solver_boundary_edge_identity(
                    corners.map(|node_id| digest(node_id as u8)),
                ),
                order: if order == ElementOrder::Tet4 {
                    BoundaryEdgeOrder::Line2
                } else {
                    BoundaryEdgeOrder::Line3
                },
                node_ids,
                adjacent_boundary_face_ids: edge_faces[index].into(),
                provenance: vec![solid.clone()],
            }
        })
        .collect();
    let node_count = if order == ElementOrder::Tet4 { 4 } else { 10 };
    let mut request = request();
    request.element_order = order;
    let mut artifact = SolverMeshArtifact {
        schema_version: SOLVER_MESH_ARTIFACT_SCHEMA_VERSION,
        canonical_digest: StableDigest::ZERO,
        root_stage_manifest_digest: digest(2),
        geometry: GeometryRevisionRef {
            source_digest: digest(3),
            geometry_revision: 1,
            persistent_mapping_version: 1,
        },
        resolved_request: request,
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
                order,
                node_ids: (1..=node_count).collect(),
                region_id: region.clone(),
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
                element_ids: vec![1],
            }],
            conformal_interfaces: Vec::new(),
            contacts: Vec::new(),
            field_topologies: vec![
                field("nodes", FieldTopologyLocation::Node, 1..=node_count),
                field("elements", FieldTopologyLocation::VolumeElement, 1..=1),
                field("faces", FieldTopologyLocation::BoundaryFace, 10..=13),
                field("edges", FieldTopologyLocation::BoundaryEdge, 20..=25),
            ],
        },
    };
    artifact.seal_canonical_digest().unwrap();
    artifact
}

/// Three separated, independently material-addressable Tet4 regions for solver tests.
pub fn three_region_tetrahedron_solver_mesh() -> SolverMeshArtifact {
    let mut template = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for (face_index, name) in [(0, "root"), (1, "tip")] {
        template.topology.boundary_faces[face_index]
            .provenance
            .push(entity(PersistentEntityKind::Face, name));
        template.topology.boundary_faces[face_index]
            .provenance
            .sort();
    }
    let mut mesh = template.clone();
    mesh.topology.nodes.clear();
    mesh.topology.volume_elements.clear();
    mesh.topology.neighbors.clear();
    mesh.topology.boundary_faces.clear();
    mesh.topology.boundary_edges.clear();
    mesh.topology.regions.clear();

    for (copy, region_name) in ["mid_aluminum", "polymer_segment", "tip_steel"]
        .into_iter()
        .enumerate()
    {
        let offset = copy as u64 * 100;
        let node_start = mesh.topology.nodes.len();
        for (local_node, node) in template.topology.nodes.iter().enumerate() {
            let mut node = node.clone();
            node.node_id += offset;
            node.stable_identity =
                StableDigest::from_bytes([32 + (copy * 4 + local_node) as u8; 32]);
            node.coordinates_m[0] += copy as f64 * 2.0;
            mesh.topology.nodes.push(node);
        }

        let mut element = template.topology.volume_elements[0].clone();
        element.element_id += offset;
        for node_id in &mut element.node_ids {
            *node_id += offset;
        }
        element.region_id = entity(PersistentEntityKind::Region, region_name);
        element.stable_identity = solver_volume_element_identity(std::array::from_fn(|index| {
            mesh.topology.nodes[node_start + index].stable_identity
        }));
        mesh.topology.volume_elements.push(element.clone());
        mesh.topology.regions.push(MeshRegion {
            region_id: element.region_id.clone(),
            element_ids: vec![element.element_id],
        });

        for neighbor in &template.topology.neighbors {
            let mut neighbor = *neighbor;
            neighbor.element_id += offset;
            neighbor.adjacent_element_id = neighbor.adjacent_element_id.map(|id| id + offset);
            mesh.topology.neighbors.push(neighbor);
        }
        for (local_face, face) in template.topology.boundary_faces.iter().enumerate() {
            let mut face = face.clone();
            face.face_id += offset;
            for node_id in &mut face.node_ids {
                *node_id += offset;
            }
            for element_id in &mut face.adjacent_volume_element_ids {
                *element_id += offset;
            }
            face.stable_identity = solver_boundary_face_identity(std::array::from_fn(|index| {
                mesh.topology.nodes
                    [node_start + face.node_ids[index] as usize - 1 - offset as usize]
                    .stable_identity
            }));
            let targets: &[&str] = match local_face {
                0 => &["interface"],
                2 if copy == 0 => &["mid_aluminum"],
                2 if copy == 2 => &["tip_steel"],
                _ => &[],
            };
            for target in targets {
                face.provenance
                    .push(entity(PersistentEntityKind::Face, target));
            }
            face.provenance.sort();
            face.provenance.dedup();
            mesh.topology.boundary_faces.push(face);
        }
        for edge in &template.topology.boundary_edges {
            let mut edge = edge.clone();
            edge.edge_id += offset;
            for node_id in &mut edge.node_ids {
                *node_id += offset;
            }
            for face_id in &mut edge.adjacent_boundary_face_ids {
                *face_id += offset;
            }
            edge.stable_identity = solver_boundary_edge_identity(std::array::from_fn(|index| {
                mesh.topology.nodes
                    [node_start + edge.node_ids[index] as usize - 1 - offset as usize]
                    .stable_identity
            }));
            mesh.topology.boundary_edges.push(edge);
        }
    }
    mesh.topology
        .regions
        .sort_by(|left, right| left.region_id.cmp(&right.region_id));
    for topology in &mut mesh.topology.field_topologies {
        topology.ordered_entity_ids = match topology.location {
            FieldTopologyLocation::Node => mesh
                .topology
                .nodes
                .iter()
                .map(|node| node.node_id)
                .collect(),
            FieldTopologyLocation::VolumeElement => mesh
                .topology
                .volume_elements
                .iter()
                .map(|element| element.element_id)
                .collect(),
            FieldTopologyLocation::BoundaryFace => mesh
                .topology
                .boundary_faces
                .iter()
                .map(|face| face.face_id)
                .collect(),
            FieldTopologyLocation::BoundaryEdge => mesh
                .topology
                .boundary_edges
                .iter()
                .map(|edge| edge.edge_id)
                .collect(),
        };
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest()
        .expect("seal multi-region fixture");
    mesh.validate()
        .expect("multi-region fixture must be canonical");
    mesh
}

/// Builds valid canonical evidence bound to a solver-mesh fixture.
pub fn canonical_tetrahedron_meshing_evidence(artifact: &SolverMeshArtifact) -> MeshingEvidence {
    let stage = MeshingStageEvidence {
        stage: MeshingStageKind::Serialization,
        partition: MeshingPartitionDescriptor {
            kind: MeshingPartitionKind::WholeStage,
            partition_index: 0,
            partition_count: 1,
            entity_range: None,
        },
        stage_result_digest: digest(41),
        entity_counts: std::collections::BTreeMap::from([(
            "serialized_elements".to_string(),
            artifact.topology.volume_elements.len() as u64,
        )]),
        invariants: vec![InvariantEvidence {
            invariant_id: "canonical_artifact_valid".to_string(),
            passed: true,
            measured: None,
            required: None,
            unit: None,
        }],
        achieved_error_distributions: std::collections::BTreeMap::new(),
        completed_work: 1,
        estimated_work: 1,
        peak_memory_bytes: 1_024,
        peak_scratch_bytes: 256,
        search_work: 1,
        maximum_recursion_depth: 1,
        iterations: 1,
        elapsed_time_ms: 1,
        cancellation_checkpoints: 1,
    };
    let evidence = MeshingEvidence {
        schema_version: MESHING_EVIDENCE_SCHEMA_VERSION,
        geometry: artifact.geometry.clone(),
        resolved_request_digest: artifact
            .resolved_request
            .canonical_digest()
            .expect("fixture request must encode canonically"),
        artifact_digest: artifact.canonical_digest,
        algorithms: artifact.resolved_request.algorithms.clone(),
        deterministic_seed: artifact.resolved_request.deterministic_seed,
        platform: PlatformBuildIdentity {
            capability_cohort: "fixture-native-exact-cad".to_string(),
            target_triple: "fixture-target".to_string(),
            build_digest: digest(42),
            exact_kernel_abi: Some("fixture-kernel".to_string()),
        },
        stages: vec![stage],
        sizing: Vec::new(),
        resources: MeshingResourceUsage {
            generated_nodes: artifact.topology.nodes.len() as u64,
            generated_elements: artifact.topology.volume_elements.len() as u64,
            peak_memory_bytes: 1_024,
            peak_scratch_bytes: 256,
            wall_time_ms: 1,
            artifact_bytes: artifact
                .canonical_encode()
                .expect("fixture artifact must encode canonically")
                .len() as u64,
            search_work: 1,
            maximum_recursion_depth: 1,
            iterations: 1,
        },
        cache_admission: CacheAdmissionDecision::Admitted,
    };
    evidence
        .validate(artifact)
        .expect("fixture evidence must validate against its artifact");
    evidence
}

fn request() -> MeshingRequest {
    MeshingRequest {
        schema_version: MESHING_REQUEST_SCHEMA_VERSION,
        element_order: ElementOrder::Tet4,
        deterministic_seed: 1,
        algorithms: AlgorithmVersionSet {
            geometry: "geometry/1".into(),
            curve: "curve/1".into(),
            surface: "surface/1".into(),
            plc: "plc/1".into(),
            tetrahedron: "tetrahedron/1".into(),
            optimization: "optimization/1".into(),
            validation: "validation/1".into(),
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
                minimum_metric_edge_length: 0.1,
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

fn field(
    topology_id: &str,
    location: FieldTopologyLocation,
    ids: std::ops::RangeInclusive<u64>,
) -> FieldTopologyMap {
    FieldTopologyMap {
        topology_id: topology_id.into(),
        location,
        ordered_entity_ids: ids.collect(),
    }
}

fn entity(kind: PersistentEntityKind, source: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: source.into(),
        assembly_path: vec!["root".into()],
    }
}

fn digest(byte: u8) -> StableDigest {
    StableDigest::from_bytes([byte; 32])
}
