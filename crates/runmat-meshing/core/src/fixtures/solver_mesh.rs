use crate::{
    solver_boundary_edge_identity, solver_boundary_face_identity, solver_midside_node_identity,
    solver_volume_element_identity, sort_solver_node_exact_parameters, AlgorithmVersionSet,
    BoundaryEdgeOrder, BoundaryFaceRole, BoundaryTriangleOrder, CancellationPolicy,
    CurveQualityTargets, ElementOrder, FieldTopologyLocation, FieldTopologyMap,
    GeometryRevisionRef, GeometryTolerancePolicy, MeshNeighbor, MeshRegion, MeshingQualityTargets,
    MeshingRequest, MeshingResourceBudget, MetricCombinationRule, MetricFieldRequest,
    MetricTensor3, PersistentEntityId, PersistentEntityKind, SolverBoundaryEdge,
    SolverBoundaryFace, SolverMeshArtifact, SolverMeshNode, SolverMeshTopology,
    SolverNodeExactParameter, SolverVolumeElement, StableDigest, SurfaceQualityTargets,
    VolumeQualityTargets, ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION, MESHING_REQUEST_SCHEMA_VERSION,
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
        schema_version: ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION,
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
