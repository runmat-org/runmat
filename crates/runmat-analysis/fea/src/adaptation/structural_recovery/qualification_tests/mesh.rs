//! Original canonical cantilever mesh fixture; geometry and inventories unchanged.
use runmat_execution::Digest;
use runmat_meshing_core::{
    solver_boundary_face_identity, solver_midside_node_identity, solver_volume_element_identity,
    sort_solver_node_exact_parameters, BoundaryFaceRole, BoundaryTriangleOrder, ElementOrder,
    FieldTopologyLocation, FieldTopologyMap, MeshNeighbor, MeshRegion, PersistentEntityId,
    PersistentEntityKind, SolverBoundaryFace, SolverMeshArtifact, SolverMeshNode,
    SolverNodeExactParameter, SolverVolumeElement, StableDigest, TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};
use std::collections::BTreeMap;
const TETRAHEDRON_FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

pub(crate) fn structured_cantilever(divisions: usize, order: ElementOrder) -> SolverMeshArtifact {
    assert!(divisions > 0);
    let mut artifact = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
    let solid = entity(PersistentEntityKind::Solid, "cantilever");
    let region = entity(PersistentEntityKind::Region, "region");
    let nodes_per_axis = divisions + 1;
    let mut nodes = Vec::with_capacity(nodes_per_axis.pow(3));
    for x in 0..=divisions {
        for y in 0..=divisions {
            for z in 0..=divisions {
                let index = nodes.len();
                nodes.push(SolverMeshNode {
                    node_id: index as u64 + 1,
                    stable_identity: digest("corner", index as u64),
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new([
                        x as f64 / divisions as f64,
                        y as f64 / divisions as f64,
                        z as f64 / divisions as f64,
                    ]),
                    provenance: vec![solid.clone()],
                    exact_parameters: Vec::new(),
                });
            }
        }
    }
    let corner_id = |x: usize, y: usize, z: usize| {
        (x * nodes_per_axis * nodes_per_axis + y * nodes_per_axis + z) as u64 + 1
    };
    let mut corner_elements = Vec::with_capacity(divisions.pow(3) * 6);
    for x in 0..divisions {
        for y in 0..divisions {
            for z in 0..divisions {
                let v000 = corner_id(x, y, z);
                let v001 = corner_id(x, y, z + 1);
                let v010 = corner_id(x, y + 1, z);
                let v011 = corner_id(x, y + 1, z + 1);
                let v100 = corner_id(x + 1, y, z);
                let v101 = corner_id(x + 1, y, z + 1);
                let v110 = corner_id(x + 1, y + 1, z);
                let v111 = corner_id(x + 1, y + 1, z + 1);
                for mut corners in [
                    [v000, v100, v110, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v011, v001, v111],
                    [v000, v001, v101, v111],
                    [v000, v101, v100, v111],
                ] {
                    if signed_six_volume(corners, &nodes) < 0.0 {
                        corners.swap(1, 2);
                    }
                    corner_elements.push(corners);
                }
            }
        }
    }

    let mut midpoint_by_edge = BTreeMap::<[u64; 2], u64>::new();
    let mut volume_elements = Vec::with_capacity(corner_elements.len());
    for corners in &corner_elements {
        let mut node_ids = corners.to_vec();
        for endpoints in TETRAHEDRON_MIDSIDE_EDGE_CORNERS
            .into_iter()
            .take(if order == ElementOrder::Tet10 { 6 } else { 0 })
        {
            let mut edge = endpoints.map(|endpoint| corners[endpoint]);
            edge.sort_unstable();
            let midpoint_id = *midpoint_by_edge.entry(edge).or_insert_with(|| {
                let [left, right] = edge.map(|id| nodes[id as usize - 1].clone());
                let node_id = nodes.len() as u64 + 1;
                nodes.push(SolverMeshNode {
                    node_id,
                    stable_identity: solver_midside_node_identity([
                        left.stable_identity,
                        right.stable_identity,
                    ]),
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new(std::array::from_fn(|axis| {
                        0.5 * (left.coordinates_m.canonical_value()[axis]
                            + right.coordinates_m.canonical_value()[axis])
                    })),
                    provenance: vec![solid.clone()],
                    exact_parameters: Vec::new(),
                });
                node_id
            });
            node_ids.push(midpoint_id);
        }
        let element_id = volume_elements.len() as u64 + 1;
        volume_elements.push(SolverVolumeElement {
            element_id,
            stable_identity: solver_volume_element_identity(
                corners.map(|id| nodes[id as usize - 1].stable_identity),
            ),
            order,
            node_ids,
            region_id: region.clone(),
            provenance: vec![solid.clone()],
        });
    }

    let mut face_owners = BTreeMap::<[u64; 3], Vec<(u64, u8)>>::new();
    for element in &volume_elements {
        for (local_face, local_nodes) in TETRAHEDRON_FACES.iter().enumerate() {
            let mut face = local_nodes.map(|index| element.node_ids[index]);
            face.sort_unstable();
            face_owners
                .entry(face)
                .or_default()
                .push((element.element_id, local_face as u8));
        }
    }
    let neighbors = volume_elements
        .iter()
        .flat_map(|element| {
            TETRAHEDRON_FACES
                .iter()
                .enumerate()
                .map(|(local_face, local_nodes)| {
                    let mut face = local_nodes.map(|index| element.node_ids[index]);
                    face.sort_unstable();
                    let adjacent_element_id = face_owners[&face]
                        .iter()
                        .find_map(|(owner, _)| (*owner != element.element_id).then_some(*owner));
                    MeshNeighbor {
                        element_id: element.element_id,
                        local_face_index: local_face as u8,
                        adjacent_element_id,
                    }
                })
        })
        .collect::<Vec<_>>();
    let mut boundary_faces = Vec::new();
    for (corners, owners) in face_owners.iter().filter(|(_, owners)| owners.len() == 1) {
        let coordinates = corners.map(|id| nodes[id as usize - 1].coordinates_m.canonical_value());
        let source = if coordinates.iter().all(|point| point[0].abs() <= 1.0e-12) {
            "root"
        } else if coordinates
            .iter()
            .all(|point| (point[0] - 1.0).abs() <= 1.0e-12)
        {
            "tip"
        } else {
            "free_surface"
        };
        let source_face = entity(PersistentEntityKind::Face, source);
        let mut node_ids = corners.to_vec();
        for mut edge in [
            [corners[0], corners[1]],
            [corners[1], corners[2]],
            [corners[2], corners[0]],
        ]
        .into_iter()
        .take(if order == ElementOrder::Tet10 { 3 } else { 0 })
        {
            edge.sort_unstable();
            let midpoint_id = midpoint_by_edge[&edge];
            let midpoint = &mut nodes[midpoint_id as usize - 1];
            midpoint.provenance.push(source_face.clone());
            midpoint.provenance.sort();
            midpoint.provenance.dedup();
            midpoint
                .exact_parameters
                .push(SolverNodeExactParameter::Surface {
                    source_face_id: source_face.clone(),
                    chart_id: digest(source, 0),
                    evaluator_uv: [0.5, 0.5],
                });
            sort_solver_node_exact_parameters(&mut midpoint.exact_parameters);
            midpoint.exact_parameters.dedup();
            node_ids.push(midpoint_id);
        }
        boundary_faces.push(SolverBoundaryFace {
            face_id: boundary_faces.len() as u64 + 1,
            stable_identity: solver_boundary_face_identity(
                corners.map(|id| nodes[id as usize - 1].stable_identity),
            ),
            order: if order == ElementOrder::Tet10 {
                BoundaryTriangleOrder::Tri6
            } else {
                BoundaryTriangleOrder::Tri3
            },
            node_ids,
            adjacent_volume_element_ids: vec![owners[0].0],
            role: BoundaryFaceRole::Exterior,
            provenance: vec![source_face],
        });
    }

    let node_ids = nodes.iter().map(|node| node.node_id).collect::<Vec<_>>();
    let element_ids = volume_elements
        .iter()
        .map(|element| element.element_id)
        .collect::<Vec<_>>();
    let face_ids = boundary_faces
        .iter()
        .map(|face| face.face_id)
        .collect::<Vec<_>>();
    artifact.topology.nodes = nodes;
    artifact.topology.volume_elements = volume_elements;
    artifact.topology.neighbors = neighbors;
    artifact.topology.boundary_faces = boundary_faces;
    artifact.topology.boundary_edges.clear();
    artifact.topology.regions = vec![MeshRegion {
        region_id: region,
        element_ids: element_ids.clone(),
    }];
    artifact.topology.field_topologies = vec![
        field("nodes", FieldTopologyLocation::Node, node_ids),
        field(
            "elements",
            FieldTopologyLocation::VolumeElement,
            element_ids,
        ),
        field("faces", FieldTopologyLocation::BoundaryFace, face_ids),
        field("edges", FieldTopologyLocation::BoundaryEdge, Vec::new()),
    ];
    artifact.resolved_request.limits.maximum_nodes = artifact.topology.nodes.len() as u64;
    artifact.resolved_request.limits.maximum_elements =
        artifact.topology.volume_elements.len() as u64;
    artifact.seal_canonical_digest().unwrap();
    artifact
}

fn signed_six_volume(corners: [u64; 4], nodes: &[SolverMeshNode]) -> f64 {
    let [a, b, c, d] = corners.map(|id| nodes[id as usize - 1].coordinates_m.canonical_value());
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
        + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0])
}

fn digest(domain: &str, ordinal: u64) -> StableDigest {
    let mut bytes = domain.as_bytes().to_vec();
    bytes.extend_from_slice(&ordinal.to_le_bytes());
    StableDigest::from_bytes(*Digest::sha256(bytes).bytes())
}

pub(super) fn entity(kind: PersistentEntityKind, source: &str) -> PersistentEntityId {
    PersistentEntityId::new(kind, source.into(), vec!["root".into()])
}

fn field(
    topology_id: &str,
    location: FieldTopologyLocation,
    ordered_entity_ids: Vec<u64>,
) -> FieldTopologyMap {
    FieldTopologyMap {
        topology_id: topology_id.into(),
        location,
        ordered_entity_ids,
    }
}
