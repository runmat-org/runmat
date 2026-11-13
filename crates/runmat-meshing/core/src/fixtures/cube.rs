//! Shared canonical cube fixture for independent scalar-physics convergence tests.
use crate::{
    solver_boundary_face_identity, solver_volume_element_identity, BoundaryFaceRole,
    BoundaryTriangleOrder, ElementOrder, FieldTopologyLocation, FieldTopologyMap, MeshNeighbor,
    MeshRegion, PersistentEntityId, PersistentEntityKind, SolverBoundaryFace, SolverMeshArtifact,
    SolverMeshNode, SolverVolumeElement, StableDigest,
};
use std::collections::BTreeMap;
const TETRAHEDRON_FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
pub fn slab_mesh(cell_count: usize) -> SolverMeshArtifact {
    slab_bar_mesh(cell_count, cell_count)
}

/// Unit cube with `axial_cells` uniform subdivisions along x and `lateral_cells` along y and z,
/// using the same six-tetrahedron cell split as [`slab_mesh`] (identical when the counts agree).
/// For one-dimensional (x-only) reference solutions this refines the axial direction alone.
pub fn slab_bar_mesh(axial_cells: usize, lateral_cells: usize) -> SolverMeshArtifact {
    assert!(
        axial_cells > 0 && lateral_cells > 0,
        "cube fixture requires a positive subdivision count"
    );
    let cell_count = axial_cells;
    let mut artifact = super::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let solid = entity(PersistentEntityKind::Solid, "slab");
    let region = entity(PersistentEntityKind::Region, "region");
    let nodes_per_axis = lateral_cells + 1;
    let mut nodes = Vec::with_capacity((cell_count + 1) * nodes_per_axis.pow(2));
    for x_index in 0..=cell_count {
        for y_index in 0..=lateral_cells {
            for z_index in 0..=lateral_cells {
                let index = nodes.len();
                nodes.push(SolverMeshNode {
                    node_id: index as u64 + 1,
                    stable_identity: node_digest(index),
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new([
                        x_index as f64 / cell_count as f64,
                        y_index as f64 / lateral_cells as f64,
                        z_index as f64 / lateral_cells as f64,
                    ]),
                    provenance: vec![solid.clone()],
                    exact_parameters: Vec::new(),
                });
            }
        }
    }

    let node_id = |x: usize, y: usize, z: usize| {
        (x * nodes_per_axis * nodes_per_axis + y * nodes_per_axis + z) as u64 + 1
    };
    let mut elements = Vec::with_capacity(cell_count * lateral_cells.pow(2) * 6);
    for x in 0..cell_count {
        for y in 0..lateral_cells {
            for z in 0..lateral_cells {
                let v000 = node_id(x, y, z);
                let v001 = node_id(x, y, z + 1);
                let v010 = node_id(x, y + 1, z);
                let v011 = node_id(x, y + 1, z + 1);
                let v100 = node_id(x + 1, y, z);
                let v101 = node_id(x + 1, y, z + 1);
                let v110 = node_id(x + 1, y + 1, z);
                let v111 = node_id(x + 1, y + 1, z + 1);
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
                    let element_id = elements.len() as u64 + 1;
                    elements.push(SolverVolumeElement {
                        element_id,
                        stable_identity: solver_volume_element_identity(
                            corners.map(|id| nodes[id as usize - 1].stable_identity),
                        ),
                        order: ElementOrder::Tet4,
                        node_ids: corners.to_vec(),
                        region_id: region.clone(),
                        provenance: vec![solid.clone()],
                    });
                }
            }
        }
    }

    let mut face_owners = BTreeMap::<[u64; 3], Vec<(u64, u8)>>::new();
    for element in &elements {
        for (local_face, local_nodes) in TETRAHEDRON_FACES.iter().enumerate() {
            let mut face = local_nodes.map(|index| element.node_ids[index]);
            face.sort_unstable();
            face_owners
                .entry(face)
                .or_default()
                .push((element.element_id, local_face as u8));
        }
    }
    let neighbors = elements
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
    let boundary_faces = face_owners
        .iter()
        .filter(|(_, owners)| owners.len() == 1)
        .enumerate()
        .map(|(index, (node_ids, owners))| {
            let x_values =
                node_ids.map(|id| nodes[id as usize - 1].coordinates_m.canonical_value()[0]);
            let source = if x_values.iter().all(|x| x.abs() <= 1.0e-12) {
                "slab_left"
            } else if x_values.iter().all(|x| (*x - 1.0).abs() <= 1.0e-12) {
                "slab_right"
            } else {
                "slab_insulated"
            };
            SolverBoundaryFace {
                face_id: index as u64 + 1,
                stable_identity: solver_boundary_face_identity(
                    node_ids.map(|id| nodes[id as usize - 1].stable_identity),
                ),
                order: BoundaryTriangleOrder::Tri3,
                node_ids: node_ids.to_vec(),
                adjacent_volume_element_ids: vec![owners[0].0],
                role: BoundaryFaceRole::Exterior,
                provenance: vec![entity(PersistentEntityKind::Face, source)],
            }
        })
        .collect::<Vec<_>>();
    let node_ids = nodes.iter().map(|node| node.node_id).collect::<Vec<_>>();
    let element_ids = elements
        .iter()
        .map(|element| element.element_id)
        .collect::<Vec<_>>();
    let face_ids = boundary_faces
        .iter()
        .map(|face| face.face_id)
        .collect::<Vec<_>>();
    artifact.topology.nodes = nodes;
    artifact.topology.volume_elements = elements;
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

/// Straight-sided quadratic version of the same shared convergence fixture.
/// Midpoints are shared by global edge identity, including the Tri6 boundary.
pub fn quadratic_slab_mesh(cell_count: usize) -> SolverMeshArtifact {
    elevate_planar_tet4_fixture(slab_mesh(cell_count))
}

/// Straight-sided quadratic version of [`slab_bar_mesh`].
pub fn quadratic_slab_bar_mesh(axial_cells: usize, lateral_cells: usize) -> SolverMeshArtifact {
    elevate_planar_tet4_fixture(slab_bar_mesh(axial_cells, lateral_cells))
}

/// Straight-sided Tet10 elevation of a Tet4 box fixture whose boundary faces are axis-aligned
/// planes ([`slab_mesh`], [`super::layered_box_mesh`]). Midside nodes of boundary edges carry
/// the source faces and planar charts of their faces.
pub fn elevate_planar_tet4_fixture(mut mesh: SolverMeshArtifact) -> SolverMeshArtifact {
    use crate::{
        solver_midside_node_identity, sort_solver_node_exact_parameters, SolverNodeExactParameter,
        TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
    };
    assert_eq!(mesh.resolved_request.element_order, ElementOrder::Tet4);
    let mut midsides = BTreeMap::new();
    let topo = &mut *mesh.topology;
    for element in &topo.volume_elements {
        for [a, b] in TETRAHEDRON_MIDSIDE_EDGE_CORNERS {
            let mut edge = [element.node_ids[a], element.node_ids[b]];
            edge.sort_unstable();
            if midsides.contains_key(&edge) {
                continue;
            }
            let endpoints = edge.map(|id| &topo.nodes[id as usize - 1]);
            let coordinates_m = std::array::from_fn(|axis| {
                (endpoints[0].coordinates_m.canonical_value()[axis]
                    + endpoints[1].coordinates_m.canonical_value()[axis])
                    * 0.5
            });
            let stable_identity =
                solver_midside_node_identity(endpoints.map(|n| n.stable_identity));
            let node_id = topo.nodes.len() as u64 + 1;
            topo.nodes.push(SolverMeshNode {
                node_id,
                stable_identity,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates_m),
                provenance: element.provenance.clone(),
                exact_parameters: Vec::new(),
            });
            midsides.insert(edge, node_id);
        }
    }
    let midpoint = |a, b| {
        let mut edge = [a, b];
        edge.sort_unstable();
        midsides[&edge]
    };
    for element in &mut topo.volume_elements {
        let corners = element.node_ids.clone();
        element.node_ids.extend(
            TETRAHEDRON_MIDSIDE_EDGE_CORNERS.map(|[a, b]| midpoint(corners[a], corners[b])),
        );
        element.order = ElementOrder::Tet10;
    }
    for face in &mut topo.boundary_faces {
        let corners = face.node_ids.clone();
        let midsides = [[0, 1], [1, 2], [2, 0]].map(|[a, b]| midpoint(corners[a], corners[b]));
        // Planar fixture charts use physical tangent coordinates and actual source IDs.
        let coordinates = [0, 1, 2].map(|i| {
            topo.nodes[corners[i] as usize - 1]
                .coordinates_m
                .canonical_value()
        });
        let normal_axis = (0..3)
            .find(|&axis| coordinates.iter().all(|p| p[axis] == coordinates[0][axis]))
            .unwrap();
        let tangent_axes = match normal_axis {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        };
        for id in midsides {
            let node = &mut topo.nodes[id as usize - 1];
            for source_face_id in &face.provenance {
                node.provenance.push(source_face_id.clone());
                node.exact_parameters
                    .push(SolverNodeExactParameter::Surface {
                        source_face_id: source_face_id.clone(),
                        chart_id: face.stable_identity,
                        evaluator_uv: tangent_axes
                            .map(|axis| node.coordinates_m.canonical_value()[axis]),
                    });
            }
            node.provenance.sort();
            node.provenance.dedup();
            sort_solver_node_exact_parameters(&mut node.exact_parameters);
        }
        face.node_ids.extend(midsides);
        face.order = BoundaryTriangleOrder::Tri6;
    }
    for field in &mut topo.field_topologies {
        if field.location == FieldTopologyLocation::Node {
            field.ordered_entity_ids = topo.nodes.iter().map(|n| n.node_id).collect();
        }
    }
    mesh.resolved_request.element_order = ElementOrder::Tet10;
    mesh.resolved_request.limits.maximum_nodes = topo.nodes.len() as u64;
    mesh.seal_canonical_digest().unwrap();
    mesh
}

pub(super) fn signed_six_volume(corners: [u64; 4], nodes: &[SolverMeshNode]) -> f64 {
    let [a, b, c, d] = corners.map(|id| nodes[id as usize - 1].coordinates_m.canonical_value());
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
        + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0])
}

pub(super) fn node_digest(index: usize) -> StableDigest {
    let mut bytes = [0_u8; 32];
    bytes[..8].copy_from_slice(&(index as u64 + 1).to_le_bytes());
    StableDigest::from_bytes(bytes)
}

pub(super) fn entity(kind: PersistentEntityKind, source: &str) -> PersistentEntityId {
    super::fixture_entity(kind, source)
}

pub(super) fn field(
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
