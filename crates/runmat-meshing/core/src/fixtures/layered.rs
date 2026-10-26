//! Conformal multi-region box fixture: horizontal layers of distinct bodies that share their
//! interface faces, as a fused CAD assembly meshes. Used by conjugate and coupled analyses.
use super::cube::{entity, field, node_digest, signed_six_volume};
use crate::{
    solver_boundary_face_identity, solver_volume_element_identity, BoundaryFaceRole,
    BoundaryTriangleOrder, ConformalInterface, ElementOrder, FieldTopologyLocation, MeshNeighbor,
    MeshRegion, PersistentEntityId, PersistentEntityKind, SolverBoundaryFace, SolverMeshArtifact,
    SolverMeshNode, SolverVolumeElement,
};
use std::collections::BTreeMap;

const TETRAHEDRON_FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

/// A box `[0, size]` split into `cells` hexahedra of six tetrahedra each. Layer `k` owns the
/// cells whose centroid y lies below `layers[k].0` and above the previous bound. Exterior faces
/// are named `<region>_<x|y|z>_<min|max>`; the shared faces of two layers are one source face
/// named `interface_<lower>_<upper>`. Regions are `Region` entities named after their layer.
pub fn layered_box_mesh(
    cells: [usize; 3],
    size_m: [f64; 3],
    layers: &[(f64, &str)],
) -> SolverMeshArtifact {
    assert!(cells.iter().all(|&n| n > 0) && !layers.is_empty());
    let mut artifact = super::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let per = [cells[0] + 1, cells[1] + 1, cells[2] + 1];
    let node_id = |x: usize, y: usize, z: usize| (x * per[1] * per[2] + y * per[2] + z) as u64 + 1;
    let mut nodes = Vec::with_capacity(per[0] * per[1] * per[2]);
    for x in 0..per[0] {
        for y in 0..per[1] {
            for z in 0..per[2] {
                let index = nodes.len();
                nodes.push(SolverMeshNode {
                    node_id: index as u64 + 1,
                    stable_identity: node_digest(index),
                    coordinates_m: runmat_types::physical::Quantity::new([
                        size_m[0] * x as f64 / cells[0] as f64,
                        size_m[1] * y as f64 / cells[1] as f64,
                        size_m[2] * z as f64 / cells[2] as f64,
                    ]),
                    provenance: Vec::new(),
                    exact_parameters: Vec::new(),
                });
            }
        }
    }
    let layer_of = |y: f64| {
        layers
            .iter()
            .position(|(bound, _)| y < *bound)
            .unwrap_or(layers.len() - 1)
    };
    let mut elements = Vec::new();
    for x in 0..cells[0] {
        for y in 0..cells[1] {
            for z in 0..cells[2] {
                let centroid_y = size_m[1] * (y as f64 + 0.5) / cells[1] as f64;
                let name = layers[layer_of(centroid_y)].1;
                let v = |dx: usize, dy: usize, dz: usize| node_id(x + dx, y + dy, z + dz);
                for mut corners in [
                    [v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(1, 1, 1)],
                    [v(0, 0, 0), v(1, 1, 0), v(0, 1, 0), v(1, 1, 1)],
                    [v(0, 0, 0), v(0, 1, 0), v(0, 1, 1), v(1, 1, 1)],
                    [v(0, 0, 0), v(0, 1, 1), v(0, 0, 1), v(1, 1, 1)],
                    [v(0, 0, 0), v(0, 0, 1), v(1, 0, 1), v(1, 1, 1)],
                    [v(0, 0, 0), v(1, 0, 1), v(1, 0, 0), v(1, 1, 1)],
                ] {
                    if signed_six_volume(corners, &nodes) < 0.0 {
                        corners.swap(1, 2);
                    }
                    elements.push(SolverVolumeElement {
                        element_id: elements.len() as u64 + 1,
                        stable_identity: solver_volume_element_identity(
                            corners.map(|id| nodes[id as usize - 1].stable_identity),
                        ),
                        order: ElementOrder::Tet4,
                        node_ids: corners.to_vec(),
                        region_id: entity(PersistentEntityKind::Region, name),
                        provenance: vec![entity(PersistentEntityKind::Solid, name)],
                    });
                }
            }
        }
    }
    for element in &elements {
        for &id in &element.node_ids {
            let node = &mut nodes[id as usize - 1];
            if !node.provenance.contains(&element.provenance[0]) {
                node.provenance.push(element.provenance[0].clone());
                node.provenance.sort();
            }
        }
    }
    let mut owners = BTreeMap::<[u64; 3], Vec<u64>>::new();
    for element in &elements {
        for local in TETRAHEDRON_FACES {
            let mut face = local.map(|i| element.node_ids[i]);
            face.sort_unstable();
            owners.entry(face).or_default().push(element.element_id);
        }
    }
    let neighbors = elements
        .iter()
        .flat_map(|element| {
            TETRAHEDRON_FACES
                .iter()
                .enumerate()
                .map(|(local, corners)| {
                    let mut face = corners.map(|i| element.node_ids[i]);
                    face.sort_unstable();
                    MeshNeighbor {
                        element_id: element.element_id,
                        local_face_index: local as u8,
                        adjacent_element_id: owners[&face]
                            .iter()
                            .copied()
                            .find(|&o| o != element.element_id),
                    }
                })
        })
        .collect::<Vec<_>>();
    let region_name = |element_id: u64| -> &str {
        &elements[element_id as usize - 1]
            .region_id
            .source_topology_id
    };
    let mut boundary_faces = Vec::new();
    let mut interfaces = BTreeMap::<String, (PersistentEntityId, String, String, Vec<u64>)>::new();
    for (face, face_owners) in &owners {
        let coordinates = face.map(|id| nodes[id as usize - 1].coordinates_m.canonical_value());
        let (role, source, adjacent) = match face_owners.as_slice() {
            [owner] => {
                let axis = (0..3)
                    .find(|&a| coordinates.iter().all(|p| p[a] == coordinates[0][a]))
                    .expect("exterior box faces are axis aligned");
                let side = if coordinates[0][axis] == 0.0 {
                    "min"
                } else {
                    "max"
                };
                let name = format!("{}_{}_{side}", region_name(*owner), ["x", "y", "z"][axis]);
                (BoundaryFaceRole::Exterior, name, vec![*owner])
            }
            [a, b] if region_name(*a) != region_name(*b) => {
                let mut pair = [region_name(*a).to_string(), region_name(*b).to_string()];
                pair.sort();
                let name = format!("interface_{}_{}", pair[0], pair[1]);
                let mut adjacent = vec![*a, *b];
                adjacent.sort_unstable();
                (BoundaryFaceRole::ConformalInterface, name, adjacent)
            }
            _ => continue,
        };
        let face_id = boundary_faces.len() as u64 + 1;
        let source_entity = entity(PersistentEntityKind::Face, &source);
        if role == BoundaryFaceRole::ConformalInterface {
            let mut pair = [
                region_name(adjacent[0]).to_string(),
                region_name(adjacent[1]).to_string(),
            ];
            pair.sort();
            let [a, b] = pair;
            let entry = interfaces
                .entry(source.clone())
                .or_insert_with(|| (source_entity.clone(), a, b, Vec::new()));
            entry.3.push(face_id);
        }
        boundary_faces.push(SolverBoundaryFace {
            face_id,
            stable_identity: solver_boundary_face_identity(
                face.map(|id| nodes[id as usize - 1].stable_identity),
            ),
            order: BoundaryTriangleOrder::Tri3,
            node_ids: face.to_vec(),
            adjacent_volume_element_ids: adjacent,
            role,
            provenance: vec![source_entity],
        });
    }
    let mut regions = BTreeMap::<PersistentEntityId, Vec<u64>>::new();
    for element in &elements {
        regions
            .entry(element.region_id.clone())
            .or_default()
            .push(element.element_id);
    }
    let mut conformal_interfaces = interfaces
        .into_values()
        .map(|(source_face_id, a, b, faces)| ConformalInterface {
            source_face_id,
            side_a_region_id: entity(PersistentEntityKind::Region, &a),
            side_b_region_id: entity(PersistentEntityKind::Region, &b),
            boundary_face_ids: faces,
        })
        .collect::<Vec<_>>();
    conformal_interfaces.sort_by(|a, b| a.source_face_id.cmp(&b.source_face_id));
    let node_ids = nodes.iter().map(|n| n.node_id).collect::<Vec<_>>();
    let element_ids = elements.iter().map(|e| e.element_id).collect::<Vec<_>>();
    let face_ids = boundary_faces.iter().map(|f| f.face_id).collect::<Vec<_>>();
    artifact.topology.nodes = nodes;
    artifact.topology.volume_elements = elements;
    artifact.topology.neighbors = neighbors;
    artifact.topology.boundary_faces = boundary_faces;
    artifact.topology.boundary_edges.clear();
    artifact.topology.regions = regions
        .into_iter()
        .map(|(region_id, element_ids)| MeshRegion {
            region_id,
            element_ids,
        })
        .collect();
    artifact.topology.conformal_interfaces = conformal_interfaces;
    artifact.topology.contacts.clear();
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
