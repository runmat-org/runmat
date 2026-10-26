mod ties;
use std::collections::{BTreeSet, VecDeque};
pub(super) use ties::{circumsphere_relation, materialized_relation};

use runmat_meshing_core::{
    quality::predicate::{orient2d, PredicateSign},
    StableDigest,
};

use super::{
    error, predicate_error, resource, topology_access::TetrahedralTopology, DelaunayInsertionError,
    DelaunayInsertionErrorKind, DelaunayVolumeNode, InsertionWork,
};

pub(super) fn cavity_is_star_shaped(
    topology: &impl TetrahedralTopology,
    cavity: &BTreeSet<u32>,
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    for index in cavity {
        let tetrahedron = topology.tetrahedron(*index).ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "cavity visibility check reached an inactive tetrahedron",
            )
        })?;
        for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
            if neighbor.is_some_and(|neighbor| cavity.contains(&neighbor)) {
                continue;
            }
            if neighbor.is_none()
                && node_lies_on_face(topology, tetrahedron.vertex_indices, opposite, node, work)?
            {
                continue;
            }
            work.predicate()?;
            let mut face = [topology.nodes()[tetrahedron.vertex_indices[0] as usize]; 3];
            let mut cursor = 0;
            for (slot, vertex) in tetrahedron.vertex_indices.iter().enumerate() {
                if slot != opposite {
                    face[cursor] = topology.nodes()[*vertex as usize];
                    cursor += 1;
                }
            }
            let interior = topology.nodes()[tetrahedron.vertex_indices[opposite] as usize];
            let interior_sign = topology
                .orient_nodes([face[0], face[1], face[2], interior])
                .map_err(predicate_error)?;
            let node_sign = topology
                .orient_nodes([face[0], face[1], face[2], node])
                .map_err(predicate_error)?;
            if node_sign == PredicateSign::Zero {
                if neighbor.is_none()
                    && node_lies_on_face(
                        topology,
                        tetrahedron.vertex_indices,
                        opposite,
                        node,
                        work,
                    )?
                {
                    continue;
                }
                return Ok(false);
            }
            if node_sign != interior_sign {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

pub(super) fn connected_cavity<T: TetrahedralTopology>(
    topology: &T,
    node: DelaunayVolumeNode,
    seed: u32,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<BTreeSet<u32>, DelaunayInsertionError> {
    let mut cavity = BTreeSet::new();
    let mut examined = BTreeSet::new();
    let mut queue = VecDeque::from([(seed, false)]);
    while let Some((index, forced)) = queue.pop_front() {
        work.checkpoint()?;
        if cavity.contains(&index) || !forced && !examined.insert(index) {
            continue;
        }
        let tetrahedron = topology.tetrahedron(index).ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "insertion cavity reached an inactive tetrahedron",
            )
        })?;
        if !forced && !in_circumsphere(topology, tetrahedron.vertex_indices, node, work)? {
            continue;
        }
        cavity.insert(index);
        if cavity.len() as u64 > work.options.maximum_cavity_tetrahedra {
            return Err(resource("cavity tetrahedron limit exceeded"));
        }
        for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
            if let Some(neighbor) = neighbor {
                if protected_faces.contains(&stable_face(
                    topology,
                    tetrahedron.vertex_indices,
                    opposite,
                )) || protect_region_boundaries
                    && topology
                        .tetrahedron(*neighbor)
                        .is_none_or(|adjacent| adjacent.region_id != tetrahedron.region_id)
                {
                    continue;
                }
                // A symbolic in-sphere tie must never leave a physical zero-volume
                // replacement across a face containing the inserted node.
                let lies_on_face =
                    node_lies_on_face(topology, tetrahedron.vertex_indices, opposite, node, work)?;
                queue.push_back((*neighbor, lies_on_face));
            }
        }
    }
    if !cavity.contains(&seed) {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            "the containing tetrahedron does not contain the node in its circumsphere",
        ));
    }
    Ok(cavity)
}

/// A protected face is an insertion barrier, but an open constrained patch can
/// be reached from its other side without traversing that face. If both
/// incident tetrahedra enter the conflict cavity, ordinary cavity replacement
/// would erase the constraint. Keep only the incident side containing the new
/// point, then discard any conflict islands disconnected from the containing
/// tetrahedron.
pub(super) fn restrict_cavity_to_protected_side(
    topology: &impl TetrahedralTopology,
    cavity: &mut BTreeSet<u32>,
    seed: u32,
    node: DelaunayVolumeNode,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    work: &mut InsertionWork<'_>,
) -> Result<(), DelaunayInsertionError> {
    if protected_faces.is_empty() {
        return Ok(());
    }
    let mut remove = BTreeSet::new();
    for index in cavity.iter().copied() {
        let tetrahedron = topology.tetrahedron(index).ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "protected-side restriction reached an inactive tetrahedron",
            )
        })?;
        for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
            let Some(neighbor) = *neighbor else {
                continue;
            };
            if neighbor <= index
                || !cavity.contains(&neighbor)
                || !protected_faces.contains(&stable_face(
                    topology,
                    tetrahedron.vertex_indices,
                    opposite,
                ))
            {
                continue;
            }
            let mut face = [topology.nodes()[tetrahedron.vertex_indices[0] as usize]; 3];
            let mut cursor = 0;
            for (slot, vertex) in tetrahedron.vertex_indices.iter().enumerate() {
                if slot != opposite {
                    face[cursor] = topology.nodes()[*vertex as usize];
                    cursor += 1;
                }
            }
            let apex = topology.nodes()[tetrahedron.vertex_indices[opposite] as usize];
            work.predicate()?;
            let apex_sign = topology
                .orient_nodes([face[0], face[1], face[2], apex])
                .map_err(predicate_error)?;
            work.predicate()?;
            let node_sign = topology
                .orient_nodes([face[0], face[1], face[2], node])
                .map_err(predicate_error)?;
            if node_sign == PredicateSign::Zero {
                let stable = stable_face(topology, tetrahedron.vertex_indices, opposite);
                if node_lies_on_face(topology, tetrahedron.vertex_indices, opposite, node, work)? {
                    return Err(error(
                        DelaunayInsertionErrorKind::InvalidNode,
                        format!(
                            "inserted node {:?} at {:?} lies on protected face {:?} at {:?}",
                            node.identity,
                            node.coordinates_m.canonical_value(),
                            stable,
                            face.map(|vertex| vertex.coordinates_m.canonical_value()),
                        ),
                    ));
                }
                // A point on the supporting plane but outside the bounded face cannot be coned
                // to that face without creating a zero-volume tetrahedron. Keep both incident
                // tetrahedra; the connected conflict component containing the insertion seed is
                // retained below.
                remove.insert(index);
                remove.insert(neighbor);
                continue;
            }
            if node_sign == apex_sign {
                remove.insert(neighbor);
            } else {
                remove.insert(index);
            }
        }
    }
    cavity.retain(|index| !remove.contains(index));
    if !cavity.contains(&seed) {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            "protected-side restriction removed the containing tetrahedron",
        ));
    }
    let mut connected = BTreeSet::new();
    let mut queue = VecDeque::from([seed]);
    while let Some(index) = queue.pop_front() {
        if !cavity.contains(&index) || !connected.insert(index) {
            continue;
        }
        let tetrahedron = topology
            .tetrahedron(index)
            .expect("restricted cavity tetrahedra remain active");
        for neighbor in tetrahedron.neighbors.into_iter().flatten() {
            if cavity.contains(&neighbor) && !connected.contains(&neighbor) {
                queue.push_back(neighbor);
            }
        }
    }
    *cavity = connected;
    Ok(())
}

pub(super) fn node_lies_on_face(
    topology: &impl TetrahedralTopology,
    vertices: [u32; 4],
    opposite: usize,
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    if !node_coplanar_with_face(topology, vertices, opposite, node, work)? {
        return Ok(false);
    }
    let mut triangle = [[0.0; 3]; 3];
    let mut cursor = 0;
    for (vertex_index, vertex) in vertices.iter().enumerate() {
        if vertex_index != opposite {
            triangle[cursor] = topology.nodes()[*vertex as usize]
                .coordinates_m
                .canonical_value();
            cursor += 1;
        }
    }
    let axes = projection_axes(triangle, work)?;
    let project = |point: [f64; 3]| [point[axes[0]], point[axes[1]]];
    let query = project(node.coordinates_m.canonical_value());
    let signs = [
        planar_orientation([project(triangle[0]), project(triangle[1]), query], work)?,
        planar_orientation([project(triangle[1]), project(triangle[2]), query], work)?,
        planar_orientation([project(triangle[2]), project(triangle[0]), query], work)?,
    ];
    let has_positive = signs.contains(&PredicateSign::Positive);
    let has_negative = signs.contains(&PredicateSign::Negative);
    Ok(!(has_positive && has_negative))
}

fn projection_axes(
    triangle: [[f64; 3]; 3],
    work: &mut InsertionWork<'_>,
) -> Result<[usize; 2], DelaunayInsertionError> {
    work.predicate()?;
    runmat_meshing_core::quality::predicate::triangle_projection_axes(triangle)
        .map_err(|failure| {
            error(
                DelaunayInsertionErrorKind::InvalidNode,
                format!("triangle projection failed: {failure:?}"),
            )
        })?
        .ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "tetrahedron face is exactly collinear",
            )
        })
}

fn planar_orientation(
    points: [[f64; 2]; 3],
    work: &mut InsertionWork<'_>,
) -> Result<PredicateSign, DelaunayInsertionError> {
    work.predicate()?;
    orient2d(points).map_err(|failure| {
        error(
            DelaunayInsertionErrorKind::InvalidNode,
            format!("planar predicate rejected coordinates: {failure:?}"),
        )
    })
}

pub(super) fn stable_face(
    topology: &impl TetrahedralTopology,
    vertices: [u32; 4],
    opposite: usize,
) -> [StableDigest; 3] {
    let mut face = [StableDigest::ZERO; 3];
    let mut cursor = 0;
    for (vertex_index, vertex) in vertices.iter().enumerate() {
        if vertex_index != opposite {
            face[cursor] = topology.nodes()[*vertex as usize].identity;
            cursor += 1;
        }
    }
    face.sort_unstable();
    face
}

pub(super) fn node_coplanar_with_face(
    topology: &impl TetrahedralTopology,
    vertices: [u32; 4],
    opposite: usize,
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    work.predicate()?;
    let mut points = [topology.nodes()[vertices[0] as usize]; 4];
    let mut cursor = 0;
    for (vertex_index, vertex) in vertices.iter().enumerate() {
        if vertex_index != opposite {
            points[cursor] = topology.nodes()[*vertex as usize];
            cursor += 1;
        }
    }
    points[3] = node;
    topology
        .orient_nodes(points)
        .map(|sign| sign == PredicateSign::Zero)
        .map_err(predicate_error)
}

fn in_circumsphere(
    topology: &impl TetrahedralTopology,
    vertices: [u32; 4],
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    // Expanding through every physical zero independently of legalization's
    // symbolic choice destroys valid tied edges and enlarges conflict cavities.
    // On-face traversal remains forced above to prevent zero-volume cells.
    circumsphere_relation(topology, vertices, node, work)
        .map(|sign| sign == PredicateSign::Positive)
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::NeverCancelled;

    use super::*;
    use crate::cdt::{
        build_delaunay_volume_topology, DelaunayInsertionOptions, DelaunayTopologyOptions,
    };

    fn node(identity: u8, coordinates_m: [f64; 3]) -> DelaunayVolumeNode {
        DelaunayVolumeNode::explicit(StableDigest::from_bytes([identity; 32]), coordinates_m)
    }

    #[test]
    fn coplanarity_does_not_force_cavity_traversal_outside_the_face() {
        let topology = build_delaunay_volume_topology(
            vec![
                node(10, [-2.0, -2.0, -2.0]),
                node(20, [2.0, -2.0, -2.0]),
                node(30, [0.0, 2.0, -2.0]),
                node(40, [0.0, 0.0, 2.0]),
            ],
            vec![[0, 1, 2, 3]],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let vertices = topology.tetrahedra[0].vertex_indices;
        let opposite = vertices
            .iter()
            .position(|vertex| {
                topology.nodes[*vertex as usize]
                    .coordinates_m
                    .canonical_value()[2]
                    == 2.0
            })
            .unwrap();
        let mut work = InsertionWork::new(DelaunayInsertionOptions::default(), &NeverCancelled);

        assert!(node_lies_on_face(
            &topology,
            vertices,
            opposite,
            node(50, [0.0, -2.0 / 3.0, -2.0]),
            &mut work,
        )
        .unwrap());
        assert!(!node_lies_on_face(
            &topology,
            vertices,
            opposite,
            node(60, [10.0, 10.0, -2.0]),
            &mut work,
        )
        .unwrap());
    }

    #[test]
    fn protected_face_restriction_keeps_only_the_inserted_point_side() {
        let topology = build_delaunay_volume_topology(
            vec![
                node(10, [0.0, 0.0, 0.0]),
                node(20, [1.0, 0.0, 0.0]),
                node(30, [0.0, 1.0, 0.0]),
                node(40, [0.0, 0.0, 1.0]),
                node(50, [0.0, 0.0, -1.0]),
            ],
            vec![[0, 1, 2, 3], [0, 2, 1, 4]],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let seed = topology
            .tetrahedra
            .iter()
            .position(|tetrahedron| {
                tetrahedron.vertex_indices.iter().any(|vertex| {
                    topology.nodes[*vertex as usize]
                        .coordinates_m
                        .canonical_value()
                        == [0.0, 0.0, 1.0]
                })
            })
            .unwrap() as u32;
        let mut cavity = BTreeSet::from([0, 1]);
        let protected = BTreeSet::from([[
            StableDigest::from_bytes([10; 32]),
            StableDigest::from_bytes([20; 32]),
            StableDigest::from_bytes([30; 32]),
        ]]);
        let mut work = InsertionWork::new(DelaunayInsertionOptions::default(), &NeverCancelled);

        restrict_cavity_to_protected_side(
            &topology,
            &mut cavity,
            seed,
            node(60, [0.2, 0.2, 0.2]),
            &protected,
            &mut work,
        )
        .unwrap();

        assert_eq!(cavity, BTreeSet::from([seed]));
    }

    #[test]
    fn protected_face_restriction_keeps_both_incident_tets_for_coplanar_outside_point() {
        let topology = build_delaunay_volume_topology(
            vec![
                node(10, [0.0, 0.0, 0.0]),
                node(20, [1.0, 0.0, 0.0]),
                node(30, [0.0, 1.0, 0.0]),
                node(40, [0.0, 0.0, 1.0]),
                node(50, [0.0, 0.0, -1.0]),
                node(60, [10.0, 10.0, 1.0]),
                node(70, [11.0, 10.0, 1.0]),
                node(80, [10.0, 11.0, 1.0]),
                node(90, [10.0, 10.0, 2.0]),
            ],
            vec![[0, 1, 2, 3], [0, 2, 1, 4], [5, 6, 7, 8]],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let protected = BTreeSet::from([[
            StableDigest::from_bytes([10; 32]),
            StableDigest::from_bytes([20; 32]),
            StableDigest::from_bytes([30; 32]),
        ]]);
        let seed = topology
            .tetrahedra
            .iter()
            .position(|tetrahedron| {
                tetrahedron.vertex_indices.iter().all(|vertex| {
                    topology.nodes[*vertex as usize]
                        .coordinates_m
                        .canonical_value()[0]
                        >= 10.0
                })
            })
            .unwrap() as u32;
        let incident = (0..topology.tetrahedra.len() as u32)
            .filter(|index| *index != seed)
            .collect::<BTreeSet<_>>();
        let mut cavity = BTreeSet::from_iter(0..topology.tetrahedra.len() as u32);
        let mut work = InsertionWork::new(DelaunayInsertionOptions::default(), &NeverCancelled);

        restrict_cavity_to_protected_side(
            &topology,
            &mut cavity,
            seed,
            node(100, [2.0, 2.0, 0.0]),
            &protected,
            &mut work,
        )
        .unwrap();

        assert_eq!(cavity, BTreeSet::from([seed]));
        assert!(incident.is_disjoint(&cavity));
    }
}
