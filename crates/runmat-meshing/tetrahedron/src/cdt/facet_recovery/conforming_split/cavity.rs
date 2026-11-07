//! The constrained Bowyer-Watson cavity of a node inserted on a recovered segment edge.
use super::*;

/// The Bowyer-Watson cavity of a node inserted on a recovered segment edge: the edge star plus
/// every tetrahedron reachable from it whose circumsphere strictly contains the node. Protected
/// faces other than the triangles split at the edge, and region interfaces, are barriers.
pub(super) fn constrained_cavity(
    topology: &TopologyMutation,
    inserted: u32,
    star: BTreeSet<u32>,
    edge: [StableDigest; 2],
    protected: &[DelaunayRecoveredFacetTriangle],
    constraint_index: u32,
) -> Result<BTreeSet<u32>, DelaunayFacetRecoveryError> {
    const MAXIMUM_CAVITY_TETRAHEDRA: usize = 1 << 16;
    let barriers = barrier_faces(protected, edge);
    let node = topology.nodes[inserted as usize];
    let mut cavity = star.clone();
    let mut frontier = star.into_iter().collect::<Vec<_>>();
    while let Some(index) = frontier.pop() {
        let tet = topology
            .tetrahedron(index)
            .expect("cavity tetrahedra remain active");
        for (slot, neighbor) in tet.neighbors.iter().enumerate() {
            let Some(neighbor) = *neighbor else { continue };
            if cavity.contains(&neighbor) {
                continue;
            }
            let other = topology
                .tetrahedron(neighbor)
                .expect("neighbor tetrahedra remain active");
            if other.region_id != tet.region_id {
                continue;
            }
            if barriers.contains(&canonical_mutation_face(topology, tet.vertex_indices, slot)) {
                continue;
            }
            let points = other
                .vertex_indices
                .map(|vertex| topology.nodes[vertex as usize]);
            let inside = crate::cdt::topology::insphere_materialized_volume_nodes([
                points[0], points[1], points[2], points[3], node,
            ])
            .map_err(|failure| {
                invalid(
                    constraint_index,
                    format!("conforming segment cavity predicate failed: {failure:?}"),
                )
            })? == PredicateSign::Positive;
            if inside {
                if cavity.len() >= MAXIMUM_CAVITY_TETRAHEDRA {
                    return Err(error(
                        DelaunayFacetRecoveryErrorKind::ResourceLimit,
                        Some(constraint_index),
                        "conforming segment cavity exceeded its tetrahedron limit",
                    ));
                }
                cavity.insert(neighbor);
                frontier.push(neighbor);
            }
        }
    }
    Ok(cavity)
}

/// A cavity face is visible from the inserted node when the cone over it is positively
/// oriented and not flat: a node constructed on a facet plane sits off that plane only by
/// rounding, and a cone over another triangle of the same facet would be a zero-volume sliver.
pub(super) fn cone_is_visible(topology: &TopologyMutation, vertices: [u32; 4]) -> bool {
    const MINIMUM_RELATIVE_VOLUME: f64 = 1.0e-10;
    let nodes = vertices.map(|vertex| topology.nodes[vertex as usize]);
    if !orient_materialized_volume_nodes(nodes).is_ok_and(|sign| sign == PredicateSign::Positive) {
        return false;
    }
    let points = nodes.map(|node| node.coordinates_m.canonical_value());
    let edge = |a: usize, b: usize| -> [f64; 3] {
        std::array::from_fn(|axis| points[b][axis] - points[a][axis])
    };
    let (u, v, w) = (edge(0, 1), edge(0, 2), edge(0, 3));
    let volume = (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0]))
        .abs();
    let longest = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        .into_iter()
        .map(|(a, b)| {
            edge(a, b)
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max);
    volume >= MINIMUM_RELATIVE_VOLUME * longest.powi(3)
}

pub(super) fn barrier_faces(
    protected: &[DelaunayRecoveredFacetTriangle],
    edge: [StableDigest; 2],
) -> BTreeSet<[StableDigest; 3]> {
    protected
        .iter()
        .filter(|triangle| {
            !edge
                .iter()
                .all(|node| triangle.node_identities.contains(node))
        })
        .map(|triangle| {
            let mut face = triangle.node_identities;
            face.sort_unstable();
            face
        })
        .collect()
}

pub(super) fn canonical_mutation_face(
    topology: &TopologyMutation,
    vertices: [u32; 4],
    opposite: usize,
) -> [StableDigest; 3] {
    let mut face = [StableDigest::ZERO; 3];
    for (cursor, vertex) in (0..4).filter(|vertex| *vertex != opposite).enumerate() {
        face[cursor] = topology.nodes[vertices[vertex] as usize].identity;
    }
    face.sort_unstable();
    face
}

/// The cavity tetrahedra reachable from the edge star across non-barrier faces.
pub(super) fn connected_from_star(
    topology: &TopologyMutation,
    cavity: &BTreeSet<u32>,
    star: &BTreeSet<u32>,
    barriers: &BTreeSet<[StableDigest; 3]>,
) -> BTreeSet<u32> {
    let mut connected = star.clone();
    let mut frontier = star.iter().copied().collect::<Vec<_>>();
    while let Some(index) = frontier.pop() {
        let tet = topology
            .tetrahedron(index)
            .expect("cavity tetrahedra remain active");
        for (slot, neighbor) in tet.neighbors.iter().enumerate() {
            let Some(neighbor) = *neighbor else { continue };
            if !cavity.contains(&neighbor)
                || connected.contains(&neighbor)
                || barriers.contains(&canonical_mutation_face(topology, tet.vertex_indices, slot))
            {
                continue;
            }
            connected.insert(neighbor);
            frontier.push(neighbor);
        }
    }
    connected
}

pub(super) fn split_faces(
    topology: &TopologyMutation,
    inserted: u32,
    hits: &[(DelaunayRecoveredFacetTriangle, PointLocation)],
    constraint_index: u32,
) -> Result<Replacement, DelaunayFacetRecoveryError> {
    let mut removed = BTreeSet::new();
    let mut created = Vec::new();
    let mut splits = Vec::new();
    for (triangle, _) in hits {
        let face = triangle.node_identities.map(|identity| {
            node_index_mutation(topology, identity)
                .ok_or_else(|| invalid(constraint_index, "facet split node is missing"))
        });
        let [a, b, c] = face;
        let face = [a?, b?, c?];
        for index in 0..topology.tetrahedron_slot_count() {
            let Some(tet) = topology.tetrahedron(index as u32) else {
                continue;
            };
            if !face.iter().all(|node| tet.vertex_indices.contains(node)) {
                continue;
            }
            removed.insert(index as u32);
            let apex = *tet
                .vertex_indices
                .iter()
                .find(|node| !face.contains(node))
                .expect("face tet has apex");
            created.extend([
                ([face[0], face[1], inserted, apex], tet.region_id.clone()),
                ([face[1], face[2], inserted, apex], tet.region_id.clone()),
                ([face[2], face[0], inserted, apex], tet.region_id.clone()),
            ]);
        }
        let identity = topology.nodes[inserted as usize].identity;
        splits.push(ProtectedFacetSplit {
            source: *triangle,
            children: vec![
                DelaunayRecoveredFacetTriangle {
                    node_identities: [
                        triangle.node_identities[0],
                        triangle.node_identities[1],
                        identity,
                    ],
                },
                DelaunayRecoveredFacetTriangle {
                    node_identities: [
                        triangle.node_identities[1],
                        triangle.node_identities[2],
                        identity,
                    ],
                },
                DelaunayRecoveredFacetTriangle {
                    node_identities: [
                        triangle.node_identities[2],
                        triangle.node_identities[0],
                        identity,
                    ],
                },
            ],
        });
    }
    Ok((removed, created, splits))
}

pub(super) fn updated_protected_faces(
    protected: &[DelaunayRecoveredFacetTriangle],
    splits: &[ProtectedFacetSplit],
) -> Vec<[StableDigest; 3]> {
    protected
        .iter()
        .filter(|triangle| !splits.iter().any(|split| split.source == **triangle))
        .flat_map(|triangle| vec![*triangle])
        .chain(splits.iter().flat_map(|split| split.children.clone()))
        .map(|triangle| {
            let mut face = triangle.node_identities;
            face.sort_unstable();
            face
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
