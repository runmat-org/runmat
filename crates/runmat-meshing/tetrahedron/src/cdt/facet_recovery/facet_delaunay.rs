//! Keeps the current facet's support triangulation consistent with the Steiner nodes that its
//! refinement places on the facet.
//!
//! A recovered support triangle `s` is a protected face. When a refinement step places a node
//! `n` on the facet strictly inside the circumcircle of `s`, every sphere through `s` contains
//! `n`, and `n` sees the cells on `s` through unprotected faces, so no constrained-Delaunay
//! complex holds both. Before such a step, the caller lifts the protection of the current
//! facet's recovered triangles that the step can encroach. After the step, triangles whose face
//! survived are protected again, and the others return to the pending support together with the
//! new nodes: each node is inserted into the facet's unrecovered triangles by a 2D
//! Bowyer-Watson step (the triangles whose circumcircle strictly contains it, bounded by
//! recovered segment edges and by triangles that stay recovered), so the pending support is the
//! facet triangulation that the 3D Delaunay insertion produced.
use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{insphere3d, orient3d, PredicateSign},
    StableDigest,
};

use super::{DelaunayRecoveredFacetTriangle, PendingSupportTriangle};
use crate::cdt::insertion::topology_access::TetrahedralTopology;

fn coordinates(topology: &impl TetrahedralTopology, identity: StableDigest) -> Option<[f64; 3]> {
    topology.node_index(identity).map(|index| {
        topology.nodes()[index as usize]
            .coordinates_m
            .canonical_value()
    })
}

fn triangle_coordinates(
    topology: &impl TetrahedralTopology,
    triangle: [StableDigest; 3],
) -> Option<[[f64; 3]; 3]> {
    let [Some(a), Some(b), Some(c)] = triangle.map(|identity| coordinates(topology, identity))
    else {
        return None;
    };
    Some([a, b, c])
}

fn sub(p: [f64; 3], q: [f64; 3]) -> [f64; 3] {
    [p[0] - q[0], p[1] - q[1], p[2] - q[2]]
}

fn dot(p: [f64; 3], q: [f64; 3]) -> f64 {
    p[0] * q[0] + p[1] * q[1] + p[2] * q[2]
}

fn cross(p: [f64; 3], q: [f64; 3]) -> [f64; 3] {
    [
        p[1] * q[2] - p[2] * q[1],
        p[2] * q[0] - p[0] * q[2],
        p[0] * q[1] - p[1] * q[0],
    ]
}

/// An explicit point off the triangle's plane. Any such point works: the sphere through the
/// triangle and the point meets the plane in the triangle's circumcircle, so the exact in-sphere
/// sign of a coplanar query decides the in-circle relation.
fn lift(triangle: [[f64; 3]; 3]) -> Option<[f64; 3]> {
    let [a, b, c] = triangle;
    let normal = cross(sub(b, a), sub(c, a));
    let length = dot(normal, normal).sqrt();
    let scale = dot(sub(b, a), sub(b, a))
        .max(dot(sub(c, a), sub(c, a)))
        .sqrt();
    if !(length > 0.0 && scale > 0.0 && length.is_finite()) {
        return None;
    }
    let lifted = [0, 1, 2].map(|axis| a[axis] + normal[axis] / length * scale);
    Some(lifted)
}

/// Whether `point`, coplanar with `triangle`, lies strictly inside its circumcircle (exact).
pub(super) fn strictly_inside_circumcircle(triangle: [[f64; 3]; 3], point: [f64; 3]) -> bool {
    let Some(apex) = lift(triangle) else {
        return false;
    };
    let [a, b, c] = triangle;
    let (Ok(orientation), Ok(sphere)) = (
        orient3d([a, b, c, apex]),
        insphere3d([a, b, c, apex, point]),
    ) else {
        return false;
    };
    orientation != PredicateSign::Zero && sphere == orientation
}

/// Whether the open segment `ends` enters the open circumdisk of `triangle` (floating point,
/// with a relative margin that admits near-tangent segments).
fn segment_enters_circumdisk(triangle: [[f64; 3]; 3], ends: [[f64; 3]; 2]) -> bool {
    let [a, b, c] = triangle;
    let (ab, ac) = (sub(b, a), sub(c, a));
    let normal = cross(ab, ac);
    let denominator = 2.0 * dot(normal, normal);
    if denominator == 0.0 {
        return false;
    }
    let first = cross(normal, ab).map(|value| value * dot(ac, ac));
    let second = cross(ac, normal).map(|value| value * dot(ab, ab));
    let offset = [0, 1, 2].map(|axis| (first[axis] + second[axis]) / denominator);
    let center = [0, 1, 2].map(|axis| a[axis] + offset[axis]);
    let radius_squared = dot(offset, offset);
    let direction = sub(ends[1], ends[0]);
    let length_squared = dot(direction, direction);
    if length_squared == 0.0 {
        return false;
    }
    let parameter = (dot(sub(center, ends[0]), direction) / length_squared).clamp(0.0, 1.0);
    let closest = [0, 1, 2].map(|axis| ends[0][axis] + direction[axis] * parameter);
    let distance_squared = dot(sub(closest, center), sub(closest, center));
    distance_squared < radius_squared * (1.0 - 1e-9)
}

/// The current facet's recovered triangles that a node on one of `missing_edges` can encroach.
pub(super) fn encroached_by_edges(
    topology: &impl TetrahedralTopology,
    recovered: &[DelaunayRecoveredFacetTriangle],
    missing_edges: &[[StableDigest; 2]],
) -> Vec<DelaunayRecoveredFacetTriangle> {
    let edges = missing_edges
        .iter()
        .filter_map(|edge| {
            let [Some(first), Some(second)] = edge.map(|identity| coordinates(topology, identity))
            else {
                return None;
            };
            Some([first, second])
        })
        .collect::<Vec<_>>();
    recovered
        .iter()
        .copied()
        .filter(|triangle| {
            triangle_coordinates(topology, triangle.node_identities).is_some_and(|corners| {
                edges
                    .iter()
                    .any(|ends| segment_enters_circumdisk(corners, *ends))
            })
        })
        .collect()
}

/// The current facet's recovered triangles whose circumcircle strictly contains `point`.
pub(super) fn encroached_by_point(
    topology: &impl TetrahedralTopology,
    recovered: &[DelaunayRecoveredFacetTriangle],
    point: [f64; 3],
) -> Vec<DelaunayRecoveredFacetTriangle> {
    recovered
        .iter()
        .copied()
        .filter(|triangle| {
            triangle_coordinates(topology, triangle.node_identities)
                .is_some_and(|corners| strictly_inside_circumcircle(corners, point))
        })
        .collect()
}

fn sorted_edge(first: StableDigest, second: StableDigest) -> [StableDigest; 2] {
    if first <= second {
        [first, second]
    } else {
        [second, first]
    }
}

/// Whether `point` lies in the closed triangle (exact orientation against the lifted apex).
fn contains(triangle: [[f64; 3]; 3], apex: [f64; 3], point: [f64; 3]) -> Option<bool> {
    let [a, b, c] = triangle;
    let reference = orient3d([a, b, c, apex]).ok()?;
    if reference == PredicateSign::Zero {
        return Some(false);
    }
    for (first, second) in [(a, b), (b, c), (c, a)] {
        let sign = orient3d([first, second, point, apex]).ok()?;
        if sign != PredicateSign::Zero && sign != reference {
            return Some(false);
        }
    }
    Some(true)
}

/// Inserts `nodes`, in order, into the unrecovered facet triangles `triangles` by 2D
/// Bowyer-Watson steps. Edges in `segment_edges` (sorted identities) bound every cavity, and so
/// does the outer boundary of `triangles`. A node outside every triangle is skipped. Returns
/// `None` when a cavity is not star-shaped from its node (the caller keeps its own split).
pub(super) fn insert_nodes(
    topology: &impl TetrahedralTopology,
    triangles: Vec<PendingSupportTriangle>,
    nodes: &[StableDigest],
    segment_edges: &BTreeSet<[StableDigest; 2]>,
) -> Option<Vec<PendingSupportTriangle>> {
    insert_nodes_at(
        &|identity| coordinates(topology, identity),
        triangles,
        nodes,
        segment_edges,
    )
}

fn insert_nodes_at(
    coordinates: &impl Fn(StableDigest) -> Option<[f64; 3]>,
    mut triangles: Vec<PendingSupportTriangle>,
    nodes: &[StableDigest],
    segment_edges: &BTreeSet<[StableDigest; 2]>,
) -> Option<Vec<PendingSupportTriangle>> {
    let triangle_coordinates = |triangle: [StableDigest; 3]| {
        let [Some(a), Some(b), Some(c)] = triangle.map(coordinates) else {
            return None;
        };
        Some([a, b, c])
    };
    for node in nodes {
        let point = coordinates(*node)?;
        if triangles
            .iter()
            .any(|entry| entry.triangle.node_identities.contains(node))
        {
            continue;
        }
        let mut corners = Vec::with_capacity(triangles.len());
        for entry in &triangles {
            corners.push(triangle_coordinates(entry.triangle.node_identities)?);
        }
        let mut seeds = Vec::new();
        for (index, triangle) in corners.iter().enumerate() {
            let Some(apex) = lift(*triangle) else {
                continue;
            };
            if contains(*triangle, apex, point)? {
                seeds.push(index);
            }
        }
        if seeds.is_empty() {
            continue;
        }
        let mut owners = BTreeMap::<[StableDigest; 2], Vec<usize>>::new();
        for (index, entry) in triangles.iter().enumerate() {
            let [a, b, c] = entry.triangle.node_identities;
            for (first, second) in [(a, b), (b, c), (c, a)] {
                owners
                    .entry(sorted_edge(first, second))
                    .or_default()
                    .push(index);
            }
        }
        let mut cavity = seeds.iter().copied().collect::<BTreeSet<_>>();
        let mut queue = seeds;
        while let Some(index) = queue.pop() {
            let [a, b, c] = triangles[index].triangle.node_identities;
            for (first, second) in [(a, b), (b, c), (c, a)] {
                let edge = sorted_edge(first, second);
                if segment_edges.contains(&edge) {
                    continue;
                }
                for neighbor in owners.get(&edge).into_iter().flatten().copied() {
                    if cavity.contains(&neighbor)
                        || !strictly_inside_circumcircle(corners[neighbor], point)
                    {
                        continue;
                    }
                    cavity.insert(neighbor);
                    queue.push(neighbor);
                }
            }
        }
        let mut created = Vec::new();
        for index in &cavity {
            let [a, b, c] = triangles[*index].triangle.node_identities;
            let apex = lift(corners[*index])?;
            let reference = {
                let [pa, pb, pc] = corners[*index];
                orient3d([pa, pb, pc, apex]).ok()?
            };
            for (first, second) in [(a, b), (b, c), (c, a)] {
                let edge = sorted_edge(first, second);
                let interior = owners.get(&edge).is_some_and(|owners| {
                    owners
                        .iter()
                        .any(|owner| owner != index && cavity.contains(owner))
                });
                if interior {
                    continue;
                }
                let sign =
                    orient3d([coordinates(first)?, coordinates(second)?, point, apex]).ok()?;
                if sign == PredicateSign::Zero {
                    // The node lies on this boundary edge: the child would have no area.
                    continue;
                }
                if sign != reference {
                    return None;
                }
                created.push(PendingSupportTriangle {
                    triangle: DelaunayRecoveredFacetTriangle {
                        node_identities: [first, second, *node],
                    },
                    lineage_support: triangles[*index].lineage_support,
                });
            }
        }
        let mut kept = created;
        kept.extend(
            triangles
                .into_iter()
                .enumerate()
                .filter(|(index, _)| !cavity.contains(index))
                .map(|(_, entry)| entry),
        );
        triangles = kept;
    }
    Some(triangles)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(index: u8) -> StableDigest {
        StableDigest::from_bytes([index + 1; 32])
    }

    fn pending(triangle: [u8; 3]) -> PendingSupportTriangle {
        let node_identities = triangle.map(id);
        PendingSupportTriangle {
            triangle: DelaunayRecoveredFacetTriangle { node_identities },
            lineage_support: node_identities,
        }
    }

    #[test]
    fn coplanar_in_circle_is_exact_on_both_windings() {
        let triangle = [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]];
        let reversed = [triangle[0], triangle[2], triangle[1]];
        for corners in [triangle, reversed] {
            assert!(strictly_inside_circumcircle(corners, [0.9, 0.9, 0.5]));
            assert!(!strictly_inside_circumcircle(corners, [1.0, 1.0, 0.5]));
            assert!(!strictly_inside_circumcircle(corners, [1.1, 1.0, 0.5]));
        }
    }

    /// A sliver support triangle with a large circumcircle next to the triangle being split:
    /// the node lands outside the sliver but inside its circumcircle, so the sliver must join
    /// the cavity and the edge it shares with the split triangle must disappear.
    #[test]
    fn node_in_a_sliver_circumcircle_retriangulates_the_sliver() {
        // 0-1-2 is a sliver (2 just above the chord 0-1); 0-3-1 lies below the chord.
        let points = [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 0.05, 0.0],
            [2.0, -2.0, 0.0],
            [2.0, -0.05, 0.0],
        ];
        let lookup = |identity: StableDigest| {
            (0..points.len() as u8)
                .find(|index| id(*index) == identity)
                .map(|index| points[index as usize])
        };
        let segment_edges = [[0, 2], [2, 1], [1, 3], [3, 0]]
            .map(|[a, b]| sorted_edge(id(a), id(b)))
            .into_iter()
            .collect();
        let triangles = vec![pending([0, 3, 1]), pending([0, 1, 2])];
        let result = insert_nodes_at(&lookup, triangles, &[id(4)], &segment_edges)
            .expect("the cavity is star-shaped");
        let has_chord = result.iter().any(|entry| {
            let nodes = entry.triangle.node_identities;
            nodes.contains(&id(0)) && nodes.contains(&id(1))
        });
        assert!(
            !has_chord,
            "the chord 0-1 of the sliver survived: {}",
            result.len()
        );
        assert_eq!(result.len(), 4);
        for entry in &result {
            let [a, b, c] = entry
                .triangle
                .node_identities
                .map(|node| lookup(node).unwrap());
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            assert!(area > 0.0, "child is not counter-clockwise");
        }
    }

    #[test]
    fn segment_edges_bound_the_cavity() {
        let points = [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 0.05, 0.0],
            [2.0, -2.0, 0.0],
            [2.0, -0.05, 0.0],
        ];
        let lookup = |identity: StableDigest| {
            (0..points.len() as u8)
                .find(|index| id(*index) == identity)
                .map(|index| points[index as usize])
        };
        let segment_edges = [[0, 2], [2, 1], [1, 3], [3, 0], [0, 1]]
            .map(|[a, b]| sorted_edge(id(a), id(b)))
            .into_iter()
            .collect();
        let triangles = vec![pending([0, 3, 1]), pending([0, 1, 2])];
        let result = insert_nodes_at(&lookup, triangles, &[id(4)], &segment_edges)
            .expect("the cavity is star-shaped");
        assert_eq!(result.len(), 4);
        assert!(result
            .iter()
            .any(|entry| entry.triangle.node_identities == [0, 1, 2].map(id)));
    }
}
