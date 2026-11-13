//! Chooses the recovered segment edge to split when a temporary support segment has no
//! constrained Delaunay result.
//!
//! Recovering a missing support edge places Steiner nodes on it. A node strictly inside the
//! diametral ball of a recovered segment edge encroaches that edge: when the facets on the edge
//! are nearly coplanar, the cells beside the edge then admit no constrained Delaunay complex,
//! and validation rejects the temporary chain. The standard remedy splits the encroached segment
//! edge instead of placing the node. This module only selects the edge; the split runs through
//! the conforming insertion and its exact validation.
use runmat_meshing_core::StableDigest;

use super::super::node_index;
use crate::cdt::insertion::topology_access::TetrahedralTopology;

/// The recovered segment edge whose open diametral ball the open support edge `edge` enters
/// deepest, excluding edges that share an endpoint with it. Floating point with a relative
/// margin: the choice only directs a split that exact validation then admits or rejects.
pub(super) fn deepest_encroached_segment_edge(
    topology: &impl TetrahedralTopology,
    edge: [StableDigest; 2],
    protected_edges: &[[StableDigest; 2]],
) -> Option<[StableDigest; 2]> {
    deepest_encroached_at(
        |identity| {
            node_index(topology, identity)
                .map(|index| topology.nodes()[index].coordinates_m.canonical_value())
        },
        edge,
        protected_edges,
    )
}

fn deepest_encroached_at(
    point: impl Fn(StableDigest) -> Option<[f64; 3]>,
    edge: [StableDigest; 2],
    protected_edges: &[[StableDigest; 2]],
) -> Option<[StableDigest; 2]> {
    let [Some(first), Some(last)] = edge.map(&point) else {
        return None;
    };
    let lower = [0, 1, 2].map(|axis| first[axis].min(last[axis]));
    let upper = [0, 1, 2].map(|axis| first[axis].max(last[axis]));
    let mut best: Option<(f64, [StableDigest; 2])> = None;
    for candidate in protected_edges.iter().copied() {
        if candidate.iter().any(|identity| edge.contains(identity)) {
            continue;
        }
        let [Some(a), Some(b)] = candidate.map(&point) else {
            continue;
        };
        let center = [0, 1, 2].map(|axis| 0.5 * (a[axis] + b[axis]));
        let radius_squared = 0.25 * distance_squared(a, b);
        if radius_squared == 0.0 {
            continue;
        }
        let radius = radius_squared.sqrt();
        if (0..3)
            .any(|axis| center[axis] + radius < lower[axis] || center[axis] - radius > upper[axis])
        {
            continue;
        }
        let ratio = closest_distance_squared(first, last, center) / radius_squared;
        if ratio < 1.0 - 1e-9 && best.is_none_or(|(current, _)| ratio < current) {
            best = Some((ratio, candidate));
        }
    }
    best.map(|(_, candidate)| candidate)
}

/// The recovered segment edge of `triangle` whose diametral ball holds the triangle's opposite
/// corner, that is, the segment edge opposite an obtuse angle. The corner encroaches that edge,
/// and the triangle's circumcenter lies beyond it, outside the facet. Ruppert's and Shewchuk's
/// refinement split the encroached segment edge before the facet; a node inside such a needle
/// lies close to all three of its corners. Floating point with a relative margin: the choice
/// only directs a split that exact validation then admits or rejects.
pub(super) fn segment_edge_encroached_by_apex(
    topology: &impl TetrahedralTopology,
    triangle: [StableDigest; 3],
    protected_edges: &[[StableDigest; 2]],
) -> Option<[StableDigest; 2]> {
    encroached_by_apex_at(
        |identity| {
            node_index(topology, identity)
                .map(|index| topology.nodes()[index].coordinates_m.canonical_value())
        },
        triangle,
        protected_edges,
    )
}

fn encroached_by_apex_at(
    point: impl Fn(StableDigest) -> Option<[f64; 3]>,
    triangle: [StableDigest; 3],
    protected_edges: &[[StableDigest; 2]],
) -> Option<[StableDigest; 2]> {
    (0..3).find_map(|apex_slot| {
        let mut edge = [triangle[(apex_slot + 1) % 3], triangle[(apex_slot + 2) % 3]];
        edge.sort_unstable();
        if !protected_edges.contains(&edge) {
            return None;
        }
        let [Some(apex), Some(a), Some(b)] = [triangle[apex_slot], edge[0], edge[1]].map(&point)
        else {
            return None;
        };
        let to_a = [0, 1, 2].map(|axis| a[axis] - apex[axis]);
        let to_b = [0, 1, 2].map(|axis| b[axis] - apex[axis]);
        let cosine_scaled: f64 = (0..3).map(|axis| to_a[axis] * to_b[axis]).sum();
        let scale = distance_squared(a, apex).sqrt() * distance_squared(b, apex).sqrt();
        (cosine_scaled < -1e-9 * scale).then_some(edge)
    })
}

/// The edge of `triangle`, among the recovered segment edges `protected_edges`, whose open
/// diametral ball holds `point` deepest.
/// `point` is the facet-interior split node of the triangle. Inside that ball the node encroaches
/// the segment edge: when the facet on the other side of the edge is nearly coplanar, the node
/// lies a few micrometres off that facet's plane, inside its circumcircle and visible from its far
/// side, and the cells beside the edge then admit no constrained Delaunay complex. Floating point
/// with a relative margin: the choice only directs a split that exact validation then admits or
/// rejects.
pub(super) fn segment_edge_encroached_by_point(
    topology: &impl TetrahedralTopology,
    triangle: [StableDigest; 3],
    point: [f64; 3],
    protected_edges: &[[StableDigest; 2]],
) -> Option<[StableDigest; 2]> {
    encroached_by_point_at(
        |identity| {
            node_index(topology, identity)
                .map(|index| topology.nodes()[index].coordinates_m.canonical_value())
        },
        triangle,
        point,
        protected_edges,
    )
}

fn encroached_by_point_at(
    coordinates: impl Fn(StableDigest) -> Option<[f64; 3]>,
    triangle: [StableDigest; 3],
    point: [f64; 3],
    protected_edges: &[[StableDigest; 2]],
) -> Option<[StableDigest; 2]> {
    let mut best: Option<(f64, [StableDigest; 2])> = None;
    for slot in 0..3 {
        let mut edge = [triangle[slot], triangle[(slot + 1) % 3]];
        edge.sort_unstable();
        if !protected_edges.contains(&edge) {
            continue;
        }
        let [Some(a), Some(b)] = edge.map(&coordinates) else {
            continue;
        };
        let radius_squared = 0.25 * distance_squared(a, b);
        if radius_squared == 0.0 {
            continue;
        }
        let center = [0, 1, 2].map(|axis| 0.5 * (a[axis] + b[axis]));
        let ratio = distance_squared(point, center) / radius_squared;
        if ratio < 1.0 - 1e-9 && best.is_none_or(|(current, _)| ratio < current) {
            best = Some((ratio, edge));
        }
    }
    best.map(|(_, edge)| edge)
}

fn distance_squared(p: [f64; 3], q: [f64; 3]) -> f64 {
    (0..3).map(|axis| (p[axis] - q[axis]).powi(2)).sum()
}

fn closest_distance_squared(first: [f64; 3], last: [f64; 3], point: [f64; 3]) -> f64 {
    let direction = [0, 1, 2].map(|axis| last[axis] - first[axis]);
    let length_squared: f64 = direction.iter().map(|value| value * value).sum();
    if length_squared == 0.0 {
        return distance_squared(first, point);
    }
    let offset: f64 = (0..3)
        .map(|axis| (point[axis] - first[axis]) * direction[axis])
        .sum();
    let parameter = (offset / length_squared).clamp(0.0, 1.0);
    let closest = [0, 1, 2].map(|axis| first[axis] + direction[axis] * parameter);
    distance_squared(closest, point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn id(index: u8) -> StableDigest {
        StableDigest::from_bytes([index + 1; 32])
    }

    // witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: A support edge that passes through the diametral ball of a nearby recovered
    //   segment edge selects that edge for splitting; an edge sharing an endpoint with the
    //   support edge and an edge whose ball the support edge misses are not selected. This local
    //   fixture does not discharge original-model meshing, physical quality, resource or
    //   scientific gates.
    #[test]
    fn support_edge_selects_the_segment_edge_whose_diametral_ball_it_enters_deepest() {
        let points = BTreeMap::from([
            (id(0), [0.0, 0.0, 0.0]),
            (id(1), [4.0, 0.0, 0.0]),
            // Segment edge 2-3: its ball (center (2, 1), radius 1.5) holds the support edge's
            // middle.
            (id(2), [0.5, 1.0, 0.0]),
            (id(3), [3.5, 1.0, 0.0]),
            // Segment edge 4-5 lies far from the support edge.
            (id(4), [0.0, 5.0, 0.0]),
            (id(5), [1.0, 5.0, 0.0]),
            // Segment edge 6-7: its ball (center (2, 1.2), radius 1.25) is entered less deeply.
            (id(6), [0.75, 1.2, 0.0]),
            (id(7), [3.25, 1.2, 0.0]),
        ]);
        let point = |identity| points.get(&identity).copied();
        let protected = [
            [id(0), id(2)],
            [id(6), id(7)],
            [id(2), id(3)],
            [id(4), id(5)],
        ];
        assert_eq!(
            deepest_encroached_at(point, [id(0), id(1)], &protected),
            Some([id(2), id(3)])
        );
        assert_eq!(
            deepest_encroached_at(point, [id(0), id(1)], &[[id(0), id(2)], [id(4), id(5)]]),
            None
        );
    }

    // witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: A facet-interior split node selects, among the candidate segment edges of its
    //   triangle, the one whose diametral ball holds it deepest; a candidate whose ball the node
    //   misses, and an edge that is no candidate, are not selected. This local fixture does not
    //   discharge original-model meshing, physical quality, resource or scientific gates.
    #[test]
    fn interior_node_selects_the_segment_edge_whose_diametral_ball_holds_it() {
        let points = BTreeMap::from([
            // The thin-wall facet of the motor-pod arm, in millimetres: the 2:1:1 node lies
            // 0.5 mm from the 1.25 mm edge 1-2, inside its 0.625 mm diametral ball, and deeper
            // inside the balls of edges 0-1 and 0-2.
            (id(0), [0.0, 0.0, 0.0]),
            (id(1), [-0.5, 1.0, -0.625]),
            (id(2), [-0.5, 1.0, 0.625]),
        ]);
        let point = |identity| points.get(&identity).copied();
        let triangle = [id(0), id(1), id(2)];
        let node = [-0.25, 0.5, 0.0];
        // The caller passes only the edges beside a recovered facet that the node encroaches.
        assert_eq!(
            encroached_by_point_at(point, triangle, node, &[[id(1), id(2)]]),
            Some([id(1), id(2)])
        );
        assert_eq!(
            encroached_by_point_at(point, triangle, node, &[[id(0), id(1)], [id(1), id(2)]]),
            Some([id(0), id(1)])
        );
        assert_eq!(
            encroached_by_point_at(point, triangle, [-0.1, 0.2, 0.0], &[[id(1), id(2)]]),
            None
        );
        assert_eq!(
            encroached_by_point_at(point, triangle, node, &[[id(0), id(3)]]),
            None
        );
    }

    // witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: A support triangle whose obtuse corner lies opposite a recovered segment edge
    //   selects that edge for splitting; the same triangle selects nothing when the edge opposite
    //   the obtuse corner is a plain support edge, and an acute triangle selects nothing. This
    //   local fixture does not discharge original-model meshing, physical quality, resource or
    //   scientific gates.
    #[test]
    fn obtuse_corner_selects_the_segment_edge_opposite_it() {
        let points = BTreeMap::from([
            // A needle with an angle of about 168 degrees at node 2, opposite edge 0-1.
            (id(0), [0.0, 0.0, 0.0]),
            (id(1), [3.57, 0.339, 0.0]),
            (id(2), [2.07, 0.0, 0.0]),
            (id(3), [1.0, 2.0, 0.0]),
        ]);
        let point = |identity| points.get(&identity).copied();
        let needle = [id(0), id(1), id(2)];
        let segments = [[id(0), id(1)], [id(0), id(2)]];
        assert_eq!(
            encroached_by_apex_at(point, needle, &segments),
            Some([id(0), id(1)])
        );
        assert_eq!(
            encroached_by_apex_at(point, needle, &[[id(0), id(2)]]),
            None
        );
        assert_eq!(
            encroached_by_apex_at(
                point,
                [id(0), id(2), id(3)],
                &[[id(0), id(2)], [id(2), id(3)]]
            ),
            None
        );
    }
}
