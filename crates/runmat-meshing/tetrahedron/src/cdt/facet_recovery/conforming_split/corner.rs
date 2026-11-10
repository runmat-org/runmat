//! Recovered segment edges that a pair-split node blocks inside a facet corner.
use super::*;

/// Recovered segment edges that a pair-split node blocks, with their midpoints. A split node
/// `n` on edge `uv` of a facet triangle `uvw` creates the child triangle `unw` (and `nvw`).
/// When a vertex `x` of the coplanar triangle across the segment edge `uw` lies strictly
/// inside the circumcircle of `unw`, every cell on `unw` has `x` inside its circumsphere, and
/// no flip may remove the segment edge `uw`, so no constrained-Delaunay complex holds `n`.
/// Splitting `uw` at its midpoint removes the conflict. The triangles come from the current
/// tetrahedralization (vertices adjacent to both ends and coplanar), so facets not yet
/// recovered count too. Each midpoint is checked the same way in turn (a bounded cascade),
/// which declines past [`MAXIMUM_ENCROACHED_CORNER_SPLITS`] splits. The circle test is a
/// floating-point trigger; the exact admission of the whole edit decides.
pub(super) fn encroached_corner_segments(
    working: &super::super::working::FacetRecoveryWorking,
    splits: &[([StableDigest; 2], DelaunayVolumeNode)],
    constraint_index: u32,
) -> Result<Result<Vec<EncroachedCornerSegment>, String>, DelaunayFacetRecoveryError> {
    let topology = working.topology();
    let index_of = |identity: StableDigest| {
        topology
            .node_index(identity)
            .ok_or_else(|| invalid(constraint_index, "facet vertex is missing"))
    };
    let coordinates = |index: u32| {
        topology.nodes[index as usize]
            .coordinates_m
            .canonical_value()
    };
    let neighbors = |index: u32| {
        topology.vertex_stars[index as usize]
            .iter()
            .filter_map(|cell| topology.tetrahedron(*cell))
            .flat_map(|cell| cell.vertex_indices)
            .filter(|vertex| *vertex != index)
            .collect::<BTreeSet<_>>()
    };
    let intervals = working
        .segments
        .iter()
        .flat_map(|segment| segment.nodes.windows(2))
        .map(|pair| {
            let mut key = [pair[0].identity, pair[1].identity];
            key.sort_unstable();
            key
        })
        .collect::<BTreeSet<_>>();
    let mut split_keys = splits
        .iter()
        .map(|(split, _)| {
            let mut key = *split;
            key.sort_unstable();
            key
        })
        .collect::<Vec<_>>();
    split_keys.sort_unstable();
    let mut found = Vec::<EncroachedCornerSegment>::new();
    let mut queue = splits
        .iter()
        .copied()
        .collect::<std::collections::VecDeque<_>>();
    while let Some((split, node)) = queue.pop_front() {
        let [u, v] = [index_of(split[0])?, index_of(split[1])?];
        let shared = neighbors(u)
            .intersection(&neighbors(v))
            .copied()
            .collect::<Vec<_>>();
        for w in shared {
            for (end, opposite) in [(u, v), (v, u)] {
                let mut key = [
                    topology.nodes[end as usize].identity,
                    topology.nodes[w as usize].identity,
                ];
                key.sort_unstable();
                if !intervals.contains(&key)
                    || split_keys.contains(&key)
                    || found.iter().any(|(edge, _, _)| *edge == key)
                {
                    continue;
                }
                let plane = [end, w, opposite].map(|vertex| topology.nodes[vertex as usize]);
                let child = [
                    coordinates(end),
                    node.coordinates_m.canonical_value(),
                    coordinates(w),
                ];
                let blocked = neighbors(end).intersection(&neighbors(w)).any(|x| {
                    *x != opposite
                        && orient_volume_nodes([
                            plane[0],
                            plane[1],
                            plane[2],
                            topology.nodes[*x as usize],
                        ])
                        .is_ok_and(|sign| sign == PredicateSign::Zero)
                        && opposite_sides(
                            coordinates(end),
                            coordinates(w),
                            coordinates(opposite),
                            coordinates(*x),
                        )
                        && strictly_inside_circumcircle(child, coordinates(*x))
                });
                if !blocked {
                    continue;
                }
                let Some(location) = segment_edge_location(&working.segments, key) else {
                    continue;
                };
                let (midpoint, recovered) = recovered_segment_midpoint_on(
                    topology,
                    &working.segments,
                    location.0,
                    location.1,
                )
                .map_err(super::super::segment_error)?;
                if topology.nodes.iter().any(|existing| {
                    crate::cdt::segment_recovery::coordinates_share_representable_location(
                        existing.coordinates_m.canonical_value(),
                        midpoint.coordinates_m.canonical_value(),
                    )
                }) {
                    return Ok(Err(
                        "an encroached segment midpoint duplicates an existing node position"
                            .to_owned(),
                    ));
                }
                if found.len() == MAXIMUM_ENCROACHED_CORNER_SPLITS {
                    return Ok(Err(format!(
                        "the encroached-segment cascade exceeds {MAXIMUM_ENCROACHED_CORNER_SPLITS} splits"
                    )));
                }
                found.push((key, midpoint, recovered));
                queue.push_back((key, midpoint));
            }
        }
    }
    Ok(Ok(found))
}

/// Whether `first` and `second` lie strictly on opposite sides of line `ab` in their common
/// plane (floating point).
pub(super) fn opposite_sides(a: [f64; 3], b: [f64; 3], first: [f64; 3], second: [f64; 3]) -> bool {
    let sub = |p: [f64; 3], q: [f64; 3]| [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
    let cross = |p: [f64; 3], q: [f64; 3]| {
        [
            p[1] * q[2] - p[2] * q[1],
            p[2] * q[0] - p[0] * q[2],
            p[0] * q[1] - p[1] * q[0],
        ]
    };
    let line = sub(b, a);
    let one = cross(line, sub(first, a));
    let two = cross(line, sub(second, a));
    (0..3).map(|axis| one[axis] * two[axis]).sum::<f64>() < 0.0
}

/// Whether `point`, coplanar with `triangle`, lies strictly inside its circumcircle, with a
/// relative margin for rounding (floating point).
pub(super) fn strictly_inside_circumcircle(triangle: [[f64; 3]; 3], point: [f64; 3]) -> bool {
    let sub = |p: [f64; 3], q: [f64; 3]| [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
    let dot = |p: [f64; 3], q: [f64; 3]| (0..3).map(|axis| p[axis] * q[axis]).sum::<f64>();
    let cross = |p: [f64; 3], q: [f64; 3]| {
        [
            p[1] * q[2] - p[2] * q[1],
            p[2] * q[0] - p[0] * q[2],
            p[0] * q[1] - p[1] * q[0],
        ]
    };
    let [a, b, c] = triangle;
    let (ab, ac) = (sub(b, a), sub(c, a));
    let normal = cross(ab, ac);
    let denominator = 2.0 * dot(normal, normal);
    if denominator == 0.0 {
        return false;
    }
    let term = |scale: f64, vector: [f64; 3]| vector.map(|value| value * scale);
    let first = term(dot(ac, ac), cross(normal, ab));
    let second = term(dot(ab, ab), cross(ac, normal));
    let offset = [0, 1, 2].map(|axis| (first[axis] + second[axis]) / denominator);
    let center = [0, 1, 2].map(|axis| a[axis] + offset[axis]);
    let radius_squared = dot(offset, offset);
    let distance_squared = dot(sub(point, center), sub(point, center));
    distance_squared < radius_squared * (1.0 - 1e-9)
}

pub(super) const MAXIMUM_ENCROACHED_CORNER_SPLITS: usize = 64;
