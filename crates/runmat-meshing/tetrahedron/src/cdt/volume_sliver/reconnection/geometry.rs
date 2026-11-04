//! Exact local reconnection geometry and deterministic candidate enumeration.
use super::*;
mod indices;

pub(super) fn local_candidates(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    source_index: usize,
    metric: runmat_meshing_size::metric::MetricTensor3,
) -> Result<Vec<LocalTetrahedronFlipCandidate>, DelaunayVolumeSliverError> {
    let mut candidates = two_to_three_candidates(topology, source_index)?;
    candidates.extend(three_to_two_candidates(topology, source_index)?);
    candidates.extend(edge_star_candidates(topology, source_index)?);
    let mut accounting = None;
    candidates.sort_by(|left, right| {
        if accounting.is_some() {
            return std::cmp::Ordering::Equal;
        }
        let right_score = match candidate_minimum_jacobian(topology, right, metric) {
            Ok(score) => score,
            Err(cause) => {
                accounting = Some(cause);
                return std::cmp::Ordering::Equal;
            }
        };
        let left_score = match candidate_minimum_jacobian(topology, left, metric) {
            Ok(score) => score,
            Err(cause) => {
                accounting = Some(cause);
                return std::cmp::Ordering::Equal;
            }
        };
        crate::cdt::volume_quality::violation_priority_key(right_score)
            .cmp(&crate::cdt::volume_quality::violation_priority_key(
                left_score,
            ))
            .then_with(|| {
                candidate_geometry_key(topology, left).cmp(&candidate_geometry_key(topology, right))
            })
            .then_with(|| candidate_key(topology, left).cmp(&candidate_key(topology, right)))
    });
    if let Some(cause) = accounting {
        return Err(error(
            DelaunayVolumeSliverErrorKind::Accounting(cause),
            "reconnection score numerical accounting failed",
        ));
    }
    candidates
        .dedup_by(|left, right| candidate_key(topology, left) == candidate_key(topology, right));
    Ok(candidates)
}

fn two_to_three_candidates(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    source_index: usize,
) -> Result<Vec<LocalTetrahedronFlipCandidate>, DelaunayVolumeSliverError> {
    let source = &topology.cell((source_index) as u32);
    let mut candidates = Vec::new();
    for neighbor_index in source.neighbors.into_iter().flatten() {
        let neighbor = &topology.cell(neighbor_index);
        if source.region_id != neighbor.region_id
            || !two_to_three_geometry_is_admissible(
                source.vertex_indices,
                neighbor.vertex_indices,
                |index| topology.node(index).predicate_point(),
            )
            .map_err(|failure| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    format!("2-to-3 reconnection predicate failed: {failure:?}"),
                )
            })?
        {
            continue;
        }
        if let Ok(candidate) = indices::flip(
            topology,
            [source_index as u32, neighbor_index],
            |[left, right], _| two_to_three_face_flip_candidate(left, right),
        ) {
            candidates.push(candidate);
        }
    }
    Ok(candidates)
}

fn three_to_two_candidates(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    source_index: usize,
) -> Result<Vec<LocalTetrahedronFlipCandidate>, DelaunayVolumeSliverError> {
    let vertices = topology.cell((source_index) as u32).vertex_indices;
    let mut candidates = Vec::new();
    for left in 0..4 {
        for right in (left + 1)..4 {
            let edge = [vertices[left], vertices[right]];
            let star = topology.edge_star(edge);
            let [first, second, third] = star.as_slice() else {
                continue;
            };
            if star.iter().any(|index| {
                topology.cell(*index).region_id != topology.cell(source_index as u32).region_id
            }) {
                continue;
            }
            let Ok(candidate) =
                indices::flip(topology, [*first, *second, *third], |cells, rank| {
                    three_to_two_edge_flip_candidate(cells, edge.map(rank))
                })
            else {
                continue;
            };
            let face = candidate
                .shared_face
                .expect("3-to-2 candidate has a ring face");
            let shared_edge = candidate.shared_edge.expect("3-to-2 candidate has an edge");
            let left_sign = orient3d_constructed([
                topology.node(face[0]).predicate_point(),
                topology.node(face[1]).predicate_point(),
                topology.node(face[2]).predicate_point(),
                topology.node(shared_edge[0]).predicate_point(),
            ])
            .map_err(|failure| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    format!("3-to-2 reconnection predicate failed: {failure:?}"),
                )
            })?;
            let right_sign = orient3d_constructed([
                topology.node(face[0]).predicate_point(),
                topology.node(face[1]).predicate_point(),
                topology.node(face[2]).predicate_point(),
                topology.node(shared_edge[1]).predicate_point(),
            ])
            .map_err(|failure| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    format!("3-to-2 reconnection predicate failed: {failure:?}"),
                )
            })?;
            if left_sign != PredicateSign::Zero
                && right_sign != PredicateSign::Zero
                && left_sign != right_sign
            {
                candidates.push(candidate);
            }
        }
    }
    Ok(candidates)
}

/// Reconnects the complete closed star of an interior edge. For a ring of `n`
/// vertices, each deterministic fan triangulation replaces `n` incident
/// tetrahedra by `2(n - 2)` tetrahedra. This includes the common 4-to-4 swap
/// that cannot be expressed as one 2-to-3 or 3-to-2 move.
fn edge_star_candidates(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    source_index: usize,
) -> Result<Vec<LocalTetrahedronFlipCandidate>, DelaunayVolumeSliverError> {
    let vertices = topology.cell((source_index) as u32).vertex_indices;
    let mut candidates = Vec::new();
    for left in 0..4 {
        for right in (left + 1)..4 {
            let mut edge = [vertices[left], vertices[right]];
            edge.sort_unstable_by_key(|vertex| topology.node(*vertex).identity);
            if topology.edge_is_boundary(edge) {
                continue;
            }
            let star = edge_star(topology, edge);
            if star.len() < 4
                || star.iter().any(|tetrahedron| {
                    topology.cell(*tetrahedron).region_id
                        != topology.cell((source_index) as u32).region_id
                })
            {
                continue;
            }
            let Some(ring) = ordered_edge_ring(topology, edge, &star) else {
                continue;
            };
            for root in 0..ring.len() {
                let rotated = (0..ring.len())
                    .map(|offset| ring[(root + offset) % ring.len()])
                    .collect::<Vec<_>>();
                let triangles = (1..(rotated.len() - 1))
                    .map(|index| [rotated[0], rotated[index], rotated[index + 1]])
                    .collect::<Vec<_>>();
                if !triangles_separate_edge(topology, edge, &triangles)? {
                    continue;
                }
                let created_tetrahedra = triangles
                    .into_iter()
                    .flat_map(|triangle| {
                        [
                            [edge[0], triangle[0], triangle[1], triangle[2]],
                            [edge[1], triangle[0], triangle[2], triangle[1]],
                        ]
                    })
                    .collect::<Vec<_>>();
                candidates.push(LocalTetrahedronFlipCandidate {
                    kind: LocalTetrahedronFlipKind::EdgeStar,
                    removed_tetrahedron_ids: star.clone(),
                    created_tetrahedra,
                    shared_face: None,
                    shared_edge: Some(edge),
                });
            }
        }
    }
    Ok(candidates)
}

fn edge_star(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    edge: [u32; 2],
) -> Vec<u32> {
    topology.edge_star(edge)
}

fn ordered_edge_ring(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    edge: [u32; 2],
    star: &[u32],
) -> Option<Vec<u32>> {
    let mut adjacency = BTreeMap::<u32, BTreeSet<u32>>::new();
    for tetrahedron in star {
        let opposite = topology
            .cell(*tetrahedron)
            .vertex_indices
            .into_iter()
            .filter(|vertex| !edge.contains(vertex))
            .collect::<Vec<_>>();
        let [left, right] = opposite.as_slice() else {
            return None;
        };
        adjacency.entry(*left).or_default().insert(*right);
        adjacency.entry(*right).or_default().insert(*left);
    }
    if adjacency.len() != star.len() || adjacency.values().any(|neighbors| neighbors.len() != 2) {
        return None;
    }
    let start = *adjacency
        .keys()
        .min_by_key(|vertex| topology.node(**vertex).identity)?;
    let mut ring = vec![start];
    let mut previous = None;
    let mut current = start;
    loop {
        let next = adjacency[&current]
            .iter()
            .copied()
            .filter(|neighbor| Some(*neighbor) != previous)
            .min_by_key(|neighbor| topology.node(*neighbor).identity)?;
        if next == start {
            break;
        }
        if ring.contains(&next) {
            return None;
        }
        ring.push(next);
        previous = Some(current);
        current = next;
    }
    (ring.len() == adjacency.len()).then_some(ring)
}

fn triangles_separate_edge(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    edge: [u32; 2],
    triangles: &[[u32; 3]],
) -> Result<bool, DelaunayVolumeSliverError> {
    for triangle in triangles {
        let sign = |apex| {
            orient3d_constructed([
                topology.node(triangle[0]).predicate_point(),
                topology.node(triangle[1]).predicate_point(),
                topology.node(triangle[2]).predicate_point(),
                topology.node(apex).predicate_point(),
            ])
        };
        let left = sign(edge[0]).map_err(|failure| {
            error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                format!("edge-star reconnection predicate failed: {failure:?}"),
            )
        })?;
        let right = sign(edge[1]).map_err(|failure| {
            error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                format!("edge-star reconnection predicate failed: {failure:?}"),
            )
        })?;
        if left == PredicateSign::Zero || right == PredicateSign::Zero || left == right {
            return Ok(false);
        }
    }
    Ok(true)
}

fn candidate_key(
    topology: &(impl VolumeTopologyRead + ?Sized),
    candidate: &LocalTetrahedronFlipCandidate,
) -> (
    LocalTetrahedronFlipKind,
    Vec<[StableDigest; 4]>,
    Vec<[StableDigest; 4]>,
) {
    (
        candidate.kind,
        {
            let mut keys = candidate
                .removed_tetrahedron_ids
                .iter()
                .map(|index| topology.cell_key(*index))
                .collect::<Vec<_>>();
            keys.sort_unstable();
            keys
        },
        candidate
            .created_tetrahedra
            .iter()
            .map(|vertices| vertices.map(|vertex| topology.node(vertex).identity))
            .collect(),
    )
}

fn candidate_geometry_key(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    candidate: &LocalTetrahedronFlipCandidate,
) -> Vec<[[u64; 3]; 4]> {
    let mut key = candidate
        .created_tetrahedra
        .iter()
        .map(|vertices| {
            let mut tetrahedron = vertices.map(|vertex| {
                topology
                    .node(vertex)
                    .coordinates_m
                    .canonical_value()
                    .map(crate::cdt::volume_quality::ordered_float_key)
            });
            tetrahedron.sort_unstable();
            tetrahedron
        })
        .collect::<Vec<_>>();
    key.sort_unstable();
    key
}

fn candidate_minimum_jacobian(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    candidate: &LocalTetrahedronFlipCandidate,
    metric: runmat_meshing_size::metric::MetricTensor3,
) -> Result<f64, runmat_simulation_usage::LedgerError> {
    candidate
        .created_tetrahedra
        .iter()
        .try_fold(f64::INFINITY, |minimum, vertices| {
            crate::cdt::volume_quality::evaluate_tetrahedron_jacobian(
                vertices.map(|vertex| topology.node(vertex).coordinates_m.canonical_value()),
                metric,
            )
            .map(|value| value.map_or(minimum, |value| minimum.min(value)))
        })
        .map(|minimum| if minimum.is_infinite() { 0.0 } else { minimum })
}
