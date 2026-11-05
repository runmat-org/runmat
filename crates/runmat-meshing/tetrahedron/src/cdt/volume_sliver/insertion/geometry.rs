//! Complete edge-star splitting and geometric kernel search.
use super::*;
mod prepared;
pub(super) use prepared::PreparedEdgeStar;

/// TetGen's `INSTAR` sliver split replaces the complete star of the selected edge rather than
/// asking an unconstrained Bowyer-Watson cavity to choose it again. This is important at a smooth
/// boundary: global Delaunay legalization can legally recreate the same flat two-facet wedge.
/// The canonical topology builder independently verifies orientation, overlap, manifoldness, and
/// incidence; the caller additionally verifies PLC conformance and strict global quality progress.
#[cfg(test)]
pub(super) fn split_complete_edge_star(
    topology: &DelaunayVolumeTopology,
    node: DelaunayVolumeNode,
    edge_identities: [StableDigest; 2],
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<EdgeStarSplit, DelaunayVolumeSliverError> {
    let edge = edge_identities.map(|identity| {
        topology
            .nodes
            .binary_search_by_key(&identity, |candidate| candidate.identity)
            .map(|index| index as u32)
    });
    let Ok(edge) = edge.into_iter().collect::<Result<Vec<_>, _>>() else {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "sliver insertion edge is absent from canonical topology",
        ));
    };
    let edge: [u32; 2] = edge.try_into().expect("sliver edge has two nodes");
    let star = topology.edge_star(edge);
    if star.len() < 3
        || topology.incidence.boundary_facets.iter().any(|facet| {
            facet.vertex_indices.contains(&edge[0]) && facet.vertex_indices.contains(&edge[1])
        })
    {
        return Ok(EdgeStarSplit::Rejected(
            "selected edge does not have a complete interior star".to_owned(),
        ));
    }
    let region = topology.cell(star[0]).region_id.clone();
    if star
        .iter()
        .any(|tetrahedron| topology.cell(*tetrahedron).region_id != &region)
    {
        return Ok(EdgeStarSplit::Rejected(
            "selected edge star crosses a region boundary".to_owned(),
        ));
    }
    let removed = star.iter().copied().collect::<BTreeSet<_>>();
    let mut nodes = topology.nodes.clone();
    let inserted = nodes
        .binary_search_by_key(&node.identity, |candidate| candidate.identity)
        .expect_err("candidate identity collision is rejected before edge-star splitting");
    nodes.insert(inserted, node);
    let inserted = inserted as u32;
    let remap = |vertex: u32| vertex + u32::from(vertex >= inserted);
    let mut tetrahedra = topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed.contains(&(*index as u32)))
        .map(|(_, tetrahedron)| {
            (
                tetrahedron.vertex_indices.map(remap),
                tetrahedron.region_id.clone(),
            )
        })
        .collect::<Vec<_>>();
    tetrahedra.extend(
        edge_star_boundary_faces(topology, &star, edge)
            .into_iter()
            .map(|face| {
                (
                    [remap(face[0]), remap(face[1]), remap(face[2]), inserted],
                    region.clone(),
                )
            }),
    );
    match build_canonical_volume_topology(
        nodes,
        tetrahedra,
        options.insertion.topology,
        cancellation,
    ) {
        Ok(topology) => Ok(EdgeStarSplit::Topology(topology)),
        Err(failure) => match failure.kind {
            DelaunayTopologyErrorKind::ResourceLimit => Err(error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                failure.to_string(),
            )),
            DelaunayTopologyErrorKind::Cancelled => Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                failure.to_string(),
            )),
            _ => Ok(EdgeStarSplit::Rejected(failure.to_string())),
        },
    }
}

pub(super) fn smooth_edge_star_point(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    star: &[u32],
    edge: [u32; 2],
    midpoint: [f64; 3],
    prepared: &PreparedEdgeStar,
    step: f64,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<[f64; 3]>, DelaunayVolumeSliverError> {
    let boundary_faces = edge_star_boundary_faces(topology, star, edge);

    let mut point = midpoint;
    let mut score = prepared.score(point)?;
    let mut improved = false;
    for iteration in 0..MAXIMUM_CAVITY_SMOOTHING_ITERATIONS {
        if iteration.is_multiple_of(16) && cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                "edge-star insertion smoothing was cancelled",
            ));
        }
        let mut best = None;
        for face in boundary_faces.iter().take(MAXIMUM_CAVITY_SEARCH_DIRECTIONS) {
            let centroid = average(
                face.iter()
                    .map(|vertex| topology.node(*vertex).coordinates_m.canonical_value()),
            );
            let candidate =
                std::array::from_fn(|axis| point[axis] + step * (centroid[axis] - point[axis]));
            if !point_is_in_edge_star_kernel(
                topology,
                star,
                edge,
                candidate,
                Point3Construction::Explicit,
            )? {
                continue;
            }
            let candidate_score = prepared.score(candidate)?;
            if candidate_score <= score {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|(best_score, best_point): &(f64, [f64; 3])| {
                    candidate_score > *best_score
                        || candidate_score == *best_score
                            && coordinate_key(candidate) < coordinate_key(*best_point)
                })
            {
                best = Some((candidate_score, candidate));
            }
        }
        let Some((best_score, best_point)) = best else {
            break;
        };
        point = best_point;
        score = best_score;
        improved = true;
    }
    Ok(improved.then_some(point))
}

/// Deterministic derivative-free maximin search in the kernel of a complete edge star.
/// Face-centroid directions are useful warm starts, but do not span the local quality gradient
/// on curved boundaries; this bounded stencil supplies the independent directions used by
/// production sliver splitters.
pub(super) fn pattern_search_edge_star(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    star: &[u32],
    edge: [u32; 2],
    start: [f64; 3],
    prepared: &PreparedEdgeStar,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<[f64; 3]>, DelaunayVolumeSliverError> {
    let local_scale = edge
        .map(|vertex| topology.node(vertex).coordinates_m.canonical_value())
        .into_iter()
        .reduce(|left, right| std::array::from_fn(|axis| left[axis] - right[axis]))
        .map(|delta| {
            delta
                .into_iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
        })
        .unwrap_or(0.0);
    if local_scale == 0.0 {
        return Ok(None);
    }
    let mut point = start;
    let initial_score = prepared.score(point)?;
    let mut score = initial_score;
    let mut radius = local_scale * 0.25;
    let minimum_radius = local_scale * 2_f64.powi(-24);
    for iteration in 0..MAXIMUM_CAVITY_SMOOTHING_ITERATIONS {
        if iteration.is_multiple_of(8) && cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                "edge-star pattern search was cancelled",
            ));
        }
        let mut best = None;
        for x in -1_i8..=1 {
            for y in -1_i8..=1 {
                for z in -1_i8..=1 {
                    if x == 0 && y == 0 && z == 0 {
                        continue;
                    }
                    let norm = f64::from(x * x + y * y + z * z).sqrt();
                    let direction = [x, y, z].map(|value| f64::from(value) / norm);
                    let candidate =
                        std::array::from_fn(|axis| point[axis] + radius * direction[axis]);
                    if !point_is_in_edge_star_kernel(
                        topology,
                        star,
                        edge,
                        candidate,
                        Point3Construction::Explicit,
                    )? {
                        continue;
                    }
                    let candidate_score = prepared.score(candidate)?;
                    if candidate_score <= score {
                        continue;
                    }
                    if best
                        .as_ref()
                        .is_none_or(|(best_score, best_point): &(f64, [f64; 3])| {
                            candidate_score > *best_score
                                || candidate_score == *best_score
                                    && candidate.map(f64::to_bits) < best_point.map(f64::to_bits)
                        })
                    {
                        best = Some((candidate_score, candidate));
                    }
                }
            }
        }
        if let Some((best_score, best_point)) = best {
            score = best_score;
            point = best_point;
        } else {
            radius *= 0.5;
            if radius < minimum_radius {
                break;
            }
        }
    }
    Ok((score > initial_score).then_some(point))
}

pub(super) fn edge_star_boundary_faces(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    star: &[u32],
    edge: [u32; 2],
) -> Vec<[u32; 3]> {
    let mut boundary_faces = star
        .iter()
        .flat_map(|tetrahedron| {
            let opposite = topology
                .cell(*tetrahedron)
                .vertex_indices
                .into_iter()
                .filter(|vertex| !edge.contains(vertex))
                .collect::<Vec<_>>();
            [
                [edge[0], opposite[0], opposite[1]],
                [edge[1], opposite[0], opposite[1]],
            ]
        })
        .map(|mut face| {
            face.sort_unstable_by_key(|vertex| topology.node(*vertex).identity);
            face
        })
        .collect::<Vec<_>>();
    boundary_faces.sort_unstable_by_key(|face| face.map(|vertex| topology.node(vertex).identity));
    boundary_faces.dedup();
    boundary_faces
}

pub(super) fn point_is_in_edge_star_kernel(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    star: &[u32],
    edge: [u32; 2],
    point: [f64; 3],
    construction: Point3Construction,
) -> Result<bool, DelaunayVolumeSliverError> {
    let probe = DelaunayVolumeNode {
        identity: kernel_probe_identity(point),
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                point,
            ),
        construction,
    };
    for tetrahedron in star {
        let opposite = topology
            .cell(*tetrahedron)
            .vertex_indices
            .into_iter()
            .filter(|vertex| !edge.contains(vertex))
            .collect::<Vec<_>>();
        for (face_endpoint, interior_endpoint) in [(edge[0], edge[1]), (edge[1], edge[0])] {
            let face = [
                topology.node(face_endpoint),
                topology.node(opposite[0]),
                topology.node(opposite[1]),
            ];
            let expected =
                orient_volume_nodes([face[0], face[1], face[2], topology.node(interior_endpoint)])
                    .map_err(|failure| {
                        error(
                            DelaunayVolumeSliverErrorKind::InvalidTopology,
                            format!(
                                "edge-star kernel predicate rejected source geometry: {failure:?}"
                            ),
                        )
                    })?;
            let observed =
                orient_volume_nodes([face[0], face[1], face[2], probe]).map_err(|failure| {
                    error(
                        DelaunayVolumeSliverErrorKind::InvalidTopology,
                        format!(
                            "edge-star kernel predicate rejected candidate geometry: {failure:?}"
                        ),
                    )
                })?;
            if expected == PredicateSign::Zero || observed != expected {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

pub(super) fn kernel_probe_identity(point: [f64; 3]) -> StableDigest {
    let mut hasher = Sha256::new();
    hasher.update(KERNEL_PROBE_IDENTITY_DOMAIN);
    for coordinate in point {
        hasher.update(coordinate.to_bits().to_be_bytes());
    }
    StableDigest::from_bytes(hasher.finalize().into())
}

pub(super) fn coordinate_key(point: [f64; 3]) -> [u64; 3] {
    point.map(f64::to_bits)
}
