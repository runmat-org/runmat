//! Deterministic local insertion direction and candidate enumeration.
use super::*;

pub(super) struct InsertionProtection<'a> {
    pub(super) faces: &'a BTreeSet<[StableDigest; 3]>,
    pub(super) edges: &'a BTreeSet<[StableDigest; 2]>,
    pub(super) segments: &'a BTreeSet<[StableDigest; 2]>,
}

pub(super) fn edge_star_choices_prepared(
    topology: &(impl VolumeTopologyRead + ?Sized),
    source: [u32; 4],
    region: &runmat_geometry_core::PersistentEntityId,
    protection: InsertionProtection<'_>,
    metrics: InsertionMetric<'_>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<EdgeStarChoice>, DelaunayVolumeSliverError> {
    let InsertionProtection {
        faces: protected_faces,
        edges: protected_edges,
        segments: protected_segments,
    } = protection;
    let mut choices = Vec::new();
    for left in 0..4 {
        for right in (left + 1)..4 {
            let mut edge = [source[left], source[right]];
            edge.sort_unstable_by_key(|vertex| topology.node(*vertex).identity);
            let edge_identities = edge.map(|vertex| topology.node(vertex).identity);
            let star = topology.edge_star(edge);
            if star.len() < 3
                || topology.edge_is_boundary(edge)
                || star.iter().any(|tetrahedron| {
                    topology.cell(*tetrahedron).region_id.as_ref() != Some(region)
                })
            {
                continue;
            }
            let ring = star
                .iter()
                .flat_map(|tetrahedron| topology.cell(*tetrahedron).vertex_indices)
                .filter(|vertex| !edge.contains(vertex))
                .collect::<BTreeSet<_>>();
            if ring.len() < 3 {
                continue;
            }
            let prepared = PreparedEdgeStar::prepare(topology, &star, edge, metrics)?;
            let endpoint_nodes = edge.map(|vertex| topology.node(vertex));
            let mut edge_identities = edge_identities;
            edge_identities.sort_unstable();
            let edge_policy = if protected_segments.contains(&edge_identities) {
                EdgeCandidatePolicy::ExcludedSegment
            } else if protected_edges.contains(&edge_identities) {
                EdgeCandidatePolicy::OffEdgeOnly
            } else {
                EdgeCandidatePolicy::IncludeMidpoint
            };
            if edge_policy == EdgeCandidatePolicy::ExcludedSegment {
                continue;
            }
            // A midpoint on a protected surface edge would be an embedded node unless the
            // surface facet were also subdivided. Smoothed candidates move strictly into the
            // edge-star kernel and preserve that facet, so—as in TetGen's INSTAR sliver split—
            // only true PLC segments exclude the edge from the search altogether.
            let Some((midpoint, midpoint_construction)) = compose_dyadic_point(
                endpoint_nodes.map(DelaunayVolumeNode::predicate_point),
                [1, 1],
                1,
            ) else {
                continue;
            };
            if edge_policy == EdgeCandidatePolicy::IncludeMidpoint {
                choices.push(EdgeStarChoice {
                    edge_identities,
                    index: 0,
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new(midpoint),
                    construction: midpoint_construction,
                    proposal_score: prepared.score(midpoint)?,
                    strategy: InsertionStrategy::EdgeStar,
                });
            }
            let boundary_faces = edge_star_boundary_faces(topology, &star, edge);
            for (direction, face) in boundary_faces
                .iter()
                .take(MAXIMUM_CAVITY_SEARCH_DIRECTIONS)
                .enumerate()
            {
                for (fraction_index, (midpoint_weight, exponent)) in
                    EDGE_STAR_DIRECTION_WEIGHTS.into_iter().enumerate()
                {
                    let Some((coordinates_m, construction)) = compose_dyadic_point(
                        [
                            SpatialPredicatePoint {
                                identity: kernel_probe_identity(midpoint),
                                coordinates: midpoint,
                                construction: midpoint_construction,
                            },
                            topology.node(face[0]).predicate_point(),
                            topology.node(face[1]).predicate_point(),
                            topology.node(face[2]).predicate_point(),
                        ],
                        [midpoint_weight, 1, 1, 1],
                        exponent,
                    ) else {
                        continue;
                    };
                    if !point_is_in_edge_star_kernel(
                        topology,
                        &star,
                        edge,
                        coordinates_m,
                        construction,
                    )? {
                        continue;
                    }
                    choices.push(EdgeStarChoice {
                        edge_identities,
                        index: 2
                            + (direction * EDGE_STAR_DIRECTION_WEIGHTS.len() + fraction_index)
                                as u16,
                        coordinates_m: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::Position,
                            _,
                        >::new(coordinates_m),
                        construction,
                        proposal_score: prepared.score(coordinates_m)?,
                        strategy: InsertionStrategy::EdgeStar,
                    });
                }
            }
            let mut smoothing_starts = vec![midpoint];
            for (search_index, step) in CAVITY_SMOOTHING_STEPS.into_iter().enumerate() {
                if let Some(coordinates_m) = smooth_edge_star_point(
                    topology,
                    &star,
                    edge,
                    midpoint,
                    &prepared,
                    step,
                    cancellation,
                )? {
                    smoothing_starts.push(coordinates_m);
                    choices.push(EdgeStarChoice {
                        edge_identities,
                        index: 1_000 + search_index as u16,
                        coordinates_m: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::Position,
                            _,
                        >::new(coordinates_m),
                        construction: Point3Construction::Explicit,
                        proposal_score: prepared.score(coordinates_m)?,
                        strategy: InsertionStrategy::EdgeStar,
                    });
                }
            }
            smoothing_starts.sort_by_key(|point| point.map(f64::to_bits));
            smoothing_starts.dedup();
            for (search_index, start) in smoothing_starts.into_iter().enumerate() {
                if let Some(coordinates_m) =
                    pattern_search_edge_star(topology, &star, edge, start, &prepared, cancellation)?
                {
                    choices.push(EdgeStarChoice {
                        edge_identities,
                        index: 2_000 + search_index as u16,
                        coordinates_m: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::Position,
                            _,
                        >::new(coordinates_m),
                        construction: Point3Construction::Explicit,
                        proposal_score: prepared.score(coordinates_m)?,
                        strategy: InsertionStrategy::EdgeStar,
                    });
                }
            }
        }
    }
    let source_index = topology
        .tetrahedron_index(source)
        .expect("quality source remains in topology");
    let mut lineage_edge = [
        topology.node(source[0]).identity,
        topology.node(source[1]).identity,
    ];
    lineage_edge.sort_unstable();
    for (slot, neighbor) in topology
        .cell(source_index)
        .neighbors
        .into_iter()
        .enumerate()
    {
        let Some(neighbor_index) = neighbor else {
            continue;
        };
        let neighbor = &topology.cell(neighbor_index);
        if neighbor.region_id.as_ref() != Some(region) {
            continue;
        }
        let vertices = neighbor
            .vertex_indices
            .map(|vertex| topology.node(vertex).coordinates_m.canonical_value());
        let neighbor_centroid: [f64; 3] = std::array::from_fn(|axis| {
            vertices[0][axis]
                + (vertices[1][axis] - vertices[0][axis] + vertices[2][axis] - vertices[0][axis]
                    + vertices[3][axis]
                    - vertices[0][axis])
                    * 0.25
        });
        let source_vertices = topology
            .cell(source_index)
            .vertex_indices
            .map(|vertex| topology.node(vertex).coordinates_m.canonical_value());
        let source_centroid = average(source_vertices.into_iter());
        for (blend_index, neighbor_weight) in [0.25, 0.5, 0.75].into_iter().enumerate() {
            let coordinates_m = std::array::from_fn(|axis| {
                source_centroid[axis]
                    + neighbor_weight * (neighbor_centroid[axis] - source_centroid[axis])
            });
            choices.push(EdgeStarChoice {
                edge_identities: lineage_edge,
                index: 10_000 + (slot * 3 + blend_index) as u16,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates_m),
                construction: Point3Construction::Explicit,
                proposal_score: 0.0,
                strategy: InsertionStrategy::Cavity { neighbor_index },
            });
        }
    }
    let source_tetrahedron = &topology.cell(source_index);
    for opposite in 0..4 {
        let mut face = source_tetrahedron
            .vertex_indices
            .into_iter()
            .enumerate()
            .filter_map(|(slot, vertex)| (slot != opposite).then_some(vertex))
            .collect::<Vec<_>>();
        face.sort_unstable_by_key(|vertex| topology.node(*vertex).identity);
        let face: [u32; 3] = face.try_into().expect("tetrahedron face has three nodes");
        let face_identities = face.map(|vertex| topology.node(vertex).identity);
        if !protected_faces.contains(&face_identities) {
            continue;
        }
        let Some(neighbor_index) = source_tetrahedron
            .neighbors
            .into_iter()
            .flatten()
            .find(|neighbor| topology.cell(*neighbor).region_id.as_ref() == Some(region))
        else {
            continue;
        };
        let points = face.map(|vertex| topology.node(vertex).coordinates_m.canonical_value());
        let first: [f64; 3] = std::array::from_fn(|axis| points[1][axis] - points[0][axis]);
        let second: [f64; 3] = std::array::from_fn(|axis| points[2][axis] - points[0][axis]);
        let mut normal = [
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        ];
        let norm = normal
            .into_iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 {
            continue;
        }
        normal.iter_mut().for_each(|value| *value /= norm);
        let centroid = average(points.into_iter());
        let interior = topology
            .node(source_tetrahedron.vertex_indices[opposite])
            .coordinates_m
            .canonical_value();
        if normal
            .into_iter()
            .enumerate()
            .map(|(axis, value)| value * (interior[axis] - centroid[axis]))
            .sum::<f64>()
            < 0.0
        {
            normal.iter_mut().for_each(|value| *value = -*value);
        }
        let shortest_edge = [(0, 1), (1, 2), (2, 0)]
            .into_iter()
            .map(|(left, right)| {
                (0..3)
                    .map(|axis| {
                        let delta = points[left][axis] - points[right][axis];
                        delta * delta
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .min_by(f64::total_cmp)
            .expect("face has edges");
        let mut lineage_edge = [face_identities[0], face_identities[1]];
        lineage_edge.sort_unstable();
        for (offset_index, factor) in [0.25, 0.5, 1.0].into_iter().enumerate() {
            let coordinates_m =
                std::array::from_fn(|axis| centroid[axis] + factor * shortest_edge * normal[axis]);
            choices.push(EdgeStarChoice {
                edge_identities: lineage_edge,
                index: 20_000 + (opposite * 3 + offset_index) as u16,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates_m),
                construction: Point3Construction::Explicit,
                proposal_score: 0.0,
                strategy: InsertionStrategy::Cavity { neighbor_index },
            });
        }
    }
    choices.sort_by(|left, right| {
        right
            .proposal_score
            .total_cmp(&left.proposal_score)
            .then_with(|| {
                left.edge_identities
                    .cmp(&right.edge_identities)
                    .then_with(|| left.index.cmp(&right.index))
            })
    });
    Ok(choices)
}
