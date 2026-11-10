//! Conforming insertion of a node on protected facets or a recovered segment edge, and the
//! canonical split with complete legalization.
use super::*;

/// How a recovered segment edge is split before legalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SplitStrategy {
    /// The constrained Bowyer-Watson cavity of the new node, corrected to stay star-shaped
    /// from it; the edge star when the corrected cavity is not star-shaped.
    ConstrainedCavity,
    /// The tetrahedra around the edge only.
    EdgeStar,
}

/// The node a conforming insertion adds, the protected facet triangles it must keep, the
/// recovered segment edge it splits (if any) and the protected triangles it lies on.
#[derive(Clone, Copy)]
pub(super) struct ConformingInsertionRequest<'a> {
    pub(super) node: DelaunayVolumeNode,
    pub(super) protected: &'a [DelaunayRecoveredFacetTriangle],
    pub(super) selected_edge: Option<[StableDigest; 2]>,
    pub(super) hits: &'a [(DelaunayRecoveredFacetTriangle, PointLocation)],
}

/// Splits with the constrained cavity first. In thin layers between parallel interfaces,
/// splitting only the edge star can leave an illegal link face that no flip removes; the
/// cavity removes those cells before refilling. The edge-star split remains the fallback.
pub(super) fn insert_conformingly(
    current: &DelaunaySegmentRecovery,
    node: DelaunayVolumeNode,
    protected: &[DelaunayRecoveredFacetTriangle],
    selected_edge: Option<[StableDigest; 2]>,
    hits: &[(DelaunayRecoveredFacetTriangle, PointLocation)],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<ConformingInsertion>, DelaunayFacetRecoveryError> {
    let request = ConformingInsertionRequest {
        node,
        protected,
        selected_edge,
        hits,
    };
    if selected_edge.is_some() {
        if let Some(insertion) = insert_conformingly_with(
            current,
            request,
            SplitStrategy::ConstrainedCavity,
            constraint_index,
            work,
        )? {
            return Ok(Some(insertion));
        }
    }
    let cavity_rejection = work.last_rejection().map(str::to_owned);
    let insertion = insert_conformingly_with(
        current,
        request,
        SplitStrategy::EdgeStar,
        constraint_index,
        work,
    )?;
    if insertion.is_none() {
        if let (Some(cavity), Some(star)) = (cavity_rejection, work.last_rejection()) {
            let combined = format!("constrained cavity: {cavity}; edge star: {star}");
            work.reject(combined);
        }
    }
    Ok(insertion)
}

pub(super) fn insert_conformingly_with(
    current: &DelaunaySegmentRecovery,
    request: ConformingInsertionRequest<'_>,
    strategy: SplitStrategy,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<ConformingInsertion>, DelaunayFacetRecoveryError> {
    let ConformingInsertionRequest {
        node,
        protected,
        selected_edge,
        hits,
    } = request;
    let mut mutation = TopologyMutation::from_canonical(current.topology.clone());
    let inserted = mutation.append_node(node).map_err(|failure| {
        error(
            DelaunayFacetRecoveryErrorKind::InvalidTopology,
            Some(constraint_index),
            failure.to_string(),
        )
    })?;
    let (removed, created, splits) = if let Some(edge) = selected_edge {
        split_edge(
            &mutation,
            inserted,
            edge,
            protected,
            strategy,
            constraint_index,
        )?
    } else {
        split_faces(&mutation, inserted, hits, constraint_index)?
    };
    if removed.is_empty() {
        work.reject("the constrained cavity of the split node is not star-shaped from it");
        return Ok(None);
    }
    if let Some((tetrahedron, _)) = created.iter().find(|(tetrahedron, _)| {
        orient_materialized_volume_nodes(tetrahedron.map(|index| mutation.nodes[index as usize]))
            .is_ok_and(|sign| sign == PredicateSign::Zero)
    }) {
        work.reject(format!(
            "conforming facet {:?} split produced exact-zero tetrahedron {:?} at coordinates {:?}; inserted construction {:?}",
            selected_edge,
            tetrahedron,
            tetrahedron.map(|index| mutation.nodes[index as usize].coordinates_m.canonical_value()),
            node.construction,
        ));
        return Ok(None);
    }
    if let Err(failure) = mutation.replace_cavity(&removed, created) {
        work.reject(format!(
            "conforming facet cavity replacement was rejected: {failure}"
        ));
        return Ok(None);
    }
    let protected_faces = updated_protected_faces(protected, &splits);
    // Every created cell contains the inserted node, so link flips around it and an exact check
    // of its star restore and prove the constrained Delaunay property locally. When an illegal
    // face remains, the trial rolls back and the complete repair below runs instead.
    let locally_legal = {
        let protected_set = protected_faces.iter().copied().collect::<BTreeSet<_>>();
        let mut kept_edges =
            crate::cdt::segment_recovery::recovered_segment_edge_keys(&current.segments);
        if let Some(edge) = selected_edge {
            kept_edges.extend(crate::cdt::segment_recovery::sorted_edge_keys(
                edge.map(|endpoint| [endpoint, node.identity]),
            ));
        }
        let mut trial = mutation.cavity_trial();
        match legalize_inserted_node_and_check_star(
            &mut trial,
            inserted,
            &protected_set,
            &kept_edges,
            work.options.segment_recovery.insertion,
            work.cancellation,
        ) {
            Ok(true) => {
                trial.commit();
                true
            }
            Ok(false) => false,
            Err(failure)
                if matches!(
                    failure.kind,
                    DelaunayInsertionErrorKind::ResourceLimit
                        | DelaunayInsertionErrorKind::Cancelled
                ) =>
            {
                return Err(insertion_error(
                    failure.kind,
                    failure.to_string(),
                    constraint_index,
                ))
            }
            Err(_) => false,
        }
    };
    let topology = match mutation.freeze(
        work.options.segment_recovery.insertion.topology,
        work.cancellation,
    ) {
        Ok(topology) => topology,
        Err(failure) if failure.kind == TopologyMutationErrorKind::InvalidTopology => {
            work.reject(format!("conforming facet topology was rejected: {failure}"));
            return Ok(None);
        }
        Err(failure) => {
            return Err(error(
                match failure.kind {
                    TopologyMutationErrorKind::ResourceLimit => {
                        DelaunayFacetRecoveryErrorKind::ResourceLimit
                    }
                    TopologyMutationErrorKind::Cancelled => {
                        DelaunayFacetRecoveryErrorKind::Cancelled
                    }
                    TopologyMutationErrorKind::InvalidTopology => unreachable!(),
                },
                Some(constraint_index),
                failure.to_string(),
            ))
        }
    };
    if locally_legal {
        return finish_conforming_insertion(current, topology, selected_edge, splits, work);
    }
    let topology = match legalize_delaunay_volume_topology_with_barriers(
        topology,
        &protected_faces,
        work.options.segment_recovery.insertion,
        work.cancellation,
    ) {
        Ok(topology) => topology,
        Err(failure)
            if matches!(
                failure.kind,
                DelaunayInsertionErrorKind::InvalidTopology
                    | DelaunayInsertionErrorKind::InvalidNode
                    | DelaunayInsertionErrorKind::PointOutsideTopology
            ) =>
        {
            work.reject(format!(
                "conforming facet legalization was rejected: {failure}"
            ));
            return Ok(None);
        }
        Err(failure) => {
            return Err(insertion_error(
                failure.kind,
                failure.to_string(),
                constraint_index,
            ))
        }
    };
    if let Err(failure) = validate_constrained_delaunay_volume_topology(
        &topology,
        &protected_faces,
        work.options.segment_recovery.insertion,
        work.cancellation,
    ) {
        return match failure.kind {
            DelaunayInsertionErrorKind::ResourceLimit | DelaunayInsertionErrorKind::Cancelled => {
                Err(insertion_error(
                    failure.kind,
                    failure.to_string(),
                    constraint_index,
                ))
            }
            _ => {
                work.reject(format!(
                    "conforming facet validation was rejected: {failure}"
                ));
                Ok(None)
            }
        };
    }
    finish_conforming_insertion(current, topology, selected_edge, splits, work)
}

pub(super) fn finish_conforming_insertion(
    current: &DelaunaySegmentRecovery,
    topology: DelaunayVolumeTopology,
    selected_edge: Option<[StableDigest; 2]>,
    splits: Vec<ProtectedFacetSplit>,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<ConformingInsertion>, DelaunayFacetRecoveryError> {
    // Built field by field: cloning `current` would copy its whole topology only to drop it.
    let mut recovery = DelaunaySegmentRecovery {
        topology,
        segments: current.segments.clone(),
        steiner_node_identities: current.steiner_node_identities.clone(),
        recovery_passes: current.recovery_passes,
    };
    if let Some(edge) = selected_edge {
        if let Some((segment_index, interval_index)) =
            recovered_segment_edge_location(current, edge)
        {
            let (_, recovered) = recovered_segment_midpoint(current, segment_index, interval_index)
                .map_err(super::super::segment_error)?;
            record_recovered_segment_split(&mut recovery, segment_index, interval_index, recovered)
                .map_err(super::super::segment_error)?;
        }
    }
    if !recovered_segments_exist(&recovery) {
        work.reject("conforming facet insertion removed a recovered segment edge");
        return Ok(None);
    }
    Ok(Some(ConformingInsertion { recovery, splits }))
}

pub(super) fn recovered_segment_edge_location(
    recovery: &DelaunaySegmentRecovery,
    edge: [StableDigest; 2],
) -> Option<(usize, usize)> {
    segment_edge_location(&recovery.segments, edge)
}

pub(super) fn point_location(
    recovery: &DelaunaySegmentRecovery,
    node: DelaunayVolumeNode,
    triangle: DelaunayRecoveredFacetTriangle,
    constraint_index: u32,
) -> Result<Option<PointLocation>, DelaunayFacetRecoveryError> {
    let vertices = triangle.node_identities.map(|identity| {
        node_index(&recovery.topology, identity)
            .map(|index| recovery.topology.nodes[index])
            .ok_or_else(|| invalid(constraint_index, "protected facet node is missing"))
    });
    let [first, second, third] = vertices;
    let vertices = [first?, second?, third?];
    if orient_volume_nodes([vertices[0], vertices[1], vertices[2], node]).map_err(|failure| {
        invalid(
            constraint_index,
            format!("facet split predicate failed: {failure:?}"),
        )
    })? != PredicateSign::Zero
    {
        return Ok(None);
    }
    let points = vertices.map(DelaunayVolumeNode::predicate_point);
    let axes = triangle_projection_axes_constructed(points)
        .map_err(|failure| {
            invalid(
                constraint_index,
                format!("facet split projection failed: {failure:?}"),
            )
        })?
        .ok_or_else(|| invalid(constraint_index, "protected facet is collinear"))?;
    let query = node.predicate_point();
    let signs = [
        projected_orient2d_constructed([points[0], points[1], query], axes),
        projected_orient2d_constructed([points[1], points[2], query], axes),
        projected_orient2d_constructed([points[2], points[0], query], axes),
    ];
    if signs.iter().any(Result::is_err) {
        return Err(invalid(
            constraint_index,
            "facet split projection predicate failed",
        ));
    }
    let signs = signs.map(Result::unwrap);
    if signs.contains(&PredicateSign::Positive) && signs.contains(&PredicateSign::Negative) {
        return Ok(None);
    }
    if let Some(slot) = signs.iter().position(|sign| *sign == PredicateSign::Zero) {
        if !points_collinear_constructed([points[slot], points[(slot + 1) % 3], query]).map_err(
            |failure| {
                invalid(
                    constraint_index,
                    format!("facet edge classification failed: {failure:?}"),
                )
            },
        )? {
            return Ok(Some(PointLocation::Interior));
        }
        let mut edge = [
            triangle.node_identities[slot],
            triangle.node_identities[(slot + 1) % 3],
        ];
        edge.sort_unstable();
        return Ok(Some(PointLocation::Edge(edge)));
    }
    Ok(Some(PointLocation::Interior))
}

pub(super) type Replacement = (
    BTreeSet<u32>,
    Vec<([u32; 4], Option<runmat_geometry_core::PersistentEntityId>)>,
    Vec<ProtectedFacetSplit>,
);

pub(super) fn split_edge(
    topology: &TopologyMutation,
    inserted: u32,
    edge: [StableDigest; 2],
    protected: &[DelaunayRecoveredFacetTriangle],
    strategy: SplitStrategy,
    constraint_index: u32,
) -> Result<Replacement, DelaunayFacetRecoveryError> {
    let left = node_index_mutation(topology, edge[0])
        .ok_or_else(|| invalid(constraint_index, "facet split edge node is missing"))?;
    let right = node_index_mutation(topology, edge[1])
        .ok_or_else(|| invalid(constraint_index, "facet split edge node is missing"))?;
    let star = topology.vertex_stars[left as usize]
        .iter()
        .copied()
        .filter(|index| {
            topology
                .tetrahedron(*index)
                .is_some_and(|tet| tet.vertex_indices.contains(&right))
        })
        .collect::<BTreeSet<_>>();
    let mut cavity = constrained_cavity(
        topology,
        inserted,
        star.clone(),
        edge,
        protected,
        constraint_index,
    )?;
    let barriers = barrier_faces(protected, edge);
    // Opposite vertices of the protected triangles split at the edge: their edges to each
    // edge endpoint must stay on the cavity boundary to carry the split children.
    let split_apexes = protected
        .iter()
        .filter(|triangle| {
            edge.iter()
                .all(|node| triangle.node_identities.contains(node))
        })
        .filter_map(|triangle| {
            triangle
                .node_identities
                .into_iter()
                .find(|node| !edge.contains(node))
                .and_then(|apex| node_index_mutation(topology, apex))
        })
        .collect::<Vec<_>>();
    // Cavity correction: drop the tetrahedra that would hide a boundary face from the node,
    // remove a protected face from both sides, or surround a split child's edge.
    for _ in 0..64 {
        let mut dropped = BTreeSet::new();
        for index in &cavity {
            let tet = topology
                .tetrahedron(*index)
                .expect("cavity tetrahedra remain active");
            for (slot, neighbor) in tet.neighbors.iter().enumerate() {
                let inside = neighbor.is_some_and(|neighbor| cavity.contains(&neighbor));
                if inside {
                    let neighbor = neighbor.expect("inside neighbor");
                    if barriers.contains(&canonical_mutation_face(
                        topology,
                        tet.vertex_indices,
                        slot,
                    )) {
                        dropped
                            .extend([*index, neighbor].into_iter().filter(|t| !star.contains(t)));
                    }
                    continue;
                }
                let face = (0..4)
                    .filter(|vertex| *vertex != slot)
                    .map(|vertex| tet.vertex_indices[vertex])
                    .collect::<Vec<_>>();
                if face.contains(&left) && face.contains(&right) {
                    continue;
                }
                let mut vertices = tet.vertex_indices;
                vertices[slot] = inserted;
                let visible = cone_is_visible(topology, vertices);
                if !visible && !star.contains(index) {
                    dropped.insert(*index);
                }
            }
        }
        for apex in &split_apexes {
            for endpoint in [left, right] {
                let around = topology.vertex_stars[endpoint as usize]
                    .iter()
                    .copied()
                    .filter(|index| {
                        topology
                            .tetrahedron(*index)
                            .is_some_and(|tet| tet.vertex_indices.contains(apex))
                    })
                    .collect::<Vec<_>>();
                if around.iter().all(|index| cavity.contains(index)) {
                    dropped.extend(around.into_iter().filter(|index| !star.contains(index)));
                }
            }
        }
        if dropped.is_empty() {
            break;
        }
        cavity.retain(|index| !dropped.contains(index));
        cavity = connected_from_star(topology, &cavity, &star, &barriers);
    }
    let fill = |removed: &BTreeSet<u32>| {
        let mut created = Vec::new();
        for index in removed {
            let tet = topology
                .tetrahedron(*index)
                .expect("cavity tetrahedra remain active");
            for (slot, neighbor) in tet.neighbors.iter().enumerate() {
                if neighbor.is_some_and(|neighbor| removed.contains(&neighbor)) {
                    continue;
                }
                let face = (0..4)
                    .filter(|vertex| *vertex != slot)
                    .map(|vertex| tet.vertex_indices[vertex])
                    .collect::<Vec<_>>();
                // The inserted node lies on the split edge, so a cavity face through that
                // edge is coplanar with it; the faces on either side of the node close it.
                if face.contains(&left) && face.contains(&right) {
                    continue;
                }
                let mut vertices = tet.vertex_indices;
                vertices[slot] = inserted;
                created.push((vertices, tet.region_id.clone()));
            }
        }
        created
    };
    // A cavity that is still not star-shaped from the node falls back to the edge star,
    // whose split is always valid; flips then legalize it.
    let star_shaped =
        |created: &Vec<([u32; 4], Option<runmat_geometry_core::PersistentEntityId>)>| {
            created
                .iter()
                .all(|(tetrahedron, _)| cone_is_visible(topology, *tetrahedron))
        };
    let created = fill(&cavity);
    let (removed, created) = if strategy == SplitStrategy::ConstrainedCavity
        && cavity.len() > star.len()
        && star_shaped(&created)
    {
        (cavity, created)
    } else if strategy == SplitStrategy::ConstrainedCavity {
        // The edge-star strategy performs this split; an empty replacement declines it here.
        (BTreeSet::new(), Vec::new())
    } else {
        let created = fill(&star);
        (star, created)
    };
    let splits = protected
        .iter()
        .filter(|triangle| {
            edge.iter()
                .all(|node| triangle.node_identities.contains(node))
        })
        .map(|triangle| {
            let slot = (0..3)
                .find(|slot| {
                    let candidate = [
                        triangle.node_identities[*slot],
                        triangle.node_identities[(*slot + 1) % 3],
                    ];
                    edge.contains(&candidate[0]) && edge.contains(&candidate[1])
                })
                .expect("protected edge belongs to triangle");
            let a = triangle.node_identities[slot];
            let b = triangle.node_identities[(slot + 1) % 3];
            let c = triangle.node_identities[(slot + 2) % 3];
            ProtectedFacetSplit {
                source: *triangle,
                children: vec![
                    DelaunayRecoveredFacetTriangle {
                        node_identities: [a, topology.nodes[inserted as usize].identity, c],
                    },
                    DelaunayRecoveredFacetTriangle {
                        node_identities: [topology.nodes[inserted as usize].identity, b, c],
                    },
                ],
            }
        })
        .collect();
    Ok((removed, created, splits))
}
