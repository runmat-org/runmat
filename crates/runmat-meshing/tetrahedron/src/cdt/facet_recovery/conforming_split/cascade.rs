//! Encroachment cascade: a recovered segment edge whose star lies in a split's cavity is split
//! first, then the original split is retried on the refined segments.
use super::*;

pub(super) const MAXIMUM_ENCROACHMENT_DEPTH: u8 = 8;

/// Triangles one facet may re-triangulate around a segment split node.
pub(super) const MAXIMUM_SPLIT_FACET_CAVITY_TRIANGLES: usize = 64;

/// Returns whether `edge` itself was split; `splits` receives every protected-facet split made,
/// in application order, including those of encroached edges split first.
#[allow(clippy::too_many_arguments)]
pub(super) fn split_with_encroachment_cascade(
    working: &mut super::super::working::FacetRecoveryWorking,
    edge: [StableDigest; 2],
    mut protected: Vec<DelaunayRecoveredFacetTriangle>,
    mut protected_edges: Vec<[StableDigest; 2]>,
    depth: u8,
    parent: Option<[StableDigest; 2]>,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
    splits: &mut Vec<ProtectedFacetSplit>,
) -> Result<bool, DelaunayFacetRecoveryError> {
    let insertion = work.options.segment_recovery.insertion;
    let cancellation = work.cancellation;
    let mut attempts = 0_u32;
    let mut reasons = Vec::new();
    // Encroached edges this call has split. A newly swallowed edge that shares an endpoint
    // with one of them is a half the last split made: the midpoint keeps swallowing the
    // shrinking halves at that vertex, so the cascade stops and the edge star is tried.
    let mut cascaded = Vec::<[StableDigest; 2]>::new();
    let mut blocked_done = Vec::<[StableDigest; 2]>::new();
    'retry: loop {
        work.support_step(constraint_index)?;
        attempts += 1;
        if attempts > u32::from(MAXIMUM_ENCROACHMENT_DEPTH) + 1 {
            reasons.push("encroachment retries exhausted".to_owned());
            work.reject(reasons.join("; "));
            return Ok(false);
        }
        let Some((segment_index, interval_index)) = segment_edge_location(&working.segments, edge)
        else {
            if attempts > 1 {
                // A nested pair split of this edge and the edge it encroached split it.
                return Ok(true);
            }
            work.reject("the edge is not an interval of a recovered segment");
            return Ok(false);
        };
        let (node, recovered) = recovered_segment_midpoint_on(
            working.topology(),
            &working.segments,
            segment_index,
            interval_index,
        )
        .map_err(super::super::segment_error)?;
        if working.topology().nodes.iter().any(|existing| {
            crate::cdt::segment_recovery::coordinates_share_representable_location(
                existing.coordinates_m.canonical_value(),
                node.coordinates_m.canonical_value(),
            )
        }) {
            work.reject("conflicting segment midpoint duplicates an existing node position");
            return Ok(false);
        }
        for strategy in [SplitStrategy::ConstrainedCavity, SplitStrategy::EdgeStar] {
            let outcome = working.edit(|topology| {
                let mut trial = topology.cavity_trial();
                let outcome = split_and_admit_locally(
                    &mut trial,
                    node,
                    edge,
                    &protected,
                    &protected_edges,
                    strategy,
                    constraint_index,
                    insertion,
                    cancellation,
                )?;
                let changed = matches!(outcome, LocalSplit::Accepted(_));
                if changed {
                    trial.commit();
                }
                Ok::<_, DelaunayFacetRecoveryError>((outcome, changed))
            })?;
            match outcome {
                LocalSplit::Accepted(accepted) => {
                    working.record_segment_split(segment_index, interval_index, recovered)?;
                    splits.extend(accepted);
                    return Ok(true);
                }
                LocalSplit::Encroached(encroached)
                    if parent.is_some_and(|parent| {
                        let mut parent = parent;
                        let mut encroached = encroached;
                        parent.sort_unstable();
                        encroached.sort_unstable();
                        parent == encroached
                    }) =>
                {
                    // Two disjoint edges encroach each other (each midpoint swallows the
                    // other's star, as on the two faces of a layer thinner than the edges):
                    // split both in one edit. The caller sees its edge split.
                    match split_pair_at_shared_vertex(
                        working,
                        edge,
                        encroached,
                        &protected,
                        &protected_edges,
                        constraint_index,
                        work,
                    )? {
                        Ok(accepted) => {
                            splits.extend(accepted);
                            return Ok(true);
                        }
                        Err(reason) => reasons.push(format!(
                            "pair split with the mutually encroaching edge declined: {reason}"
                        )),
                    }
                }
                LocalSplit::Encroached(encroached)
                    if encroached.iter().any(|node| edge.contains(node)) =>
                {
                    // Midpoint splits of two segments meeting at an acute vertex encroach each
                    // other without end. Split both at the same distance from the shared vertex
                    // in one admitted edit: neither new sub-edge at that vertex then holds the
                    // other's point in its diametral ball.
                    match split_pair_at_shared_vertex(
                        working,
                        edge,
                        encroached,
                        &protected,
                        &protected_edges,
                        constraint_index,
                        work,
                    )? {
                        Ok(accepted) => {
                            splits.extend(accepted);
                            return Ok(true);
                        }
                        Err(reason) => reasons.push(format!(
                            "shared-vertex pair split with encroached edge declined: {reason}"
                        )),
                    }
                }
                LocalSplit::Encroached(encroached)
                    if depth < MAXIMUM_ENCROACHMENT_DEPTH
                        && !cascaded
                            .iter()
                            .any(|done| done.iter().any(|node| encroached.contains(node))) =>
                {
                    cascaded.push(encroached);
                    let first = splits.len();
                    if !split_with_encroachment_cascade(
                        working,
                        encroached,
                        protected.clone(),
                        protected_edges.clone(),
                        depth + 1,
                        Some(edge),
                        constraint_index,
                        work,
                        splits,
                    )? {
                        reasons.push(format!(
                            "in-place split of encroached edge at depth {} declined: {}",
                            depth + 1,
                            work.last_rejection().unwrap_or("no reason recorded")
                        ));
                        continue;
                    }
                    refresh_constraint_views(
                        working,
                        &splits[first..],
                        &mut protected,
                        &mut protected_edges,
                    );
                    continue 'retry;
                }
                LocalSplit::Encroached(encroached) => reasons.push(format!(
                    "in-place {strategy:?} split: the midpoint encroaches recovered segment edge \
                     {:?} beyond the cascade depth or its split was declined",
                    encroached.map(|identity| working.topology().node_index(identity).map(
                        |index| working.topology().nodes[index as usize]
                            .coordinates_m
                            .canonical_value()
                    ))
                )),
                LocalSplit::Declined(reason) => {
                    reasons.push(format!("in-place {strategy:?} split declined: {reason}"))
                }
            }
        }
        // Both splits declined. A segment edge at an endpoint of `edge` whose coplanar opposite
        // vertex lies in the circumcircle of a child triangle of the split blocks it: no
        // constrained-Delaunay complex holds the midpoint while that edge stays. Split the
        // blocking edge, then retry.
        if depth < MAXIMUM_ENCROACHMENT_DEPTH {
            if let Ok(blocking) =
                encroached_corner_segments(working, &[(edge, node)], constraint_index)?
            {
                if let Some((blocked, _, _)) = blocking
                    .into_iter()
                    .find(|(blocked, _, _)| !blocked_done.contains(blocked))
                {
                    blocked_done.push(blocked);
                    let first = splits.len();
                    if split_with_encroachment_cascade(
                        working,
                        blocked,
                        protected.clone(),
                        protected_edges.clone(),
                        depth + 1,
                        Some(edge),
                        constraint_index,
                        work,
                        splits,
                    )? {
                        refresh_constraint_views(
                            working,
                            &splits[first..],
                            &mut protected,
                            &mut protected_edges,
                        );
                        continue 'retry;
                    }
                    reasons.push(format!(
                        "split of blocking segment edge at depth {} declined: {}",
                        depth + 1,
                        work.last_rejection().unwrap_or("no reason recorded")
                    ));
                }
            }
        }
        work.reject(reasons.join("; "));
        return Ok(false);
    }
}

/// Refreshes the caller-side constraint views with splits made by a nested cascade.
pub(super) fn refresh_constraint_views(
    working: &super::super::working::FacetRecoveryWorking,
    applied: &[ProtectedFacetSplit],
    protected: &mut Vec<DelaunayRecoveredFacetTriangle>,
    protected_edges: &mut Vec<[StableDigest; 2]>,
) {
    apply_protected_facet_splits(
        &mut [],
        protected,
        applied
            .iter()
            .map(|split| ProtectedFacetSplit {
                source: split.source,
                children: split.children.clone(),
            })
            .collect(),
    );
    *protected_edges = working
        .segments
        .iter()
        .flat_map(|segment| segment.nodes.windows(2))
        .map(|pair| {
            let mut key = [pair[0].identity, pair[1].identity];
            key.sort_unstable();
            key
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
}

/// A recovered segment edge, other than the split edge, whose every incident cell lies in the
/// replacement cavity.
pub(super) fn encroached_segment_edge(
    topology: &TopologyMutation,
    removed: &BTreeSet<u32>,
    protected_edges: &[[StableDigest; 2]],
    split_edge: [StableDigest; 2],
) -> Option<[StableDigest; 2]> {
    let mut candidates = BTreeSet::new();
    for index in removed {
        let vertices = topology
            .tetrahedron(*index)
            .expect("cavity cells are active")
            .vertex_indices;
        for first in 0..4 {
            for second in first + 1..4 {
                let mut key = [
                    topology.nodes[vertices[first] as usize].identity,
                    topology.nodes[vertices[second] as usize].identity,
                ];
                key.sort_unstable();
                if key != split_edge {
                    candidates.insert((key, [vertices[first], vertices[second]]));
                }
            }
        }
    }
    let keys = candidates
        .iter()
        .map(|(key, _)| *key)
        .collect::<BTreeSet<_>>();
    let segment_edges = protected_edges
        .iter()
        .filter(|edge| keys.contains(*edge))
        .copied()
        .collect::<BTreeSet<_>>();
    candidates
        .into_iter()
        .filter(|(key, _)| segment_edges.contains(key))
        .find(|(_, vertices)| {
            topology
                .edge_tetrahedra(*vertices)
                .iter()
                .all(|cell| removed.contains(cell))
        })
        .map(|(key, _)| key)
}

pub(in crate::cdt::facet_recovery) fn segment_edge_location(
    segments: &[crate::cdt::DelaunayRecoveredSegment],
    mut edge: [StableDigest; 2],
) -> Option<(usize, usize)> {
    edge.sort_unstable();
    segments
        .iter()
        .enumerate()
        .find_map(|(segment_index, segment)| {
            segment
                .nodes
                .windows(2)
                .enumerate()
                .find_map(|(interval_index, interval)| {
                    let mut candidate = [interval[0].identity, interval[1].identity];
                    candidate.sort_unstable();
                    (candidate == edge).then_some((segment_index, interval_index))
                })
        })
}
