use std::collections::BTreeSet;

use runmat_meshing_core::{
    quality::predicate::{
        points_collinear_constructed, projected_orient2d_constructed,
        triangle_projection_axes_constructed, PredicateSign,
    },
    StableDigest,
};

use super::{
    cavity::recovered_segments_exist, error, node_index, DelaunayFacetRecoveryError,
    DelaunayFacetRecoveryErrorKind, DelaunayRecoveredFacetTriangle, DelaunaySegmentRecovery,
    FacetRecoveryWork,
};
use crate::cdt::{
    insertion::{
        legalize_delaunay_volume_topology_with_barriers, legalize_inserted_node_and_check_star,
        topology_access::TetrahedralTopology, validate_constrained_delaunay_volume_topology,
    },
    mutation::{TopologyMutation, TopologyMutationErrorKind},
    segment_recovery::{
        record_recovered_segment_split, recovered_segment_midpoint, recovered_segment_midpoint_on,
    },
    topology::{orient_materialized_volume_nodes, orient_volume_nodes},
    DelaunayInsertionErrorKind, DelaunayVolumeNode, DelaunayVolumeTopology,
};

pub(super) struct ProtectedFacetSplit {
    pub(super) source: DelaunayRecoveredFacetTriangle,
    pub(super) children: Vec<DelaunayRecoveredFacetTriangle>,
}

pub(super) struct ConformingInsertion {
    pub(super) recovery: DelaunaySegmentRecovery,
    pub(super) splits: Vec<ProtectedFacetSplit>,
}

pub(super) fn apply_protected_facet_splits(
    facets: &mut [super::DelaunayRecoveredFacet],
    protected: &mut Vec<DelaunayRecoveredFacetTriangle>,
    splits: Vec<ProtectedFacetSplit>,
) {
    for split in splits {
        for facet in facets.iter_mut() {
            if let Some(index) = facet
                .triangles
                .iter()
                .position(|triangle| *triangle == split.source)
            {
                facet
                    .triangles
                    .splice(index..=index, split.children.clone());
            }
        }
        if let Some(index) = protected
            .iter()
            .position(|triangle| *triangle == split.source)
        {
            protected.splice(index..=index, split.children);
        }
    }
}

enum PointLocation {
    Interior,
    Edge([StableDigest; 2]),
}

pub(super) fn try_insert_on_protected_facets(
    current: &DelaunaySegmentRecovery,
    node: DelaunayVolumeNode,
    protected: &[DelaunayRecoveredFacetTriangle],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<ConformingInsertion>, DelaunayFacetRecoveryError> {
    let mut hits = Vec::new();
    for triangle in protected {
        work.support_step(constraint_index)?;
        if let Some(location) = point_location(current, node, *triangle, constraint_index)? {
            hits.push((*triangle, location));
        }
    }
    if hits.is_empty() {
        return Ok(None);
    }
    let edge = hits.iter().find_map(|(_, location)| match location {
        PointLocation::Interior => None,
        PointLocation::Edge(edge) => Some(*edge),
    });
    insert_conformingly(
        current,
        node,
        protected,
        edge,
        &hits,
        constraint_index,
        work,
    )
}

pub(super) fn try_split_recovered_segment_edge(
    current: &DelaunaySegmentRecovery,
    edge: [StableDigest; 2],
    protected: &[DelaunayRecoveredFacetTriangle],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<ConformingInsertion>, DelaunayFacetRecoveryError> {
    work.support_step(constraint_index)?;
    let Some((segment_index, interval_index)) = recovered_segment_edge_location(current, edge)
    else {
        return Ok(None);
    };
    let (node, _) = recovered_segment_midpoint(current, segment_index, interval_index)
        .map_err(super::segment_error)?;
    if current.topology.nodes.iter().any(|existing| {
        crate::cdt::segment_recovery::coordinates_share_representable_location(
            existing.coordinates_m.canonical_value(),
            node.coordinates_m.canonical_value(),
        )
    }) {
        work.reject("conflicting segment midpoint duplicates an existing node position");
        return Ok(None);
    }
    insert_conformingly(
        current,
        node,
        protected,
        Some(edge),
        &[],
        constraint_index,
        work,
    )
}

/// Splits a recovered segment edge directly on the working topology: the split, Lawson flips in
/// the new node's link and an exact admission of every changed cell run in a cavity trial, so
/// the cost follows the size of the change. Returns `None` with the topology unchanged when the
/// local repair cannot finish; the caller then runs the canonical split with complete
/// legalization.
///
/// The constrained Bowyer-Watson cavity of the midpoint can contain the whole star of another
/// recovered segment edge. The midpoint then lies strictly inside every circumsphere around
/// that edge, so the edge has no empty circumsphere once the midpoint exists and no
/// constrained-Delaunay complex contains both. That encroached edge is split first (bounded
/// cascade), and the original split is retried on the refined segments.
pub(super) fn split_recovered_segment_edge_in_place(
    working: &mut super::working::FacetRecoveryWorking,
    edge: [StableDigest; 2],
    protected: &[DelaunayRecoveredFacetTriangle],
    protected_edges: &[[StableDigest; 2]],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<Vec<ProtectedFacetSplit>>, DelaunayFacetRecoveryError> {
    let protected = protected.to_vec();
    let protected_edges = protected_edges.to_vec();
    let mut trial_splits = Vec::new();
    // The cascade edits the working topology through committed splits; a declined cascade
    // leaves those committed encroachment splits in place, which are themselves valid
    // constrained-Delaunay refinements of recovered segments.
    let result = split_with_encroachment_cascade(
        working,
        edge,
        protected,
        protected_edges,
        0,
        None,
        constraint_index,
        work,
        &mut trial_splits,
    )?;
    Ok(if result || !trial_splits.is_empty() {
        Some(trial_splits)
    } else {
        None
    })
}

const MAXIMUM_ENCROACHMENT_DEPTH: u8 = 8;

/// Returns whether `edge` itself was split; `splits` receives every protected-facet split made,
/// in application order, including those of encroached edges split first.
#[allow(clippy::too_many_arguments)]
fn split_with_encroachment_cascade(
    working: &mut super::working::FacetRecoveryWorking,
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
        .map_err(super::segment_error)?;
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
fn refresh_constraint_views(
    working: &super::working::FacetRecoveryWorking,
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

/// Splits `edge` and `other`, which share one endpoint, at dyadic points the same distance from
/// that endpoint, in one cavity trial. The distance is half the shorter edge: `edge` is split at
/// its midpoint unless `other` is shorter, in which case `edge` is split at half the length of
/// `other`. A larger distance would place the point on `other` near its far end, which leaves a
/// sliver facet triangle there and encroaches the segments at that end. `other` is split through
/// its edge star first, so `edge` survives for its constrained Bowyer-Watson split; link flips
/// around both nodes and the exact admission of every changed cell decide the edit.
#[allow(clippy::too_many_arguments)]
fn split_pair_at_shared_vertex(
    working: &mut super::working::FacetRecoveryWorking,
    edge: [StableDigest; 2],
    other: [StableDigest; 2],
    protected: &[DelaunayRecoveredFacetTriangle],
    protected_edges: &[[StableDigest; 2]],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Result<Vec<ProtectedFacetSplit>, String>, DelaunayFacetRecoveryError> {
    let (Some(edge_location), Some(other_location)) = (
        segment_edge_location(&working.segments, edge),
        segment_edge_location(&working.segments, other),
    ) else {
        return Ok(Err("an edge is not a recovered segment interval".to_owned()));
    };
    let [(edge_node, edge_recovered), (other_node, other_recovered)] =
        match other.iter().copied().find(|node| edge.contains(node)) {
            Some(shared) => match shared_vertex_split_points(
                working.topology(),
                &working.segments,
                [edge, other],
                shared,
                [edge_location, other_location],
                constraint_index,
            )? {
                Ok(points) => points,
                Err(reason) => return Ok(Err(reason)),
            },
            // Disjoint edges that encroach each other (the midpoint split of either swallows
            // the other's star) are split at both midpoints in one edit.
            None => [edge_location, other_location]
                .map(|(segment, interval)| {
                    recovered_segment_midpoint_on(
                        working.topology(),
                        &working.segments,
                        segment,
                        interval,
                    )
                })
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .map_err(super::segment_error)?
                .try_into()
                .expect("two split points"),
        };
    if [edge_node, other_node].iter().any(|candidate| {
        working.topology().nodes.iter().any(|existing| {
            crate::cdt::segment_recovery::coordinates_share_representable_location(
                existing.coordinates_m.canonical_value(),
                candidate.coordinates_m.canonical_value(),
            )
        })
    }) {
        return Ok(Err(
            "a split point duplicates an existing node position".to_owned()
        ));
    }
    // Recovered segment edges that an earlier attempt of this edit found blocking: an illegal
    // face whose only repair removes that edge (see `observed_change_reflex_edges`). Each is
    // split in the same edit on the next attempt.
    let mut forced = Vec::<EncroachedCornerSegment>::new();
    loop {
        let mut seeds = vec![(edge, edge_node), (other, other_node)];
        seeds.extend(forced.iter().map(|(split, node, _)| (*split, *node)));
        let mut encroached = match encroached_corner_segments(working, &seeds, constraint_index)? {
            Ok(encroached) => encroached,
            Err(reason) => return Ok(Err(reason)),
        };
        // Split first: the blocking edges are kept by every other split of the edit.
        encroached.extend(forced.iter().cloned());
        let mut blocking = Vec::<[StableDigest; 2]>::new();
        let insertion = work.options.segment_recovery.insertion;
        let cancellation = work.cancellation;
        let outcome = working.edit(|topology| {
        let mut trial = topology.cavity_trial();
        let outcome = (|| -> Result<Result<Vec<ProtectedFacetSplit>, String>, DelaunayFacetRecoveryError> {
            let mut observed = trial.observe_edge_changes();
            let mut protected = protected.to_vec();
            let mut all_splits = Vec::new();
            let mut halves = vec![
                [edge[0], edge_node.identity],
                [edge[1], edge_node.identity],
                [other[0], other_node.identity],
                [other[1], other_node.identity],
            ];
            for (split, node, _) in &encroached {
                halves.extend(split.map(|endpoint| [endpoint, node.identity]));
            }
            // Link flips keep every recovered segment edge, both split edges until they are
            // split, and the four halves.
            let kept_edges = crate::cdt::segment_recovery::sorted_edge_keys(
                protected_edges
                    .iter()
                    .copied()
                    .chain([edge, other])
                    .chain(encroached.iter().map(|(split, _, _)| *split))
                    .chain(halves.iter().copied()),
            );
            let protected_face_set = |protected: &[DelaunayRecoveredFacetTriangle]| {
                protected
                    .iter()
                    .map(|triangle| {
                        let mut face = triangle.node_identities;
                        face.sort_unstable();
                        face
                    })
                    .collect::<BTreeSet<_>>()
            };
            // Each node is legalized before the next is placed, so the second split starts from
            // a constrained-Delaunay complex: its constrained cavity is then exact, and link
            // flips only repair what visibility or barriers left out of that cavity.
            // Segments that the pair nodes block are split first, the last of the cascade first,
            // each by its constrained cavity (the edge star when that cavity would delete a kept
            // edge). Split later, their still-kept edges would push the pair nodes' cavities onto
            // edge stars, whose link flips can stall.
            let order = encroached
                .iter()
                .rev()
                .map(|(split, node, _)| (*node, *split, SplitStrategy::ConstrainedCavity))
                .chain([
                    (other_node, other, SplitStrategy::EdgeStar),
                    (edge_node, edge, SplitStrategy::ConstrainedCavity),
                ])
                .collect::<Vec<_>>();
            for (node, split, strategy) in order {
                let Ok(index) = observed.append_node(node) else {
                    return Ok(Err("a split node was rejected".to_owned()));
                };
                let (removed, created, splits) =
                    split_edge(&observed, index, split, &protected, strategy, constraint_index)?;
                // A constrained cavity that holds the whole star of a kept edge (a recovered
                // segment edge or a half of the first split) would delete that edge, and no link
                // flip can bring it back. The edge-star split removes only cells around the split
                // edge and keeps every other edge, so it takes over.
                let mut split_key = split;
                split_key.sort_unstable();
                let kept_list = kept_edges.iter().copied().collect::<Vec<_>>();
                let swallows_kept_edge = !removed.is_empty()
                    && encroached_segment_edge(&observed, &removed, &kept_list, split_key).is_some();
                let (removed, created, splits) = if removed.is_empty() || swallows_kept_edge {
                    split_edge(
                        &observed,
                        index,
                        split,
                        &protected,
                        SplitStrategy::EdgeStar,
                        constraint_index,
                    )?
                } else {
                    (removed, created, splits)
                };
                if removed.is_empty() {
                    return Ok(Err("a split produced no replacement".to_owned()));
                }
                if let Err(failure) = observed.replace_cavity(&removed, created) {
                    return Ok(Err(format!("a split replacement was rejected: {failure}")));
                }
                apply_protected_facet_splits(
                    &mut [],
                    &mut protected,
                    splits
                        .iter()
                        .map(|split| ProtectedFacetSplit {
                            source: split.source,
                            children: split.children.clone(),
                        })
                        .collect(),
                );
                all_splits.extend(splits);
                match crate::cdt::insertion::legalize_inserted_nodes_with_facet_flips(
                    &mut observed,
                    &[index],
                    &protected_face_set(&protected),
                    &kept_edges,
                    true,
                    insertion,
                    cancellation,
                ) {
                    Ok(flips) => {
                        let Some(flip_splits) = facet_flip_splits(&observed, &mut protected, &flips)
                        else {
                            return Ok(Err(
                                "a facet-plane flip names a protected triangle that is not recorded"
                                    .to_owned(),
                            ));
                        };
                        all_splits.extend(flip_splits);
                    }
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
                    Err(failure) => return Ok(Err(format!("link flips failed: {failure}"))),
                }
            }
            let protected_faces = protected_face_set(&protected);
            let mut replaced = [edge, other]
                .into_iter()
                .chain(encroached.iter().map(|(split, _, _)| *split))
                .collect::<Vec<_>>();
            for key in &mut replaced {
                key.sort_unstable();
            }
            let mut checkpoint = || Ok::<(), DelaunayFacetRecoveryError>(());
            let removed_edges = observed.removed_edge_candidates(&mut checkpoint)?;
            let lost = protected_edges
                .iter()
                .filter(|candidate| !replaced.contains(candidate))
                .copied()
                .map(|candidate| (candidate, false))
                .chain(halves.into_iter().map(|half| (half, true)))
                .find(|(candidate, half)| {
                    let [Some(first), Some(second)] =
                        candidate.map(|identity| observed.node_index(identity))
                    else {
                        return true;
                    };
                    let mut key = [first, second];
                    key.sort_unstable();
                    (*half || removed_edges.contains(&key)) && !observed.has_edge(key)
                });
            if let Some((lost, _)) = lost {
                return Ok(Err(format!(
                    "the pair split lost recovered segment edge {:?}",
                    lost.map(|identity| observed.node_index(identity).map(
                        |index| observed.nodes[index as usize].coordinates_m.canonical_value()
                    ))
                )));
            }
            match crate::cdt::insertion::observed_change_violation(
                &observed,
                &observed.changed_cells(),
                &protected_faces,
                insertion,
                cancellation,
            ) {
                Ok(None) => Ok(Ok(all_splits)),
                Ok(Some(violation)) => {
                    let reflex = crate::cdt::insertion::observed_change_reflex_edges(
                        &observed,
                        &observed.changed_cells(),
                        &protected_faces,
                        insertion,
                        cancellation,
                    )
                    .map_err(|failure| {
                        insertion_error(failure.kind, failure.to_string(), constraint_index)
                    })?;
                    for pair in reflex {
                        let mut key = pair.map(|index| observed.nodes[index as usize].identity);
                        key.sort_unstable();
                        if !replaced.contains(&key)
                            && protected_edges.iter().any(|candidate| {
                                let mut candidate = *candidate;
                                candidate.sort_unstable();
                                candidate == key
                            })
                        {
                            blocking.push(key);
                        }
                    }
                    Ok(Err(violation))
                }
                Err(failure) => Err(insertion_error(
                    failure.kind,
                    failure.to_string(),
                    constraint_index,
                )),
            }
        })()?;
        let changed = outcome.is_ok();
        if changed {
            trial.commit();
        }
        Ok::<_, DelaunayFacetRecoveryError>((outcome, changed))
    })?;
        if outcome.is_err() && encroached.len() < MAXIMUM_ENCROACHED_CORNER_SPLITS {
            if let Some(location) = blocking
                .first()
                .and_then(|key| segment_edge_location(&working.segments, *key).map(|at| (*key, at)))
            {
                let (key, (segment, interval)) = location;
                let (midpoint, recovered) = recovered_segment_midpoint_on(
                    working.topology(),
                    &working.segments,
                    segment,
                    interval,
                )
                .map_err(super::segment_error)?;
                if !working.topology().nodes.iter().any(|existing| {
                    crate::cdt::segment_recovery::coordinates_share_representable_location(
                        existing.coordinates_m.canonical_value(),
                        midpoint.coordinates_m.canonical_value(),
                    )
                }) {
                    forced.push((key, midpoint, recovered));
                    continue;
                }
            }
        }
        if outcome.is_ok() {
            working.record_segment_split(other_location.0, other_location.1, other_recovered)?;
            let edge_location =
                segment_edge_location(&working.segments, edge).ok_or_else(|| {
                    invalid(constraint_index, "split edge left its recovered segment")
                })?;
            working.record_segment_split(edge_location.0, edge_location.1, edge_recovered)?;
            for (split, _, recovered) in encroached {
                let location =
                    segment_edge_location(&working.segments, split).ok_or_else(|| {
                        invalid(
                            constraint_index,
                            "encroached edge left its recovered segment",
                        )
                    })?;
                working.record_segment_split(location.0, location.1, recovered)?;
            }
        }
        return Ok(outcome);
    }
}

type EncroachedCornerSegment = (
    [StableDigest; 2],
    DelaunayVolumeNode,
    crate::cdt::DelaunayRecoveredSegmentNode,
);

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
fn encroached_corner_segments(
    working: &super::working::FacetRecoveryWorking,
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
                .map_err(super::segment_error)?;
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
fn opposite_sides(a: [f64; 3], b: [f64; 3], first: [f64; 3], second: [f64; 3]) -> bool {
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
fn strictly_inside_circumcircle(triangle: [[f64; 3]; 3], point: [f64; 3]) -> bool {
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

const MAXIMUM_ENCROACHED_CORNER_SPLITS: usize = 64;

/// The pair-split points of `edges = [edge, other]`, which share the vertex `shared`: dyadic
/// points the same distance from `shared`, that distance being half the shorter edge (see
/// [`split_pair_at_shared_vertex`]). `locations` are the edges' segment intervals.
pub(super) fn shared_vertex_split_points(
    topology: &impl crate::cdt::insertion::topology_access::TetrahedralTopology,
    segments: &[crate::cdt::DelaunayRecoveredSegment],
    edges: [[StableDigest; 2]; 2],
    shared: StableDigest,
    locations: [(usize, usize); 2],
    constraint_index: u32,
) -> Result<
    Result<[(DelaunayVolumeNode, crate::cdt::DelaunayRecoveredSegmentNode); 2], String>,
    DelaunayFacetRecoveryError,
> {
    let [edge, other] = edges;
    let [edge_location, other_location] = locations;
    let point = |identity: StableDigest| {
        topology
            .node_index(identity)
            .map(|index| {
                topology.nodes()[index as usize]
                    .coordinates_m
                    .canonical_value()
            })
            .ok_or_else(|| invalid(constraint_index, "split segment vertex is missing"))
    };
    let distance = |first: [f64; 3], second: [f64; 3]| {
        (0..3)
            .map(|axis| (first[axis] - second[axis]).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let shared_point = point(shared)?;
    let other_length_m = distance(point(other[0])?, point(other[1])?);
    let edge_length_m = distance(point(edge[0])?, point(edge[1])?);
    let (edge_node, edge_recovered) = if other_length_m < edge_length_m {
        let Some(split) = crate::cdt::segment_recovery::recovered_segment_point_on(
            topology,
            segments,
            edge_location.0,
            edge_location.1,
            Some((shared, 0.5 * other_length_m)),
        )
        .map_err(super::segment_error)?
        else {
            return Ok(Err(format!(
                "no dyadic point lies {} m from the shared vertex inside the split edge",
                0.5 * other_length_m
            )));
        };
        split
    } else {
        recovered_segment_midpoint_on(topology, segments, edge_location.0, edge_location.1)
            .map_err(super::segment_error)?
    };
    let distance_m = distance(edge_node.coordinates_m.canonical_value(), shared_point);
    let Some(other_split) = crate::cdt::segment_recovery::recovered_segment_point_on(
        topology,
        segments,
        other_location.0,
        other_location.1,
        Some((shared, distance_m)),
    )
    .map_err(super::segment_error)?
    else {
        return Ok(Err(format!(
            "no dyadic point lies {distance_m} m from the shared vertex inside the encroached edge"
        )));
    };
    Ok(Ok([(edge_node, edge_recovered), other_split]))
}

enum LocalSplit {
    Accepted(Vec<ProtectedFacetSplit>),
    /// The cavity holds the whole star of this other recovered segment edge.
    Encroached([StableDigest; 2]),
    Declined(String),
}

#[allow(clippy::too_many_arguments)]
fn split_and_admit_locally(
    topology: &mut TopologyMutation,
    node: DelaunayVolumeNode,
    edge: [StableDigest; 2],
    protected: &[DelaunayRecoveredFacetTriangle],
    protected_edges: &[[StableDigest; 2]],
    strategy: SplitStrategy,
    constraint_index: u32,
    options: crate::cdt::DelaunayInsertionOptions,
    cancellation: &dyn runmat_meshing_core::MeshingCancellationSignal,
) -> Result<LocalSplit, DelaunayFacetRecoveryError> {
    let mut observed = topology.observe_edge_changes();
    let Ok(inserted) = observed.append_node(node) else {
        return Ok(LocalSplit::Declined("duplicate node".to_owned()));
    };
    let mut split_edge_key = edge;
    split_edge_key.sort_unstable();
    let (removed, created, mut splits) = split_edge(
        &observed,
        inserted,
        edge,
        protected,
        strategy,
        constraint_index,
    )?;
    if removed.is_empty() {
        return Ok(LocalSplit::Declined("empty replacement".to_owned()));
    }
    if let Some(encroached) =
        encroached_segment_edge(&observed, &removed, protected_edges, split_edge_key)
    {
        return Ok(LocalSplit::Encroached(encroached));
    }
    if created.iter().any(|(tetrahedron, _)| {
        orient_materialized_volume_nodes(tetrahedron.map(|index| observed.nodes[index as usize]))
            .is_ok_and(|sign| sign == PredicateSign::Zero)
    }) || observed.replace_cavity(&removed, created).is_err()
    {
        return Ok(LocalSplit::Declined(
            "zero-volume or rejected cavity replacement".to_owned(),
        ));
    }
    let protected_faces = updated_protected_faces(protected, &splits)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let kept_edges = crate::cdt::segment_recovery::sorted_edge_keys(
        protected_edges
            .iter()
            .copied()
            .chain(edge.map(|endpoint| [endpoint, node.identity])),
    );
    let flips = match crate::cdt::insertion::legalize_inserted_node_and_check_star_with_facet_flips(
        &mut observed,
        inserted,
        &protected_faces,
        &kept_edges,
        true,
        options,
        cancellation,
    ) {
        Ok((true, flips)) => flips,
        Ok((false, _)) => {
            return Ok(LocalSplit::Declined(
                "an illegal face remains in the new node's star after link flips".to_owned(),
            ))
        }
        Err(failure)
            if matches!(
                failure.kind,
                DelaunayInsertionErrorKind::ResourceLimit | DelaunayInsertionErrorKind::Cancelled
            ) =>
        {
            return Err(insertion_error(
                failure.kind,
                failure.to_string(),
                constraint_index,
            ))
        }
        Err(failure) => {
            return Ok(LocalSplit::Declined(format!(
                "link flips failed: {failure}"
            )))
        }
    };
    // Facet-plane flips retriangulate protected triangles; they join the split records, and the
    // admission below checks the flipped protection.
    let protected_faces = if flips.is_empty() {
        protected_faces
    } else {
        let mut current = protected.to_vec();
        apply_protected_facet_splits(&mut [], &mut current, copy_splits(&splits));
        let Some(flip_splits) = facet_flip_splits(&observed, &mut current, &flips) else {
            return Ok(LocalSplit::Declined(
                "a facet-plane flip names a protected triangle that is not recorded".to_owned(),
            ));
        };
        splits.extend(flip_splits);
        protected_face_keys(&current)
    };
    // The split edge is replaced by its two halves; every other recovered segment edge that a
    // removed cell carried must survive.
    let halves = edge.map(|endpoint| [endpoint, node.identity]);
    let mut checkpoint = || Ok::<(), DelaunayFacetRecoveryError>(());
    let removed_edges = observed.removed_edge_candidates(&mut checkpoint)?;
    let segment_edge_lost = protected_edges
        .iter()
        .filter(|candidate| **candidate != split_edge_key)
        .copied()
        .chain(halves)
        .any(|candidate| {
            let [Some(first), Some(second)] =
                candidate.map(|identity| observed.node_index(identity))
            else {
                return true;
            };
            let mut key = [first, second];
            key.sort_unstable();
            (removed_edges.contains(&key) || candidate.contains(&node.identity))
                && !observed.has_edge(key)
        });
    if segment_edge_lost {
        return Ok(LocalSplit::Declined(
            "a recovered segment edge was lost".to_owned(),
        ));
    }
    let violation = crate::cdt::insertion::observed_change_violation(
        &observed,
        &observed.changed_cells(),
        &protected_faces,
        options,
        cancellation,
    )
    .map_err(|failure| insertion_error(failure.kind, failure.to_string(), constraint_index))?;
    Ok(match violation {
        None => LocalSplit::Accepted(splits),
        Some(violation) => LocalSplit::Declined(violation),
    })
}

fn copy_splits(splits: &[ProtectedFacetSplit]) -> Vec<ProtectedFacetSplit> {
    splits
        .iter()
        .map(|split| ProtectedFacetSplit {
            source: split.source,
            children: split.children.clone(),
        })
        .collect()
}

fn protected_face_keys(
    protected: &[DelaunayRecoveredFacetTriangle],
) -> BTreeSet<[StableDigest; 3]> {
    protected
        .iter()
        .map(|triangle| {
            let mut face = triangle.node_identities;
            face.sort_unstable();
            face
        })
        .collect()
}

/// Records facet-plane flips as protected-facet changes and applies them to `protected`: the
/// first removed triangle is replaced by both created triangles, oriented like it, and the
/// second is dropped. Both lie in one facet (their shared edge is not a segment edge), so each
/// facet's triangles still cover the same region. Returns `None` when a removed triangle is not
/// in `protected`.
fn facet_flip_splits(
    topology: &TopologyMutation,
    protected: &mut Vec<DelaunayRecoveredFacetTriangle>,
    flips: &[crate::cdt::insertion::FacetFlip],
) -> Option<Vec<ProtectedFacetSplit>> {
    let point = |identity: StableDigest| {
        node_index_mutation(topology, identity).map(|index| topology.nodes[index as usize])
    };
    let mut records = Vec::new();
    for flip in flips {
        let find = |face: [StableDigest; 3]| {
            protected.iter().copied().find(|triangle| {
                let mut key = triangle.node_identities;
                key.sort_unstable();
                key == face
            })
        };
        let (first, second) = (find(flip.removed[0])?, find(flip.removed[1])?);
        let off_plane = point(flip.off_plane)?;
        let orientation = |triangle: [StableDigest; 3]| -> Option<PredicateSign> {
            let [a, b, c] = triangle.map(point);
            orient_materialized_volume_nodes([a?, b?, c?, off_plane]).ok()
        };
        let sense = orientation(first.node_identities)?;
        if sense == PredicateSign::Zero {
            return None;
        }
        let children = flip
            .created
            .iter()
            .map(|face| {
                let mut ordered = *face;
                match orientation(ordered)? {
                    PredicateSign::Zero => return None,
                    current if current != sense => ordered.swap(0, 1),
                    _ => {}
                }
                Some(DelaunayRecoveredFacetTriangle {
                    node_identities: ordered,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let applied = vec![
            ProtectedFacetSplit {
                source: first,
                children,
            },
            ProtectedFacetSplit {
                source: second,
                children: Vec::new(),
            },
        ];
        apply_protected_facet_splits(&mut [], protected, copy_splits(&applied));
        records.extend(applied);
    }
    Some(records)
}

/// A recovered segment edge, other than the split edge, whose every incident cell lies in the
/// replacement cavity.
fn encroached_segment_edge(
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

pub(super) fn segment_edge_location(
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

/// How a recovered segment edge is split before legalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SplitStrategy {
    /// The constrained Bowyer-Watson cavity of the new node, corrected to stay star-shaped
    /// from it; the edge star when the corrected cavity is not star-shaped.
    ConstrainedCavity,
    /// The tetrahedra around the edge only.
    EdgeStar,
}

/// Splits with the constrained cavity first. In thin layers between parallel interfaces,
/// splitting only the edge star can leave an illegal link face that no flip removes; the
/// cavity removes those cells before refilling. The edge-star split remains the fallback.
fn insert_conformingly(
    current: &DelaunaySegmentRecovery,
    node: DelaunayVolumeNode,
    protected: &[DelaunayRecoveredFacetTriangle],
    selected_edge: Option<[StableDigest; 2]>,
    hits: &[(DelaunayRecoveredFacetTriangle, PointLocation)],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<ConformingInsertion>, DelaunayFacetRecoveryError> {
    if selected_edge.is_some() {
        if let Some(insertion) = insert_conformingly_with(
            current,
            node,
            protected,
            selected_edge,
            hits,
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
        node,
        protected,
        selected_edge,
        hits,
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

#[allow(clippy::too_many_arguments)]
fn insert_conformingly_with(
    current: &DelaunaySegmentRecovery,
    node: DelaunayVolumeNode,
    protected: &[DelaunayRecoveredFacetTriangle],
    selected_edge: Option<[StableDigest; 2]>,
    hits: &[(DelaunayRecoveredFacetTriangle, PointLocation)],
    strategy: SplitStrategy,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<ConformingInsertion>, DelaunayFacetRecoveryError> {
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

fn finish_conforming_insertion(
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
                .map_err(super::segment_error)?;
            record_recovered_segment_split(&mut recovery, segment_index, interval_index, recovered)
                .map_err(super::segment_error)?;
        }
    }
    if !recovered_segments_exist(&recovery) {
        work.reject("conforming facet insertion removed a recovered segment edge");
        return Ok(None);
    }
    Ok(Some(ConformingInsertion { recovery, splits }))
}

fn recovered_segment_edge_location(
    recovery: &DelaunaySegmentRecovery,
    edge: [StableDigest; 2],
) -> Option<(usize, usize)> {
    segment_edge_location(&recovery.segments, edge)
}

fn point_location(
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

type Replacement = (
    BTreeSet<u32>,
    Vec<([u32; 4], Option<runmat_geometry_core::PersistentEntityId>)>,
    Vec<ProtectedFacetSplit>,
);

fn split_edge(
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

/// The Bowyer-Watson cavity of a node inserted on a recovered segment edge: the edge star plus
/// every tetrahedron reachable from it whose circumsphere strictly contains the node. Protected
/// faces other than the triangles split at the edge, and region interfaces, are barriers.
fn constrained_cavity(
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
fn cone_is_visible(topology: &TopologyMutation, vertices: [u32; 4]) -> bool {
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

fn barrier_faces(
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

fn canonical_mutation_face(
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
fn connected_from_star(
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

fn split_faces(
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

fn updated_protected_faces(
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

fn node_index_mutation(topology: &TopologyMutation, identity: StableDigest) -> Option<u32> {
    topology
        .nodes
        .iter()
        .position(|node| node.identity == identity)
        .map(|index| index as u32)
}

fn insertion_error(
    kind: DelaunayInsertionErrorKind,
    reason: String,
    constraint_index: u32,
) -> DelaunayFacetRecoveryError {
    let kind = match kind {
        DelaunayInsertionErrorKind::ResourceLimit => DelaunayFacetRecoveryErrorKind::ResourceLimit,
        DelaunayInsertionErrorKind::Cancelled => DelaunayFacetRecoveryErrorKind::Cancelled,
        _ => DelaunayFacetRecoveryErrorKind::InvalidTopology,
    };
    error(kind, Some(constraint_index), reason)
}

fn invalid(constraint_index: u32, reason: impl Into<String>) -> DelaunayFacetRecoveryError {
    error(
        DelaunayFacetRecoveryErrorKind::InvalidTopology,
        Some(constraint_index),
        reason,
    )
}
