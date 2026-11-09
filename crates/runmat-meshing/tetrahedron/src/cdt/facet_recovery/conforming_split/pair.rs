//! Pair splits: two recovered segment edges that share an endpoint, split the same distance from
//! it in one cavity trial.
use super::*;

/// Splits `edge` and `other`, which share one endpoint, at dyadic points the same distance from
/// that endpoint, in one cavity trial. The distance is half the shorter edge: `edge` is split at
/// its midpoint unless `other` is shorter, in which case `edge` is split at half the length of
/// `other`. A larger distance would place the point on `other` near its far end, which leaves a
/// sliver facet triangle there and encroaches the segments at that end. `other` is split through
/// its edge star first, so `edge` survives for its constrained Bowyer-Watson split; link flips
/// around both nodes and the exact admission of every changed cell decide the edit.
#[allow(clippy::too_many_arguments)]
pub(super) fn split_pair_at_shared_vertex(
    working: &mut super::super::working::FacetRecoveryWorking,
    edge: [StableDigest; 2],
    other: [StableDigest; 2],
    protected: &[DelaunayRecoveredFacetTriangle],
    protected_edges: &[[StableDigest; 2]],
    placement: DisjointPairPlacement,
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
            None if placement == DisjointPairPlacement::Aligned => {
                let (edge_node, edge_recovered) = recovered_segment_midpoint_on(
                    working.topology(),
                    &working.segments,
                    edge_location.0,
                    edge_location.1,
                )
                .map_err(super::super::segment_error)?;
                match aligned_split_point(
                    working.topology(),
                    &working.segments,
                    other,
                    other_location,
                    edge_node.coordinates_m.canonical_value(),
                    constraint_index,
                )? {
                    Ok(other_split) => [(edge_node, edge_recovered), other_split],
                    Err(reason) => return Ok(Err(reason)),
                }
            }
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
                .map_err(super::super::segment_error)?
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
        let mut swallowed = Vec::<[StableDigest; 2]>::new();
        let unsplit_intervals = {
            let mut splitting = [edge, other]
                .into_iter()
                .chain(encroached.iter().map(|(split, _, _)| *split))
                .collect::<Vec<_>>();
            for key in &mut splitting {
                key.sort_unstable();
            }
            protected_edges
                .iter()
                .copied()
                .filter(|candidate| {
                    let mut key = *candidate;
                    key.sort_unstable();
                    !splitting.contains(&key)
                })
                .collect::<Vec<_>>()
        };
        let insertion = work.options.segment_recovery.insertion;
        let cancellation = work.cancellation;
        let outcome = working.edit(|topology| {
        let mut trial = topology.cavity_trial();
        let outcome = (|| -> Result<Result<Vec<ProtectedFacetSplit>, String>, DelaunayFacetRecoveryError> {
            let mut observed = trial.observe_edge_changes();
            let mut protected = protected.to_vec();
            let mut all_splits = Vec::new();
            let mut retriangulated = Vec::new();
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
                // Each split re-triangulates the facet triangles its node encroaches, bounded by
                // the kept edges: the triangles between two nearly parallel split segments are
                // fanned around the first node, then around the second.
                let CavitySplit {
                    removed,
                    created,
                    splits,
                    retriangulated: fans,
                } = split_edge_with_facet_cavities(
                    &observed,
                    index,
                    node,
                    split,
                    &protected,
                    &kept_edges,
                    strategy,
                    constraint_index,
                )?;
                // A constrained cavity that holds the whole star of a kept edge (a recovered
                // segment edge or a half of the first split) would delete that edge, and no link
                // flip can bring it back. The edge-star split removes only cells around the split
                // edge and keeps every other edge, so it takes over.
                let mut split_key = split;
                split_key.sort_unstable();
                let kept_list = kept_edges.iter().copied().collect::<Vec<_>>();
                let swallows_kept_edge = !removed.is_empty()
                    && encroached_segment_edge(&observed, &removed, &kept_list, split_key).is_some();
                // A swallowed recovered segment interval that this edit does not split cannot
                // coexist with the node in any constrained-Delaunay complex; the edge-star split
                // below keeps it, and if the edit is then declined it is split on the next try.
                if swallows_kept_edge {
                    if let Some(interval) = encroached_segment_edge(
                        &observed,
                        &removed,
                        &unsplit_intervals,
                        split_key,
                    ) {
                        swallowed.push(interval);
                    }
                }
                let CavitySplit {
                    removed,
                    created,
                    splits,
                    retriangulated: fans,
                } = if removed.is_empty() || swallows_kept_edge {
                    split_edge_with_facet_cavities(
                        &observed,
                        index,
                        node,
                        split,
                        &protected,
                        &kept_edges,
                        SplitStrategy::EdgeStar,
                        constraint_index,
                    )?
                } else {
                    CavitySplit {
                        removed,
                        created,
                        splits,
                        retriangulated: fans,
                    }
                };
                retriangulated.extend(fans);
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
            if retriangulated_face_missing(&observed, &retriangulated, &protected_faces) {
                return Ok(Err(
                    "a re-triangulated facet triangle is not a face after link flips".to_owned(),
                ));
            }
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
        blocking.extend(swallowed);
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
                .map_err(super::super::segment_error)?;
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

/// Where a pair split of two disjoint segment edges places its points.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum DisjointPairPlacement {
    /// Both dyadic midpoints.
    Midpoints,
    /// The midpoint of `edge`, and the dyadic point of `other` nearest the projection of that
    /// midpoint onto `other`. Two nearly parallel edges a small gap apart then get split points
    /// facing each other, and the facet triangles between them are fanned into short rungs.
    Aligned,
}

/// The dyadic split point of `other` nearest the projection of `point` onto it.
fn aligned_split_point(
    topology: &impl crate::cdt::insertion::topology_access::TetrahedralTopology,
    segments: &[crate::cdt::DelaunayRecoveredSegment],
    other: [StableDigest; 2],
    location: (usize, usize),
    point: [f64; 3],
    constraint_index: u32,
) -> Result<
    Result<(DelaunayVolumeNode, crate::cdt::DelaunayRecoveredSegmentNode), String>,
    DelaunayFacetRecoveryError,
> {
    let at = |identity: StableDigest| {
        topology
            .node_index(identity)
            .map(|index| {
                topology.nodes()[index as usize]
                    .coordinates_m
                    .canonical_value()
            })
            .ok_or_else(|| invalid(constraint_index, "split segment vertex is missing"))
    };
    let (start, end) = (at(other[0])?, at(other[1])?);
    let direction = [0, 1, 2].map(|axis| end[axis] - start[axis]);
    let length_squared = direction.iter().map(|value| value * value).sum::<f64>();
    if length_squared <= 0.0 || !length_squared.is_finite() {
        return Ok(Err("the aligned split edge has no length".to_owned()));
    }
    let along = [0, 1, 2]
        .map(|axis| (point[axis] - start[axis]) * direction[axis])
        .iter()
        .sum::<f64>()
        / length_squared;
    // Keep the point well inside the interval; the dyadic search rounds it to the nearest
    // parameter that lies strictly inside.
    let along = along.clamp(0.25, 0.75);
    let distance_m = along * length_squared.sqrt();
    Ok(crate::cdt::segment_recovery::recovered_segment_point_on(
        topology,
        segments,
        location.0,
        location.1,
        Some((other[0], distance_m)),
    )
    .map_err(super::super::segment_error)?
    .ok_or_else(|| {
        format!("no dyadic point lies {distance_m} m from the aligned edge's first vertex")
    }))
}

pub(super) type EncroachedCornerSegment = (
    [StableDigest; 2],
    DelaunayVolumeNode,
    crate::cdt::DelaunayRecoveredSegmentNode,
);

/// The pair-split points of `edges = [edge, other]`, which share the vertex `shared`: dyadic
/// points the same distance from `shared`, that distance being half the shorter edge (see
/// [`split_pair_at_shared_vertex`]). `locations` are the edges' segment intervals.
pub(in crate::cdt::facet_recovery) fn shared_vertex_split_points(
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
        .map_err(super::super::segment_error)?
        else {
            return Ok(Err(format!(
                "no dyadic point lies {} m from the shared vertex inside the split edge",
                0.5 * other_length_m
            )));
        };
        split
    } else {
        recovered_segment_midpoint_on(topology, segments, edge_location.0, edge_location.1)
            .map_err(super::super::segment_error)?
    };
    let distance_m = distance(edge_node.coordinates_m.canonical_value(), shared_point);
    let Some(other_split) = crate::cdt::segment_recovery::recovered_segment_point_on(
        topology,
        segments,
        other_location.0,
        other_location.1,
        Some((shared, distance_m)),
    )
    .map_err(super::super::segment_error)?
    else {
        return Ok(Err(format!(
            "no dyadic point lies {distance_m} m from the shared vertex inside the encroached edge"
        )));
    };
    Ok(Ok([(edge_node, edge_recovered), other_split]))
}
