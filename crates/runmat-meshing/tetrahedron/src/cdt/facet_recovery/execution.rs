//! Ordered facet recovery over one mutable topology owner. Artifact types and
//! independent terminal verification remain in the parent and validation modules.
use super::*;

pub(in crate::cdt) fn construct_delaunay_facet_recovery(
    segment_recovery: DelaunaySegmentRecovery,
    constraints: &DelaunayConstraints,
    options: DelaunayFacetRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayFacetRecovery, DelaunayFacetRecoveryError> {
    let mut working = FacetRecoveryWorking::new(&segment_recovery);
    let mut work = FacetRecoveryWork::new(options, cancellation);
    let mut facets = Vec::with_capacity(constraints.facets.len());
    let mut protection = FacetProtection::default();
    let mut protected_edges = recovered_segment_edges(&segment_recovery);
    let mut steiner_insertions = Vec::new();
    for constraint_index in 0..constraints.facets.len() {
        let progress_span = constraint_index.is_multiple_of(64).then(|| {
            tracing::info_span!(
                "meshing.tetrahedron.facet_recovery.progress",
                constraint_index,
                facet_count = constraints.facets.len()
            )
        });
        let _progress_guard = progress_span.as_ref().map(|span| span.enter());
        work.begin_constraint();
        // Recovery mutates the tetrahedral topology. Deriving every support up front would leave
        // later facets holding diagonals that an earlier recovery may legitimately remove.
        let initial_support = facet_support_on(
            working.topology(),
            &working.segments,
            constraints,
            constraint_index as u32,
            &mut work,
        )?;
        let support_is_topological = initial_support.iter().all(|triangle| {
            triangle_edges(triangle.node_identities)
                .into_iter()
                .all(|edge| support_edge_exists(working.topology(), edge))
        });
        if !support_is_topological {
            let footprint = constraints.facets[constraint_index]
                .vertex_indices
                .map(|index| constraints.nodes[index as usize].identity);
            if try_recover_facet_patch_in_place(
                &mut working,
                footprint,
                &initial_support,
                &protection.triangles,
                &protected_edges,
                constraint_index as u32,
                &mut work,
            )? {
                for triangle in &initial_support {
                    protection.insert(*triangle);
                }
                facets.push(DelaunayRecoveredFacet {
                    constraint_index: constraint_index as u32,
                    triangles: initial_support,
                });
                continue;
            }
        }
        let mut pending = initial_support
            .into_iter()
            .map(|triangle| PendingSupportTriangle {
                lineage_support: triangle.node_identities,
                triangle,
            })
            .collect::<VecDeque<_>>();
        if pending.len() as u64 > options.maximum_pending_support_triangles {
            return Err(resource(
                constraint_index as u32,
                "facet support queue exceeded its configured triangle limit",
            ));
        }
        let mut support = Vec::new();
        let mut refinement_rank = 0_u64;
        while let Some(pending_triangle) = pending.pop_front() {
            if !triangle_has_positive_area_on(
                working.topology(),
                pending_triangle.triangle.node_identities,
                constraint_index as u32,
            )? {
                continue;
            }
            work.clear_rejection();
            let mut refinement_rejection = None;
            let mut suspended = Vec::new();
            let edge_state =
                if working.support_edges_exist(pending_triangle.triangle.node_identities) {
                    SupportTriangleEdgeState::Ready
                } else {
                    let missing_edges = triangle_edges(pending_triangle.triangle.node_identities)
                        .into_iter()
                        .filter(|edge| !support_edge_exists(working.topology(), *edge))
                        .collect::<Vec<_>>();
                    suspended = facet_delaunay::encroached_by_edges(
                        working.topology(),
                        &support,
                        &missing_edges,
                    );
                    suspend(&mut protection, &suspended);
                    let protected_faces = protection.ordered_faces();
                    let context = || SupportRefinementContext {
                        protected_triangles: &protection.triangles,
                        protected_faces: &protected_faces,
                        protected_edges: &protected_edges,
                        constraint_index: constraint_index as u32,
                        refinement_rank,
                    };
                    match ensure_support_triangle_edges_in_place(
                        &mut working,
                        &pending_triangle,
                        context(),
                        &mut work,
                    )? {
                        Some(state) => state,
                        None => {
                            let in_place = work.last_rejection().map(str::to_owned);
                            let state = working.canonical_call(
                                constraint_index as u32,
                                &mut work,
                                |source, work| {
                                    ensure_support_triangle_edges(
                                        source,
                                        &pending_triangle,
                                        context(),
                                        work,
                                    )
                                },
                            )?;
                            if let (SupportTriangleEdgeState::Unavailable, Some(in_place)) =
                                (&state, in_place)
                            {
                                let canonical = work
                                    .last_rejection()
                                    .unwrap_or("no reason recorded")
                                    .to_owned();
                                work.reject(format!("{canonical}; in-place path: {in_place}"));
                            }
                            match state {
                                SupportTriangleEdgeState::Unavailable => {
                                    match split_segment_encroached_by_support_edge(
                                        &mut working,
                                        &pending_triangle,
                                        &protection.triangles,
                                        &protected_edges,
                                        constraint_index as u32,
                                        &mut work,
                                    )? {
                                        Some(protected_splits) => {
                                            SupportTriangleEdgeState::RefinedSegment {
                                                protected_splits,
                                            }
                                        }
                                        None => SupportTriangleEdgeState::Unavailable,
                                    }
                                }
                                state => state,
                            }
                        }
                    }
                };
            let vanished = settle_suspended(&working, &mut protection, &mut support, suspended);
            match edge_state {
                SupportTriangleEdgeState::Ready => push_vanished(&mut pending, &vanished),
                SupportTriangleEdgeState::Unavailable => {
                    push_vanished(&mut pending, &vanished);
                    refinement_rejection = work.last_rejection().map(str::to_owned);
                }
                SupportTriangleEdgeState::Split {
                    triangles,
                    insertions,
                    protected_splits,
                } => {
                    tracing::info!(
                        event_code = "meshing.tetrahedron.facet_recovery.support_split",
                        constraint_index,
                        refinement_rank,
                        child_count = triangles.len(),
                        inserted_node_count = insertions.len(),
                        "facet support split"
                    );
                    apply_splits_with_current(
                        &mut protection,
                        &mut facets,
                        &mut support,
                        constraint_index as u32,
                        protected_splits,
                    );
                    refinement_rank += 1;
                    push_split(
                        &working,
                        &mut pending,
                        pending_triangle,
                        triangles,
                        vanished,
                        &insertions,
                        &protected_edges,
                        options.maximum_pending_support_triangles,
                        constraint_index as u32,
                    )?;
                    steiner_insertions.extend(insertions);
                    continue;
                }
                SupportTriangleEdgeState::RefinedSegment { protected_splits } => {
                    push_vanished(&mut pending, &vanished);
                    let split_span = tracing::info_span!(
                        "meshing.tetrahedron.facet_recovery.segment_split",
                        constraint_index,
                        refined_segment_node_count = working.steiner_node_count(),
                        protected_facet_split_count = protected_splits.len(),
                    );
                    let _split_guard = split_span.enter();
                    tracing::info!(
                        event_code = "meshing.tetrahedron.facet_recovery.segment_split",
                        constraint_index,
                        refined_segment_node_count = working.steiner_node_count(),
                        protected_facet_split_count = protected_splits.len(),
                        "facet recovery split a protected segment"
                    );
                    apply_splits_with_current(
                        &mut protection,
                        &mut facets,
                        &mut support,
                        constraint_index as u32,
                        protected_splits,
                    );
                    let refined_edges = working
                        .segments
                        .iter()
                        .flat_map(|segment| segment.nodes.windows(2))
                        .map(|pair| {
                            let mut edge = [pair[0].identity, pair[1].identity];
                            edge.sort_unstable();
                            edge
                        })
                        .collect::<std::collections::BTreeSet<_>>();
                    pending.push_front(pending_triangle);
                    insert_segment_split_nodes(
                        &working,
                        &mut pending,
                        &protected_edges,
                        &refined_edges,
                        options.maximum_pending_support_triangles,
                        constraint_index as u32,
                    )?;
                    protected_edges = refined_edges.into_iter().collect();
                    continue;
                }
            }
            let triangle = pending_triangle.triangle;
            if !triangle_has_positive_area_on(
                working.topology(),
                triangle.node_identities,
                constraint_index as u32,
            )? {
                continue;
            }
            if !face_exists(
                working.topology(),
                triangle.node_identities,
                constraint_index as u32,
                &mut work,
            )? && !working.edit(|topology| {
                try_recover_facet_with_local_edge_flip(
                    topology,
                    triangle.node_identities,
                    &protected_edges,
                    &protection.faces,
                    constraint_index as u32,
                    &mut work,
                )
                .map(|recovered| (recovered, recovered))
            })? {
                // The facet cavity runs on the working topology first; the canonical provider
                // decides only when the in-place path declines, and its topology then becomes
                // the working topology inside canonical_call.
                let cavity_insertions = match working.atomic(|working| {
                    let recovered = try_recover_facet_with_cavity_in_place(
                        working,
                        triangle.node_identities,
                        &protection.triangles,
                        &protected_edges,
                        constraint_index as u32,
                        &mut work,
                    )?;
                    let changed = recovered.is_some();
                    Ok((recovered, changed))
                })? {
                    Some(insertions) => Some(insertions),
                    None => {
                        let in_place = work.last_rejection().map(str::to_owned);
                        let canonical = working.canonical_call(
                            constraint_index as u32,
                            &mut work,
                            |source, work| {
                                Ok(try_recover_facet_with_cavity(
                                    source,
                                    triangle.node_identities,
                                    &protection.triangles,
                                    constraint_index as u32,
                                    work,
                                )?
                                .map(|updated| {
                                    source.topology = updated.topology;
                                    updated.steiner_insertions
                                }))
                            },
                        )?;
                        if let (None, Some(in_place)) = (&canonical, in_place) {
                            let reason = work
                                .last_rejection()
                                .unwrap_or("no reason recorded")
                                .to_owned();
                            work.reject(format!("{reason}; in-place cavity path: {in_place}"));
                        }
                        canonical
                    }
                };
                if let Some(insertions) = cavity_insertions {
                    steiner_insertions.extend(insertions);
                } else {
                    let recovery_rejection = work.last_rejection().map(str::to_owned);
                    let interior_suspended = match interior_support_node(
                        working.topology(),
                        &pending_triangle,
                        &protected_edges,
                        constraint_index as u32,
                        refinement_rank,
                        &mut work,
                    )? {
                        Some((node, _, _)) => facet_delaunay::encroached_by_point(
                            working.topology(),
                            &support,
                            node.coordinates_m.canonical_value(),
                        ),
                        None => Vec::new(),
                    };
                    work.clear_rejection();
                    suspend(&mut protection, &interior_suspended);
                    let protected_faces = protection.ordered_faces();
                    let interior_context = || SupportRefinementContext {
                        protected_triangles: &protection.triangles,
                        protected_faces: &protected_faces,
                        protected_edges: &protected_edges,
                        constraint_index: constraint_index as u32,
                        refinement_rank,
                    };
                    let interior = match working.atomic(|working| {
                        let state = refine_support_triangle_interior_in_place(
                            working,
                            &pending_triangle,
                            interior_context(),
                            &mut work,
                        )?;
                        let changed = matches!(
                            state,
                            Some(
                                SupportTriangleEdgeState::Split { .. }
                                    | SupportTriangleEdgeState::RefinedSegment { .. }
                            )
                        );
                        Ok((state, changed))
                    })? {
                        Some(state) => state,
                        None => {
                            let in_place = work.last_rejection().map(str::to_owned);
                            let state = working.canonical_call(
                                constraint_index as u32,
                                &mut work,
                                |source, work| {
                                    refine_support_triangle_interior(
                                        source,
                                        &pending_triangle,
                                        interior_context(),
                                        work,
                                    )
                                },
                            )?;
                            if let (SupportTriangleEdgeState::Unavailable, Some(in_place)) =
                                (&state, in_place)
                            {
                                let reason = work
                                    .last_rejection()
                                    .unwrap_or("no reason recorded")
                                    .to_owned();
                                work.reject(format!(
                                    "{reason}; in-place interior path: {in_place}"
                                ));
                            }
                            state
                        }
                    };
                    let vanished = settle_suspended(
                        &working,
                        &mut protection,
                        &mut support,
                        interior_suspended,
                    );
                    if !matches!(interior, SupportTriangleEdgeState::Split { .. }) {
                        push_vanished(&mut pending, &vanished);
                    }
                    match interior {
                        SupportTriangleEdgeState::Split {
                            triangles,
                            insertions,
                            protected_splits,
                        } => {
                            tracing::info!(
                                event_code = "meshing.tetrahedron.facet_recovery.interior_split",
                                constraint_index,
                                refinement_rank,
                                child_count = triangles.len(),
                                inserted_node_count = insertions.len(),
                                "facet interior split"
                            );
                            apply_splits_with_current(
                                &mut protection,
                                &mut facets,
                                &mut support,
                                constraint_index as u32,
                                protected_splits,
                            );
                            refinement_rank += 1;
                            push_split(
                                &working,
                                &mut pending,
                                pending_triangle,
                                triangles,
                                vanished,
                                &insertions,
                                &protected_edges,
                                options.maximum_pending_support_triangles,
                                constraint_index as u32,
                            )?;
                            steiner_insertions.extend(insertions);
                            continue;
                        }
                        SupportTriangleEdgeState::RefinedSegment { protected_splits } => {
                            tracing::info!(
                                event_code = "meshing.tetrahedron.facet_recovery.segment_split",
                                constraint_index,
                                refined_segment_node_count = working.steiner_node_count(),
                                protected_facet_split_count = protected_splits.len(),
                                "facet recovery split a protected segment"
                            );
                            apply_splits_with_current(
                                &mut protection,
                                &mut facets,
                                &mut support,
                                constraint_index as u32,
                                protected_splits,
                            );
                            let refined_edges = working
                                .segments
                                .iter()
                                .flat_map(|segment| segment.nodes.windows(2))
                                .map(|pair| {
                                    let mut edge = [pair[0].identity, pair[1].identity];
                                    edge.sort_unstable();
                                    edge
                                })
                                .collect::<std::collections::BTreeSet<_>>();
                            pending.push_front(pending_triangle);
                            insert_segment_split_nodes(
                                &working,
                                &mut pending,
                                &protected_edges,
                                &refined_edges,
                                options.maximum_pending_support_triangles,
                                constraint_index as u32,
                            )?;
                            protected_edges = refined_edges.into_iter().collect();
                            continue;
                        }
                        SupportTriangleEdgeState::Ready | SupportTriangleEdgeState::Unavailable => {
                        }
                    }
                    let recovery_rejection =
                        work.last_rejection()
                            .unwrap_or(recovery_rejection.as_deref().unwrap_or(
                            "neither the protected flip nor cavity recovery admitted a candidate",
                        ));
                    let detail = refinement_rejection.map_or_else(
                        || recovery_rejection.to_owned(),
                        |refinement| {
                            format!(
                                "support refinement rejected: {refinement}; facet recovery rejected: {recovery_rejection}"
                            )
                        },
                    );
                    let coordinates = triangle.node_identities.map(|identity| {
                        node_index(working.topology(), identity).map(|index| {
                            working.topology().nodes()[index]
                                .coordinates_m
                                .canonical_value()
                        })
                    });
                    return Err(error(
                        DelaunayFacetRecoveryErrorKind::UnsatisfiableConstraint,
                        Some(constraint_index as u32),
                        format!(
                            "facet support {:?} at {coordinates:?} is absent and recovery rejected every candidate: {detail}",
                            triangle.node_identities,
                        ),
                    ));
                }
            }
            protection.insert(triangle);
            support.push(triangle);
        }
        facets.push(DelaunayRecoveredFacet {
            constraint_index: constraint_index as u32,
            triangles: support,
        });
    }
    steiner_insertions.sort_by_key(|insertion| {
        (
            insertion.constraint_index,
            insertion.support_node_identities,
            insertion.insertion_round,
            insertion.candidate_rank,
            insertion.node_identity,
        )
    });
    let (topology, recovered_segments) = working.finish(&work)?;
    let recovery = DelaunayFacetRecovery {
        segment_recovery,
        recovered_segments,
        topology,
        facets,
        steiner_insertions,
    };
    Ok(recovery)
}

/// Applies protected-facet changes to every recovered facet, including the triangles the current
/// constraint has recovered so far.
fn apply_splits_with_current(
    protection: &mut FacetProtection,
    facets: &mut Vec<DelaunayRecoveredFacet>,
    support: &mut Vec<DelaunayRecoveredFacetTriangle>,
    constraint_index: u32,
    splits: Vec<super::conforming_split::ProtectedFacetSplit>,
) {
    facets.push(DelaunayRecoveredFacet {
        constraint_index,
        triangles: std::mem::take(support),
    });
    protection.apply_splits(facets, splits);
    *support = facets.pop().expect("current facet").triangles;
}

pub(super) fn support_edge_exists(
    topology: &impl TetrahedralTopology,
    endpoints: [StableDigest; 2],
) -> bool {
    let [Some(first), Some(second)] = endpoints.map(|id| topology.node_index(id)) else {
        return false;
    };
    topology.has_edge([first, second])
}

/// Lifts the protection of the current facet's recovered triangles that the next refinement
/// step can encroach (see [`facet_delaunay`]).
fn suspend(protection: &mut FacetProtection, triangles: &[DelaunayRecoveredFacetTriangle]) {
    for triangle in triangles {
        protection.remove(*triangle);
    }
}

/// Protects again each suspended triangle whose face survived the step. The others leave the
/// current facet's recovered support and are returned.
fn settle_suspended(
    working: &FacetRecoveryWorking,
    protection: &mut FacetProtection,
    support: &mut Vec<DelaunayRecoveredFacetTriangle>,
    suspended: Vec<DelaunayRecoveredFacetTriangle>,
) -> Vec<DelaunayRecoveredFacetTriangle> {
    let mut vanished = Vec::new();
    for triangle in suspended {
        let [Some(a), Some(b), Some(c)] = triangle
            .node_identities
            .map(|identity| working.topology().node_index(identity))
        else {
            vanished.push(triangle);
            continue;
        };
        if working.topology().face_incidence([a, b, c]).is_empty() {
            vanished.push(triangle);
        } else {
            protection.insert(triangle);
        }
    }
    support.retain(|triangle| !vanished.contains(triangle));
    vanished
}

/// Returns triangles that a step removed without a node in their circumcircle to the pending
/// support as they are.
fn push_vanished(
    pending: &mut VecDeque<PendingSupportTriangle>,
    vanished: &[DelaunayRecoveredFacetTriangle],
) {
    for triangle in vanished.iter().rev() {
        pending.push_front(PendingSupportTriangle {
            triangle: *triangle,
            lineage_support: triangle.node_identities,
        });
    }
}

/// Places the nodes that a segment split added on the current facet's unrecovered support: a
/// split node on the facet boundary enters the pending triangles by the same 2D Bowyer-Watson
/// step as a facet Steiner node, so no pending triangle keeps an edge that is no longer a
/// segment interval. Nodes off the pending triangles are skipped.
fn insert_segment_split_nodes(
    working: &FacetRecoveryWorking,
    pending: &mut VecDeque<PendingSupportTriangle>,
    previous_edges: &[[StableDigest; 2]],
    refined_edges: &std::collections::BTreeSet<[StableDigest; 2]>,
    maximum_pending: u64,
    constraint_index: u32,
) -> Result<(), DelaunayFacetRecoveryError> {
    let previous_nodes = previous_edges
        .iter()
        .flatten()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    // Each new node, in chain order, with the edge it split: between the last node already
    // placed before it and the next node that existed before the split.
    let mut nodes = Vec::new();
    for segment in &working.segments {
        let chain = &segment.nodes;
        let mut placed = None;
        for (index, entry) in chain.iter().enumerate() {
            if previous_nodes.contains(&entry.identity) {
                placed = Some(entry.identity);
                continue;
            }
            let next = chain[index + 1..]
                .iter()
                .find(|later| previous_nodes.contains(&later.identity));
            if let (Some(left), Some(right)) = (placed, next) {
                nodes.push((entry.identity, [left, right.identity]));
            }
            placed = Some(entry.identity);
        }
    }
    if nodes.is_empty() {
        return Ok(());
    }
    let region = pending.drain(..).collect::<Vec<_>>();
    let Some(triangles) = facet_delaunay::insert_segment_nodes(
        working.topology(),
        region.clone(),
        &nodes,
        refined_edges,
    ) else {
        pending.extend(region);
        return Ok(());
    };
    if triangles.len() as u64 > maximum_pending {
        return Err(resource(
            constraint_index,
            "facet support queue exceeded its configured triangle limit",
        ));
    }
    pending.extend(triangles);
    Ok(())
}

/// Queues the support that replaces a split support triangle. The split's new nodes are inserted
/// into the facet's unrecovered triangles (the split triangle, the recovered triangles the step
/// removed and the pending ones) by 2D Bowyer-Watson steps, so the pending support matches the
/// facet triangulation that the 3D Delaunay insertion produced and fan splits leave no slivers.
/// A split without new nodes, or a cavity that is not star-shaped from its node, keeps the
/// split's own children.
#[allow(clippy::too_many_arguments)]
fn push_split(
    working: &FacetRecoveryWorking,
    pending: &mut VecDeque<PendingSupportTriangle>,
    split: PendingSupportTriangle,
    children: Vec<PendingSupportTriangle>,
    vanished: Vec<DelaunayRecoveredFacetTriangle>,
    insertions: &[DelaunayFacetSteinerInsertion],
    protected_edges: &[[StableDigest; 2]],
    maximum_pending: u64,
    constraint_index: u32,
) -> Result<(), DelaunayFacetRecoveryError> {
    if !insertions.is_empty() {
        let mut region = Vec::with_capacity(pending.len() + vanished.len() + 1);
        region.push(PendingSupportTriangle {
            triangle: split.triangle,
            lineage_support: split.lineage_support,
        });
        region.extend(vanished.iter().map(|triangle| PendingSupportTriangle {
            triangle: *triangle,
            lineage_support: triangle.node_identities,
        }));
        region.extend(pending.iter().map(|entry| PendingSupportTriangle {
            triangle: entry.triangle,
            lineage_support: entry.lineage_support,
        }));
        let nodes = insertions
            .iter()
            .map(|insertion| insertion.node_identity)
            .collect::<Vec<_>>();
        let segment_edges = protected_edges.iter().copied().collect();
        if let Some(triangles) =
            facet_delaunay::insert_nodes(working.topology(), region, &nodes, &segment_edges)
        {
            if triangles.len() as u64 > maximum_pending {
                return Err(resource(
                    constraint_index,
                    "facet support queue exceeded its configured triangle limit",
                ));
            }
            pending.clear();
            pending.extend(triangles);
            return Ok(());
        }
        // The cavity is not star-shaped from a node: keep the split's own children.
        push_vanished(pending, &vanished);
    }
    ensure_pending_capacity(pending, children.len(), maximum_pending, constraint_index)?;
    for triangle in children.into_iter().rev() {
        pending.push_front(triangle);
    }
    Ok(())
}
