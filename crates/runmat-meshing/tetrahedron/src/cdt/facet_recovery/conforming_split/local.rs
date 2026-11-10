//! Local split and exact admission of a segment split on the working topology, with the
//! re-triangulation of protected facets around the split node.
use super::*;

pub(super) enum LocalSplit {
    Accepted(Vec<ProtectedFacetSplit>),
    /// The cavity holds the whole star of this other recovered segment edge.
    Encroached([StableDigest; 2]),
    /// Link flips leave an illegal face in the node's star that carries this other recovered
    /// segment interval as a face edge.
    Blocked {
        reason: String,
        segment: [StableDigest; 2],
    },
    Declined(String),
}

/// A segment split replacement together with the re-triangulation of the protected facet
/// triangles that the split node encroaches (see `segment_split_cavities`).
pub(super) struct CavitySplit {
    pub(super) removed: BTreeSet<u32>,
    pub(super) created: Vec<([u32; 4], Option<runmat_geometry_core::PersistentEntityId>)>,
    pub(super) splits: Vec<ProtectedFacetSplit>,
    /// The fan triangles of the facet cavities; the caller requires each one that stays
    /// protected to be a face of the admitted topology.
    pub(super) retriangulated: Vec<DelaunayRecoveredFacetTriangle>,
}

/// A recovered segment edge, the node that splits it and how its cells are split.
#[derive(Clone, Copy)]
pub(super) struct SegmentEdgeSplit {
    pub(super) node: DelaunayVolumeNode,
    pub(super) edge: [StableDigest; 2],
    pub(super) strategy: SplitStrategy,
}

/// Splits `edge` at the appended node `inserted`. Protected triangles of the facets on the edge
/// whose circumcircle holds the node leave protection and are re-triangulated in their facet;
/// the split and the link flips then see them as ordinary faces, and the caller's admission
/// requires their replacements. `segment_edges` bounds the facet cavities and includes `edge`.
pub(super) fn split_edge_with_facet_cavities(
    topology: &TopologyMutation,
    inserted: u32,
    split: SegmentEdgeSplit,
    protected: &[DelaunayRecoveredFacetTriangle],
    segment_edges: &BTreeSet<[StableDigest; 2]>,
    constraint_index: u32,
) -> Result<CavitySplit, DelaunayFacetRecoveryError> {
    let SegmentEdgeSplit {
        node,
        edge,
        strategy,
    } = split;
    let cavities = super::super::facet_delaunay::segment_split_cavities(
        topology,
        protected,
        edge,
        node.identity,
        node.coordinates_m.canonical_value(),
        segment_edges,
        MAXIMUM_SPLIT_FACET_CAVITY_TRIANGLES,
    );
    let released = cavities
        .iter()
        .flat_map(|cavity| cavity.sources.iter().copied())
        .collect::<Vec<_>>();
    let split_protection = protected
        .iter()
        .copied()
        .filter(|triangle| {
            !released.contains(triangle)
                || edge
                    .iter()
                    .all(|endpoint| triangle.node_identities.contains(endpoint))
        })
        .collect::<Vec<_>>();
    let (removed, created, mut splits) = split_edge(
        topology,
        inserted,
        edge,
        &split_protection,
        strategy,
        constraint_index,
    )?;
    let retriangulated = cavities
        .iter()
        .flat_map(|cavity| cavity.children.iter().copied())
        .collect::<Vec<_>>();
    if !cavities.is_empty() {
        tracing::info!(
            event_code = "meshing.tetrahedron.facet_recovery.split_facet_cavity",
            constraint_index,
            facet_count = cavities.len(),
            released_triangle_count = released.len(),
            child_triangle_count = retriangulated.len(),
            "a segment split node encroaches protected triangles of a facet on the edge, which \
             are re-triangulated around it"
        );
        splits.retain(|split| !released.contains(&split.source));
        for cavity in cavities {
            let mut sources = cavity.sources.into_iter();
            if let Some(first) = sources.next() {
                splits.push(ProtectedFacetSplit {
                    source: first,
                    children: cavity.children,
                });
            }
            splits.extend(sources.map(|source| ProtectedFacetSplit {
                source,
                children: Vec::new(),
            }));
        }
    }
    Ok(CavitySplit {
        removed,
        created,
        splits,
        retriangulated,
    })
}

/// Whether a re-triangulated facet triangle that is still protected is missing as a face.
pub(super) fn retriangulated_face_missing(
    topology: &TopologyMutation,
    retriangulated: &[DelaunayRecoveredFacetTriangle],
    protected_faces: &BTreeSet<[StableDigest; 3]>,
) -> bool {
    retriangulated.iter().any(|triangle| {
        let mut face = triangle.node_identities;
        face.sort_unstable();
        if !protected_faces.contains(&face) {
            return false;
        }
        let [Some(a), Some(b), Some(c)] =
            triangle.node_identities.map(|identity| topology.node_index(identity))
        else {
            return true;
        };
        topology.face_incidence([a, b, c]).is_empty()
    })
}

pub(super) fn split_and_admit_locally(
    topology: &mut TopologyMutation,
    split: SegmentEdgeSplit,
    protected: &[DelaunayRecoveredFacetTriangle],
    protected_edges: &[[StableDigest; 2]],
    constraint_index: u32,
    options: crate::cdt::DelaunayInsertionOptions,
    cancellation: &dyn runmat_meshing_core::MeshingCancellationSignal,
) -> Result<LocalSplit, DelaunayFacetRecoveryError> {
    let SegmentEdgeSplit { node, edge, .. } = split;
    let mut split_edge_key = edge;
    split_edge_key.sort_unstable();
    let mut observed = topology.observe_edge_changes();
    let Ok(inserted) = observed.append_node(node) else {
        return Ok(LocalSplit::Declined("duplicate node".to_owned()));
    };
    let CavitySplit {
        removed,
        created,
        mut splits,
        retriangulated,
    } = split_edge_with_facet_cavities(
        &observed,
        inserted,
        split,
        protected,
        &protected_edges
            .iter()
            .copied()
            .chain([split_edge_key])
            .collect(),
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
            let illegal = illegal_star_face(&observed, inserted, &protected_faces, &kept_edges);
            let reason = format!(
                "an illegal face remains in the new node's star after link flips (split edge {:?}): {}",
                edge.map(|identity| observed
                    .node_index(identity)
                    .map(|index| observed.nodes[index as usize].coordinates_m.canonical_value())),
                illegal.description
            );
            // A recovered segment interval on the illegal face, other than a half of the split,
            // keeps the face: no link flip removes it while that interval stays unsplit.
            return Ok(
                match illegal.kept_edges.into_iter().find(|kept| {
                    !kept.contains(&node.identity)
                        && *kept != split_edge_key
                        && protected_edges.iter().any(|candidate| {
                            let mut candidate = *candidate;
                            candidate.sort_unstable();
                            candidate == *kept
                        })
                }) {
                    Some(segment) => LocalSplit::Blocked { reason, segment },
                    None => LocalSplit::Declined(reason),
                },
            );
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
    // The re-triangulated facet triangles that the flips kept must exist as faces.
    if retriangulated_face_missing(&observed, &retriangulated, &protected_faces) {
        return Ok(LocalSplit::Declined(
            "a re-triangulated facet triangle is not a face after link flips".to_owned(),
        ));
    }
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

/// Describes the first unprotected face in the star of `inserted` whose neighbor apex lies
/// strictly inside the cell's circumsphere, for the decline reason: its corners, the apex, the
/// face edges that must stay, and the protected triangles that share an edge with the face.
pub(super) struct IllegalStarFace {
    pub(super) description: String,
    /// The face edges in `kept_edges`, sorted.
    pub(super) kept_edges: Vec<[StableDigest; 2]>,
}

pub(super) fn illegal_star_face(
    topology: &TopologyMutation,
    inserted: u32,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    kept_edges: &BTreeSet<[StableDigest; 2]>,
) -> IllegalStarFace {
    let point = |vertex: u32| topology.nodes[vertex as usize].coordinates_m.canonical_value();
    for left_index in topology.vertex_stars[inserted as usize].iter().copied() {
        let Some(left) = topology.tetrahedron(left_index) else {
            continue;
        };
        for (opposite, neighbor) in left.neighbors.iter().enumerate() {
            let Some(right) = neighbor.and_then(|index| topology.tetrahedron(index)) else {
                continue;
            };
            let face = canonical_mutation_face(topology, left.vertex_indices, opposite);
            if left.region_id != right.region_id || protected_faces.contains(&face) {
                continue;
            }
            let Some(apex) = right
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !left.vertex_indices.contains(vertex))
            else {
                continue;
            };
            let nodes = left.vertex_indices.map(|vertex| topology.nodes[vertex as usize]);
            if !topology
                .insphere_nodes([
                    nodes[0],
                    nodes[1],
                    nodes[2],
                    nodes[3],
                    topology.nodes[apex as usize],
                ])
                .is_ok_and(|sign| sign == PredicateSign::Positive)
            {
                continue;
            }
            let corners = (0..4)
                .filter(|slot| *slot != opposite)
                .map(|slot| left.vertex_indices[slot])
                .collect::<Vec<_>>();
            let at = |identity: &StableDigest| topology.node_index(*identity).map(point);
            let kept_keys = [(0, 1), (1, 2), (0, 2)]
                .into_iter()
                .filter_map(|(a, b)| {
                    let mut key = [face[a], face[b]];
                    key.sort_unstable();
                    kept_edges.contains(&key).then_some(key)
                })
                .collect::<Vec<_>>();
            let kept = kept_keys
                .iter()
                .map(|key| key.map(|identity| at(&identity)))
                .collect::<Vec<_>>();
            let adjacent = protected_faces
                .iter()
                .filter(|other| other.iter().filter(|node| face.contains(node)).count() == 2)
                .map(|other| other.map(|identity| at(&identity)))
                .collect::<Vec<_>>();
            let description = format!(
                "face {:?} (opposite cell vertex {:?}) with apex {:?}; {} kept face edges {kept:?}, \
                 {} protected triangles share a face edge {adjacent:?}, inserted node on face: {}",
                corners.iter().map(|vertex| point(*vertex)).collect::<Vec<_>>(),
                point(left.vertex_indices[opposite]),
                point(apex),
                kept.len(),
                adjacent.len(),
                corners.contains(&inserted)
            );
            return IllegalStarFace {
                description,
                kept_edges: kept_keys,
            };
        }
    }
    IllegalStarFace {
        description: "not located".to_owned(),
        kept_edges: Vec::new(),
    }
}

pub(super) fn copy_splits(splits: &[ProtectedFacetSplit]) -> Vec<ProtectedFacetSplit> {
    splits
        .iter()
        .map(|split| ProtectedFacetSplit {
            source: split.source,
            children: split.children.clone(),
        })
        .collect()
}

pub(super) fn protected_face_keys(
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
pub(super) fn facet_flip_splits(
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
