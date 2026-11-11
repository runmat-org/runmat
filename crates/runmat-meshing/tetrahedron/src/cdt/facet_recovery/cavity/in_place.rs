//! Facet cavity recovery and Steiner insertion on the working topology. Each step plans on the
//! generic topology, replaces the cavity in a cavity trial and keeps the edit only when the exact
//! local admission holds: every face of every created cell is constrained Delaunay, every
//! protected face and the facet exist, and every recovered segment edge survives. These are the
//! conditions the canonical providers check on the whole complex. A declined step rolls back.

use std::collections::BTreeSet;

use runmat_meshing_core::StableDigest;

use super::steiner::{
    candidate_node, candidate_refills_cavity, facet_edges_exist, steiner_candidate_groups,
    steiner_lineage, CandidateInsertion, SteinerCandidateGroup,
};
use super::{
    canonical_face, facet_recovery_refill_options, plan_facet_cavity,
    progressive_facet_recovery_refill_options, refill_side_exact_with_options,
    resource_or_cancelled, segment_edge_removed, try_recover_facet_patch_in_place, CavityPlan,
};
use crate::cavity::constrained::{
    ConstrainedCavity, ConstrainedCavityNode, ConstrainedCavityRefillOptions,
};
use crate::cdt::facet_recovery::{
    working::FacetRecoveryWorking, DelaunayFacetRecoveryError, DelaunayFacetRecoveryErrorKind,
    DelaunayRecoveredFacetTriangle, FacetRecoveryWork,
};
use crate::cdt::{
    insertion::topology_access::TetrahedralTopology, mutation::EdgeChangeObserver,
    DelaunayFacetSteinerInsertion, DelaunayInsertionErrorKind, DelaunayVolumeNode,
};

/// The outcome of inserting one Steiner node on the working topology.
pub(in crate::cdt::facet_recovery) enum InPlaceCandidate {
    Inserted,
    /// The insertion removed these recovered segment edges; nothing was kept.
    SegmentConflicts(Vec<[StableDigest; 2]>),
    Declined(String),
}

/// Recovers `facet` with its constrained cavity on the working topology, adding interior Steiner
/// nodes when a cavity side cannot be refilled from its own nodes. The candidate order, node
/// identities and lineage are those of [`super::try_recover_facet_with_cavity`]. Returns the
/// Steiner lineage on success; `None` leaves the topology as it was before the last declined step
/// (earlier progressive insertions stay, as the canonical provider's do not) and records why.
pub(in crate::cdt::facet_recovery) fn try_recover_facet_with_cavity_in_place(
    working: &mut FacetRecoveryWorking,
    facet: [StableDigest; 3],
    protected_facets: &[DelaunayRecoveredFacetTriangle],
    protected_edges: &[[StableDigest; 2]],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<Vec<DelaunayFacetSteinerInsertion>>, DelaunayFacetRecoveryError> {
    let cap = [DelaunayRecoveredFacetTriangle {
        node_identities: facet,
    }];
    let protected_faces = protected_facets
        .iter()
        .map(|triangle| canonical_face(triangle.node_identities))
        .chain([canonical_face(facet)])
        .collect::<BTreeSet<_>>();
    let barrier_faces = protected_facets
        .iter()
        .map(|triangle| canonical_face(triangle.node_identities))
        .collect::<Vec<_>>();
    let mut insertion_round = 0_u64;
    let mut steiner_insertions = Vec::new();
    loop {
        let cavities =
            match plan_facet_cavity(working.topology(), facet, &[facet], constraint_index, work)? {
                CavityPlan::Refill { .. } => {
                    // The patch path recomputes the same plan and admits it locally.
                    return Ok(try_recover_facet_patch_in_place(
                        working,
                        facet,
                        &cap,
                        protected_facets,
                        protected_edges,
                        constraint_index,
                        work,
                    )?
                    .then_some(steiner_insertions));
                }
                CavityPlan::Unavailable => return Ok(None),
                CavityPlan::NeedsSteiner(cavities) => cavities,
            };
        let groups =
            steiner_candidate_groups(working.topology(), &cavities, constraint_index, work)?;
        let request = |inserted: Vec<DelaunayVolumeNode>, refill_options| SideRefill {
            inserted,
            facet,
            protected_faces: &protected_faces,
            protected_edges,
            cavities: &cavities,
            refill_options,
            constraint_index,
        };
        if groups.len() == 2 {
            let first_candidates = groups[0].ordered();
            let second_candidates = groups[1].ordered();
            let mut second_refillable = vec![None; second_candidates.len()];
            for (first_rank, first_coordinates) in first_candidates.iter().copied().enumerate() {
                if !candidate_refills_cavity(
                    &cavities[0],
                    working.topology(),
                    first_coordinates,
                    constraint_index,
                    work,
                )? {
                    continue;
                }
                let Some(first_node) = candidate_node(
                    working.topology(),
                    explicit_candidate(first_coordinates, facet, insertion_round, first_rank),
                    constraint_index,
                    work,
                )?
                else {
                    continue;
                };
                for (second_rank, second_coordinates) in
                    second_candidates.iter().copied().enumerate()
                {
                    let refillable = match second_refillable[second_rank] {
                        Some(refillable) => refillable,
                        None => {
                            let refillable = candidate_refills_cavity(
                                &cavities[1],
                                working.topology(),
                                second_coordinates,
                                constraint_index,
                                work,
                            )?;
                            second_refillable[second_rank] = Some(refillable);
                            refillable
                        }
                    };
                    if !refillable {
                        continue;
                    }
                    let Some(second_node) = candidate_node(
                        working.topology(),
                        explicit_candidate(
                            second_coordinates,
                            facet,
                            insertion_round + 1,
                            second_rank,
                        ),
                        constraint_index,
                        work,
                    )?
                    else {
                        continue;
                    };
                    let Some(retained) = refill_sides_in_place(
                        working,
                        request(
                            vec![first_node, second_node],
                            progressive_facet_recovery_refill_options(),
                        ),
                        work,
                    )?
                    else {
                        continue;
                    };
                    for (node, round, rank) in [
                        (first_node, insertion_round, first_rank as u64),
                        (second_node, insertion_round + 1, second_rank as u64),
                    ] {
                        if retained.contains(&node.identity) {
                            work.cavity_steiner_node(constraint_index)?;
                            steiner_insertions.push(steiner_lineage(
                                facet,
                                constraint_index,
                                round,
                                rank,
                                node.identity,
                            ));
                        }
                    }
                    return Ok(Some(steiner_insertions));
                }
            }
            work.reject("no jointly valid Steiner pair refilled both cavity halves");
            return Ok(None);
        }
        let SteinerCandidateGroup { progressive, exact } =
            groups.into_iter().next().unwrap_or_default();
        let exact_rank_offset = progressive.len() as u64;
        let mut advanced = false;
        for (candidate_rank, coordinates_m) in progressive.into_iter().enumerate() {
            let Some(node) = candidate_node(
                working.topology(),
                explicit_candidate(coordinates_m, facet, insertion_round, candidate_rank),
                constraint_index,
                work,
            )?
            else {
                continue;
            };
            if refill_sides_in_place(
                working,
                request(vec![node], facet_recovery_refill_options()),
                work,
            )?
            .is_some()
            {
                work.cavity_steiner_node(constraint_index)?;
                steiner_insertions.push(steiner_lineage(
                    facet,
                    constraint_index,
                    insertion_round,
                    candidate_rank as u64,
                    node.identity,
                ));
                return Ok(Some(steiner_insertions));
            }
            // Progressive insertion: keep the node only when the facet's three edges then exist,
            // and recover the facet again around it.
            match insert_candidate_in_place(
                working,
                InPlaceCandidateRequest {
                    node,
                    required_facet_edges: Some(facet),
                    near: facet[0],
                },
                &barrier_faces,
                protected_edges,
                constraint_index,
                work,
            )? {
                InPlaceCandidate::Inserted => {
                    work.cavity_steiner_node(constraint_index)?;
                    steiner_insertions.push(steiner_lineage(
                        facet,
                        constraint_index,
                        insertion_round,
                        candidate_rank as u64,
                        node.identity,
                    ));
                    insertion_round += 1;
                    advanced = true;
                    break;
                }
                InPlaceCandidate::SegmentConflicts(_) | InPlaceCandidate::Declined(_) => {}
            }
        }
        if advanced {
            continue;
        }
        let progressive_rejection = work
            .last_rejection()
            .unwrap_or("progressive insertion returned no topology")
            .to_owned();
        let mut admitted = Vec::new();
        for (candidate_rank, coordinates_m) in exact.into_iter().enumerate() {
            let candidate_rank = exact_rank_offset + candidate_rank as u64;
            let Some(node) = candidate_node(
                working.topology(),
                explicit_candidate(
                    coordinates_m,
                    facet,
                    insertion_round,
                    candidate_rank as usize,
                ),
                constraint_index,
                work,
            )?
            else {
                continue;
            };
            if refill_sides_in_place(
                working,
                request(vec![node], facet_recovery_refill_options()),
                work,
            )?
            .is_some()
            {
                work.cavity_steiner_node(constraint_index)?;
                steiner_insertions.push(steiner_lineage(
                    facet,
                    constraint_index,
                    insertion_round,
                    candidate_rank,
                    node.identity,
                ));
                return Ok(Some(steiner_insertions));
            }
            admitted.push((node, candidate_rank));
        }
        if !admitted.is_empty() {
            let nodes = admitted.iter().map(|(node, _)| *node).collect::<Vec<_>>();
            if let Some(retained) = refill_sides_in_place(
                working,
                request(nodes, facet_recovery_refill_options()),
                work,
            )? {
                for (node, candidate_rank) in admitted
                    .iter()
                    .copied()
                    .filter(|(node, _)| retained.contains(&node.identity))
                {
                    work.cavity_steiner_node(constraint_index)?;
                    steiner_insertions.push(steiner_lineage(
                        facet,
                        constraint_index,
                        insertion_round,
                        candidate_rank,
                        node.identity,
                    ));
                }
                return Ok(Some(steiner_insertions));
            }
        }
        let batch_rejection = work
            .last_rejection()
            .unwrap_or("atomic Steiner-inventory refill returned no topology")
            .to_owned();
        work.reject(format!(
            "no valid Steiner candidate refilled the cavity in place: admitted_candidates={}, \
             atomic inventory rejected: {batch_rejection}; progressive candidates rejected: \
             {progressive_rejection}",
            admitted.len(),
        ));
        return Ok(None);
    }
}

fn explicit_candidate(
    coordinates_m: [f64; 3],
    facet: [StableDigest; 3],
    insertion_round: u64,
    candidate_rank: usize,
) -> CandidateInsertion {
    CandidateInsertion {
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                coordinates_m,
            ),
        construction: runmat_meshing_core::quality::predicate::Point3Construction::Explicit,
        facet,
        insertion_round,
        candidate_rank: candidate_rank as u64,
    }
}

struct SideRefill<'a> {
    inserted: Vec<DelaunayVolumeNode>,
    facet: [StableDigest; 3],
    /// Protected faces including the facet.
    protected_faces: &'a BTreeSet<[StableDigest; 3]>,
    protected_edges: &'a [[StableDigest; 2]],
    cavities: &'a [ConstrainedCavity],
    refill_options: ConstrainedCavityRefillOptions,
    constraint_index: u32,
}

/// Refills one or both cavity sides with their own nodes and `inserted`, as one cavity
/// replacement on the working topology. Inserted nodes that no refill cell uses are not added.
/// Returns the identities of the inserted nodes kept, or `None` after recording why.
fn refill_sides_in_place(
    working: &mut FacetRecoveryWorking,
    request: SideRefill<'_>,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<Vec<StableDigest>>, DelaunayFacetRecoveryError> {
    let SideRefill {
        inserted,
        facet,
        protected_faces,
        protected_edges,
        cavities,
        refill_options,
        constraint_index,
    } = request;
    if cavities.is_empty()
        || cavities.len() > 2
        || cavities
            .windows(2)
            .any(|pair| pair[0].removed_tetrahedron_ids != pair[1].removed_tetrahedron_ids)
    {
        work.reject("atomic facet refill received inconsistent cavity sides");
        return Ok(None);
    }
    let topology = working.topology();
    let mut positions = BTreeSet::new();
    let mut identities = BTreeSet::new();
    if inserted.iter().any(|node| {
        let coordinates = node.coordinates_m.canonical_value();
        topology.node_index(node.identity).is_some()
            || topology.contains_position(coordinates)
            || !identities.insert(node.identity)
            || !positions.insert(coordinates.map(|value| (value + 0.0).to_bits()))
    }) {
        work.reject("atomic facet refill received duplicate Steiner node evidence");
        return Ok(None);
    }
    let base = u32::try_from(topology.nodes().len()).map_err(|_| {
        resource_or_cancelled(
            DelaunayFacetRecoveryErrorKind::ResourceLimit,
            constraint_index,
            "facet cavity node index space is exhausted".to_owned(),
        )
    })?;
    let removed = cavities[0]
        .removed_tetrahedron_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut local = removed
        .iter()
        .flat_map(|index| {
            topology
                .tetrahedron(*index)
                .expect("cavity cells are active")
                .vertex_indices
        })
        .map(|node| {
            (
                node,
                topology.nodes()[node as usize]
                    .coordinates_m
                    .canonical_value(),
            )
        })
        .collect::<Vec<_>>();
    local.sort_unstable_by_key(|(node, _)| *node);
    local.dedup_by_key(|(node, _)| *node);
    local.extend(
        inserted
            .iter()
            .enumerate()
            .map(|(offset, node)| (base + offset as u32, node.coordinates_m.canonical_value())),
    );
    let cavity_nodes = local
        .into_iter()
        .map(|(node_id, coordinates)| ConstrainedCavityNode {
            node_id,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(coordinates),
        })
        .collect::<Vec<_>>();
    let mut refills = Vec::new();
    for cavity in cavities {
        let Some(refill) = refill_side_exact_with_options(
            cavity,
            &cavity_nodes,
            refill_options,
            constraint_index,
            work,
        )?
        else {
            let detail = work
                .last_rejection()
                .unwrap_or("the cavity refiller returned no candidate")
                .to_owned();
            work.reject(format!(
                "atomic facet cavity side could not be retriangulated: {detail}"
            ));
            return Ok(None);
        };
        refills.extend(refill);
    }
    // Keep only inserted nodes the refill uses, in order, at consecutive appended indices.
    let used = refills.iter().flatten().copied().collect::<BTreeSet<_>>();
    let mut remap = Vec::with_capacity(inserted.len());
    let mut kept = Vec::new();
    for (offset, node) in inserted.iter().enumerate() {
        if used.contains(&(base + offset as u32)) {
            remap.push(Some(base + kept.len() as u32));
            kept.push(*node);
        } else {
            remap.push(None);
        }
    }
    let refills = refills
        .into_iter()
        .map(|cell| {
            cell.map(|node| match node.checked_sub(base) {
                Some(offset) => remap[offset as usize].expect("used inserted node is kept"),
                None => node,
            })
        })
        .collect::<Vec<_>>();
    let insertion = work.options.segment_recovery.insertion;
    let cancellation = work.cancellation;
    let retained = kept.iter().map(|node| node.identity).collect::<Vec<_>>();
    let accepted = working.edit(|topology| {
        let mut trial = topology.cavity_trial();
        let rejection = {
            let mut observed = trial.observe_edge_changes();
            let mut rejection = None;
            for node in &kept {
                if observed.append_node(*node).is_err() {
                    rejection = Some("an atomic facet refill node was rejected".to_owned());
                    break;
                }
            }
            match rejection {
                Some(rejection) => Some(rejection),
                None => match observed
                    .replace_cavity(&removed, refills.iter().map(|cell| (*cell, None)).collect())
                {
                    Err(failure) => Some(format!(
                        "atomic facet cavity topology was invalid: {failure}"
                    )),
                    Ok(_) => admission(
                        &observed,
                        &[facet],
                        protected_faces,
                        protected_edges,
                        insertion,
                        cancellation,
                        constraint_index,
                    )?,
                },
            }
        };
        let accepted = rejection.is_none();
        match rejection {
            Some(rejection) => work.reject(rejection),
            None => trial.commit(),
        }
        Ok::<_, DelaunayFacetRecoveryError>((accepted, accepted))
    })?;
    Ok(accepted.then_some(retained))
}

/// The Steiner node [`insert_candidate_in_place`] inserts, the facet whose edges must survive
/// the insertion, and the node whose star seeds point location.
#[derive(Clone, Copy)]
pub(in crate::cdt::facet_recovery) struct InPlaceCandidateRequest {
    pub(in crate::cdt::facet_recovery) node: DelaunayVolumeNode,
    pub(in crate::cdt::facet_recovery) required_facet_edges: Option<[StableDigest; 3]>,
    pub(in crate::cdt::facet_recovery) near: StableDigest,
}

/// Inserts `node` with its constrained Bowyer-Watson cavity (protected faces and region
/// interfaces are barriers), then link flips keep every recovered segment edge. With
/// `required_facet_edges`, the insertion is kept only when that facet's three edges exist after
/// it. Point location starts from a cell of `near`'s star (a node close to the candidate).
pub(in crate::cdt::facet_recovery) fn insert_candidate_in_place(
    working: &mut FacetRecoveryWorking,
    candidate: InPlaceCandidateRequest,
    protected_faces: &[[StableDigest; 3]],
    protected_edges: &[[StableDigest; 2]],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<InPlaceCandidate, DelaunayFacetRecoveryError> {
    let InPlaceCandidateRequest {
        node,
        required_facet_edges,
        near,
    } = candidate;
    let insertion = work.options.segment_recovery.insertion;
    let cancellation = work.cancellation;
    let protected_set = protected_faces
        .iter()
        .copied()
        .map(canonical_face)
        .collect::<BTreeSet<_>>();
    let kept_edges =
        crate::cdt::segment_recovery::sorted_edge_keys(protected_edges.iter().copied());
    let hint = working
        .topology()
        .node_index(near)
        .and_then(|vertex| working.topology().vertex_star(vertex).first().copied());
    let map_hard = |failure: crate::cdt::DelaunayInsertionError| match failure.kind {
        DelaunayInsertionErrorKind::Cancelled => Some(resource_or_cancelled(
            DelaunayFacetRecoveryErrorKind::Cancelled,
            constraint_index,
            failure.to_string(),
        )),
        DelaunayInsertionErrorKind::ResourceLimit => Some(resource_or_cancelled(
            DelaunayFacetRecoveryErrorKind::ResourceLimit,
            constraint_index,
            failure.to_string(),
        )),
        _ => None,
    };
    working.edit(|topology| {
        let mut trial = topology.cavity_trial();
        let outcome = (|| -> Result<InPlaceCandidate, DelaunayFacetRecoveryError> {
            let mut observed = trial.observe_edge_changes();
            if let Err(failure) =
                crate::cdt::insertion::insert_delaunay_volume_node_into_mutation_with_prepared_barriers_from_hint(
                    &mut observed,
                    node,
                    &protected_set,
                    true,
                    hint,
                    insertion,
                    cancellation,
                )
            {
                if let Some(hard) = map_hard(failure.clone()) {
                    return Err(hard);
                }
                return Ok(InPlaceCandidate::Declined(format!(
                    "in-place Steiner insertion was rejected: {failure}"
                )));
            }
            let Some(inserted) = observed.node_index(node.identity) else {
                return Ok(InPlaceCandidate::Declined(
                    "in-place Steiner insertion lost its node".to_owned(),
                ));
            };
            if let Err(failure) = crate::cdt::insertion::legalize_inserted_nodes(
                &mut observed,
                &[inserted],
                &protected_set,
                &kept_edges,
                insertion,
                cancellation,
            ) {
                if let Some(hard) = map_hard(failure.clone()) {
                    return Err(hard);
                }
                return Ok(InPlaceCandidate::Declined(format!(
                    "in-place Steiner link flips failed: {failure}"
                )));
            }
            let conflicts = removed_segment_edges(&observed, protected_edges)?;
            if !conflicts.is_empty() {
                return Ok(InPlaceCandidate::SegmentConflicts(conflicts));
            }
            if let Some(facet) = required_facet_edges {
                if !facet_edges_exist(&*observed, facet) {
                    return Ok(InPlaceCandidate::Declined(
                        "the in-place Steiner insertion did not recover the facet edges".to_owned(),
                    ));
                }
            }
            Ok(
                match admission(
                    &observed,
                    &[],
                    &protected_set,
                    protected_edges,
                    insertion,
                    cancellation,
                    constraint_index,
                )? {
                    None => InPlaceCandidate::Inserted,
                    Some(violation) => InPlaceCandidate::Declined(violation),
                },
            )
        })()?;
        let changed = matches!(outcome, InPlaceCandidate::Inserted);
        if changed {
            trial.commit();
        }
        Ok((outcome, changed))
    })
    .inspect(|outcome| {
        if let InPlaceCandidate::Declined(reason) = outcome {
            work.reject(reason.clone());
        }
    })
}

fn removed_segment_edges(
    observed: &EdgeChangeObserver<'_>,
    protected_edges: &[[StableDigest; 2]],
) -> Result<Vec<[StableDigest; 2]>, DelaunayFacetRecoveryError> {
    let mut checkpoint = || Ok::<(), DelaunayFacetRecoveryError>(());
    let removed = observed.removed_edge_candidates(&mut checkpoint)?;
    if removed.is_empty() {
        return Ok(Vec::new());
    }
    Ok(protected_edges
        .iter()
        .copied()
        .filter(|edge| {
            let [Some(first), Some(second)] = edge.map(|identity| observed.node_index(identity))
            else {
                return false;
            };
            let mut key = [first, second];
            key.sort_unstable();
            removed.contains(&key) && !observed.has_edge(key)
        })
        .collect())
}

/// The exact local admission of an observed edit: every cap triangle exists, no recovered segment
/// edge was removed, and every changed cell is constrained Delaunay with its protected faces kept.
fn admission(
    observed: &EdgeChangeObserver<'_>,
    cap: &[[StableDigest; 3]],
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protected_edges: &[[StableDigest; 2]],
    insertion: crate::cdt::DelaunayInsertionOptions,
    cancellation: &dyn runmat_meshing_core::MeshingCancellationSignal,
    constraint_index: u32,
) -> Result<Option<String>, DelaunayFacetRecoveryError> {
    if let Some(missing) = cap.iter().find(|triangle| {
        let [Some(a), Some(b), Some(c)] = triangle.map(|identity| observed.node_index(identity))
        else {
            return true;
        };
        observed.face_incidence([a, b, c]).is_empty()
    }) {
        return Ok(Some(format!("the cavity refill omitted facet {missing:?}")));
    }
    if segment_edge_removed(observed, protected_edges)? {
        return Ok(Some("the edit removed a recovered segment edge".to_owned()));
    }
    crate::cdt::insertion::observed_change_violation(
        observed,
        &observed.changed_cells(),
        protected_faces,
        insertion,
        cancellation,
    )
    .map(|violation| {
        violation.map(|violation| format!("the edit was not constrained-Delaunay: {violation}"))
    })
    .map_err(|failure| {
        resource_or_cancelled(
            if failure.kind == DelaunayInsertionErrorKind::Cancelled {
                DelaunayFacetRecoveryErrorKind::Cancelled
            } else {
                DelaunayFacetRecoveryErrorKind::ResourceLimit
            },
            constraint_index,
            failure.to_string(),
        )
    })
}
