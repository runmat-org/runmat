use std::collections::BTreeSet;

use runmat_meshing_core::{
    quality::predicate::{orient2d, Point3Construction, PredicateSign},
    StableDigest,
};

use super::conforming_split::{
    split_recovered_segment_edge_in_place, try_insert_on_protected_facets,
    try_split_recovered_segment_edge, ProtectedFacetSplit,
};
use super::{
    cavity::{
        candidate_node, insert_candidate_in_place, recovered_segments_exist, InPlaceCandidate, segment_hits_triangle_interior, steiner_lineage,
        try_insert_candidate_with_conflicts, CandidateInsertion, CandidateInsertionResult,
    },
    error, node_index, segment_error, support_edge_exists, DelaunayFacetRecoveryError,
    DelaunayFacetRecoveryErrorKind, DelaunayFacetSteinerInsertion, DelaunayRecoveredFacetTriangle,
    DelaunaySegmentRecovery, FacetRecoveryWork,
};
use crate::cdt::segment_recovery::{
    recover_delaunay_edges_with_local_flips_in_place, recover_temporary_delaunay_segment,
    recover_temporary_delaunay_segment_in_place, try_recover_delaunay_edges_with_local_flips,
    InPlaceTemporarySegment, TemporarySegmentRequest,
};
use crate::cdt::DelaunaySegmentRecoveryErrorKind;

mod encroachment;

#[derive(Clone)]
pub(super) struct PendingSupportTriangle {
    pub(super) triangle: DelaunayRecoveredFacetTriangle,
    pub(super) lineage_support: [StableDigest; 3],
}

pub(super) enum SupportTriangleEdgeState {
    Ready,
    Split {
        triangles: Vec<PendingSupportTriangle>,
        insertions: Vec<DelaunayFacetSteinerInsertion>,
        protected_splits: Vec<ProtectedFacetSplit>,
    },
    RefinedSegment {
        protected_splits: Vec<ProtectedFacetSplit>,
    },
    Unavailable,
}

enum SupportNodeInsertionState {
    Inserted {
        protected_splits: Vec<ProtectedFacetSplit>,
    },
    RefinedSegment {
        protected_splits: Vec<ProtectedFacetSplit>,
    },
    Unavailable,
}

pub(super) struct SupportRefinementContext<'a> {
    pub(super) protected_triangles: &'a [DelaunayRecoveredFacetTriangle],
    pub(super) protected_faces: &'a [[StableDigest; 3]],
    pub(super) protected_edges: &'a [[StableDigest; 2]],
    pub(super) constraint_index: u32,
    pub(super) refinement_rank: u64,
}

/// Recovers a support triangle's missing edges on the working topology. Local flips, the
/// temporary support segment and a conflicting segment edge's split each run in a cavity trial
/// with an exact admission of the changed cells. Returns `None`, with every declined edit rolled
/// back, when a step needs the complete repair; the caller then runs
/// [`ensure_support_triangle_edges`] on the canonical topology.
pub(super) fn ensure_support_triangle_edges_in_place(
    working: &mut super::working::FacetRecoveryWorking,
    pending: &PendingSupportTriangle,
    context: SupportRefinementContext<'_>,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<SupportTriangleEdgeState>, DelaunayFacetRecoveryError> {
    let SupportRefinementContext {
        protected_triangles,
        protected_faces,
        protected_edges,
        constraint_index,
        refinement_rank,
    } = context;
    let triangle = pending.triangle.node_identities;
    let missing_edges = triangle_edges(triangle)
        .into_iter()
        .filter(|endpoints| !support_edge_exists(working.topology(), *endpoints))
        .collect::<Vec<_>>();
    if missing_edges.is_empty() {
        return Ok(Some(SupportTriangleEdgeState::Ready));
    }
    let segment_options = work.options.segment_recovery;
    let cancellation = work.cancellation;
    working.edit(|topology| {
        recover_delaunay_edges_with_local_flips_in_place(
            topology,
            &missing_edges,
            protected_edges,
            protected_faces,
            constraint_index,
            segment_options,
            cancellation,
        )
        .map(|changed| ((), changed))
        .map_err(segment_error)
    })?;
    let Some(missing_edge_slot) = triangle_edges(triangle)
        .into_iter()
        .position(|edge| !support_edge_exists(working.topology(), edge))
    else {
        return Ok(Some(SupportTriangleEdgeState::Ready));
    };
    let edge = triangle_edges(triangle)[missing_edge_slot];
    let insertion_round = 0;
    let mut lineage = Vec::new();
    let mut segment_options = work.options.segment_recovery;
    segment_options.maximum_steiner_nodes = segment_options
        .maximum_steiner_nodes
        .min(work.options.maximum_cavity_steiner_nodes);
    segment_options.maximum_split_depth = segment_options.maximum_split_depth.min(16);
    let outcome = working.edit(|topology| {
        recover_temporary_delaunay_segment_in_place(
            topology,
            TemporarySegmentRequest {
                endpoints: edge,
                protected_edges,
                protected_faces,
                constraint_index,
                options: segment_options,
                cancellation,
            },
            |numerator, exponent, _coordinates_m| {
                let candidate_rank =
                    temporary_segment_candidate_rank(refinement_rank, numerator, exponent);
                let identity = super::facet_steiner_identity(
                    pending.lineage_support,
                    insertion_round,
                    candidate_rank,
                );
                lineage.push((identity, candidate_rank));
                identity
            },
        )
        .map(|outcome| {
            let changed = matches!(outcome, InPlaceTemporarySegment::Recovered { .. });
            (outcome, changed)
        })
        .map_err(segment_error)
    })?;
    match outcome {
        InPlaceTemporarySegment::Declined(reason) => {
            work.reject(format!("in-place temporary support segment declined: {reason}"));
            Ok(None)
        }
        InPlaceTemporarySegment::Conflicts(conflicts) => {
            let Some(splits) = split_recovered_segment_edge_in_place(
                working,
                conflicts[0],
                protected_triangles,
                protected_edges,
                constraint_index,
                work,
            )?
            else {
                let reason = work
                    .last_rejection()
                    .unwrap_or("no reason recorded")
                    .to_owned();
                work.reject(format!(
                    "in-place split of conflicting recovered segment edge declined: {reason}"
                ));
                return Ok(None);
            };
            work.cavity_steiner_node(constraint_index)?;
            Ok(Some(SupportTriangleEdgeState::RefinedSegment {
                protected_splits: splits,
            }))
        }
        InPlaceTemporarySegment::Recovered {
            chain,
            inserted_nodes,
        } => {
            for _ in &inserted_nodes {
                work.cavity_steiner_node(constraint_index)?;
            }
            let insertions = temporary_insertions(
                &inserted_nodes,
                &lineage,
                pending,
                constraint_index,
                insertion_round,
            )?;
            Ok(Some(SupportTriangleEdgeState::Split {
                triangles: fan_support_triangle(
                    working.topology(),
                    pending,
                    missing_edge_slot,
                    &chain,
                    constraint_index,
                )?,
                insertions,
                protected_splits: Vec::new(),
            }))
        }
    }
}

/// After both temporary support paths declined: when the pending triangle's missing edge enters
/// the diametral ball of a recovered segment edge, Steiner nodes on it encroach that edge, so the
/// edge is split on the working topology (with its encroachment cascade) and the triangle is
/// retried on the refined segments. Returns the protected-facet splits, or `None` with the
/// rejection kept when no edge is encroached or the split declines.
pub(super) fn split_segment_encroached_by_support_edge(
    working: &mut super::working::FacetRecoveryWorking,
    pending: &PendingSupportTriangle,
    protected_triangles: &[DelaunayRecoveredFacetTriangle],
    protected_edges: &[[StableDigest; 2]],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<Vec<ProtectedFacetSplit>>, DelaunayFacetRecoveryError> {
    let Some(edge) = triangle_edges(pending.triangle.node_identities)
        .into_iter()
        .find(|edge| !support_edge_exists(working.topology(), *edge))
    else {
        return Ok(None);
    };
    let Some(encroached) =
        encroachment::deepest_encroached_segment_edge(working.topology(), edge, protected_edges)
    else {
        return Ok(None);
    };
    let declined = work.last_rejection().unwrap_or("no reason recorded").to_owned();
    work.clear_rejection();
    let Some(splits) = split_recovered_segment_edge_in_place(
        working,
        encroached,
        protected_triangles,
        protected_edges,
        constraint_index,
        work,
    )?
    else {
        let detail = work.last_rejection().unwrap_or("no reason recorded").to_owned();
        work.reject(format!(
            "{declined}; the recovered segment edge that the support edge encroaches could not \
             be split: {detail}"
        ));
        return Ok(None);
    };
    work.cavity_steiner_node(constraint_index)?;
    tracing::info!(
        event_code = "meshing.tetrahedron.facet_recovery.encroached_segment_split",
        constraint_index,
        protected_facet_split_count = splits.len(),
        "a missing support edge encroached a recovered segment edge, which was split"
    );
    Ok(Some(splits))
}

fn temporary_insertions(
    inserted_nodes: &[crate::cdt::DelaunayVolumeNode],
    lineage: &[(StableDigest, u64)],
    pending: &PendingSupportTriangle,
    constraint_index: u32,
    insertion_round: u64,
) -> Result<Vec<DelaunayFacetSteinerInsertion>, DelaunayFacetRecoveryError> {
    inserted_nodes
        .iter()
        .map(|node| {
            let candidate_rank = lineage
                .iter()
                .find_map(|(identity, rank)| (*identity == node.identity).then_some(*rank))
                .ok_or_else(|| {
                    error(
                        DelaunayFacetRecoveryErrorKind::InvalidTopology,
                        Some(constraint_index),
                        "temporary support node is missing facet lineage",
                    )
                })?;
            Ok(steiner_lineage(
                pending.lineage_support,
                constraint_index,
                insertion_round,
                candidate_rank,
                node.identity,
            ))
        })
        .collect()
}

pub(super) fn ensure_support_triangle_edges(
    recovery: &mut DelaunaySegmentRecovery,
    pending: &PendingSupportTriangle,
    context: SupportRefinementContext<'_>,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<SupportTriangleEdgeState, DelaunayFacetRecoveryError> {
    let SupportRefinementContext {
        protected_triangles,
        protected_faces,
        protected_edges,
        constraint_index,
        refinement_rank,
    } = context;
    let triangle = pending.triangle.node_identities;
    let missing_edges = triangle_edges(triangle)
        .into_iter()
        .filter(|endpoints| !support_edge_exists(&recovery.topology, *endpoints))
        .collect::<Vec<_>>();
    if !missing_edges.is_empty() {
        if let Some(topology) = try_recover_delaunay_edges_with_local_flips(
            &recovery.topology,
            &missing_edges,
            protected_edges,
            protected_faces,
            constraint_index,
            work.options.segment_recovery,
            work.cancellation,
        )
        .map_err(segment_error)?
        {
            let mut candidate = recovery.clone();
            candidate.topology = topology;
            if recovered_segments_exist(&candidate) {
                recovery.topology = candidate.topology;
            }
        }
    }
    if triangle_edges(triangle)
        .into_iter()
        .all(|edge| support_edge_exists(&recovery.topology, edge))
    {
        return Ok(SupportTriangleEdgeState::Ready);
    }

    let Some(missing_edge_slot) = triangle_edges(triangle)
        .into_iter()
        .position(|edge| !support_edge_exists(&recovery.topology, edge))
    else {
        return Ok(SupportTriangleEdgeState::Ready);
    };
    let edge = triangle_edges(triangle)[missing_edge_slot];
    let insertion_round = 0;
    let mut lineage = Vec::new();
    let mut segment_options = work.options.segment_recovery;
    segment_options.maximum_steiner_nodes = segment_options
        .maximum_steiner_nodes
        .min(work.options.maximum_cavity_steiner_nodes);
    segment_options.maximum_split_depth = segment_options.maximum_split_depth.min(16);
    let temporary = recover_temporary_delaunay_segment(
        &recovery.topology,
        TemporarySegmentRequest {
            endpoints: edge,
            protected_edges,
            protected_faces,
            constraint_index,
            options: segment_options,
            cancellation: work.cancellation,
        },
        |numerator, exponent, _coordinates_m| {
            let candidate_rank =
                temporary_segment_candidate_rank(refinement_rank, numerator, exponent);
            let identity = super::facet_steiner_identity(
                pending.lineage_support,
                insertion_round,
                candidate_rank,
            );
            lineage.push((identity, candidate_rank));
            identity
        },
    );
    // The temporary chain is a trial on a copy of the topology. When its points admit no
    // constrained Delaunay tetrahedralization with the current barriers (validation rejects
    // the legalized copy) or no split clears a witness, decline it, so the caller tries the
    // facet cavity and interior refinement. Budget, cancellation and option errors still end
    // recovery.
    let temporary = match temporary {
        Ok(temporary) => temporary,
        Err(failure)
            if matches!(
                failure.kind,
                DelaunaySegmentRecoveryErrorKind::InvalidTopology
                    | DelaunaySegmentRecoveryErrorKind::UnsatisfiableConstraint
            ) =>
        {
            work.reject(format!("temporary support segment declined: {failure}"));
            return Ok(SupportTriangleEdgeState::Unavailable);
        }
        Err(failure) => return Err(segment_error(failure)),
    };
    if let Some(conflict) = temporary.conflicting_protected_edges.first().copied() {
        work.clear_rejection();
        let Some(refined) = try_split_recovered_segment_edge(
            recovery,
            conflict,
            protected_triangles,
            constraint_index,
            work,
        )?
        else {
            let coordinates = conflict.map(|identity| {
                node_index(&recovery.topology, identity)
                    .map(|index| recovery.topology.nodes[index].coordinates_m.canonical_value())
            });
            let detail = work
                .last_rejection()
                .unwrap_or("the edge is not an interval of a recovered segment")
                .to_owned();
            work.reject(format!(
                "temporary support segment for edge {:?} conflicts with recovered segment edge \
                 {coordinates:?}, which could not be split: {detail}",
                edge.map(|identity| node_index(&recovery.topology, identity)
                    .map(|index| recovery.topology.nodes[index].coordinates_m.canonical_value())),
            ));
            return Ok(SupportTriangleEdgeState::Unavailable);
        };
        *recovery = refined.recovery;
        work.cavity_steiner_node(constraint_index)?;
        return Ok(SupportTriangleEdgeState::RefinedSegment {
            protected_splits: refined.splits,
        });
    }
    for _ in &temporary.inserted_nodes {
        work.cavity_steiner_node(constraint_index)?;
    }
    recovery.topology = temporary.topology;
    let insertions = temporary_insertions(
        &temporary.inserted_nodes,
        &lineage,
        pending,
        constraint_index,
        insertion_round,
    )?;
    Ok(SupportTriangleEdgeState::Split {
        triangles: fan_support_triangle(
            &recovery.topology,
            pending,
            missing_edge_slot,
            &temporary.chain,
            constraint_index,
        )?,
        insertions,
        protected_splits: Vec::new(),
    })
}

/// Splits a topologically bounded support triangle when its three edges exist but the face cannot
/// be recovered by constrained flips or cavity replacement. The dyadic 2:1:1 point is strictly
/// inside every nondegenerate triangle and remains exactly coplanar in constructed predicates.
/// The facet-interior split node of `pending`, with its insertion round and candidate rank, or
/// `None` (with a rejection) when the triangle cannot take one.
pub(super) fn interior_support_node(
    topology: &impl crate::cdt::insertion::topology_access::TetrahedralTopology,
    pending: &PendingSupportTriangle,
    protected_edges: &[[StableDigest; 2]],
    constraint_index: u32,
    refinement_rank: u64,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<(crate::cdt::DelaunayVolumeNode, u64, u64)>, DelaunayFacetRecoveryError> {
    let triangle = pending.triangle.node_identities;
    if !triangle_edges(triangle)
        .into_iter()
        .all(|edge| support_edge_exists(topology, edge))
    {
        work.reject("facet-interior refinement requires all three support edges");
        return Ok(None);
    }
    let triangle_indices = triangle.map(|identity| {
        node_index(topology, identity)
            .map(|index| index as u32)
            .ok_or_else(|| {
                error(
                    DelaunayFacetRecoveryErrorKind::InvalidTopology,
                    Some(constraint_index),
                    "facet-interior refinement node is missing from the topology",
                )
            })
    });
    let [first, second, third] = triangle_indices;
    let triangle_indices = [first?, second?, third?];
    for edge in protected_edges.iter().copied() {
        let [Some(first), Some(second)] =
            edge.map(|identity| node_index(topology, identity))
        else {
            return Err(error(
                DelaunayFacetRecoveryErrorKind::InvalidTopology,
                Some(constraint_index),
                "recovered segment node is missing during facet-intersection validation",
            ));
        };
        if segment_hits_triangle_interior(
            topology,
            [first as u32, second as u32],
            triangle_indices,
            constraint_index,
            work,
        )? {
            work.reject(format!(
                "recovered segment edge {edge:?} crosses the facet interior"
            ));
            return Ok(None);
        }
    }
    let mut vertices = triangle
        .map(|identity| {
            node_index(topology, identity)
                .map(|index| {
                    (
                        identity,
                        topology.nodes()[index]
                            .coordinates_m
                            .canonical_value(),
                    )
                })
                .ok_or_else(|| {
                    error(
                        DelaunayFacetRecoveryErrorKind::InvalidTopology,
                        Some(constraint_index),
                        "facet-interior refinement node is missing from the topology",
                    )
                })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    vertices.sort_by_key(|(identity, _)| *identity);
    let construction_vertices = std::array::from_fn(|index| vertices[index].1);
    let coordinates_m = std::array::from_fn(|axis| {
        (2.0 * construction_vertices[0][axis]
            + construction_vertices[1][axis]
            + construction_vertices[2][axis])
            * 0.25
    });
    let insertion_round = 1;
    let candidate_rank = u64::MAX - refinement_rank;
    let Some(node) = candidate_node(
        topology,
        CandidateInsertion {
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(coordinates_m),
            construction: Point3Construction::TriangleDyadic {
                vertices: construction_vertices,
                weights_numerator: [2, 1, 1],
                weight_exponent: 2,
            },
            facet: pending.lineage_support,
            insertion_round,
            candidate_rank,
        },
        constraint_index,
        work,
    )?
    else {
        return Ok(None);
    };
    Ok(Some((node, insertion_round, candidate_rank)))
}

/// [`refine_support_triangle_interior`] on the working topology: the split node is inserted with
/// its constrained cavity and kept under the exact local admission; a recovered segment edge it
/// would remove is split in place instead. Returns `None`, with every declined edit rolled back,
/// when the canonical provider must decide.
pub(super) fn refine_support_triangle_interior_in_place(
    working: &mut super::working::FacetRecoveryWorking,
    pending: &PendingSupportTriangle,
    context: SupportRefinementContext<'_>,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<SupportTriangleEdgeState>, DelaunayFacetRecoveryError> {
    let SupportRefinementContext {
        protected_triangles,
        protected_faces,
        protected_edges,
        constraint_index,
        refinement_rank,
    } = context;
    let triangle = pending.triangle.node_identities;
    let Some((node, insertion_round, candidate_rank)) = interior_support_node(
        working.topology(),
        pending,
        protected_edges,
        constraint_index,
        refinement_rank,
        work,
    )?
    else {
        return Ok(Some(SupportTriangleEdgeState::Unavailable));
    };
    match insert_candidate_in_place(
        working,
        node,
        protected_faces,
        protected_edges,
        None,
        triangle[0],
        constraint_index,
        work,
    )? {
        InPlaceCandidate::Inserted => {
            work.cavity_steiner_node(constraint_index)?;
            Ok(Some(interior_split(
                pending,
                node,
                insertion_round,
                candidate_rank,
                constraint_index,
                Vec::new(),
            )))
        }
        InPlaceCandidate::SegmentConflicts(conflicts) => {
            let Some(splits) = split_recovered_segment_edge_in_place(
                working,
                conflicts[0],
                protected_triangles,
                protected_edges,
                constraint_index,
                work,
            )?
            else {
                let reason = work.last_rejection().unwrap_or("no reason recorded").to_owned();
                work.reject(format!(
                    "in-place interior node conflicts with recovered segment edge {:?}; its split \
                     declined: {reason}",
                    conflicts[0]
                ));
                return Ok(None);
            };
            work.cavity_steiner_node(constraint_index)?;
            Ok(Some(SupportTriangleEdgeState::RefinedSegment {
                protected_splits: splits,
            }))
        }
        InPlaceCandidate::Declined(_) => Ok(None),
    }
}

fn interior_split(
    pending: &PendingSupportTriangle,
    node: crate::cdt::DelaunayVolumeNode,
    insertion_round: u64,
    candidate_rank: u64,
    constraint_index: u32,
    protected_splits: Vec<ProtectedFacetSplit>,
) -> SupportTriangleEdgeState {
    let triangle = pending.triangle.node_identities;
    let triangles = triangle
        .into_iter()
        .zip(triangle.into_iter().cycle().skip(1))
        .take(3)
        .map(|(first, second)| PendingSupportTriangle {
            triangle: DelaunayRecoveredFacetTriangle {
                node_identities: [first, second, node.identity],
            },
            lineage_support: pending.lineage_support,
        })
        .collect();
    SupportTriangleEdgeState::Split {
        triangles,
        insertions: vec![steiner_lineage(
            pending.lineage_support,
            constraint_index,
            insertion_round,
            candidate_rank,
            node.identity,
        )],
        protected_splits,
    }
}

pub(super) fn refine_support_triangle_interior(
    recovery: &mut DelaunaySegmentRecovery,
    pending: &PendingSupportTriangle,
    context: SupportRefinementContext<'_>,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<SupportTriangleEdgeState, DelaunayFacetRecoveryError> {
    let SupportRefinementContext {
        protected_triangles,
        protected_faces,
        protected_edges,
        constraint_index,
        refinement_rank,
    } = context;
    let Some((node, insertion_round, candidate_rank)) = interior_support_node(
        &recovery.topology,
        pending,
        protected_edges,
        constraint_index,
        refinement_rank,
        work,
    )?
    else {
        return Ok(SupportTriangleEdgeState::Unavailable);
    };
    let protected_splits = match try_insert_support_node(
        recovery,
        node,
        protected_triangles,
        protected_faces,
        constraint_index,
        work,
    )? {
        SupportNodeInsertionState::Inserted { protected_splits } => protected_splits,
        SupportNodeInsertionState::RefinedSegment { protected_splits } => {
            return Ok(SupportTriangleEdgeState::RefinedSegment { protected_splits });
        }
        SupportNodeInsertionState::Unavailable => {
            return Ok(SupportTriangleEdgeState::Unavailable);
        }
    };
    Ok(interior_split(
        pending,
        node,
        insertion_round,
        candidate_rank,
        constraint_index,
        protected_splits,
    ))
}

fn temporary_segment_candidate_rank(refinement_rank: u64, numerator: u64, exponent: u8) -> u64 {
    let dyadic = (u64::from(exponent) << 48) | (numerator & ((1_u64 << 48) - 1));
    refinement_rank.rotate_left(17) ^ dyadic
}

fn try_insert_support_node(
    recovery: &mut DelaunaySegmentRecovery,
    node: crate::cdt::DelaunayVolumeNode,
    protected_triangles: &[DelaunayRecoveredFacetTriangle],
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<SupportNodeInsertionState, DelaunayFacetRecoveryError> {
    let conforming = try_insert_on_protected_facets(
        recovery,
        node,
        protected_triangles,
        constraint_index,
        work,
    )?;
    let conforming_rejection = work.last_rejection().map(str::to_owned);
    if let Some(conforming) = conforming {
        *recovery = conforming.recovery;
        work.cavity_steiner_node(constraint_index)?;
        return Ok(SupportNodeInsertionState::Inserted {
            protected_splits: conforming.splits,
        });
    }
    match try_insert_candidate_with_conflicts(
        recovery,
        node,
        protected_faces,
        constraint_index,
        work,
    )? {
        CandidateInsertionResult::Inserted(updated) => {
            *recovery = updated;
            work.cavity_steiner_node(constraint_index)?;
            Ok(SupportNodeInsertionState::Inserted {
                protected_splits: Vec::new(),
            })
        }
        CandidateInsertionResult::SegmentConflicts(conflicts) => {
            let before_split = work.last_rejection().map(str::to_owned);
            let Some(refined) = try_split_recovered_segment_edge(
                recovery,
                conflicts[0],
                protected_triangles,
                constraint_index,
                work,
            )?
            else {
                let split_rejection = work
                    .last_rejection()
                    .filter(|reason| Some(*reason) != before_split.as_deref())
                    .unwrap_or("the conflicting edge is not a recovered segment interval")
                    .to_owned();
                let source_state =
                    match crate::cdt::insertion::validate_constrained_delaunay_volume_topology(
                        &recovery.topology,
                        protected_faces,
                        work.options.segment_recovery.insertion,
                        work.cancellation,
                    ) {
                        Ok(()) => "constrained-Delaunay".to_owned(),
                        Err(failure) => format!("already invalid: {failure}"),
                    };
                work.reject(format!(
                    "support node {:?} conflicts with recovered segment edge {:?} at {:?} ({} conflicts); \
                     splitting it was rejected: {split_rejection}; on-facet insertion was \
                     rejected: {}; the topology before insertion was {source_state}",
                    node.coordinates_m.canonical_value(),
                    conflicts[0],
                    conflicts[0].map(|identity| node_index(&recovery.topology, identity)
                        .map(|index| recovery.topology.nodes[index].coordinates_m.canonical_value())),
                    conflicts.len(),
                    conforming_rejection.as_deref().unwrap_or("no reason recorded")
                ));
                return Ok(SupportNodeInsertionState::Unavailable);
            };
            *recovery = refined.recovery;
            work.cavity_steiner_node(constraint_index)?;
            Ok(SupportNodeInsertionState::RefinedSegment {
                protected_splits: refined.splits,
            })
        }
        CandidateInsertionResult::Unavailable => {
            if let Some(conforming_rejection) = conforming_rejection {
                let fallback = work
                    .last_rejection()
                    .unwrap_or("generic facet insertion was unavailable");
                work.reject(format!(
                    "{conforming_rejection}; fallback rejected: {fallback}"
                ));
            }
            Ok(SupportNodeInsertionState::Unavailable)
        }
    }
}

fn fan_support_triangle(
    topology: &impl crate::cdt::insertion::topology_access::TetrahedralTopology,
    pending: &PendingSupportTriangle,
    edge_slot: usize,
    chain: &[crate::cdt::DelaunayRecoveredSegmentNode],
    constraint_index: u32,
) -> Result<Vec<PendingSupportTriangle>, DelaunayFacetRecoveryError> {
    let triangle = pending.triangle.node_identities;
    let opposite = triangle[(edge_slot + 2) % 3];
    chain
        .windows(2)
        .map(|pair| [pair[0].identity, pair[1].identity, opposite])
        .filter_map(|node_identities| {
            match triangle_has_positive_area_on(topology, node_identities, constraint_index) {
                Ok(true) => Some(Ok(PendingSupportTriangle {
                    triangle: DelaunayRecoveredFacetTriangle { node_identities },
                    lineage_support: pending.lineage_support,
                })),
                Ok(false) => None,
                Err(failure) => Some(Err(failure)),
            }
        })
        .collect()
}

pub(super) fn triangle_has_positive_area_on(
    topology: &impl crate::cdt::insertion::topology_access::TetrahedralTopology,
    triangle: [StableDigest; 3],
    constraint_index: u32,
) -> Result<bool, DelaunayFacetRecoveryError> {
    let points = triangle.map(|identity| {
        node_index(topology, identity)
            .map(|index| topology.nodes()[index].coordinates_m.canonical_value())
            .ok_or_else(|| {
                error(
                    DelaunayFacetRecoveryErrorKind::InvalidTopology,
                    Some(constraint_index),
                    "split facet support node is missing from the tetrahedral topology",
                )
            })
    });
    let [first, second, third] = points;
    let points = [first?, second?, third?];
    for axes in [[0, 1], [0, 2], [1, 2]] {
        let sign =
            orient2d(points.map(|point| [point[axes[0]], point[axes[1]]])).map_err(|failure| {
                error(
                    DelaunayFacetRecoveryErrorKind::InvalidTopology,
                    Some(constraint_index),
                    format!("split facet support predicate failed: {failure:?}"),
                )
            })?;
        if sign != PredicateSign::Zero {
            return Ok(true);
        }
    }
    Ok(false)
}

fn triangle_edges(triangle: [StableDigest; 3]) -> [[StableDigest; 2]; 3] {
    [
        [triangle[0], triangle[1]],
        [triangle[1], triangle[2]],
        [triangle[2], triangle[0]],
    ]
}

pub(super) fn recovered_segment_edges(
    recovery: &DelaunaySegmentRecovery,
) -> Vec<[StableDigest; 2]> {
    let mut edges = BTreeSet::new();
    for segment in &recovery.segments {
        for pair in segment.nodes.windows(2) {
            let mut edge = [pair[0].identity, pair[1].identity];
            edge.sort_unstable();
            edges.insert(edge);
        }
    }
    edges.into_iter().collect()
}
