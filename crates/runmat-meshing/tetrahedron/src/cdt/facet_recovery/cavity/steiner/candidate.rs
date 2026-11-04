use std::collections::BTreeSet;

use runmat_meshing_core::{quality::predicate::Point3Construction, StableDigest};

use super::super::super::FacetRecoveryWork;
use super::super::{
    facet_recovery_refill_options, invalid_topology, progressive_facet_recovery_refill_options,
    recovered_segments_exist, resource_or_cancelled,
};
use crate::cdt::segment_recovery::coordinates_share_representable_location;
use crate::cdt::segment_recovery::try_restore_delaunay_edges_with_local_flips;
use crate::{
    cavity::constrained::{
        generate_constrained_cavity_interior_steiner_candidates_with_diagnostics,
        retriangulate_constrained_cavity_from_nodes, ConstrainedCavity, ConstrainedCavityNode,
        ConstrainedCavityRefillBudget, ConstrainedCavityRefillError,
        ConstrainedCavitySteinerCandidateBudget, ConstrainedCavityValidationError,
    },
    cdt::{
        insertion::{
            insert_delaunay_volume_node_locally_with_barriers,
            legalize_delaunay_volume_topology_with_barriers,
            validate_constrained_delaunay_volume_topology,
        },
        DelaunayFacetRecoveryError, DelaunayFacetRecoveryErrorKind, DelaunaySegmentRecovery,
        DelaunayVolumeNode,
    },
};

pub(in crate::cdt::facet_recovery) struct CandidateInsertion {
    pub(in crate::cdt::facet_recovery) coordinates_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
    pub(in crate::cdt::facet_recovery) construction: Point3Construction,
    pub(in crate::cdt::facet_recovery) facet: [StableDigest; 3],
    pub(in crate::cdt::facet_recovery) insertion_round: u64,
    pub(in crate::cdt::facet_recovery) candidate_rank: u64,
}

pub(in crate::cdt::facet_recovery) enum CandidateInsertionResult {
    Inserted(DelaunaySegmentRecovery),
    SegmentConflicts(Vec<[StableDigest; 2]>),
    Unavailable,
}

#[derive(Default)]
pub(super) struct SteinerCandidateGroup {
    pub(super) progressive: Vec<[f64; 3]>,
    pub(super) exact: Vec<[f64; 3]>,
}

impl SteinerCandidateGroup {
    pub(super) fn ordered(&self) -> Vec<[f64; 3]> {
        self.progressive
            .iter()
            .chain(&self.exact)
            .copied()
            .collect()
    }
}

pub(super) fn facet_edges_exist(
    topology: &crate::cdt::DelaunayVolumeTopology,
    facet: [StableDigest; 3],
) -> bool {
    [
        [facet[0], facet[1]],
        [facet[1], facet[2]],
        [facet[2], facet[0]],
    ]
    .into_iter()
    .all(|edge| {
        let Ok(first) = topology
            .nodes
            .binary_search_by_key(&edge[0], |node| node.identity)
        else {
            return false;
        };
        let Ok(second) = topology
            .nodes
            .binary_search_by_key(&edge[1], |node| node.identity)
        else {
            return false;
        };
        topology.incidence.vertex_stars[first]
            .iter()
            .any(|tetrahedron| {
                topology.tetrahedra[*tetrahedron as usize]
                    .vertex_indices
                    .contains(&(second as u32))
            })
    })
}

pub(in crate::cdt::facet_recovery) fn candidate_node(
    current: &DelaunaySegmentRecovery,
    insertion: CandidateInsertion,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<DelaunayVolumeNode>, DelaunayFacetRecoveryError> {
    work.cavity_steiner_insertion_attempt(constraint_index)?;
    let node = DelaunayVolumeNode {
        identity: super::steiner_identity(
            insertion.facet,
            insertion.insertion_round,
            insertion.candidate_rank,
        ),
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                insertion.coordinates_m.canonical_value(),
            ),
        construction: insertion.construction,
    };
    if let Some(existing) = current
        .topology
        .nodes
        .iter()
        .find(|existing| existing.identity == node.identity)
    {
        if existing.coordinates_m.canonical_value() != node.coordinates_m.canonical_value() {
            return Err(invalid_topology(
                constraint_index,
                "facet cavity Steiner identity collision",
            ));
        }
        work.reject("facet cavity Steiner candidate identity already exists");
        return Ok(None);
    }
    if current.topology.nodes.iter().any(|existing| {
        coordinates_share_representable_location(
            existing.coordinates_m.canonical_value(),
            node.coordinates_m.canonical_value(),
        )
    }) {
        work.reject("facet cavity Steiner candidate duplicates an existing node position");
        return Ok(None);
    }
    Ok(Some(node))
}

pub(in crate::cdt::facet_recovery) fn try_insert_candidate(
    current: &DelaunaySegmentRecovery,
    node: DelaunayVolumeNode,
    protected: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<DelaunaySegmentRecovery>, DelaunayFacetRecoveryError> {
    Ok(
        match try_insert_candidate_with_conflicts(current, node, protected, constraint_index, work)?
        {
            CandidateInsertionResult::Inserted(recovery) => Some(recovery),
            CandidateInsertionResult::SegmentConflicts(_)
            | CandidateInsertionResult::Unavailable => None,
        },
    )
}

pub(in crate::cdt::facet_recovery) fn try_insert_candidate_with_conflicts(
    current: &DelaunaySegmentRecovery,
    node: DelaunayVolumeNode,
    protected: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<CandidateInsertionResult, DelaunayFacetRecoveryError> {
    let mut candidate = match insert_delaunay_volume_node_locally_with_barriers(
        current.topology.clone(),
        node,
        protected,
        work.options.segment_recovery.insertion,
        work.cancellation,
    ) {
        Ok(candidate) => candidate,
        Err(failure) => match failure.kind {
            crate::cdt::DelaunayInsertionErrorKind::ResourceLimit => {
                return Err(resource_or_cancelled(
                    DelaunayFacetRecoveryErrorKind::ResourceLimit,
                    constraint_index,
                    failure.to_string(),
                ));
            }
            crate::cdt::DelaunayInsertionErrorKind::Cancelled => {
                return Err(resource_or_cancelled(
                    DelaunayFacetRecoveryErrorKind::Cancelled,
                    constraint_index,
                    failure.to_string(),
                ));
            }
            crate::cdt::DelaunayInsertionErrorKind::InvalidOptions => {
                return Err(invalid_topology(constraint_index, failure.to_string()));
            }
            crate::cdt::DelaunayInsertionErrorKind::InvalidNode => {
                work.reject(format!(
                    "facet cavity Steiner candidate is not admissible: {failure}"
                ));
                return Ok(CandidateInsertionResult::Unavailable);
            }
            crate::cdt::DelaunayInsertionErrorKind::InvalidTopology
            | crate::cdt::DelaunayInsertionErrorKind::PointOutsideTopology => {
                work.reject(format!(
                    "facet cavity Steiner insertion was rejected: {failure}"
                ));
                return Ok(CandidateInsertionResult::Unavailable);
            }
        },
    };
    candidate = match legalize_delaunay_volume_topology_with_barriers(
        candidate,
        protected,
        work.options.segment_recovery.insertion,
        work.cancellation,
    ) {
        Ok(candidate) => candidate,
        Err(failure) => match failure.kind {
            crate::cdt::DelaunayInsertionErrorKind::ResourceLimit => {
                return Err(resource_or_cancelled(
                    DelaunayFacetRecoveryErrorKind::ResourceLimit,
                    constraint_index,
                    failure.to_string(),
                ));
            }
            crate::cdt::DelaunayInsertionErrorKind::Cancelled => {
                return Err(resource_or_cancelled(
                    DelaunayFacetRecoveryErrorKind::Cancelled,
                    constraint_index,
                    failure.to_string(),
                ));
            }
            crate::cdt::DelaunayInsertionErrorKind::InvalidOptions => {
                return Err(invalid_topology(constraint_index, failure.to_string()));
            }
            crate::cdt::DelaunayInsertionErrorKind::InvalidTopology
            | crate::cdt::DelaunayInsertionErrorKind::InvalidNode
            | crate::cdt::DelaunayInsertionErrorKind::PointOutsideTopology => {
                work.reject(format!(
                    "facet cavity Steiner insertion could not legalize its unconstrained faces: {failure}"
                ));
                return Ok(CandidateInsertionResult::Unavailable);
            }
        },
    };
    let required_edges = recovered_segment_edges(current);
    let missing_edges = missing_recovered_segment_edges(current, &candidate);
    if !missing_edges.is_empty() {
        candidate = match try_restore_delaunay_edges_with_local_flips(
            &candidate,
            &required_edges,
            protected,
            constraint_index,
            work.options.segment_recovery,
            work.cancellation,
        )
        .map_err(super::super::super::segment_error)?
        {
            Some(candidate) => candidate,
            None => {
                work.reject(
                    "facet cavity Steiner insertion could not restore every recovered segment",
                );
                return Ok(CandidateInsertionResult::SegmentConflicts(missing_edges));
            }
        };
    }
    if let Err(failure) = validate_constrained_delaunay_volume_topology(
        &candidate,
        protected,
        work.options.segment_recovery.insertion,
        work.cancellation,
    ) {
        return match failure.kind {
            crate::cdt::DelaunayInsertionErrorKind::ResourceLimit => Err(resource_or_cancelled(
                DelaunayFacetRecoveryErrorKind::ResourceLimit,
                constraint_index,
                failure.to_string(),
            )),
            crate::cdt::DelaunayInsertionErrorKind::Cancelled => Err(resource_or_cancelled(
                DelaunayFacetRecoveryErrorKind::Cancelled,
                constraint_index,
                failure.to_string(),
            )),
            crate::cdt::DelaunayInsertionErrorKind::InvalidOptions => {
                Err(invalid_topology(constraint_index, failure.to_string()))
            }
            crate::cdt::DelaunayInsertionErrorKind::InvalidNode => {
                work.reject(format!(
                    "facet cavity Steiner candidate is not admissible after insertion: {failure}"
                ));
                Ok(CandidateInsertionResult::Unavailable)
            }
            crate::cdt::DelaunayInsertionErrorKind::InvalidTopology
            | crate::cdt::DelaunayInsertionErrorKind::PointOutsideTopology => {
                work.reject(format!(
                    "facet cavity Steiner insertion did not preserve constrained Delaunay topology: {failure}"
                ));
                Ok(CandidateInsertionResult::Unavailable)
            }
        };
    }
    let mut candidate_recovery = current.clone();
    candidate_recovery.topology = candidate;
    if recovered_segments_exist(&candidate_recovery) {
        Ok(CandidateInsertionResult::Inserted(candidate_recovery))
    } else {
        work.reject("facet cavity Steiner insertion removed a recovered segment");
        Ok(CandidateInsertionResult::SegmentConflicts(
            missing_recovered_segment_edges(current, &candidate_recovery.topology),
        ))
    }
}

fn recovered_segment_edges(current: &DelaunaySegmentRecovery) -> Vec<[StableDigest; 2]> {
    let mut edges = BTreeSet::new();
    for segment in &current.segments {
        for pair in segment.nodes.windows(2) {
            let mut edge = [pair[0].identity, pair[1].identity];
            edge.sort_unstable();
            edges.insert(edge);
        }
    }
    edges.into_iter().collect()
}

fn missing_recovered_segment_edges(
    current: &DelaunaySegmentRecovery,
    topology: &crate::cdt::DelaunayVolumeTopology,
) -> Vec<[StableDigest; 2]> {
    recovered_segment_edges(current)
        .into_iter()
        .filter(|edge| {
            let [Ok(first), Ok(second)] = edge.map(|identity| {
                topology
                    .nodes
                    .binary_search_by_key(&identity, |node| node.identity)
            }) else {
                return true;
            };
            !topology.incidence.vertex_stars[first]
                .iter()
                .any(|tetrahedron| {
                    topology.tetrahedra[*tetrahedron as usize]
                        .vertex_indices
                        .contains(&(second as u32))
                })
        })
        .collect()
}

pub(super) fn steiner_candidate_groups(
    recovery: &DelaunaySegmentRecovery,
    cavities: &[ConstrainedCavity],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Vec<SteinerCandidateGroup>, DelaunayFacetRecoveryError> {
    let nodes = recovery
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(node_id, node)| ConstrainedCavityNode {
            node_id: node_id as u32,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(node.coordinates_m.canonical_value()),
        })
        .collect::<Vec<_>>();
    let budget = ConstrainedCavitySteinerCandidateBudget {
        maximum_candidates: work.options.maximum_cavity_steiner_candidates,
        maximum_evaluations: work
            .options
            .maximum_cavity_steiner_candidate_evaluations_per_round,
        cancellation_check_interval: cancellation_interval(work),
    };
    let mut groups = Vec::with_capacity(cavities.len());
    for cavity in cavities {
        let topology_slots = budget.maximum_candidates / 2;
        let progressive_budget = ConstrainedCavitySteinerCandidateBudget {
            maximum_candidates: budget.maximum_candidates - topology_slots,
            ..budget
        };
        let (mut generated, progressive_diagnostics) =
            generate_constrained_cavity_interior_steiner_candidates_with_diagnostics(
                cavity,
                &nodes,
                progressive_facet_recovery_refill_options(),
                progressive_budget,
                work.cancellation,
            )
            .map_err(|failure| map_candidate_error(failure, constraint_index))?;
        let topology = if topology_slots == 0 {
            None
        } else {
            Some(
                generate_constrained_cavity_interior_steiner_candidates_with_diagnostics(
                    cavity,
                    &nodes,
                    facet_recovery_refill_options(),
                    ConstrainedCavitySteinerCandidateBudget {
                        maximum_candidates: topology_slots,
                        ..budget
                    },
                    work.cancellation,
                )
                .map_err(|failure| map_candidate_error(failure, constraint_index))?,
            )
        };
        let mut exact = topology
            .as_ref()
            .map(|(candidates, _)| candidates.clone())
            .unwrap_or_default();
        if generated.is_empty() && exact.is_empty() {
            work.reject(format!(
                "facet cavity generated no interior Steiner candidates: progressive={progressive_diagnostics:?}, topology={:?}",
                topology.as_ref().map(|(_, diagnostics)| diagnostics)
            ));
        }
        let mut seen = BTreeSet::new();
        generated.retain(|point| seen.insert(point.map(f64::to_bits)));
        exact.retain(|point| seen.insert(point.map(f64::to_bits)));
        groups.push(SteinerCandidateGroup {
            progressive: generated,
            exact,
        });
    }
    Ok(groups)
}

pub(super) fn candidate_refills_cavity(
    cavity: &ConstrainedCavity,
    recovery: &DelaunaySegmentRecovery,
    coordinates_m: [f64; 3],
    constraint_index: u32,
    work: &FacetRecoveryWork<'_>,
) -> Result<bool, DelaunayFacetRecoveryError> {
    candidate_refills_cavity_with_options(
        cavity,
        recovery,
        coordinates_m,
        facet_recovery_refill_options(),
        constraint_index,
        work,
    )
}

fn candidate_refills_cavity_with_options(
    cavity: &ConstrainedCavity,
    recovery: &DelaunaySegmentRecovery,
    coordinates_m: [f64; 3],
    options: crate::cavity::constrained::ConstrainedCavityRefillOptions,
    constraint_index: u32,
    work: &FacetRecoveryWork<'_>,
) -> Result<bool, DelaunayFacetRecoveryError> {
    let mut nodes = recovery
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(node_id, node)| ConstrainedCavityNode {
            node_id: node_id as u32,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(node.coordinates_m.canonical_value()),
        })
        .collect::<Vec<_>>();
    let node_id = u32::try_from(nodes.len()).map_err(|_| {
        invalid_topology(
            constraint_index,
            "facet cavity Steiner node identity space is exhausted",
        )
    })?;
    nodes.push(ConstrainedCavityNode {
        node_id,
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                coordinates_m,
            ),
    });
    let budget = ConstrainedCavityRefillBudget {
        maximum_nodes: work.options.maximum_cavity_nodes,
        maximum_boundary_faces: work.options.maximum_cavity_boundary_faces,
        maximum_candidate_tetrahedra: work.options.maximum_cavity_candidate_tetrahedra,
        maximum_candidate_evaluations: work.options.maximum_cavity_candidate_evaluations,
        maximum_search_attempts: work.options.maximum_cavity_exact_cover_attempts,
        maximum_expansion_rounds: work.options.maximum_cavity_expansion_rounds,
        cancellation_check_interval: cancellation_interval(work),
    };
    retriangulate_constrained_cavity_from_nodes(cavity, &nodes, options, budget, work.cancellation)
        .map(|refill| refill.is_some())
        .map_err(|failure| map_candidate_error(failure, constraint_index))
}

fn cancellation_interval(work: &FacetRecoveryWork<'_>) -> u64 {
    work.options
        .segment_recovery
        .constraints
        .cancellation_check_interval
}

fn map_candidate_error(
    failure: ConstrainedCavityRefillError,
    constraint_index: u32,
) -> DelaunayFacetRecoveryError {
    match failure {
        ConstrainedCavityRefillError::Accounting(cause)
        | ConstrainedCavityRefillError::Validation(ConstrainedCavityValidationError::Accounting(
            cause,
        )) => resource_or_cancelled(
            DelaunayFacetRecoveryErrorKind::Accounting(cause),
            constraint_index,
            "numerical accounting denied cavity candidate".to_owned(),
        ),
        ConstrainedCavityRefillError::ResourceLimit { reason } => resource_or_cancelled(
            DelaunayFacetRecoveryErrorKind::ResourceLimit,
            constraint_index,
            reason,
        ),
        ConstrainedCavityRefillError::Cancelled => resource_or_cancelled(
            DelaunayFacetRecoveryErrorKind::Cancelled,
            constraint_index,
            "cancelled".to_owned(),
        ),
        failure => invalid_topology(
            constraint_index,
            format!("facet cavity Steiner generation failed: {failure:?}"),
        ),
    }
}
