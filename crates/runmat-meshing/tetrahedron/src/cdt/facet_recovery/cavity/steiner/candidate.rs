use std::collections::BTreeSet;

use runmat_meshing_core::StableDigest;

use super::super::super::FacetRecoveryWork;
use super::super::{invalid_topology, recovered_segments_exist, resource_or_cancelled};
use crate::{
    cavity::constrained::{
        generate_constrained_cavity_interior_steiner_candidates,
        retriangulate_constrained_cavity_from_nodes, ConstrainedCavity, ConstrainedCavityNode,
        ConstrainedCavityRefillBudget, ConstrainedCavityRefillError,
        ConstrainedCavityRefillOptions, ConstrainedCavitySteinerCandidateBudget,
    },
    cdt::{
        insertion::insert_delaunay_volume_node_with_barriers, DelaunayFacetRecoveryError,
        DelaunayFacetRecoveryErrorKind, DelaunaySegmentRecovery, DelaunayVolumeNode,
    },
};

pub(super) struct CandidateInsertion<'a> {
    pub(super) coordinates_m: [f64; 3],
    pub(super) facet: [StableDigest; 3],
    pub(super) insertion_round: u64,
    pub(super) candidate_rank: u64,
    pub(super) protected: &'a [[StableDigest; 3]],
}

pub(super) fn insert_first_candidate(
    current: &DelaunaySegmentRecovery,
    candidates: Vec<[f64; 3]>,
    facet: [StableDigest; 3],
    insertion_round: u64,
    protected: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<(DelaunaySegmentRecovery, StableDigest, u64)>, DelaunayFacetRecoveryError> {
    for (candidate_index, coordinates_m) in candidates.into_iter().enumerate() {
        let candidate_rank = candidate_index as u64;
        if let Some((candidate, node_identity)) = try_insert_candidate(
            current,
            CandidateInsertion {
                coordinates_m,
                facet,
                insertion_round,
                candidate_rank,
                protected,
            },
            constraint_index,
            work,
        )? {
            return Ok(Some((candidate, node_identity, candidate_rank)));
        }
    }
    Ok(None)
}

pub(super) fn try_insert_candidate(
    current: &DelaunaySegmentRecovery,
    insertion: CandidateInsertion<'_>,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<(DelaunaySegmentRecovery, StableDigest)>, DelaunayFacetRecoveryError> {
    work.cavity_steiner_insertion_attempt(constraint_index)?;
    let node = DelaunayVolumeNode {
        identity: super::steiner_identity(
            insertion.facet,
            insertion.insertion_round,
            insertion.candidate_rank,
        ),
        coordinates_m: insertion.coordinates_m,
    };
    if let Some(existing) = current
        .topology
        .nodes
        .iter()
        .find(|existing| existing.identity == node.identity)
    {
        if existing.coordinates_m != node.coordinates_m {
            return Err(invalid_topology(
                constraint_index,
                "facet cavity Steiner identity collision",
            ));
        }
        return Ok(None);
    }
    if current
        .topology
        .nodes
        .iter()
        .any(|existing| existing.coordinates_m == node.coordinates_m)
    {
        return Ok(None);
    }
    let candidate = match insert_delaunay_volume_node_with_barriers(
        current.topology.clone(),
        node,
        insertion.protected,
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
            crate::cdt::DelaunayInsertionErrorKind::InvalidOptions
            | crate::cdt::DelaunayInsertionErrorKind::InvalidNode => {
                return Err(invalid_topology(constraint_index, failure.to_string()));
            }
            crate::cdt::DelaunayInsertionErrorKind::InvalidTopology
            | crate::cdt::DelaunayInsertionErrorKind::PointOutsideTopology => return Ok(None),
        },
    };
    let mut candidate_recovery = current.clone();
    candidate_recovery.topology = candidate;
    Ok(
        recovered_segments_exist(&candidate_recovery)
            .then_some((candidate_recovery, node.identity)),
    )
}

pub(super) fn steiner_candidate_groups(
    recovery: &DelaunaySegmentRecovery,
    cavities: &[ConstrainedCavity],
    constraint_index: u32,
    work: &FacetRecoveryWork<'_>,
) -> Result<Vec<Vec<[f64; 3]>>, DelaunayFacetRecoveryError> {
    let nodes = recovery
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(node_id, node)| ConstrainedCavityNode {
            node_id: node_id as u32,
            coordinates_m: node.coordinates_m,
        })
        .collect::<Vec<_>>();
    let options = refill_options();
    let budget = ConstrainedCavitySteinerCandidateBudget {
        maximum_candidates: work.options.maximum_cavity_steiner_candidates,
        maximum_evaluations: work
            .options
            .maximum_cavity_steiner_candidate_evaluations_per_round,
        cancellation_check_interval: cancellation_interval(work),
    };
    let mut groups = Vec::with_capacity(cavities.len());
    for cavity in cavities {
        let generated = generate_constrained_cavity_interior_steiner_candidates(
            cavity,
            &nodes,
            options,
            budget,
            work.cancellation,
        )
        .map_err(|failure| map_candidate_error(failure, constraint_index))?;
        let mut seen = BTreeSet::new();
        groups.push(
            generated
                .into_iter()
                .filter(|point| seen.insert(point.map(f64::to_bits)))
                .collect(),
        );
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
    let mut nodes = recovery
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(node_id, node)| ConstrainedCavityNode {
            node_id: node_id as u32,
            coordinates_m: node.coordinates_m,
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
        coordinates_m,
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
    retriangulate_constrained_cavity_from_nodes(
        cavity,
        &nodes,
        refill_options(),
        budget,
        work.cancellation,
    )
    .map(|refill| refill.is_some())
    .map_err(|failure| map_candidate_error(failure, constraint_index))
}

fn refill_options() -> ConstrainedCavityRefillOptions {
    ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..ConstrainedCavityRefillOptions::default()
    }
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
