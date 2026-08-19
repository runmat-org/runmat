use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use super::super::FacetRecoveryWork;
use super::{
    recovered_segments_exist, resource_or_cancelled, try_recover_facet_cavity_once,
    FacetCavityAttempt,
};
use crate::{
    cavity::constrained::{
        generate_constrained_cavity_interior_steiner_candidates, ConstrainedCavityNode,
        ConstrainedCavityRefillError, ConstrainedCavityRefillOptions,
        ConstrainedCavitySteinerCandidateBudget,
    },
    cdt::{
        insertion::insert_delaunay_volume_node_with_barriers, DelaunayFacetRecoveryError,
        DelaunayFacetRecoveryErrorKind, DelaunayRecoveredFacetTriangle, DelaunaySegmentRecovery,
        DelaunayVolumeNode, DelaunayVolumeTopology,
    },
};
use runmat_meshing_core::StableDigest;

const FACET_STEINER_IDENTITY_DOMAIN: &[u8] = b"runmat:cdt-facet-steiner-node:v1";

/// Retries the same checked cavity mutation after each legal barrier-aware insertion. The
/// `FacetRecoveryWork` counters are shared across all retries and facets, so failure is atomic and
/// the loop cannot outlive the configured node or candidate limits.
pub(in crate::cdt::facet_recovery) fn try_recover_facet_with_cavity(
    recovery: &DelaunaySegmentRecovery,
    facet: [StableDigest; 3],
    protected_facets: &[DelaunayRecoveredFacetTriangle],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayFacetRecoveryError> {
    let mut current = recovery.clone();
    let mut insertion_round = 0_u64;
    loop {
        match try_recover_facet_cavity_once(
            &current,
            facet,
            protected_facets,
            constraint_index,
            work,
        )? {
            FacetCavityAttempt::Recovered(topology) => return Ok(Some(topology)),
            FacetCavityAttempt::Unavailable => return Ok(None),
            FacetCavityAttempt::NeedsSteiner(cavities) => {
                let candidates = steiner_candidates(&current, &cavities, constraint_index, work)?;
                let protected = protected_facets
                    .iter()
                    .map(|triangle| {
                        let mut identities = triangle.node_identities;
                        identities.sort_unstable();
                        identities
                    })
                    .collect::<Vec<_>>();
                let mut inserted = None;
                for (candidate_index, coordinates_m) in candidates.into_iter().enumerate() {
                    work.cavity_steiner_insertion_attempt(constraint_index)?;
                    let node = DelaunayVolumeNode {
                        identity: steiner_identity(facet, insertion_round, candidate_index as u64),
                        coordinates_m,
                    };
                    if let Some(existing) = current
                        .topology
                        .nodes
                        .iter()
                        .find(|existing| existing.identity == node.identity)
                    {
                        if existing.coordinates_m != node.coordinates_m {
                            return Err(super::invalid_topology(
                                constraint_index,
                                "facet cavity Steiner identity collision",
                            ));
                        }
                        continue;
                    }
                    if current
                        .topology
                        .nodes
                        .iter()
                        .any(|existing| existing.coordinates_m == node.coordinates_m)
                    {
                        continue;
                    }
                    let candidate = match insert_delaunay_volume_node_with_barriers(
                        current.topology.clone(),
                        node,
                        &protected,
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
                                return Err(super::invalid_topology(
                                    constraint_index,
                                    failure.to_string(),
                                ));
                            }
                            crate::cdt::DelaunayInsertionErrorKind::InvalidTopology
                            | crate::cdt::DelaunayInsertionErrorKind::PointOutsideTopology => {
                                continue;
                            }
                        },
                    };
                    let mut candidate_recovery = current.clone();
                    candidate_recovery.topology = candidate;
                    if recovered_segments_exist(&candidate_recovery) {
                        inserted = Some(candidate_recovery);
                        break;
                    }
                }
                let Some(next) = inserted else {
                    return Ok(None);
                };
                work.cavity_steiner_node(constraint_index)?;
                insertion_round += 1;
                current = next;
            }
        }
    }
}

fn steiner_candidates(
    recovery: &DelaunaySegmentRecovery,
    cavities: &[crate::cavity::constrained::ConstrainedCavity],
    constraint_index: u32,
    work: &FacetRecoveryWork<'_>,
) -> Result<Vec<[f64; 3]>, DelaunayFacetRecoveryError> {
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
    let options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..ConstrainedCavityRefillOptions::default()
    };
    let budget = ConstrainedCavitySteinerCandidateBudget {
        maximum_candidates: work.options.maximum_cavity_steiner_candidates,
        maximum_evaluations: work
            .options
            .maximum_cavity_steiner_candidate_evaluations_per_round,
        cancellation_check_interval: work
            .options
            .segment_recovery
            .constraints
            .cancellation_check_interval,
    };
    let mut seen = BTreeSet::new();
    let mut candidates = Vec::new();
    for cavity in cavities {
        let generated = generate_constrained_cavity_interior_steiner_candidates(
            cavity,
            &nodes,
            options,
            budget,
            work.cancellation,
        )
        .map_err(|failure| map_candidate_error(failure, constraint_index))?;
        for point in generated {
            let bits = point.map(f64::to_bits);
            if seen.insert(bits) {
                candidates.push(point);
            }
        }
    }
    Ok(candidates)
}

fn steiner_identity(
    facet: [StableDigest; 3],
    insertion_round: u64,
    candidate_index: u64,
) -> StableDigest {
    // The candidate construction algorithm and canonical rank define identity. Floating-point
    // coordinates are derived evidence and deliberately excluded from the stable hash.
    let mut facet = facet;
    facet.sort_unstable();
    let mut hasher = Sha256::new();
    hasher.update(FACET_STEINER_IDENTITY_DOMAIN);
    for identity in facet {
        hasher.update(identity.bytes());
    }
    hasher.update(insertion_round.to_be_bytes());
    hasher.update(candidate_index.to_be_bytes());
    StableDigest::from_bytes(hasher.finalize().into())
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
        failure => super::invalid_topology(
            constraint_index,
            format!("facet cavity Steiner generation failed: {failure:?}"),
        ),
    }
}
