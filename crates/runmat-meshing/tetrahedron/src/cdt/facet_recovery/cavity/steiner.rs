use sha2::{Digest, Sha256};

use super::super::FacetRecoveryWork;
use super::{try_recover_facet_cavity_once, FacetCavityAttempt};
use crate::cdt::{
    DelaunayFacetRecoveryError, DelaunayFacetSteinerInsertion, DelaunayRecoveredFacetTriangle,
    DelaunaySegmentRecovery, DelaunayVolumeTopology,
};
use runmat_meshing_core::StableDigest;

mod candidate;

use candidate::{
    candidate_refills_cavity, insert_first_candidate, steiner_candidate_groups,
    try_insert_candidate, CandidateInsertion,
};

const FACET_STEINER_IDENTITY_DOMAIN: &[u8] = b"runmat:cdt-facet-steiner-node:v1";

#[derive(Clone, Debug, PartialEq)]
pub(in crate::cdt::facet_recovery) struct FacetCavityRecovery {
    pub(in crate::cdt::facet_recovery) topology: DelaunayVolumeTopology,
    pub(in crate::cdt::facet_recovery) steiner_insertions: Vec<DelaunayFacetSteinerInsertion>,
}

/// Retries a single unfillable side after each checked insertion. When both sides of one authored
/// facet need an interior node, it searches bounded candidate pairs against the same immutable
/// cavity and commits only a jointly valid constrained-Delaunay refill. Work counters are shared
/// across every trial and facet, so failure is atomic and the search cannot outlive its limits.
pub(in crate::cdt::facet_recovery) fn try_recover_facet_with_cavity(
    recovery: &DelaunaySegmentRecovery,
    facet: [StableDigest; 3],
    protected_facets: &[DelaunayRecoveredFacetTriangle],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<FacetCavityRecovery>, DelaunayFacetRecoveryError> {
    let mut current = recovery.clone();
    let mut insertion_round = 0_u64;
    let mut steiner_insertions = Vec::new();
    loop {
        match try_recover_facet_cavity_once(
            &current,
            facet,
            protected_facets,
            constraint_index,
            work,
        )? {
            FacetCavityAttempt::Recovered(topology) => {
                return Ok(Some(FacetCavityRecovery {
                    topology,
                    steiner_insertions,
                }));
            }
            FacetCavityAttempt::Unavailable => return Ok(None),
            FacetCavityAttempt::NeedsSteiner(cavities) => {
                let candidate_groups =
                    steiner_candidate_groups(&current, &cavities, constraint_index, work)?;
                let protected = protected_facets
                    .iter()
                    .map(|triangle| {
                        let mut identities = triangle.node_identities;
                        identities.sort_unstable();
                        identities
                    })
                    .collect::<Vec<_>>();
                if candidate_groups.len() == 2 {
                    let original = current.clone();
                    let mut second_refillable = vec![None; candidate_groups[1].len()];
                    for (first_rank, first_coordinates) in
                        candidate_groups[0].iter().copied().enumerate()
                    {
                        if !candidate_refills_cavity(
                            &cavities[0],
                            &original,
                            first_coordinates,
                            constraint_index,
                            work,
                        )? {
                            continue;
                        }
                        let Some((first, first_identity)) = try_insert_candidate(
                            &original,
                            CandidateInsertion {
                                coordinates_m: first_coordinates,
                                facet,
                                insertion_round,
                                candidate_rank: first_rank as u64,
                                protected: &protected,
                            },
                            constraint_index,
                            work,
                        )?
                        else {
                            continue;
                        };
                        for (second_rank, second_coordinates) in
                            candidate_groups[1].iter().copied().enumerate()
                        {
                            let refillable = match second_refillable[second_rank] {
                                Some(refillable) => refillable,
                                None => {
                                    let refillable = candidate_refills_cavity(
                                        &cavities[1],
                                        &original,
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
                            let Some((inserted, second_identity)) = try_insert_candidate(
                                &first,
                                CandidateInsertion {
                                    coordinates_m: second_coordinates,
                                    facet,
                                    insertion_round: insertion_round + 1,
                                    candidate_rank: second_rank as u64,
                                    protected: &protected,
                                },
                                constraint_index,
                                work,
                            )?
                            else {
                                continue;
                            };
                            let Some(topology) = super::refill_simultaneous_sides(
                                &original,
                                &inserted,
                                facet,
                                protected_facets,
                                &cavities,
                                constraint_index,
                                work,
                            )?
                            else {
                                continue;
                            };
                            work.cavity_steiner_node(constraint_index)?;
                            work.cavity_steiner_node(constraint_index)?;
                            let mut support_node_identities = facet;
                            support_node_identities.sort_unstable();
                            steiner_insertions.extend([
                                DelaunayFacetSteinerInsertion {
                                    constraint_index,
                                    support_node_identities,
                                    insertion_round,
                                    candidate_rank: first_rank as u64,
                                    node_identity: first_identity,
                                },
                                DelaunayFacetSteinerInsertion {
                                    constraint_index,
                                    support_node_identities,
                                    insertion_round: insertion_round + 1,
                                    candidate_rank: second_rank as u64,
                                    node_identity: second_identity,
                                },
                            ]);
                            return Ok(Some(FacetCavityRecovery {
                                topology,
                                steiner_insertions,
                            }));
                        }
                    }
                    return Ok(None);
                }
                let Some((next, node_identity, candidate_rank)) = insert_first_candidate(
                    &current,
                    candidate_groups.into_iter().flatten().collect(),
                    facet,
                    insertion_round,
                    &protected,
                    constraint_index,
                    work,
                )?
                else {
                    return Ok(None);
                };
                work.cavity_steiner_node(constraint_index)?;
                let mut support_node_identities = facet;
                support_node_identities.sort_unstable();
                steiner_insertions.push(DelaunayFacetSteinerInsertion {
                    constraint_index,
                    support_node_identities,
                    insertion_round,
                    candidate_rank,
                    node_identity,
                });
                insertion_round += 1;
                current = next;
            }
        }
    }
}

pub(in crate::cdt::facet_recovery) fn steiner_identity(
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
