use sha2::{Digest, Sha256};

use super::super::FacetRecoveryWork;
use super::{try_recover_facet_cavity_once, FacetCavityAttempt};
use crate::cdt::{
    DelaunayFacetRecoveryError, DelaunayFacetSteinerInsertion, DelaunayRecoveredFacetTriangle,
    DelaunaySegmentRecovery, DelaunayVolumeTopology,
};
use runmat_meshing_core::StableDigest;

mod candidate;

pub(in crate::cdt::facet_recovery) use candidate::{
    candidate_node, try_insert_candidate, try_insert_candidate_with_conflicts, CandidateInsertion,
    CandidateInsertionResult,
};
pub(in crate::cdt::facet_recovery) use candidate::{
    candidate_refills_cavity, facet_edges_exist, steiner_candidate_groups, SteinerCandidateGroup,
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
    'recovery: loop {
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
                    steiner_candidate_groups(&current.topology, &cavities, constraint_index, work)?;
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
                    let first_candidates = candidate_groups[0].ordered();
                    let second_candidates = candidate_groups[1].ordered();
                    let mut second_refillable = vec![None; second_candidates.len()];
                    for (first_rank, first_coordinates) in
                        first_candidates.iter().copied().enumerate()
                    {
                        if !candidate_refills_cavity(
                            &cavities[0],
                            &original.topology,
                            first_coordinates,
                            constraint_index,
                            work,
                        )? {
                            continue;
                        }
                        let Some(first_node) = candidate_node(
                            &original.topology,
                            CandidateInsertion {
                                coordinates_m: runmat_types::physical::Quantity::<runmat_types::physical::kind::Position,_>::new(first_coordinates),
                                construction: runmat_meshing_core::quality::predicate::Point3Construction::Explicit,
                                facet,
                                insertion_round,
                                candidate_rank: first_rank as u64,
                            },
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
                                        &original.topology,
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
                                &original.topology,
                                CandidateInsertion {
                                    coordinates_m: runmat_types::physical::Quantity::<runmat_types::physical::kind::Position,_>::new(second_coordinates),
                                    construction: runmat_meshing_core::quality::predicate::Point3Construction::Explicit,
                                    facet,
                                    insertion_round: insertion_round + 1,
                                    candidate_rank: second_rank as u64,
                                },
                                constraint_index,
                                work,
                            )?
                            else {
                                continue;
                            };
                            let Some(topology) = super::refill_cavity_sides(
                                &original,
                                super::CavityRefillRequest {
                                    inserted_nodes: &[first_node, second_node],
                                    facet,
                                    protected_facets,
                                    cavities: &cavities,
                                    refill_options:
                                        super::progressive_facet_recovery_refill_options(),
                                    constraint_index,
                                },
                                work,
                            )?
                            else {
                                continue;
                            };
                            for (node, round, rank) in [
                                (first_node, insertion_round, first_rank as u64),
                                (second_node, insertion_round + 1, second_rank as u64),
                            ] {
                                if topology
                                    .nodes
                                    .binary_search_by_key(&node.identity, |candidate| {
                                        candidate.identity
                                    })
                                    .is_err()
                                {
                                    continue;
                                }
                                work.cavity_steiner_node(constraint_index)?;
                                steiner_insertions.push(steiner_lineage(
                                    facet,
                                    constraint_index,
                                    round,
                                    rank,
                                    node.identity,
                                ));
                            }
                            return Ok(Some(FacetCavityRecovery {
                                topology,
                                steiner_insertions,
                            }));
                        }
                    }
                    work.reject("no jointly valid Steiner pair refilled both cavity halves");
                    return Ok(None);
                }
                let SteinerCandidateGroup { progressive, exact } =
                    candidate_groups.into_iter().next().unwrap_or_default();
                let exact_rank_offset = progressive.len() as u64;
                for (candidate_rank, coordinates_m) in progressive.into_iter().enumerate() {
                    let candidate_rank = candidate_rank as u64;
                    let Some(node) = candidate_node(
                        &current.topology,
                        CandidateInsertion {
                            coordinates_m: runmat_types::physical::Quantity::<runmat_types::physical::kind::Position,_>::new(coordinates_m),
                            construction: runmat_meshing_core::quality::predicate::Point3Construction::Explicit,
                            facet,
                            insertion_round,
                            candidate_rank,
                        },
                        constraint_index,
                        work,
                    )?
                    else {
                        continue;
                    };
                    let node_identity = node.identity;
                    if let Some(topology) = super::refill_cavity_sides(
                        &current,
                        super::CavityRefillRequest {
                            inserted_nodes: &[node],
                            facet,
                            protected_facets,
                            cavities: &cavities,
                            refill_options: super::facet_recovery_refill_options(),
                            constraint_index,
                        },
                        work,
                    )? {
                        work.cavity_steiner_node(constraint_index)?;
                        steiner_insertions.push(steiner_lineage(
                            facet,
                            constraint_index,
                            insertion_round,
                            candidate_rank,
                            node_identity,
                        ));
                        return Ok(Some(FacetCavityRecovery {
                            topology,
                            steiner_insertions,
                        }));
                    }
                    let Some(next) =
                        try_insert_candidate(&current, node, &protected, constraint_index, work)?
                    else {
                        continue;
                    };
                    if facet_edges_exist(&next.topology, facet) {
                        work.cavity_steiner_node(constraint_index)?;
                        steiner_insertions.push(steiner_lineage(
                            facet,
                            constraint_index,
                            insertion_round,
                            candidate_rank,
                            node.identity,
                        ));
                        insertion_round += 1;
                        current = next;
                        continue 'recovery;
                    }
                }
                let progressive_rejection = work
                    .last_rejection()
                    .unwrap_or("progressive insertion returned no topology")
                    .to_owned();
                let mut admitted = Vec::new();
                for (candidate_rank, coordinates_m) in exact.into_iter().enumerate() {
                    let candidate_rank = exact_rank_offset + candidate_rank as u64;
                    let Some(node) = candidate_node(
                        &current.topology,
                        CandidateInsertion {
                            coordinates_m: runmat_types::physical::Quantity::<runmat_types::physical::kind::Position,_>::new(coordinates_m),
                            construction: runmat_meshing_core::quality::predicate::Point3Construction::Explicit,
                            facet,
                            insertion_round,
                            candidate_rank,
                        },
                        constraint_index,
                        work,
                    )?
                    else {
                        continue;
                    };
                    if let Some(topology) = super::refill_cavity_sides(
                        &current,
                        super::CavityRefillRequest {
                            inserted_nodes: &[node],
                            facet,
                            protected_facets,
                            cavities: &cavities,
                            refill_options: super::facet_recovery_refill_options(),
                            constraint_index,
                        },
                        work,
                    )? {
                        work.cavity_steiner_node(constraint_index)?;
                        steiner_insertions.push(steiner_lineage(
                            facet,
                            constraint_index,
                            insertion_round,
                            candidate_rank,
                            node.identity,
                        ));
                        return Ok(Some(FacetCavityRecovery {
                            topology,
                            steiner_insertions,
                        }));
                    }
                    admitted.push((node, candidate_rank));
                }
                if !admitted.is_empty() {
                    let nodes = admitted.iter().map(|(node, _)| *node).collect::<Vec<_>>();
                    if let Some(topology) = super::refill_cavity_sides(
                        &current,
                        super::CavityRefillRequest {
                            inserted_nodes: &nodes,
                            facet,
                            protected_facets,
                            cavities: &cavities,
                            refill_options: super::facet_recovery_refill_options(),
                            constraint_index,
                        },
                        work,
                    )? {
                        for (node, candidate_rank) in
                            admitted.iter().copied().filter(|(node, _)| {
                                topology
                                    .nodes
                                    .binary_search_by_key(&node.identity, |candidate| {
                                        candidate.identity
                                    })
                                    .is_ok()
                            })
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
                        return Ok(Some(FacetCavityRecovery {
                            topology,
                            steiner_insertions,
                        }));
                    }
                }
                let batch_rejection = work
                    .last_rejection()
                    .unwrap_or("atomic Steiner-inventory refill returned no topology");
                let admitted_coordinates = admitted
                    .iter()
                    .take(8)
                    .map(|(node, _)| node.coordinates_m.canonical_value())
                    .collect::<Vec<_>>();
                work.reject(format!(
                    "no valid Steiner candidate refilled the cavity: admitted_candidates={}, admitted_coordinates={admitted_coordinates:?}, atomic inventory rejected: {batch_rejection}; progressive candidates rejected: {progressive_rejection}",
                    admitted.len(),
                ));
                return Ok(None);
            }
        }
    }
}

pub(in crate::cdt::facet_recovery) fn steiner_lineage(
    facet: [StableDigest; 3],
    constraint_index: u32,
    insertion_round: u64,
    candidate_rank: u64,
    node_identity: StableDigest,
) -> DelaunayFacetSteinerInsertion {
    let mut support_node_identities = facet;
    support_node_identities.sort_unstable();
    DelaunayFacetSteinerInsertion {
        constraint_index,
        support_node_identities,
        insertion_round,
        candidate_rank,
        node_identity,
    }
}

pub(in crate::cdt::facet_recovery) fn steiner_identity(
    facet: [StableDigest; 3],
    insertion_round: u64,
    candidate_index: u64,
) -> StableDigest {
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
