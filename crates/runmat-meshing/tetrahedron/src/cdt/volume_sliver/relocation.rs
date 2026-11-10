use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{predicate::tetrahedron_volume, MeshingCancellationSignal, StableDigest};
use sha2::{Digest, Sha256};

use super::admission::{
    candidate_quality_is_admissible, evaluate_candidate, evaluate_candidate_unranked,
    CandidateAdmissionPolicy, CandidateTopologyContract,
};
use super::working::rejected::{RankedOutCandidate, SearchRecording};
use super::retriangulate::PreparedVertexRemoval;
use super::work::OptimizationWork;
use super::{
    error, DelaunayVolumeSliverDeletion, DelaunayVolumeSliverError, DelaunayVolumeSliverErrorKind,
    DelaunayVolumeSliverMutation, DelaunayVolumeSliverOptions, DelaunayVolumeSliverRelocation,
    SliverSearchPolicy, CAVITY_SMOOTHING_STEPS, MAXIMUM_CAVITY_SEARCH_DIRECTIONS,
    MAXIMUM_CAVITY_SMOOTHING_ITERATIONS,
};
use crate::cdt::topology::VolumeTopologyRead;
use crate::cdt::{
    topology::{
        build_delaunay_volume_topology_with_regions,
        orient_materialized_volume_nodes as orient_volume_nodes,
    },
    DelaunayTetrahedronQuality, DelaunayTopologyErrorKind, DelaunayVolumeNode,
    DelaunayVolumeProvenance, DelaunayVolumeQuality, DelaunayVolumeRefinementInput,
    DelaunayVolumeTopology,
};

mod block;
mod cavity;
mod changes;
mod collapse;
mod geometry;
mod materialization;
mod neighborhood;
use collapse::collapse_interior_edge;
use materialization::rebuild_with_relocated_node;
pub(super) mod working;
use geometry::{
    neighbor_centroid, sliver_height_target, smooth_vertex_star_points_with_metrics,
    volume_weighted_star_centroid,
};

const RELOCATION_IDENTITY_DOMAIN: &[u8] = b"runmat/meshing/cdt/sliver-relocation/1\0";

pub(super) struct RelocationCandidate {
    pub(super) topology: DelaunayVolumeTopology,
    pub(super) quality: DelaunayVolumeQuality,
    pub(super) mutation: DelaunayVolumeSliverMutation,
}

pub(super) struct RelocationSearch {
    pub(super) candidate: Option<RelocationCandidate>,
    pub(super) rejection_counts: std::collections::BTreeMap<&'static str, u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RelocationSearchScope {
    /// Search every supported topological and geometric relocation.
    Complete,
    /// Same topology-changing strategies after the shared working owner already
    /// exhausted fixed-star coordinates. Do not materialize those trials again.
    TopologyChanging,
    /// Keep the current cavity connectivity. Compound searches use this scope so a bounded
    /// lookahead cannot recursively launch another global Delaunay legalization.
    #[cfg(test)]
    FixedStarOnly,
}

impl RelocationSearchScope {
    fn topology_changes(self) -> bool {
        matches!(self, Self::Complete | Self::TopologyChanging)
    }
}

/// The sliver one relocation search repairs, the topology it starts from and the search policy.
pub(super) struct RelocationSearchRequest<'a> {
    pub(super) input: DelaunayVolumeRefinementInput<'a>,
    pub(super) source: &'a DelaunayTetrahedronQuality,
    pub(super) current_spectrum: &'a [f64],
    pub(super) policy: SliverSearchPolicy,
    pub(super) scope: RelocationSearchScope,
}

#[cfg(test)]
pub(super) fn relocation_candidates(
    input: DelaunayVolumeRefinementInput<'_>,
    source: &DelaunayTetrahedronQuality,
    current_spectrum: &[f64],
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    policy: SliverSearchPolicy,
    scope: RelocationSearchScope,
) -> Result<RelocationSearch, DelaunayVolumeSliverError> {
    search(
        RelocationSearchRequest {
            input,
            source,
            current_spectrum,
            policy,
            scope,
        },
        cancellation,
        work,
        None,
    )
}

/// [`relocation_candidates`] that also records, in `recording`, the nodes of every cell a
/// produced candidate added or removed and the changed cells' quality of every candidate that
/// lost only the quality ranking, so a caller can tell when a rejected search would decide the
/// same.
pub(super) fn relocation_candidates_recording_changes(
    request: RelocationSearchRequest<'_>,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    recording: &mut SearchRecording,
) -> Result<RelocationSearch, DelaunayVolumeSliverError> {
    search(request, cancellation, work, Some(recording))
}

fn search(
    request: RelocationSearchRequest<'_>,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    mut recording: Option<&mut SearchRecording>,
) -> Result<RelocationSearch, DelaunayVolumeSliverError> {
    let RelocationSearchRequest {
        input,
        source,
        current_spectrum,
        policy,
        scope,
    } = request;
    let SliverSearchPolicy {
        options,
        refinement: refinement_policy,
        objective,
    } = policy;
    let topology = input.topology;
    let protected = protected_nodes(topology, input.provenance, input.fixed_node_identities);
    let boundary = topology
        .incidence
        .boundary_facets
        .iter()
        .flat_map(|facet| facet.vertex_indices)
        .map(|index| topology.nodes[index as usize].identity)
        .collect::<BTreeSet<_>>();
    let protected_faces = input
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<Vec<_>>();
    let evaluation = CandidateEvaluation {
        input,
        protected_faces: &protected_faces,
        current_spectrum,
        cancellation,
    };
    let mut source_nodes = source.node_identities;
    source_nodes.sort_unstable();
    let eligible_sources = source_nodes
        .into_iter()
        .filter(|identity| !protected.contains(identity) && !boundary.contains(identity))
        .collect::<Vec<_>>();
    const FRACTIONS: [f64; 4] = [1.0, 0.5, 0.25, 0.125];
    let mut rejection_counts = std::collections::BTreeMap::new();
    let mut prepared_removals = BTreeMap::new();

    // A conforming interior-edge collapse is the least disruptive topological
    // repair for a free volume vertex. In particular, it completes the standard
    // split-then-collapse escape for slivers whose edge star cannot be improved
    // by a direct swap. Only the deleted endpoint must be an unconstrained
    // interior vertex: an unchanged boundary vertex is also a valid target.
    // Excluding boundary targets strands free vertices in thin boundary collars.
    // The topology and PLC admission below independently enforce conformity,
    // region separation, and preservation of every protected facet.
    for source_identity in eligible_sources
        .iter()
        .copied()
        .filter(|_| scope.topology_changes())
    {
        let source_index = topology
            .nodes
            .binary_search_by_key(&source_identity, |node| node.identity)
            .map_err(|_| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    "sliver source node is absent from canonical topology",
                )
            })?;
        let mut targets = topology.incidence.vertex_stars[source_index]
            .iter()
            .flat_map(|tetrahedron| topology.tetrahedra[*tetrahedron as usize].vertex_indices)
            .map(|vertex| topology.nodes[vertex as usize].identity)
            .filter(|identity| *identity != source_identity)
            .collect::<Vec<_>>();
        targets.sort_unstable();
        targets.dedup();
        for target_identity in targets {
            work.evaluate()?;
            work.global_candidate(topology.nodes.len(), topology.tetrahedra.len())?;
            let Some(candidate_topology) = collapse_interior_edge(
                topology,
                source_index,
                target_identity,
                options,
                cancellation,
            )?
            else {
                reject(
                    &mut rejection_counts,
                    "interior_edge_collapse_not_admissible",
                );
                continue;
            };
            let Some(quality) = evaluate_recorded(
                recording.as_deref_mut(),
                topology,
                &candidate_topology,
                &evaluation,
                CandidateAdmissionPolicy {
                    topology_contract: CandidateTopologyContract::CertifiedConformingCavity,
                    refinement: refinement_policy,
                    objective,
                    options,
                },
                &mut rejection_counts,
            )?
            else {
                continue;
            };
            return Ok(RelocationSearch {
                candidate: Some(RelocationCandidate {
                    topology: candidate_topology,
                    quality,
                    mutation: DelaunayVolumeSliverMutation::Deleted(DelaunayVolumeSliverDeletion {
                        source_node_identity: source_identity,
                        source_tetrahedron_node_identities: source.node_identities,
                    }),
                }),
                rejection_counts,
            });
        }
    }

    // If no adjacent volume vertex is a legal collapse target, refill the
    // complete deleted-vertex cavity before searching relocation coordinates.
    for source_identity in eligible_sources
        .iter()
        .copied()
        .filter(|_| scope.topology_changes())
    {
        let Some(source_index) = topology
            .nodes
            .binary_search_by_key(&source_identity, |node| node.identity)
            .ok()
        else {
            return Err(error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                "sliver source node is absent from canonical topology",
            ));
        };
        work.evaluate()?;
        work.global_candidate(topology.nodes.len(), topology.tetrahedra.len())?;
        if let std::collections::btree_map::Entry::Vacant(entry) =
            prepared_removals.entry(source_index)
        {
            entry.insert(PreparedVertexRemoval::prepare(
                topology,
                source_index,
                options.insertion,
                cancellation,
            )?);
        }
        let candidate_topology = match prepared_removals
            .get(&source_index)
            .and_then(Option::as_ref)
        {
            Some(prepared) => prepared.materialize(&protected_faces, cancellation)?,
            None => None,
        };
        if let Some(candidate_topology) = candidate_topology {
            if let Some(quality) = evaluate_recorded(
                recording.as_deref_mut(),
                topology,
                &candidate_topology,
                &evaluation,
                CandidateAdmissionPolicy {
                    topology_contract: CandidateTopologyContract::Delaunay,
                    refinement: refinement_policy,
                    objective,
                    options,
                },
                &mut rejection_counts,
            )? {
                return Ok(RelocationSearch {
                    candidate: Some(RelocationCandidate {
                        topology: candidate_topology,
                        quality,
                        mutation: DelaunayVolumeSliverMutation::Deleted(
                            DelaunayVolumeSliverDeletion {
                                source_node_identity: source_identity,
                                source_tetrahedron_node_identities: source.node_identities,
                            },
                        ),
                    }),
                    rejection_counts,
                });
            }
        } else {
            reject(&mut rejection_counts, "vertex_star_could_not_be_refilled");
        }
    }
    for source_identity in eligible_sources {
        let source_index = topology
            .nodes
            .binary_search_by_key(&source_identity, |node| node.identity)
            .map_err(|_| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    "sliver source node is absent from canonical topology",
                )
            })?;
        let source_coordinates = topology.nodes[source_index].coordinates_m.canonical_value();
        let mut candidate_coordinates = smooth_vertex_star_points_with_metrics(
            topology,
            source_index,
            &|index| input.quality.tetrahedra[index as usize].resolved_metric,
            geometry::SmoothingObjective::for_policy(policy, input.quality_options),
            cancellation,
        )?;
        let targets = [
            sliver_height_target(
                topology,
                source,
                source_identity,
                input.quality_options.minimum_metric_scaled_jacobian,
            )?,
            Some(volume_weighted_star_centroid(topology, source_index)?),
            Some(neighbor_centroid(topology, source_index)?),
        ];
        for target in targets.into_iter().flatten() {
            for fraction in FRACTIONS {
                let coordinates = std::array::from_fn(|axis| {
                    source_coordinates[axis] + fraction * (target[axis] - source_coordinates[axis])
                });
                if !candidate_coordinates.contains(&coordinates) {
                    candidate_coordinates.push(coordinates);
                }
            }
        }
        for coordinates_m in candidate_coordinates {
            work.evaluate()?;
            let replacement_node =
                DelaunayVolumeNode::explicit(replacement_identity(source_identity), coordinates_m);
            if scope != RelocationSearchScope::TopologyChanging {
                work.global_candidate(topology.nodes.len(), topology.tetrahedra.len())?;
                if let Some(candidate_topology) = rebuild_with_relocated_node(
                    topology,
                    source_index,
                    replacement_node,
                    options,
                    cancellation,
                )? {
                    if let Some(candidate_quality) = evaluate_recorded(
                        recording.as_deref_mut(),
                        topology,
                        &candidate_topology,
                        &evaluation,
                        CandidateAdmissionPolicy {
                            topology_contract: CandidateTopologyContract::CertifiedConformingCavity,
                            refinement: refinement_policy,
                            objective,
                            options,
                        },
                        &mut rejection_counts,
                    )? {
                        return Ok(RelocationSearch {
                            candidate: Some(RelocationCandidate {
                                topology: candidate_topology,
                                quality: candidate_quality,
                                mutation: relocation_mutation(
                                    source_identity,
                                    replacement_node,
                                    source.node_identities,
                                ),
                            }),
                            rejection_counts,
                        });
                    }
                } else {
                    reject(
                        &mut rejection_counts,
                        "fixed_star_relocation_not_admissible",
                    );
                }
            }
            #[cfg(test)]
            if matches!(scope, RelocationSearchScope::FixedStarOnly) {
                continue;
            }
            if let std::collections::btree_map::Entry::Vacant(entry) =
                prepared_removals.entry(source_index)
            {
                entry.insert(PreparedVertexRemoval::prepare(
                    topology,
                    source_index,
                    options.insertion,
                    cancellation,
                )?);
            }
            let candidate_topology = match prepared_removals
                .get(&source_index)
                .and_then(Option::as_ref)
            {
                Some(prepared) => {
                    work.global_candidate(topology.nodes.len(), topology.tetrahedra.len())?;
                    prepared.relocate(replacement_node, &protected_faces, cancellation)?
                }
                None => None,
            };
            let Some(candidate_topology) = candidate_topology else {
                reject(
                    &mut rejection_counts,
                    "relocated_vertex_star_could_not_be_refilled",
                );
                continue;
            };
            let Some(candidate_quality) = evaluate_recorded(
                recording.as_deref_mut(),
                topology,
                &candidate_topology,
                &evaluation,
                CandidateAdmissionPolicy {
                    topology_contract: CandidateTopologyContract::Delaunay,
                    refinement: refinement_policy,
                    objective,
                    options,
                },
                &mut rejection_counts,
            )?
            else {
                continue;
            };
            return Ok(RelocationSearch {
                candidate: Some(RelocationCandidate {
                    topology: candidate_topology,
                    quality: candidate_quality,
                    mutation: relocation_mutation(
                        source_identity,
                        replacement_node,
                        source.node_identities,
                    ),
                }),
                rejection_counts,
            });
        }
    }
    Ok(RelocationSearch {
        candidate: None,
        rejection_counts,
    })
}

fn relocation_mutation(
    source_node_identity: StableDigest,
    replacement_node: DelaunayVolumeNode,
    source_tetrahedron_node_identities: [StableDigest; 4],
) -> DelaunayVolumeSliverMutation {
    DelaunayVolumeSliverMutation::Relocated(DelaunayVolumeSliverRelocation {
        source_node_identity,
        replacement_node,
        source_tetrahedron_node_identities,
    })
}

/// Records one atomic multi-node relocation as an equivalent ordered sequence.
/// Each source tetrahedron is captured before the group moves; members that an
/// earlier entry already relocated are named by their replacement identity, so
/// sequential lineage replay sees exactly the nodes present at each step.
fn grouped_relocation_mutations(
    moves: impl IntoIterator<Item = (StableDigest, DelaunayVolumeNode, [StableDigest; 4])>,
) -> Vec<DelaunayVolumeSliverMutation> {
    let mut moved = BTreeMap::new();
    moves
        .into_iter()
        .map(|(source, replacement, tetrahedron)| {
            let tetrahedron =
                tetrahedron.map(|identity| moved.get(&identity).copied().unwrap_or(identity));
            moved.insert(source, replacement.identity);
            relocation_mutation(source, replacement, tetrahedron)
        })
        .collect()
}

fn reject(counts: &mut std::collections::BTreeMap<&'static str, u64>, reason: &'static str) {
    *counts.entry(reason).or_default() += 1;
}

/// Moves one eligible sliver vertex away from its opposite face far enough to make a useful
/// quality change. Fractions of this target are checked by the caller because the rest of the
/// vertex star, protected facets, and the exact Delaunay contract remain authoritative gates.
fn protected_nodes(
    topology: &DelaunayVolumeTopology,
    provenance: &DelaunayVolumeProvenance,
    fixed_node_identities: &[StableDigest],
) -> BTreeSet<StableDigest> {
    let mut protected = provenance
        .nodes
        .iter()
        .map(|binding| binding.node_identity)
        .collect::<BTreeSet<_>>();
    protected.extend(
        provenance
            .segments
            .iter()
            .flat_map(|binding| binding.node_identities),
    );
    protected.extend(fixed_node_identities.iter().copied());
    protected.extend(
        provenance
            .facets
            .iter()
            .flat_map(|binding| binding.node_identities),
    );
    protected.retain(|identity| {
        topology
            .nodes
            .binary_search_by_key(identity, |node| node.identity)
            .is_ok()
    });
    protected
}

fn replacement_identity(source: StableDigest) -> StableDigest {
    let mut hasher = Sha256::new();
    hasher.update(RELOCATION_IDENTITY_DOMAIN);
    hasher.update(source.bytes());
    StableDigest::from_bytes(hasher.finalize().into())
}

pub(super) fn relocation_identity_is_valid(relocation: &DelaunayVolumeSliverRelocation) -> bool {
    relocation.replacement_node.identity == replacement_identity(relocation.source_node_identity)
}

#[cfg(test)]
mod tests;

/// The search-wide inputs every candidate in one relocation search is evaluated against.
#[derive(Clone, Copy)]
struct CandidateEvaluation<'a> {
    input: DelaunayVolumeRefinementInput<'a>,
    protected_faces: &'a [[StableDigest; 3]],
    current_spectrum: &'a [f64],
    cancellation: &'a dyn MeshingCancellationSignal,
}

/// [`evaluate_candidate`] that, with a recording, also records the nodes of the cells the
/// candidate changed, and the changed cells' quality ratios when the candidate loses only the
/// quality ranking.
fn evaluate_recorded(
    recording: Option<&mut SearchRecording>,
    current: &DelaunayVolumeTopology,
    candidate: &DelaunayVolumeTopology,
    evaluation: &CandidateEvaluation<'_>,
    policy: CandidateAdmissionPolicy,
    rejection_counts: &mut BTreeMap<&'static str, u64>,
) -> Result<Option<DelaunayVolumeQuality>, DelaunayVolumeSliverError> {
    let CandidateEvaluation {
        input,
        protected_faces,
        current_spectrum,
        cancellation,
    } = *evaluation;
    let Some(recording) = recording else {
        return evaluate_candidate(
            candidate,
            input,
            protected_faces,
            current_spectrum,
            policy,
            cancellation,
            rejection_counts,
        );
    };
    let changes = changes::changed_cells(current, candidate);
    changes::record_changed_nodes(current, candidate, &changes, &mut recording.changed);
    let Some(quality) = evaluate_candidate_unranked(
        candidate,
        input,
        protected_faces,
        policy.topology_contract,
        policy.options,
        cancellation,
        rejection_counts,
    )?
    else {
        return Ok(None);
    };
    if candidate_quality_is_admissible(
        &quality,
        input,
        current_spectrum,
        policy.refinement,
        policy.objective,
        rejection_counts,
    ) {
        return Ok(Some(quality));
    }
    let ratios = |cell: &DelaunayTetrahedronQuality| {
        [
            cell.optimization_violation_ratio,
            cell.refinement_violation_ratio,
        ]
    };
    recording.ranked_out.push(RankedOutCandidate {
        removed: changes
            .removed
            .iter()
            .map(|index| ratios(&input.quality.tetrahedra[*index]))
            .collect(),
        added: changes
            .added
            .iter()
            .map(|index| ratios(&quality.tetrahedra[*index]))
            .collect(),
    });
    Ok(None)
}
