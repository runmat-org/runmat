use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{orient3d_constructed, PredicateSign},
    MeshingCancellationSignal, StableDigest,
};

use super::admission::{
    candidate_quality_is_admissible, compare_spectra, evaluate_candidate_unranked,
    quality_spectrum, CandidateTopologyContract,
};
use super::relocation::relocation_candidates;
use super::work::OptimizationWork;
use super::{
    error, DelaunayVolumeReconnectionKind, DelaunayVolumeSliverError,
    DelaunayVolumeSliverErrorKind, DelaunayVolumeSliverMutation, DelaunayVolumeSliverOptions,
    DelaunayVolumeSliverReconnection, SliverSearchPolicy,
};
use crate::{
    cdt::{
        topology::build_delaunay_volume_topology_with_regions, DelaunayTetrahedronQuality,
        DelaunayTopologyErrorKind, DelaunayVolumeQuality, DelaunayVolumeRefinementInput,
        DelaunayVolumeTopology,
    },
    reconnect::{
        local_tetrahedron_boundary_faces, three_to_two_edge_flip_candidate,
        two_to_three_face_flip_candidate, two_to_three_geometry_is_admissible, LocalTetrahedron,
        LocalTetrahedronFlipCandidate, LocalTetrahedronFlipKind,
    },
};

pub(in crate::cdt) struct ReconnectionCandidate {
    pub(in crate::cdt) topology: DelaunayVolumeTopology,
    pub(in crate::cdt) quality: DelaunayVolumeQuality,
    pub(in crate::cdt) mutations: Vec<DelaunayVolumeSliverMutation>,
}

pub(in crate::cdt) struct ReconnectionSearch {
    pub(in crate::cdt) candidate: Option<ReconnectionCandidate>,
    pub(in crate::cdt) rejection_counts: BTreeMap<&'static str, u64>,
}

struct LookaheadSeed {
    topology: DelaunayVolumeTopology,
    quality: DelaunayVolumeQuality,
    mutations: Vec<DelaunayVolumeSliverMutation>,
}

struct LookaheadFrontier<'a> {
    protected_faces: &'a [[StableDigest; 3]],
    seeds: Vec<LookaheadSeed>,
}

#[derive(Clone, Copy)]
pub(super) struct ReconnectionMode {
    pub(super) allow_lookahead: bool,
    pub(super) maximum_candidates: Option<u64>,
}

pub(super) fn reconnection_candidates(
    input: DelaunayVolumeRefinementInput<'_>,
    source: &DelaunayTetrahedronQuality,
    current_spectrum: &[f64],
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    policy: SliverSearchPolicy,
    mode: ReconnectionMode,
) -> Result<ReconnectionSearch, DelaunayVolumeSliverError> {
    let SliverSearchPolicy {
        options,
        refinement: refinement_policy,
        objective,
    } = policy;
    let ReconnectionMode {
        allow_lookahead,
        maximum_candidates,
    } = mode;
    let source_index = input
        .topology
        .tetrahedra
        .iter()
        .position(|tetrahedron| {
            identities(input.topology, tetrahedron.vertex_indices) == source.node_identities
        })
        .ok_or_else(|| {
            error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                "optimization source tetrahedron is absent from canonical topology",
            )
        })?;
    let protected_faces = input
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<Vec<_>>();
    let candidates = local_candidates(input.topology, source_index, source.resolved_metric)?;

    let mut rejection_counts = BTreeMap::new();
    let mut lookahead_seeds = Vec::new();
    for (candidate_index, candidate) in candidates.into_iter().enumerate() {
        if maximum_candidates.is_some_and(|limit| candidate_index as u64 >= limit) {
            break;
        }
        work.evaluate()?;
        let Some(topology) = apply_candidate(input.topology, &candidate, options, cancellation)?
        else {
            reject(
                &mut rejection_counts,
                "reconnection_geometry_not_admissible",
            );
            continue;
        };
        let Some(quality) = evaluate_candidate_unranked(
            &topology,
            input,
            &protected_faces,
            CandidateTopologyContract::CertifiedConformingCavity,
            options,
            cancellation,
            &mut rejection_counts,
        )?
        else {
            continue;
        };
        let mutation = reconnection_mutation(input.topology, &candidate, source.node_identities);
        if candidate_quality_is_admissible(
            &quality,
            input,
            current_spectrum,
            refinement_policy,
            objective,
            &mut rejection_counts,
        ) {
            return Ok(ReconnectionSearch {
                candidate: Some(ReconnectionCandidate {
                    topology,
                    quality,
                    mutations: vec![mutation],
                }),
                rejection_counts,
            });
        }
        lookahead_seeds.push(LookaheadSeed {
            topology,
            quality,
            mutations: vec![mutation],
        });
    }
    if allow_lookahead {
        if let Some(candidate) = bounded_reconnection_lookahead(
            input,
            current_spectrum,
            cancellation,
            LookaheadFrontier {
                protected_faces: &protected_faces,
                seeds: lookahead_seeds,
            },
            &mut rejection_counts,
            work,
            policy,
        )? {
            return Ok(ReconnectionSearch {
                candidate: Some(candidate),
                rejection_counts,
            });
        }
    }
    Ok(ReconnectionSearch {
        candidate: None,
        rejection_counts,
    })
}

fn local_candidates(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
    metric: runmat_meshing_size::metric::MetricTensor3,
) -> Result<Vec<LocalTetrahedronFlipCandidate>, DelaunayVolumeSliverError> {
    let mut candidates = two_to_three_candidates(topology, source_index)?;
    candidates.extend(three_to_two_candidates(topology, source_index)?);
    candidates.extend(edge_star_candidates(topology, source_index)?);
    candidates.sort_by(|left, right| {
        super::super::volume_quality::violation_priority_key(candidate_minimum_jacobian(
            topology, right, metric,
        ))
        .cmp(&super::super::volume_quality::violation_priority_key(
            candidate_minimum_jacobian(topology, left, metric),
        ))
        .then_with(|| {
            candidate_geometry_key(topology, left).cmp(&candidate_geometry_key(topology, right))
        })
        .then_with(|| candidate_key(left).cmp(&candidate_key(right)))
    });
    candidates.dedup_by(|left, right| candidate_key(left) == candidate_key(right));
    Ok(candidates)
}

fn bounded_reconnection_lookahead(
    input: DelaunayVolumeRefinementInput<'_>,
    current_spectrum: &[f64],
    cancellation: &dyn MeshingCancellationSignal,
    frontier: LookaheadFrontier<'_>,
    rejection_counts: &mut BTreeMap<&'static str, u64>,
    work: &mut OptimizationWork<'_>,
    policy: SliverSearchPolicy,
) -> Result<Option<ReconnectionCandidate>, DelaunayVolumeSliverError> {
    let LookaheadFrontier {
        protected_faces,
        seeds: lookahead_seeds,
    } = frontier;
    let SliverSearchPolicy {
        options,
        refinement: refinement_policy,
        objective,
    } = policy;
    const MAXIMUM_LOOKAHEAD_DEPTH: usize = 4;
    const MAXIMUM_LOOKAHEAD_WIDTH: usize = 8;
    let mut frontier = lookahead_seeds;
    for _depth in 2..=MAXIMUM_LOOKAHEAD_DEPTH {
        let mut next_frontier = Vec::new();
        let mut seen = BTreeSet::new();
        for state in frontier {
            let Some(next_source) =
                super::admission::worst_quality_source(&state.quality, &state.topology, objective)
            else {
                continue;
            };
            let Some(next_source_index) =
                state.topology.tetrahedra.iter().position(|tetrahedron| {
                    identities(&state.topology, tetrahedron.vertex_indices)
                        == next_source.node_identities
                })
            else {
                return Err(error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    "lookahead quality source is absent from its candidate topology",
                ));
            };
            for next in local_candidates(
                &state.topology,
                next_source_index,
                next_source.resolved_metric,
            )? {
                work.evaluate()?;
                let Some(next_topology) =
                    apply_candidate(&state.topology, &next, options, cancellation)?
                else {
                    continue;
                };
                let Some(next_quality) = evaluate_candidate_unranked(
                    &next_topology,
                    input,
                    protected_faces,
                    CandidateTopologyContract::CertifiedConformingCavity,
                    options,
                    cancellation,
                    rejection_counts,
                )?
                else {
                    continue;
                };
                let mut mutations = state.mutations.clone();
                mutations.push(reconnection_mutation(
                    &state.topology,
                    &next,
                    next_source.node_identities,
                ));
                if candidate_quality_is_admissible(
                    &next_quality,
                    input,
                    current_spectrum,
                    refinement_policy,
                    objective,
                    rejection_counts,
                ) {
                    return Ok(Some(ReconnectionCandidate {
                        topology: next_topology,
                        quality: next_quality,
                        mutations,
                    }));
                }
                let key = topology_key(&next_topology);
                if seen.insert(key) {
                    next_frontier.push(LookaheadSeed {
                        topology: next_topology,
                        quality: next_quality,
                        mutations,
                    });
                }
            }
            let state_input = DelaunayVolumeRefinementInput {
                topology: &state.topology,
                metric_request: input.metric_request,
                provenance: input.provenance,
                quality: &state.quality,
                quality_options: input.quality_options,
                fixed_node_identities: input.fixed_node_identities,
            };
            let relocation = relocation_candidates(
                state_input,
                next_source,
                current_spectrum,
                cancellation,
                work,
                SliverSearchPolicy {
                    options,
                    refinement: refinement_policy,
                    objective,
                },
                super::relocation::RelocationSearchScope::FixedStarOnly,
            )?;
            if let Some(relocation) = relocation.candidate {
                let mut mutations = state.mutations;
                mutations.push(relocation.mutation);
                return Ok(Some(ReconnectionCandidate {
                    topology: relocation.topology,
                    quality: relocation.quality,
                    mutations,
                }));
            }
        }
        next_frontier.sort_by(|left, right| {
            compare_spectra(
                &quality_spectrum(&left.quality, objective),
                &quality_spectrum(&right.quality, objective),
            )
            .then_with(|| topology_key(&left.topology).cmp(&topology_key(&right.topology)))
        });
        next_frontier.truncate(MAXIMUM_LOOKAHEAD_WIDTH);
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }
    Ok(None)
}

fn topology_key(topology: &DelaunayVolumeTopology) -> Vec<[StableDigest; 4]> {
    topology
        .tetrahedra
        .iter()
        .map(|tetrahedron| {
            let mut key = identities(topology, tetrahedron.vertex_indices);
            key.sort_unstable();
            key
        })
        .collect()
}

fn reconnection_mutation(
    topology: &DelaunayVolumeTopology,
    candidate: &LocalTetrahedronFlipCandidate,
    source_tetrahedron_node_identities: [StableDigest; 4],
) -> DelaunayVolumeSliverMutation {
    let kind = match candidate.kind {
        LocalTetrahedronFlipKind::TwoToThreeFace => DelaunayVolumeReconnectionKind::TwoToThreeFace,
        LocalTetrahedronFlipKind::ThreeToTwoEdge => DelaunayVolumeReconnectionKind::ThreeToTwoEdge,
        LocalTetrahedronFlipKind::EdgeStar => DelaunayVolumeReconnectionKind::EdgeStar,
    };
    DelaunayVolumeSliverMutation::Reconnected(DelaunayVolumeSliverReconnection {
        kind,
        source_tetrahedron_node_identities,
        removed_tetrahedron_node_identities: candidate
            .removed_tetrahedron_ids
            .iter()
            .map(|index| {
                identities(
                    topology,
                    topology.tetrahedra[*index as usize].vertex_indices,
                )
            })
            .collect(),
        created_tetrahedron_node_identities: candidate
            .created_tetrahedra
            .iter()
            .map(|vertices| identities(topology, *vertices))
            .collect(),
    })
}

fn two_to_three_candidates(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
) -> Result<Vec<LocalTetrahedronFlipCandidate>, DelaunayVolumeSliverError> {
    let source = &topology.tetrahedra[source_index];
    let mut candidates = Vec::new();
    for neighbor_index in source.neighbors.into_iter().flatten() {
        let neighbor = &topology.tetrahedra[neighbor_index as usize];
        if source.region_id != neighbor.region_id
            || !two_to_three_geometry_is_admissible(
                source.vertex_indices,
                neighbor.vertex_indices,
                |index| topology.nodes[index as usize].predicate_point(),
            )
            .map_err(|failure| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    format!("2-to-3 reconnection predicate failed: {failure:?}"),
                )
            })?
        {
            continue;
        }
        if let Ok(candidate) = two_to_three_face_flip_candidate(
            local(topology, source_index),
            local(topology, neighbor_index as usize),
        ) {
            candidates.push(candidate);
        }
    }
    Ok(candidates)
}

fn three_to_two_candidates(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
) -> Result<Vec<LocalTetrahedronFlipCandidate>, DelaunayVolumeSliverError> {
    let vertices = topology.tetrahedra[source_index].vertex_indices;
    let mut candidates = Vec::new();
    for left in 0..4 {
        for right in (left + 1)..4 {
            let edge = [vertices[left], vertices[right]];
            let star = topology.incidence.vertex_stars[edge[0] as usize]
                .iter()
                .copied()
                .filter(|tetrahedron| {
                    topology.incidence.vertex_stars[edge[1] as usize]
                        .binary_search(tetrahedron)
                        .is_ok()
                })
                .collect::<Vec<_>>();
            let [first, second, third] = star.as_slice() else {
                continue;
            };
            let tetrahedra = [
                local(topology, *first as usize),
                local(topology, *second as usize),
                local(topology, *third as usize),
            ];
            if tetrahedra.iter().any(|tetrahedron| {
                topology.tetrahedra[tetrahedron.tetrahedron_id as usize].region_id
                    != topology.tetrahedra[source_index].region_id
            }) {
                continue;
            }
            let Ok(candidate) = three_to_two_edge_flip_candidate(tetrahedra, edge) else {
                continue;
            };
            let face = candidate
                .shared_face
                .expect("3-to-2 candidate has a ring face");
            let shared_edge = candidate.shared_edge.expect("3-to-2 candidate has an edge");
            let left_sign = orient3d_constructed([
                topology.nodes[face[0] as usize].predicate_point(),
                topology.nodes[face[1] as usize].predicate_point(),
                topology.nodes[face[2] as usize].predicate_point(),
                topology.nodes[shared_edge[0] as usize].predicate_point(),
            ])
            .map_err(|failure| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    format!("3-to-2 reconnection predicate failed: {failure:?}"),
                )
            })?;
            let right_sign = orient3d_constructed([
                topology.nodes[face[0] as usize].predicate_point(),
                topology.nodes[face[1] as usize].predicate_point(),
                topology.nodes[face[2] as usize].predicate_point(),
                topology.nodes[shared_edge[1] as usize].predicate_point(),
            ])
            .map_err(|failure| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    format!("3-to-2 reconnection predicate failed: {failure:?}"),
                )
            })?;
            if left_sign != PredicateSign::Zero
                && right_sign != PredicateSign::Zero
                && left_sign != right_sign
            {
                candidates.push(candidate);
            }
        }
    }
    Ok(candidates)
}

/// Reconnects the complete closed star of an interior edge. For a ring of `n`
/// vertices, each deterministic fan triangulation replaces `n` incident
/// tetrahedra by `2(n - 2)` tetrahedra. This includes the common 4-to-4 swap
/// that cannot be expressed as one 2-to-3 or 3-to-2 move.
fn edge_star_candidates(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
) -> Result<Vec<LocalTetrahedronFlipCandidate>, DelaunayVolumeSliverError> {
    let vertices = topology.tetrahedra[source_index].vertex_indices;
    let mut candidates = Vec::new();
    for left in 0..4 {
        for right in (left + 1)..4 {
            let mut edge = [vertices[left], vertices[right]];
            edge.sort_unstable();
            if topology.incidence.boundary_facets.iter().any(|facet| {
                facet.vertex_indices.contains(&edge[0]) && facet.vertex_indices.contains(&edge[1])
            }) {
                continue;
            }
            let star = edge_star(topology, edge);
            if star.len() < 4
                || star.iter().any(|tetrahedron| {
                    topology.tetrahedra[*tetrahedron as usize].region_id
                        != topology.tetrahedra[source_index].region_id
                })
            {
                continue;
            }
            let Some(ring) = ordered_edge_ring(topology, edge, &star) else {
                continue;
            };
            for root in 0..ring.len() {
                let rotated = (0..ring.len())
                    .map(|offset| ring[(root + offset) % ring.len()])
                    .collect::<Vec<_>>();
                let triangles = (1..(rotated.len() - 1))
                    .map(|index| [rotated[0], rotated[index], rotated[index + 1]])
                    .collect::<Vec<_>>();
                if !triangles_separate_edge(topology, edge, &triangles)? {
                    continue;
                }
                let created_tetrahedra = triangles
                    .into_iter()
                    .flat_map(|triangle| {
                        [
                            [edge[0], triangle[0], triangle[1], triangle[2]],
                            [edge[1], triangle[0], triangle[2], triangle[1]],
                        ]
                    })
                    .collect::<Vec<_>>();
                candidates.push(LocalTetrahedronFlipCandidate {
                    kind: LocalTetrahedronFlipKind::EdgeStar,
                    removed_tetrahedron_ids: star.clone(),
                    created_tetrahedra,
                    shared_face: None,
                    shared_edge: Some(edge),
                });
            }
        }
    }
    Ok(candidates)
}

fn edge_star(topology: &DelaunayVolumeTopology, edge: [u32; 2]) -> Vec<u32> {
    topology.incidence.vertex_stars[edge[0] as usize]
        .iter()
        .copied()
        .filter(|tetrahedron| {
            topology.incidence.vertex_stars[edge[1] as usize]
                .binary_search(tetrahedron)
                .is_ok()
        })
        .collect()
}

fn ordered_edge_ring(
    topology: &DelaunayVolumeTopology,
    edge: [u32; 2],
    star: &[u32],
) -> Option<Vec<u32>> {
    let mut adjacency = BTreeMap::<u32, BTreeSet<u32>>::new();
    for tetrahedron in star {
        let opposite = topology.tetrahedra[*tetrahedron as usize]
            .vertex_indices
            .into_iter()
            .filter(|vertex| !edge.contains(vertex))
            .collect::<Vec<_>>();
        let [left, right] = opposite.as_slice() else {
            return None;
        };
        adjacency.entry(*left).or_default().insert(*right);
        adjacency.entry(*right).or_default().insert(*left);
    }
    if adjacency.len() != star.len() || adjacency.values().any(|neighbors| neighbors.len() != 2) {
        return None;
    }
    let start = *adjacency.keys().next()?;
    let mut ring = vec![start];
    let mut previous = None;
    let mut current = start;
    loop {
        let next = adjacency[&current]
            .iter()
            .copied()
            .find(|neighbor| Some(*neighbor) != previous)?;
        if next == start {
            break;
        }
        if ring.contains(&next) {
            return None;
        }
        ring.push(next);
        previous = Some(current);
        current = next;
    }
    (ring.len() == adjacency.len()).then_some(ring)
}

fn triangles_separate_edge(
    topology: &DelaunayVolumeTopology,
    edge: [u32; 2],
    triangles: &[[u32; 3]],
) -> Result<bool, DelaunayVolumeSliverError> {
    for triangle in triangles {
        let sign = |apex| {
            orient3d_constructed([
                topology.nodes[triangle[0] as usize].predicate_point(),
                topology.nodes[triangle[1] as usize].predicate_point(),
                topology.nodes[triangle[2] as usize].predicate_point(),
                topology.nodes[apex as usize].predicate_point(),
            ])
        };
        let left = sign(edge[0]).map_err(|failure| {
            error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                format!("edge-star reconnection predicate failed: {failure:?}"),
            )
        })?;
        let right = sign(edge[1]).map_err(|failure| {
            error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                format!("edge-star reconnection predicate failed: {failure:?}"),
            )
        })?;
        if left == PredicateSign::Zero || right == PredicateSign::Zero || left == right {
            return Ok(false);
        }
    }
    Ok(true)
}

fn apply_candidate(
    topology: &DelaunayVolumeTopology,
    candidate: &LocalTetrahedronFlipCandidate,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeSliverError> {
    let removed = candidate
        .removed_tetrahedron_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let old_local = candidate
        .removed_tetrahedron_ids
        .iter()
        .map(|index| topology.tetrahedra[*index as usize].vertex_indices)
        .collect::<Vec<_>>();
    if local_tetrahedron_boundary_faces(&old_local)
        != local_tetrahedron_boundary_faces(&candidate.created_tetrahedra)
    {
        return Ok(None);
    }
    let region = topology.tetrahedra[candidate.removed_tetrahedron_ids[0] as usize]
        .region_id
        .clone();
    let mut tetrahedra = topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed.contains(&(*index as u32)))
        .map(|(_, tetrahedron)| (tetrahedron.vertex_indices, tetrahedron.region_id.clone()))
        .collect::<Vec<_>>();
    tetrahedra.extend(
        candidate
            .created_tetrahedra
            .iter()
            .copied()
            .map(|tetrahedron| (tetrahedron, region.clone())),
    );
    match build_delaunay_volume_topology_with_regions(
        topology.nodes.clone(),
        tetrahedra,
        options.insertion.topology,
        cancellation,
    ) {
        Ok(topology) => Ok(Some(topology)),
        Err(failure) => match failure.kind {
            DelaunayTopologyErrorKind::ResourceLimit => Err(error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                failure.to_string(),
            )),
            DelaunayTopologyErrorKind::Cancelled => Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                failure.to_string(),
            )),
            _ => Ok(None),
        },
    }
}

fn local(topology: &DelaunayVolumeTopology, index: usize) -> LocalTetrahedron {
    LocalTetrahedron {
        tetrahedron_id: index as u32,
        node_ids: topology.tetrahedra[index].vertex_indices,
    }
}

fn identities(topology: &DelaunayVolumeTopology, vertices: [u32; 4]) -> [StableDigest; 4] {
    vertices.map(|vertex| topology.nodes[vertex as usize].identity)
}

fn candidate_key(
    candidate: &LocalTetrahedronFlipCandidate,
) -> (LocalTetrahedronFlipKind, Vec<u32>, Vec<[u32; 4]>) {
    (
        candidate.kind,
        candidate.removed_tetrahedron_ids.clone(),
        candidate.created_tetrahedra.clone(),
    )
}

fn candidate_geometry_key(
    topology: &DelaunayVolumeTopology,
    candidate: &LocalTetrahedronFlipCandidate,
) -> Vec<[[u64; 3]; 4]> {
    let mut key = candidate
        .created_tetrahedra
        .iter()
        .map(|vertices| {
            let mut tetrahedron = vertices.map(|vertex| {
                topology.nodes[vertex as usize]
                    .coordinates_m
                    .map(super::super::volume_quality::ordered_float_key)
            });
            tetrahedron.sort_unstable();
            tetrahedron
        })
        .collect::<Vec<_>>();
    key.sort_unstable();
    key
}

fn candidate_minimum_jacobian(
    topology: &DelaunayVolumeTopology,
    candidate: &LocalTetrahedronFlipCandidate,
    metric: runmat_meshing_size::metric::MetricTensor3,
) -> f64 {
    candidate
        .created_tetrahedra
        .iter()
        .filter_map(|vertices| {
            super::super::volume_quality::evaluate_tetrahedron_geometry(
                vertices.map(|vertex| topology.nodes[vertex as usize].coordinates_m),
                metric,
            )
            .map(|quality| quality.metric_scaled_jacobian)
        })
        .min_by(f64::total_cmp)
        .unwrap_or(0.0)
}

fn reject(counts: &mut BTreeMap<&'static str, u64>, reason: &'static str) {
    *counts.entry(reason).or_default() += 1;
}
