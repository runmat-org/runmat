use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{orient3d_constructed, PredicateSign},
    MeshingCancellationSignal, StableDigest,
};

#[cfg(test)]
use super::admission::{candidate_quality_is_admissible, evaluate_canonical_candidate_unranked};
#[cfg(test)]
use super::relocation::relocation_candidates;
use super::work::OptimizationWork;
use super::{
    error, DelaunayVolumeReconnectionKind, DelaunayVolumeSliverError,
    DelaunayVolumeSliverErrorKind, DelaunayVolumeSliverMutation, DelaunayVolumeSliverOptions,
    DelaunayVolumeSliverReconnection, SliverSearchPolicy,
};
use crate::{
    cdt::{topology::VolumeTopologyRead, DelaunayTetrahedronQuality},
    reconnect::{
        three_to_two_edge_flip_candidate, two_to_three_face_flip_candidate,
        two_to_three_geometry_is_admissible, LocalTetrahedron, LocalTetrahedronFlipCandidate,
        LocalTetrahedronFlipKind,
    },
};

#[cfg(test)]
use super::retained_state::{connectivity, QualityPatch, TopologyPatch};
mod geometry;
#[cfg(test)]
mod lookahead;
pub(super) mod working;
#[cfg(test)]
use connectivity::connectivity_delta;
use geometry::local_candidates;
#[cfg(test)]
use lookahead::bounded_reconnection_lookahead;

#[cfg(test)]
pub(in crate::cdt) struct ReconnectionCandidate {
    pub(in crate::cdt) topology: DelaunayVolumeTopology,
    pub(in crate::cdt) quality: DelaunayVolumeQuality,
    pub(in crate::cdt) mutations: Vec<DelaunayVolumeSliverMutation>,
}

#[cfg(test)]
pub(in crate::cdt) struct ReconnectionSearch {
    pub(in crate::cdt) candidate: Option<ReconnectionCandidate>,
    pub(in crate::cdt) rejection_counts: BTreeMap<&'static str, u64>,
}

#[cfg(test)]
struct LookaheadSeed {
    patch: TopologyPatch,
    quality: QualityPatch,
    mutations: Vec<DelaunayVolumeSliverMutation>,
}

#[cfg(test)]
struct LookaheadFrontier<'a> {
    protected_faces: &'a [[StableDigest; 3]],
    seeds: Vec<LookaheadSeed>,
}

#[derive(Clone, Copy)]
pub(super) struct ReconnectionMode {
    pub(super) allow_lookahead: bool,
    pub(super) maximum_candidates: Option<u64>,
}

#[cfg(test)]
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
        let Some(quality) = evaluate_canonical_candidate_unranked(
            &topology,
            input,
            &protected_faces,
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
                    topology: topology.into_inner(),
                    quality,
                    mutations: vec![mutation],
                }),
                rejection_counts,
            });
        }
        if allow_lookahead {
            lookahead_seeds.push(LookaheadSeed {
                patch: TopologyPatch::between(input.topology, &topology)?,
                quality: QualityPatch::between(input.quality, &quality)?,
                mutations: vec![mutation],
            });
        }
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

fn reconnection_mutation(
    topology: &(impl VolumeTopologyRead + ?Sized),
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
            .map(|index| identities(topology, topology.cell(*index).vertex_indices))
            .collect(),
        created_tetrahedron_node_identities: candidate
            .created_tetrahedra
            .iter()
            .map(|vertices| identities(topology, *vertices))
            .collect(),
    })
}

#[cfg(test)]
fn apply_candidate(
    topology: &DelaunayVolumeTopology,
    candidate: &LocalTetrahedronFlipCandidate,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<CanonicalVolumeTopology>, DelaunayVolumeSliverError> {
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
    match build_canonical_volume_topology(
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

fn identities(
    topology: &(impl VolumeTopologyRead + ?Sized),
    vertices: [u32; 4],
) -> [StableDigest; 4] {
    vertices.map(|vertex| topology.node(vertex).identity)
}

fn reject(counts: &mut BTreeMap<&'static str, u64>, reason: &'static str) {
    *counts.entry(reason).or_default() += 1;
}

#[cfg(test)]
use crate::cdt::{
    topology::{build_canonical_volume_topology, CanonicalVolumeTopology},
    DelaunayTopologyErrorKind, DelaunayVolumeQuality, DelaunayVolumeRefinementInput,
    DelaunayVolumeTopology,
};

#[cfg(test)]
use crate::reconnect::local_tetrahedron_boundary_faces;
