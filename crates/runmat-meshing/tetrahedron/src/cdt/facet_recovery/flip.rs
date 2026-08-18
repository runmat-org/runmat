use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    StableDigest,
};

use super::{
    error, node_index, DelaunayFacetRecoveryError, DelaunayFacetRecoveryErrorKind,
    DelaunaySegmentRecovery, DelaunayVolumeTopology, FacetRecoveryWork,
};
use crate::cdt::{
    topology::build_delaunay_volume_topology_with_regions, validate_delaunay_volume_topology,
    DelaunayInsertionErrorKind, DelaunayTopologyErrorKind,
};

pub(super) fn try_recover_facet_with_edge_flip(
    recovery: &DelaunaySegmentRecovery,
    facet: [StableDigest; 3],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayFacetRecoveryError> {
    let topology = &recovery.topology;
    let facet_indices = facet.map(|identity| {
        node_index(topology, identity).ok_or_else(|| {
            invalid_topology(constraint_index, "facet-flip node is missing from topology")
        })
    });
    let [first, second, third] = facet_indices;
    let mut facet_indices = [first? as u32, second? as u32, third? as u32];
    facet_indices.sort_unstable();
    let protected_edges = protected_segment_edges(recovery, constraint_index)?;
    let edge_stars = edge_stars(topology, constraint_index, work)?;

    for (edge, tetrahedra) in edge_stars {
        if tetrahedra.len() != 3 || protected_edges.contains(&edge) {
            continue;
        }
        let ring = tetrahedra
            .iter()
            .flat_map(|index| topology.tetrahedra[*index as usize].vertex_indices)
            .filter(|vertex| !edge.contains(vertex))
            .collect::<BTreeSet<_>>();
        if ring.iter().copied().collect::<Vec<_>>() != facet_indices {
            continue;
        }
        work.flip_attempt(constraint_index)?;
        let regions = tetrahedra
            .iter()
            .map(|index| &topology.tetrahedra[*index as usize].region_id)
            .collect::<BTreeSet<_>>();
        if regions.len() != 1 || !opposite_sides(topology, edge, facet_indices, constraint_index)? {
            continue;
        }
        let region = topology.tetrahedra[tetrahedra[0] as usize]
            .region_id
            .clone();
        let removed = tetrahedra.into_iter().collect::<BTreeSet<_>>();
        let mut replacements = topology
            .tetrahedra
            .iter()
            .enumerate()
            .filter(|(index, _)| !removed.contains(&(*index as u32)))
            .map(|(_, tetrahedron)| (tetrahedron.vertex_indices, tetrahedron.region_id.clone()))
            .collect::<Vec<_>>();
        replacements.push((
            [
                edge[0],
                facet_indices[0],
                facet_indices[1],
                facet_indices[2],
            ],
            region.clone(),
        ));
        replacements.push((
            [
                edge[1],
                facet_indices[0],
                facet_indices[2],
                facet_indices[1],
            ],
            region,
        ));
        let candidate = match build_delaunay_volume_topology_with_regions(
            topology.nodes.clone(),
            replacements,
            work.options.segment_recovery.insertion.topology,
            work.cancellation,
        ) {
            Ok(candidate) => candidate,
            Err(candidate_error) => match candidate_error.kind {
                DelaunayTopologyErrorKind::ResourceLimit => {
                    return Err(resource_or_cancelled(
                        DelaunayFacetRecoveryErrorKind::ResourceLimit,
                        constraint_index,
                        candidate_error.to_string(),
                    ));
                }
                DelaunayTopologyErrorKind::Cancelled => {
                    return Err(resource_or_cancelled(
                        DelaunayFacetRecoveryErrorKind::Cancelled,
                        constraint_index,
                        candidate_error.to_string(),
                    ));
                }
                _ => continue,
            },
        };
        match validate_delaunay_volume_topology(
            &candidate,
            work.options.segment_recovery.insertion,
            work.cancellation,
        ) {
            Ok(()) => return Ok(Some(candidate)),
            Err(validation) => match validation.kind {
                DelaunayInsertionErrorKind::ResourceLimit => {
                    return Err(resource_or_cancelled(
                        DelaunayFacetRecoveryErrorKind::ResourceLimit,
                        constraint_index,
                        validation.to_string(),
                    ));
                }
                DelaunayInsertionErrorKind::Cancelled => {
                    return Err(resource_or_cancelled(
                        DelaunayFacetRecoveryErrorKind::Cancelled,
                        constraint_index,
                        validation.to_string(),
                    ));
                }
                _ => continue,
            },
        }
    }
    Ok(None)
}

fn edge_stars(
    topology: &DelaunayVolumeTopology,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<BTreeMap<[u32; 2], Vec<u32>>, DelaunayFacetRecoveryError> {
    let mut stars = BTreeMap::<[u32; 2], Vec<u32>>::new();
    for (tetrahedron_index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        for first in 0..4 {
            for second in first + 1..4 {
                work.search_step(constraint_index)?;
                let mut edge = [
                    tetrahedron.vertex_indices[first],
                    tetrahedron.vertex_indices[second],
                ];
                edge.sort_unstable();
                stars
                    .entry(edge)
                    .or_default()
                    .push(tetrahedron_index as u32);
            }
        }
    }
    Ok(stars)
}

fn protected_segment_edges(
    recovery: &DelaunaySegmentRecovery,
    constraint_index: u32,
) -> Result<BTreeSet<[u32; 2]>, DelaunayFacetRecoveryError> {
    recovery
        .segments
        .iter()
        .flat_map(|segment| segment.nodes.windows(2))
        .map(|pair| {
            let mut edge = [
                node_index(&recovery.topology, pair[0].identity).ok_or_else(|| {
                    invalid_topology(constraint_index, "recovered segment node is missing")
                })? as u32,
                node_index(&recovery.topology, pair[1].identity).ok_or_else(|| {
                    invalid_topology(constraint_index, "recovered segment node is missing")
                })? as u32,
            ];
            edge.sort_unstable();
            Ok(edge)
        })
        .collect()
}

fn opposite_sides(
    topology: &DelaunayVolumeTopology,
    edge: [u32; 2],
    facet: [u32; 3],
    constraint_index: u32,
) -> Result<bool, DelaunayFacetRecoveryError> {
    let coordinates = |index: u32| topology.nodes[index as usize].coordinates_m;
    let left = orient3d([facet[0], facet[1], facet[2], edge[0]].map(coordinates)).map_err(
        |predicate| {
            error(
                DelaunayFacetRecoveryErrorKind::InvalidTopology,
                Some(constraint_index),
                format!("facet-flip orientation predicate failed: {predicate:?}"),
            )
        },
    )?;
    let right = orient3d([facet[0], facet[1], facet[2], edge[1]].map(coordinates)).map_err(
        |predicate| {
            error(
                DelaunayFacetRecoveryErrorKind::InvalidTopology,
                Some(constraint_index),
                format!("facet-flip orientation predicate failed: {predicate:?}"),
            )
        },
    )?;
    Ok(matches!(
        (left, right),
        (PredicateSign::Positive, PredicateSign::Negative)
            | (PredicateSign::Negative, PredicateSign::Positive)
    ))
}

fn invalid_topology(constraint_index: u32, reason: &'static str) -> DelaunayFacetRecoveryError {
    error(
        DelaunayFacetRecoveryErrorKind::InvalidTopology,
        Some(constraint_index),
        reason,
    )
}

fn resource_or_cancelled(
    kind: DelaunayFacetRecoveryErrorKind,
    constraint_index: u32,
    reason: String,
) -> DelaunayFacetRecoveryError {
    error(kind, Some(constraint_index), reason)
}
