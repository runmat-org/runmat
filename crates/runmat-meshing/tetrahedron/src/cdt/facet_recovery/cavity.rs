//! Bounded recovery for a missing facet support triangle whose surrounding edge stars form a
//! closed local cavity. Exact orientation predicates partition the existing nodes; recovery does
//! not invent rounded points on the authoritative facet plane.

use std::collections::BTreeSet;

use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    StableDigest,
};

use super::{
    error, node_index, DelaunayFacetRecoveryError, DelaunayFacetRecoveryErrorKind,
    DelaunayRecoveredFacetTriangle, DelaunaySegmentRecovery, DelaunayVolumeTopology,
    FacetRecoveryWork,
};
use crate::{
    cavity::constrained::{
        retriangulate_constrained_cavity_from_nodes, ConstrainedCavity, ConstrainedCavityNode,
        ConstrainedCavityRefillBudget, ConstrainedCavityRefillError,
        ConstrainedCavityRefillOptions,
    },
    cdt::{
        insertion::validate_constrained_delaunay_volume_topology,
        topology::build_delaunay_volume_topology_with_regions, DelaunayInsertionErrorKind,
        DelaunayTopologyErrorKind,
    },
};

mod sides;
pub(super) mod star;
mod steiner;

use sides::side_cavities;
use star::star_refill;
pub(super) use steiner::try_recover_facet_with_edge_star_cavity;

pub(super) enum FacetCavityAttempt {
    Recovered(DelaunayVolumeTopology),
    NeedsSteiner(Vec<ConstrainedCavity>),
    Unavailable,
}

pub(super) struct BoundaryFace {
    pub(super) nodes: [u32; 3],
    pub(super) interior_node: u32,
    pub(super) outside_tetrahedron: Option<u32>,
}

pub(super) fn try_recover_facet_cavity_once(
    recovery: &DelaunaySegmentRecovery,
    facet: [StableDigest; 3],
    protected_facets: &[DelaunayRecoveredFacetTriangle],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<FacetCavityAttempt, DelaunayFacetRecoveryError> {
    let facet_nodes = facet_indices(&recovery.topology, facet, constraint_index)?;
    let signs = plane_signs(&recovery.topology, facet_nodes, constraint_index, work)?;
    let mut removed = edge_star_seed(&recovery.topology, facet_nodes, constraint_index, work)?;
    if removed.is_empty() {
        return Ok(FacetCavityAttempt::Unavailable);
    }
    expand_crossing_boundary(
        &recovery.topology,
        &signs,
        &mut removed,
        constraint_index,
        work,
    )?;
    let boundary = cavity_boundary(&recovery.topology, &removed, constraint_index, work)?;
    let Some((positive, negative)) = side_cavities(
        &recovery.topology,
        facet_nodes,
        &signs,
        &removed,
        &boundary,
        constraint_index,
    )?
    else {
        return Ok(FacetCavityAttempt::Unavailable);
    };
    let cavity_nodes = removed
        .iter()
        .flat_map(|tetrahedron| recovery.topology.tetrahedra[*tetrahedron as usize].vertex_indices)
        .collect::<BTreeSet<_>>();
    if cavity_nodes.len() as u64 > work.options.maximum_cavity_nodes {
        return Err(resource_or_cancelled(
            DelaunayFacetRecoveryErrorKind::ResourceLimit,
            constraint_index,
            "facet cavity node limit exceeded".to_owned(),
        ));
    }
    let nodes = cavity_nodes
        .into_iter()
        .map(|node_id| ConstrainedCavityNode {
            node_id,
            coordinates_m: recovery.topology.nodes[node_id as usize].coordinates_m,
        })
        .collect::<Vec<_>>();
    let positive_refill = refill_side(
        &positive,
        &nodes,
        &recovery.topology,
        constraint_index,
        work,
    )?;
    let negative_refill = refill_side(
        &negative,
        &nodes,
        &recovery.topology,
        constraint_index,
        work,
    )?;
    let (positive_refill, negative_refill) = match (positive_refill, negative_refill) {
        (Some(positive_refill), Some(negative_refill)) => (positive_refill, negative_refill),
        (positive_refill, negative_refill) => {
            let mut cavities = Vec::new();
            if positive_refill.is_none() {
                cavities.push(positive);
            }
            if negative_refill.is_none() {
                cavities.push(negative);
            }
            return Ok(FacetCavityAttempt::NeedsSteiner(cavities));
        }
    };

    let mut tetrahedra = recovery
        .topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed.contains(&(*index as u32)))
        .map(|(_, tetrahedron)| (tetrahedron.vertex_indices, tetrahedron.region_id.clone()))
        .collect::<Vec<_>>();
    tetrahedra.extend(
        positive_refill
            .into_iter()
            .chain(negative_refill)
            .map(|tetrahedron| (tetrahedron, None)),
    );
    let candidate = match build_delaunay_volume_topology_with_regions(
        recovery.topology.nodes.clone(),
        tetrahedra,
        work.options.segment_recovery.insertion.topology,
        work.cancellation,
    ) {
        Ok(candidate) => candidate,
        Err(failure) => match failure.kind {
            DelaunayTopologyErrorKind::ResourceLimit => {
                return Err(resource_or_cancelled(
                    DelaunayFacetRecoveryErrorKind::ResourceLimit,
                    constraint_index,
                    failure.to_string(),
                ));
            }
            DelaunayTopologyErrorKind::Cancelled => {
                return Err(resource_or_cancelled(
                    DelaunayFacetRecoveryErrorKind::Cancelled,
                    constraint_index,
                    failure.to_string(),
                ));
            }
            _ => return Ok(FacetCavityAttempt::Unavailable),
        },
    };
    let mut protected = protected_facets
        .iter()
        .map(|triangle| canonical_face(triangle.node_identities))
        .collect::<Vec<_>>();
    protected.push(canonical_face(facet));
    protected.sort_unstable();
    protected.dedup();
    match validate_constrained_delaunay_volume_topology(
        &candidate,
        &protected,
        work.options.segment_recovery.insertion,
        work.cancellation,
    ) {
        Ok(()) => {}
        Err(failure) => match failure.kind {
            DelaunayInsertionErrorKind::ResourceLimit => {
                return Err(resource_or_cancelled(
                    DelaunayFacetRecoveryErrorKind::ResourceLimit,
                    constraint_index,
                    failure.to_string(),
                ));
            }
            DelaunayInsertionErrorKind::Cancelled => {
                return Err(resource_or_cancelled(
                    DelaunayFacetRecoveryErrorKind::Cancelled,
                    constraint_index,
                    failure.to_string(),
                ));
            }
            _ => return Ok(FacetCavityAttempt::Unavailable),
        },
    }
    let mut candidate_recovery = recovery.clone();
    candidate_recovery.topology = candidate.clone();
    if recovered_segments_exist(&candidate_recovery) {
        Ok(FacetCavityAttempt::Recovered(candidate))
    } else {
        Ok(FacetCavityAttempt::Unavailable)
    }
}

pub(in crate::cdt::facet_recovery) fn refill_side(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    topology: &DelaunayVolumeTopology,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<Vec<[u32; 4]>>, DelaunayFacetRecoveryError> {
    if let Some(star) = star_refill(cavity, topology, constraint_index, work)? {
        return Ok(Some(star));
    }
    let refill_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..ConstrainedCavityRefillOptions::default()
    };
    let budget = ConstrainedCavityRefillBudget {
        maximum_nodes: work.options.maximum_cavity_nodes,
        maximum_boundary_faces: work.options.maximum_cavity_boundary_faces,
        maximum_candidate_tetrahedra: work.options.maximum_cavity_candidate_tetrahedra,
        maximum_candidate_evaluations: work.options.maximum_cavity_candidate_evaluations,
        maximum_search_attempts: work.options.maximum_cavity_exact_cover_attempts,
        maximum_expansion_rounds: work.options.maximum_cavity_expansion_rounds,
        cancellation_check_interval: work
            .options
            .segment_recovery
            .constraints
            .cancellation_check_interval,
    };
    match retriangulate_constrained_cavity_from_nodes(
        cavity,
        nodes,
        refill_options,
        budget,
        work.cancellation,
    ) {
        Ok(refill) => Ok(refill.map(|refill| {
            refill
                .tetrahedra
                .into_iter()
                .map(|tetrahedron| tetrahedron.node_ids)
                .collect()
        })),
        Err(ConstrainedCavityRefillError::ResourceLimit { reason }) => Err(resource_or_cancelled(
            DelaunayFacetRecoveryErrorKind::ResourceLimit,
            constraint_index,
            reason,
        )),
        Err(ConstrainedCavityRefillError::Cancelled) => Err(resource_or_cancelled(
            DelaunayFacetRecoveryErrorKind::Cancelled,
            constraint_index,
            "cancelled".to_owned(),
        )),
        Err(failure) => Err(invalid_topology(
            constraint_index,
            format!("facet cavity refill failed: {failure:?}"),
        )),
    }
}

fn recovered_segments_exist(recovery: &DelaunaySegmentRecovery) -> bool {
    // The caller independently validated the evidence before mutation. Nodes are immutable here,
    // so candidate admission only needs to prove that every protected chain edge survived.
    recovery.segments.iter().all(|segment| {
        segment.nodes.windows(2).all(|pair| {
            let Some(first) = node_index(&recovery.topology, pair[0].identity) else {
                return false;
            };
            let Some(second) = node_index(&recovery.topology, pair[1].identity) else {
                return false;
            };
            recovery.topology.incidence.vertex_stars[first]
                .iter()
                .any(|tetrahedron| {
                    recovery.topology.tetrahedra[*tetrahedron as usize]
                        .vertex_indices
                        .contains(&(second as u32))
                })
        })
    })
}

fn facet_indices(
    topology: &DelaunayVolumeTopology,
    facet: [StableDigest; 3],
    constraint_index: u32,
) -> Result<[u32; 3], DelaunayFacetRecoveryError> {
    facet
        .map(|identity| {
            node_index(topology, identity)
                .map(|index| index as u32)
                .ok_or_else(|| invalid_topology(constraint_index, "facet cavity node is missing"))
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map(|indices| [indices[0], indices[1], indices[2]])
}

fn plane_signs(
    topology: &DelaunayVolumeTopology,
    facet: [u32; 3],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Vec<PredicateSign>, DelaunayFacetRecoveryError> {
    let mut signs = Vec::with_capacity(topology.nodes.len());
    for node in &topology.nodes {
        work.search_step(constraint_index)?;
        signs.push(
            orient3d([
                topology.nodes[facet[0] as usize].coordinates_m,
                topology.nodes[facet[1] as usize].coordinates_m,
                topology.nodes[facet[2] as usize].coordinates_m,
                node.coordinates_m,
            ])
            .map_err(|failure| {
                invalid_topology(
                    constraint_index,
                    format!("facet cavity predicate failed: {failure:?}"),
                )
            })?,
        );
    }
    Ok(signs)
}

fn edge_star_seed(
    topology: &DelaunayVolumeTopology,
    facet: [u32; 3],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<BTreeSet<u32>, DelaunayFacetRecoveryError> {
    let mut removed = BTreeSet::new();
    for (index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        work.cavity_step(constraint_index)?;
        if facet
            .iter()
            .filter(|node| tetrahedron.vertex_indices.contains(node))
            .count()
            >= 2
        {
            removed.insert(index as u32);
            if removed.len() as u64 > work.options.maximum_cavity_tetrahedra {
                return Err(resource_or_cancelled(
                    DelaunayFacetRecoveryErrorKind::ResourceLimit,
                    constraint_index,
                    "facet cavity tetrahedron limit exceeded".to_owned(),
                ));
            }
        }
    }
    Ok(removed)
}

fn expand_crossing_boundary(
    topology: &DelaunayVolumeTopology,
    signs: &[PredicateSign],
    removed: &mut BTreeSet<u32>,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<(), DelaunayFacetRecoveryError> {
    loop {
        let boundary = cavity_boundary(topology, removed, constraint_index, work)?;
        let crossing = boundary.into_iter().find(|face| {
            let face_signs = face.nodes.map(|node| signs[node as usize]);
            face_signs.contains(&PredicateSign::Positive)
                && face_signs.contains(&PredicateSign::Negative)
        });
        let Some(crossing) = crossing else {
            return Ok(());
        };
        let Some(outside) = crossing.outside_tetrahedron else {
            return Err(invalid_topology(
                constraint_index,
                "facet plane crosses the admitted topology hull",
            ));
        };
        work.cavity_step(constraint_index)?;
        removed.insert(outside);
        if removed.len() as u64 > work.options.maximum_cavity_tetrahedra {
            return Err(resource_or_cancelled(
                DelaunayFacetRecoveryErrorKind::ResourceLimit,
                constraint_index,
                "facet cavity tetrahedron limit exceeded".to_owned(),
            ));
        }
    }
}

fn cavity_boundary(
    topology: &DelaunayVolumeTopology,
    removed: &BTreeSet<u32>,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Vec<BoundaryFace>, DelaunayFacetRecoveryError> {
    let mut boundary = Vec::new();
    for tetrahedron_index in removed {
        let tetrahedron = &topology.tetrahedra[*tetrahedron_index as usize];
        for opposite in 0..4 {
            work.search_step(constraint_index)?;
            if tetrahedron.neighbors[opposite].is_some_and(|neighbor| removed.contains(&neighbor)) {
                continue;
            }
            let mut nodes = [0; 3];
            let mut cursor = 0;
            for (index, node) in tetrahedron.vertex_indices.iter().enumerate() {
                if index != opposite {
                    nodes[cursor] = *node;
                    cursor += 1;
                }
            }
            boundary.push(BoundaryFace {
                nodes,
                interior_node: tetrahedron.vertex_indices[opposite],
                outside_tetrahedron: tetrahedron.neighbors[opposite],
            });
            if boundary.len() as u64 > work.options.maximum_cavity_boundary_faces {
                return Err(resource_or_cancelled(
                    DelaunayFacetRecoveryErrorKind::ResourceLimit,
                    constraint_index,
                    "facet cavity boundary-face limit exceeded".to_owned(),
                ));
            }
        }
    }
    Ok(boundary)
}

fn canonical_face(mut face: [StableDigest; 3]) -> [StableDigest; 3] {
    face.sort_unstable();
    face
}

pub(super) fn invalid_topology(
    constraint_index: u32,
    reason: impl Into<String>,
) -> DelaunayFacetRecoveryError {
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
