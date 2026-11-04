//! Bounded recovery for a missing facet support triangle whose intersected tetrahedra form a
//! closed local cavity. Exact orientation predicates partition the existing nodes; recovery does
//! not invent rounded points on the authoritative facet plane.

use std::collections::{BTreeMap, BTreeSet};

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
        retriangulate_constrained_cavity_from_nodes_with_diagnostics, ConstrainedCavity,
        ConstrainedCavityNode, ConstrainedCavityRefillBudget, ConstrainedCavityRefillError,
        ConstrainedCavityRefillOptions, ConstrainedCavityValidationError,
    },
    cdt::{
        insertion::validate_constrained_delaunay_volume_topology,
        topology::build_delaunay_volume_topology_with_regions, DelaunayInsertionErrorKind,
        DelaunayTopologyErrorKind,
    },
};

mod footprint;
mod sides;
mod simultaneous;
pub(super) mod star;
mod steiner;

use footprint::facet_footprint_crosses_face;
pub(super) use footprint::segment_hits_triangle_interior;
use sides::{side_cavities, SideCavityConstruction};
use simultaneous::{refill_cavity_sides, CavityRefillRequest};
use star::star_refill;
pub(super) use steiner::{
    candidate_node, steiner_identity, steiner_lineage, try_insert_candidate_with_conflicts,
    try_recover_facet_with_cavity, CandidateInsertion, CandidateInsertionResult,
};

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
    try_recover_facet_cavity_once_with_cap(
        recovery,
        facet,
        &[facet],
        protected_facets,
        constraint_index,
        work,
    )
}

pub(super) fn try_recover_facet_patch_with_cavity(
    recovery: &DelaunaySegmentRecovery,
    footprint: [StableDigest; 3],
    cap: &[DelaunayRecoveredFacetTriangle],
    protected_facets: &[DelaunayRecoveredFacetTriangle],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayFacetRecoveryError> {
    let cap = cap
        .iter()
        .map(|triangle| triangle.node_identities)
        .collect::<Vec<_>>();
    match try_recover_facet_cavity_once_with_cap(
        recovery,
        footprint,
        &cap,
        protected_facets,
        constraint_index,
        work,
    )? {
        FacetCavityAttempt::Recovered(topology) => Ok(Some(topology)),
        FacetCavityAttempt::NeedsSteiner(_) => {
            work.reject("atomic facet-patch cavity requires Steiner nodes");
            Ok(None)
        }
        FacetCavityAttempt::Unavailable => Ok(None),
    }
}

fn try_recover_facet_cavity_once_with_cap(
    recovery: &DelaunaySegmentRecovery,
    footprint: [StableDigest; 3],
    cap: &[[StableDigest; 3]],
    protected_facets: &[DelaunayRecoveredFacetTriangle],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<FacetCavityAttempt, DelaunayFacetRecoveryError> {
    if cap.is_empty() {
        work.reject("atomic facet-patch cavity received an empty cap");
        return Ok(FacetCavityAttempt::Unavailable);
    }
    let facet_nodes = facet_indices(&recovery.topology, footprint, constraint_index)?;
    let cap_nodes = cap
        .iter()
        .copied()
        .map(|triangle| facet_indices(&recovery.topology, triangle, constraint_index))
        .collect::<Result<Vec<_>, _>>()?;
    let cap_boundary_edges = patch_boundary_edges(&cap_nodes);
    let signs = plane_signs(&recovery.topology, facet_nodes, constraint_index, work)?;
    let mut removed =
        facet_intersection_seed(&recovery.topology, facet_nodes, constraint_index, work)?;
    if removed.is_empty() {
        work.reject("no tetrahedron face crosses the missing facet footprint");
        return Ok(FacetCavityAttempt::Unavailable);
    }
    let mandatory = removed.clone();
    expand_crossing_boundary(
        &recovery.topology,
        facet_nodes,
        &signs,
        &mut removed,
        constraint_index,
        work,
    )?;
    if !regularize_nonmanifold_edge_stars(
        &recovery.topology,
        &mut removed,
        &mandatory,
        &cap_boundary_edges,
        &signs,
        constraint_index,
        work,
    )? {
        let detail = work
            .last_rejection()
            .unwrap_or(
                "intersected tetrahedra have no manifold cavity repair that preserves the facet cap boundary",
            )
            .to_owned();
        let footprint_coordinates = facet_nodes.map(|node| {
            recovery.topology.nodes[node as usize]
                .coordinates_m
                .canonical_value()
        });
        work.reject(format!(
            "{detail}; facet footprint is {footprint_coordinates:?}"
        ));
        return Ok(FacetCavityAttempt::Unavailable);
    }
    let boundary = cavity_boundary(&recovery.topology, &removed, constraint_index, work)?;
    let cavities = match side_cavities(
        &recovery.topology,
        &cap_nodes,
        &removed,
        &boundary,
        constraint_index,
    )? {
        SideCavityConstruction::Ready(cavities) => cavities,
        SideCavityConstruction::Unavailable { reason, open_edge } => {
            let reason = open_edge.map_or(reason.clone(), |edge| {
                format!(
                    "{reason}; {}",
                    edge_star_diagnostic(&recovery.topology, edge, &removed, &signs)
                )
            });
            work.reject(reason);
            return Ok(FacetCavityAttempt::Unavailable);
        }
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
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(
                recovery.topology.nodes[node_id as usize]
                    .coordinates_m
                    .canonical_value(),
            ),
        })
        .collect::<Vec<_>>();
    let mut refills = Vec::with_capacity(cavities.len());
    let mut unfilled = Vec::new();
    for cavity in cavities {
        match refill_side(&cavity, &nodes, &recovery.topology, constraint_index, work)? {
            Some(refill) => refills.push(refill),
            None => unfilled.push(cavity),
        }
    }
    if !unfilled.is_empty() {
        return Ok(FacetCavityAttempt::NeedsSteiner(unfilled));
    }

    let mut tetrahedra = recovery
        .topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed.contains(&(*index as u32)))
        .map(|(_, tetrahedron)| (tetrahedron.vertex_indices, tetrahedron.region_id.clone()))
        .collect::<Vec<_>>();
    tetrahedra.extend(
        refills
            .into_iter()
            .flatten()
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
            _ => {
                work.reject(format!("candidate cavity topology was invalid: {failure}"));
                return Ok(FacetCavityAttempt::Unavailable);
            }
        },
    };
    let mut protected = protected_facets
        .iter()
        .map(|triangle| canonical_face(triangle.node_identities))
        .collect::<Vec<_>>();
    protected.extend(cap.iter().copied().map(canonical_face));
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
            _ => {
                work.reject(format!(
                    "candidate cavity was not constrained-Delaunay: {failure}"
                ));
                return Ok(FacetCavityAttempt::Unavailable);
            }
        },
    }
    let mut candidate_recovery = recovery.clone();
    candidate_recovery.topology = candidate.clone();
    if recovered_segments_exist(&candidate_recovery) {
        Ok(FacetCavityAttempt::Recovered(candidate))
    } else {
        work.reject("candidate cavity removed a previously recovered segment");
        Ok(FacetCavityAttempt::Unavailable)
    }
}

fn edge_star_diagnostic(
    topology: &DelaunayVolumeTopology,
    edge: [u32; 2],
    removed: &BTreeSet<u32>,
    signs: &[PredicateSign],
) -> String {
    let mut total = 0usize;
    let mut selected = 0usize;
    let mut sign_patterns = BTreeMap::<[u8; 2], usize>::new();
    for (tetrahedron_id, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        if !edge
            .iter()
            .all(|node| tetrahedron.vertex_indices.contains(node))
        {
            continue;
        }
        total += 1;
        if removed.contains(&(tetrahedron_id as u32)) {
            selected += 1;
        }
        let mut other = tetrahedron
            .vertex_indices
            .into_iter()
            .filter(|node| !edge.contains(node))
            .map(|node| match signs[node as usize] {
                PredicateSign::Negative => 0,
                PredicateSign::Zero => 1,
                PredicateSign::Positive => 2,
            })
            .collect::<Vec<_>>();
        other.sort_unstable();
        if other.len() == 2 {
            *sign_patterns.entry([other[0], other[1]]).or_default() += 1;
        }
    }
    format!(
        "open-edge star has {selected}/{total} selected tetrahedra and opposite-node sign patterns {sign_patterns:?}"
    )
}

pub(in crate::cdt::facet_recovery) fn refill_side(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    topology: &DelaunayVolumeTopology,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<Vec<[u32; 4]>>, DelaunayFacetRecoveryError> {
    refill_side_with_options(
        cavity,
        nodes,
        topology,
        progressive_facet_recovery_refill_options(),
        constraint_index,
        work,
    )
}

pub(in crate::cdt::facet_recovery) fn refill_side_with_options(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    topology: &DelaunayVolumeTopology,
    refill_options: ConstrainedCavityRefillOptions,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<Vec<[u32; 4]>>, DelaunayFacetRecoveryError> {
    if let Some(star) = star_refill(cavity, topology, constraint_index, work)? {
        return Ok(Some(star));
    }
    refill_side_exact_with_options(cavity, nodes, refill_options, constraint_index, work)
}

pub(in crate::cdt::facet_recovery) fn refill_side_exact_with_options(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    refill_options: ConstrainedCavityRefillOptions,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<Vec<[u32; 4]>>, DelaunayFacetRecoveryError> {
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
    match retriangulate_constrained_cavity_from_nodes_with_diagnostics(
        cavity,
        nodes,
        refill_options,
        budget,
        work.cancellation,
    ) {
        Ok(result) => {
            if result.refill.is_none() {
                work.reject(format!(
                    "facet cavity exact-cover produced no refill: terminal={}, accepted_nodes={}, accepted_node_sample={:?}, candidate_tetrahedra={}, candidate_rejections={:?}, dead_end_reasons={:?}, first_dead_end={:?}",
                    result.terminal_reason,
                    result.accepted_node_count,
                    result.sampled_accepted_nodes,
                    result.candidate_tetrahedron_count,
                    result.candidate_rejection_counts,
                    result.dead_end_reason_counts,
                    result.first_dead_end,
                ));
            }
            Ok(result.refill.map(|refill| {
                refill
                    .tetrahedra
                    .into_iter()
                    .map(|tetrahedron| tetrahedron.node_ids)
                    .collect()
            }))
        }
        Err(ConstrainedCavityRefillError::Accounting(cause))
        | Err(ConstrainedCavityRefillError::Validation(
            ConstrainedCavityValidationError::Accounting(cause),
        )) => Err(resource_or_cancelled(
            DelaunayFacetRecoveryErrorKind::Accounting(cause),
            constraint_index,
            "numerical accounting denied cavity refill".to_owned(),
        )),
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
        Err(failure) => {
            work.reject(format!(
                "facet cavity refill rejected its candidate: {failure:?}"
            ));
            Ok(None)
        }
    }
}

pub(in crate::cdt::facet_recovery) fn facet_recovery_refill_options(
) -> ConstrainedCavityRefillOptions {
    ConstrainedCavityRefillOptions {
        min_volume_m3: f64::MIN_POSITIVE,
        max_aspect_ratio: f64::MAX,
        min_scaled_jacobian: 0.0,
        classification_tolerance: runmat_meshing_core::quality::tolerance::MeshingTolerance {
            absolute_m: runmat_types::physical::Length::new(0.0),
            relative: 0.0,
        },
        ..ConstrainedCavityRefillOptions::default()
    }
}

pub(in crate::cdt::facet_recovery) fn progressive_facet_recovery_refill_options(
) -> ConstrainedCavityRefillOptions {
    ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..ConstrainedCavityRefillOptions::default()
    }
}

pub(super) fn recovered_segments_exist(recovery: &DelaunaySegmentRecovery) -> bool {
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
                topology.nodes[facet[0] as usize]
                    .coordinates_m
                    .canonical_value(),
                topology.nodes[facet[1] as usize]
                    .coordinates_m
                    .canonical_value(),
                topology.nodes[facet[2] as usize]
                    .coordinates_m
                    .canonical_value(),
                node.coordinates_m.canonical_value(),
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

fn facet_intersection_seed(
    topology: &DelaunayVolumeTopology,
    facet: [u32; 3],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<BTreeSet<u32>, DelaunayFacetRecoveryError> {
    let mut removed = BTreeSet::new();
    for (index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        work.cavity_step(constraint_index)?;
        let mut intersects = false;
        for opposite in 0..4 {
            let mut face = [0; 3];
            let mut cursor = 0;
            for (vertex_index, vertex) in tetrahedron.vertex_indices.iter().enumerate() {
                if vertex_index != opposite {
                    face[cursor] = *vertex;
                    cursor += 1;
                }
            }
            if facet_footprint_crosses_face(topology, facet, face, constraint_index, work)? {
                intersects = true;
                break;
            }
        }
        if intersects {
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
    facet: [u32; 3],
    signs: &[PredicateSign],
    removed: &mut BTreeSet<u32>,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<(), DelaunayFacetRecoveryError> {
    loop {
        let boundary = cavity_boundary(topology, removed, constraint_index, work)?;
        let crossing = boundary.into_iter().find_map(|face| {
            let face_signs = face.nodes.map(|node| signs[node as usize]);
            if !face_signs.contains(&PredicateSign::Positive)
                || !face_signs.contains(&PredicateSign::Negative)
            {
                return None;
            }
            match facet_footprint_crosses_face(topology, facet, face.nodes, constraint_index, work)
            {
                Ok(true) => Some(Ok(face)),
                Ok(false) => None,
                Err(failure) => Some(Err(failure)),
            }
        });
        let Some(crossing) = crossing else {
            return Ok(());
        };
        let crossing = crossing?;
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

fn regularize_nonmanifold_edge_stars(
    topology: &DelaunayVolumeTopology,
    removed: &mut BTreeSet<u32>,
    mandatory: &BTreeSet<u32>,
    protected_edges: &BTreeSet<[u32; 2]>,
    plane_signs: &[PredicateSign],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<bool, DelaunayFacetRecoveryError> {
    loop {
        let boundary = cavity_boundary(topology, removed, constraint_index, work)?;
        let mut edge_uses = BTreeMap::<[u32; 2], usize>::new();
        for face in boundary {
            for mut edge in [
                [face.nodes[0], face.nodes[1]],
                [face.nodes[1], face.nodes[2]],
                [face.nodes[2], face.nodes[0]],
            ] {
                edge.sort_unstable();
                *edge_uses.entry(edge).or_default() += 1;
            }
        }
        let Some(edge) = edge_uses
            .into_iter()
            .find_map(|(edge, uses)| (uses > 2).then_some(edge))
        else {
            return Ok(true);
        };
        let incident = topology.incidence.vertex_stars[edge[0] as usize]
            .iter()
            .copied()
            .filter(|tetrahedron| {
                topology.tetrahedra[*tetrahedron as usize]
                    .vertex_indices
                    .contains(&edge[1])
            })
            .collect::<BTreeSet<_>>();
        if let Some(expansion) =
            disconnected_edge_star_gaps(topology, edge, &incident, removed, protected_edges)
        {
            let mut expanded = false;
            for tetrahedron in expansion {
                work.cavity_step(constraint_index)?;
                expanded |= removed.insert(tetrahedron);
                if removed.len() as u64 > work.options.maximum_cavity_tetrahedra {
                    return Err(resource_or_cancelled(
                        DelaunayFacetRecoveryErrorKind::ResourceLimit,
                        constraint_index,
                        "facet cavity tetrahedron limit exceeded while regularizing a nonmanifold edge star"
                            .to_owned(),
                    ));
                }
            }
            if expanded {
                continue;
            }
        }
        if let Some(contraction) =
            removable_selected_edge_star_components(topology, edge, &incident, removed, mandatory)
        {
            let mut contracted = false;
            for tetrahedron in contraction {
                work.cavity_step(constraint_index)?;
                contracted |= removed.remove(&tetrahedron);
            }
            if contracted {
                continue;
            }
        }
        {
            let coordinates = edge.map(|node| {
                topology.nodes[node as usize]
                    .coordinates_m
                    .canonical_value()
            });
            let signs = edge.map(|node| plane_signs[node as usize]);
            let selected = incident
                .intersection(removed)
                .copied()
                .collect::<BTreeSet<_>>();
            let components = edge_star_components(topology, edge, &selected);
            let component_evidence = components
                .iter()
                .map(|component| {
                    (
                        component.clone(),
                        component
                            .iter()
                            .filter(|tetrahedron| mandatory.contains(tetrahedron))
                            .count(),
                    )
                })
                .collect::<Vec<_>>();
            work.reject(format!(
                "nonmanifold cavity edge {edge:?} at {coordinates:?} with facet-plane signs {signs:?} has neither a boundary-preserving expansion nor a contraction disjoint from the facet intersection; selected edge-star components and mandatory counts are {component_evidence:?}; facet-cap boundary has {} edges",
                protected_edges.len(),
            ));
            return Ok(false);
        }
    }
}

fn removable_selected_edge_star_components(
    topology: &DelaunayVolumeTopology,
    edge: [u32; 2],
    incident: &BTreeSet<u32>,
    removed: &BTreeSet<u32>,
    mandatory: &BTreeSet<u32>,
) -> Option<Vec<u32>> {
    let selected = incident
        .intersection(removed)
        .copied()
        .collect::<BTreeSet<_>>();
    let mut components = edge_star_components(topology, edge, &selected);
    if components.len() < 2 {
        return None;
    }
    components.sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
    let mandatory_components = components
        .iter()
        .enumerate()
        .filter_map(|(index, component)| {
            component
                .iter()
                .any(|tetrahedron| mandatory.contains(tetrahedron))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    if mandatory_components.len() > 1 {
        return None;
    }
    let retained = mandatory_components.first().copied().unwrap_or(0);
    Some(
        components
            .into_iter()
            .enumerate()
            .filter(|(index, _)| *index != retained)
            .flat_map(|(_, component)| component)
            .collect(),
    )
}

fn edge_star_components(
    topology: &DelaunayVolumeTopology,
    edge: [u32; 2],
    members: &BTreeSet<u32>,
) -> Vec<Vec<u32>> {
    let mut unseen = members.clone();
    let mut components = Vec::new();
    while let Some(first) = unseen.pop_first() {
        let mut pending = vec![first];
        let mut component = Vec::new();
        while let Some(tetrahedron_id) = pending.pop() {
            component.push(tetrahedron_id);
            let tetrahedron = &topology.tetrahedra[tetrahedron_id as usize];
            for (opposite, node) in tetrahedron.vertex_indices.iter().enumerate() {
                if edge.contains(node) {
                    continue;
                }
                let Some(neighbor) = tetrahedron.neighbors[opposite] else {
                    continue;
                };
                if members.contains(&neighbor) && unseen.remove(&neighbor) {
                    pending.push(neighbor);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components
}

fn disconnected_edge_star_gaps(
    topology: &DelaunayVolumeTopology,
    edge: [u32; 2],
    incident: &BTreeSet<u32>,
    removed: &BTreeSet<u32>,
    protected_edges: &BTreeSet<[u32; 2]>,
) -> Option<Vec<u32>> {
    let unselected = incident
        .difference(removed)
        .copied()
        .collect::<BTreeSet<_>>();
    let mut gaps = edge_star_components(topology, edge, &unselected);
    gaps.sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
    let open_star = incident.iter().any(|tetrahedron_id| {
        let tetrahedron = &topology.tetrahedra[*tetrahedron_id as usize];
        tetrahedron
            .vertex_indices
            .iter()
            .enumerate()
            .filter(|(_, node)| !edge.contains(node))
            .any(|(opposite, _)| tetrahedron.neighbors[opposite].is_none())
    });
    let retained_gap_candidates = if open_star {
        std::iter::once(None)
            .chain((0..gaps.len()).map(Some))
            .collect::<Vec<_>>()
    } else {
        (0..gaps.len()).map(Some).collect()
    };
    for retained_gap in retained_gap_candidates {
        let expansion = gaps
            .iter()
            .enumerate()
            .filter(|(index, _)| Some(*index) != retained_gap)
            .flat_map(|(_, gap)| gap.iter().copied())
            .collect::<Vec<_>>();
        let expansion_set = expansion.iter().copied().collect::<BTreeSet<_>>();
        let preserves_cap_boundary = protected_edges.iter().all(|protected| {
            topology.incidence.vertex_stars[protected[0] as usize]
                .iter()
                .copied()
                .filter(|tetrahedron| {
                    topology.tetrahedra[*tetrahedron as usize]
                        .vertex_indices
                        .contains(&protected[1])
                })
                .any(|tetrahedron| {
                    !removed.contains(&tetrahedron) && !expansion_set.contains(&tetrahedron)
                })
        });
        if preserves_cap_boundary {
            return Some(expansion);
        }
    }
    None
}

fn patch_boundary_edges(cap: &[[u32; 3]]) -> BTreeSet<[u32; 2]> {
    let mut uses = BTreeMap::<[u32; 2], usize>::new();
    for triangle in cap {
        for mut edge in [
            [triangle[0], triangle[1]],
            [triangle[1], triangle[2]],
            [triangle[2], triangle[0]],
        ] {
            edge.sort_unstable();
            *uses.entry(edge).or_default() += 1;
        }
    }
    uses.into_iter()
        .filter_map(|(edge, count)| (count == 1).then_some(edge))
        .collect()
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

#[cfg(test)]
mod tests {
    use super::{disconnected_edge_star_gaps, removable_selected_edge_star_components};
    use crate::cdt::{build_delaunay_volume_topology, DelaunayTopologyOptions, DelaunayVolumeNode};
    use runmat_meshing_core::{NeverCancelled, StableDigest};
    use std::collections::BTreeSet;

    #[test]
    fn edge_star_regularization_retains_one_canonical_gap() {
        const RING: usize = 6;
        let mut nodes = vec![
            DelaunayVolumeNode::explicit(StableDigest::from_bytes([1; 32]), [0.0, 0.0, -1.0]),
            DelaunayVolumeNode::explicit(StableDigest::from_bytes([2; 32]), [0.0, 0.0, 1.0]),
        ];
        for index in 0..RING {
            let angle = std::f64::consts::TAU * index as f64 / RING as f64;
            nodes.push(DelaunayVolumeNode::explicit(
                StableDigest::from_bytes([(index + 3) as u8; 32]),
                [angle.cos(), angle.sin(), 0.0],
            ));
        }
        let tetrahedra = (0..RING)
            .map(|index| [0, 1, index as u32 + 2, (index + 1) as u32 % RING as u32 + 2])
            .collect();
        let topology = build_delaunay_volume_topology(
            nodes,
            tetrahedra,
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let incident = (0..RING as u32).collect::<BTreeSet<_>>();
        let removed = BTreeSet::from([0, 2, 4]);

        assert_eq!(
            disconnected_edge_star_gaps(&topology, [0, 1], &incident, &removed, &BTreeSet::new()),
            Some(vec![3])
        );
        assert_eq!(
            removable_selected_edge_star_components(
                &topology,
                [0, 1],
                &incident,
                &removed,
                &BTreeSet::from([0])
            ),
            Some(vec![4])
        );
        assert_eq!(
            removable_selected_edge_star_components(
                &topology,
                [0, 1],
                &incident,
                &removed,
                &BTreeSet::from([0, 4])
            ),
            None
        );
    }
}
