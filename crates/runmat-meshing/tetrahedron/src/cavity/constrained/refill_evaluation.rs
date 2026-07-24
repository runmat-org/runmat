use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{point_in_closed_triangle_surface, PointInClosedSurface},
    quality::tolerance::MeshingTolerance,
};

use super::{
    boundary_nodes::{
        boundary_node_coordinates, candidate_respects_protected_boundary_distance,
        cavity_boundary_node_ids, cavity_boundary_triangles,
    },
    refill_candidates::{
        boundary_node_refill_candidate, centroid_interior_refill_candidate,
        multi_interior_node_refill_candidate, single_tetrahedron_refill_candidate,
        two_interior_node_refill_candidate,
    },
    refill_tetrahedra::{
        record_refill_rejection, refill_is_better, refill_validation_reason,
        star_refill_candidate_with_rejection_reason, LocalTetrahedronReconnectionDiagnostics,
    },
    validation::{validate_constrained_cavity, validate_refill_options},
    ConstrainedCavity, ConstrainedCavityNode, ConstrainedCavityRefill,
    ConstrainedCavityRefillError, ConstrainedCavityRefillEvaluation,
    ConstrainedCavityRefillOptions,
};

pub fn generate_constrained_cavity_refill_candidates(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    interior_candidate_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<ConstrainedCavityRefill, ConstrainedCavityRefillError> {
    let evaluation = evaluate_constrained_cavity_refill_candidates(
        cavity,
        boundary_nodes,
        interior_candidate_nodes,
        options,
    )?;
    evaluation
        .refill
        .ok_or(ConstrainedCavityRefillError::NoValidCandidate {
            rejected_by_reason: evaluation.rejected_by_reason,
        })
}

pub fn evaluate_constrained_cavity_refill_candidates(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    interior_candidate_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<ConstrainedCavityRefillEvaluation, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let mut rejected_by_reason = BTreeMap::<String, usize>::new();
    let mut local_reconnection_diagnostics = LocalTetrahedronReconnectionDiagnostics::default();

    if interior_candidate_nodes.is_empty() {
        if boundary_node_ids.len() == 4 {
            let Some(refill) =
                single_tetrahedron_refill_candidate(cavity, &boundary_node_map, options)
                    .map_err(ConstrainedCavityRefillError::Validation)?
            else {
                record_refill_rejection(
                    &mut rejected_by_reason,
                    "single_tetrahedron_refill_rejected",
                );
                return Ok(refill_evaluation(
                    None,
                    rejected_by_reason,
                    local_reconnection_diagnostics,
                ));
            };
            return Ok(refill_evaluation(
                Some(refill),
                rejected_by_reason,
                local_reconnection_diagnostics,
            ));
        };
        match boundary_node_refill_candidate(cavity, &boundary_node_map, options) {
            Ok(Ok(candidate)) => {
                merge_local_reconnection_diagnostics(
                    &mut local_reconnection_diagnostics,
                    candidate.local_reconnection_diagnostics,
                );
                return Ok(refill_evaluation(
                    Some(candidate.refill),
                    rejected_by_reason,
                    local_reconnection_diagnostics,
                ));
            }
            Ok(Err(reason)) => record_refill_rejection(&mut rejected_by_reason, reason),
            Err(err) => {
                record_refill_rejection(&mut rejected_by_reason, refill_validation_reason(&err))
            }
        }
        match centroid_interior_refill_candidate(
            cavity,
            &boundary_node_map,
            &boundary_triangles,
            options,
        ) {
            Ok(Ok(refill)) => {
                return Ok(refill_evaluation(
                    Some(refill),
                    rejected_by_reason,
                    local_reconnection_diagnostics,
                ));
            }
            Ok(Err(reason)) => record_refill_rejection(&mut rejected_by_reason, reason),
            Err(err) => {
                record_refill_rejection(&mut rejected_by_reason, refill_validation_reason(&err))
            }
        }
        return Ok(refill_evaluation(
            None,
            rejected_by_reason,
            local_reconnection_diagnostics,
        ));
    }

    let mut seen_interior_nodes = BTreeSet::<u32>::new();
    let tolerance = MeshingTolerance::default();
    let mut best = None::<ConstrainedCavityRefill>;
    let mut valid_interior_nodes = Vec::<ConstrainedCavityNode>::new();
    for node in interior_candidate_nodes {
        if !seen_interior_nodes.insert(node.node_id) {
            return Err(ConstrainedCavityRefillError::DuplicateInteriorNode {
                node_id: node.node_id,
            });
        }
        if boundary_node_ids.contains(&node.node_id) {
            return Err(
                ConstrainedCavityRefillError::InteriorNodeReusesBoundaryNode {
                    node_id: node.node_id,
                },
            );
        }
        if !candidate_respects_protected_boundary_distance(
            cavity,
            &boundary_node_map,
            node.coordinates_m,
            options,
        ) {
            record_refill_rejection(&mut rejected_by_reason, "protected_boundary_distance");
            continue;
        }
        if point_in_closed_triangle_surface(node.coordinates_m, &boundary_triangles, tolerance)
            != PointInClosedSurface::Inside
        {
            record_refill_rejection(&mut rejected_by_reason, "interior_point_outside_cavity");
            continue;
        }
        valid_interior_nodes.push(node.clone());
        let refill = match star_refill_candidate_with_rejection_reason(
            cavity,
            &boundary_node_map,
            node.clone(),
            options,
        ) {
            Ok(Ok(refill)) => refill,
            Ok(Err(reason)) => {
                record_refill_rejection(&mut rejected_by_reason, reason);
                continue;
            }
            Err(err) => {
                record_refill_rejection(&mut rejected_by_reason, refill_validation_reason(&err));
                continue;
            }
        };
        if best
            .as_ref()
            .is_none_or(|candidate| refill_is_better(&refill, candidate))
        {
            best = Some(refill);
        }
    }
    if best.is_none() && valid_interior_nodes.len() >= 2 {
        match two_interior_node_refill_candidate(
            cavity,
            &boundary_node_map,
            &boundary_triangles,
            &valid_interior_nodes,
            options,
        ) {
            Ok(Ok(refill)) => best = Some(refill),
            Ok(Err(reason)) => record_refill_rejection(&mut rejected_by_reason, reason),
            Err(err) => {
                record_refill_rejection(&mut rejected_by_reason, refill_validation_reason(&err))
            }
        }
    }
    if best.is_none() && valid_interior_nodes.len() >= 3 {
        match multi_interior_node_refill_candidate(
            cavity,
            &boundary_node_map,
            &boundary_triangles,
            &valid_interior_nodes,
            options,
        ) {
            Ok(Ok(refill)) => best = Some(refill),
            Ok(Err(reason)) => record_refill_rejection(&mut rejected_by_reason, reason),
            Err(err) => {
                record_refill_rejection(&mut rejected_by_reason, refill_validation_reason(&err))
            }
        }
    }
    if best.is_none() && boundary_node_ids.len() > 4 {
        match boundary_node_refill_candidate(cavity, &boundary_node_map, options) {
            Ok(Ok(candidate)) => {
                merge_local_reconnection_diagnostics(
                    &mut local_reconnection_diagnostics,
                    candidate.local_reconnection_diagnostics,
                );
                best = Some(candidate.refill);
            }
            Ok(Err(reason)) => record_refill_rejection(&mut rejected_by_reason, reason),
            Err(err) => {
                record_refill_rejection(&mut rejected_by_reason, refill_validation_reason(&err))
            }
        }
    }

    Ok(refill_evaluation(
        best,
        rejected_by_reason,
        local_reconnection_diagnostics,
    ))
}

fn refill_evaluation(
    refill: Option<ConstrainedCavityRefill>,
    rejected_by_reason: BTreeMap<String, usize>,
    local_reconnection_diagnostics: LocalTetrahedronReconnectionDiagnostics,
) -> ConstrainedCavityRefillEvaluation {
    ConstrainedCavityRefillEvaluation {
        refill,
        rejected_by_reason,
        local_reconnection_attempt_count: local_reconnection_diagnostics
            .attempted_reconnection_count,
        local_reconnection_accepted_count: local_reconnection_diagnostics
            .accepted_reconnection_count,
        local_reconnection_rejected_count: local_reconnection_diagnostics
            .rejected_reconnection_count,
        local_reconnection_rejected_by_reason: local_reconnection_diagnostics.rejected_by_reason,
    }
}

fn merge_local_reconnection_diagnostics(
    target: &mut LocalTetrahedronReconnectionDiagnostics,
    source: LocalTetrahedronReconnectionDiagnostics,
) {
    target.attempted_reconnection_count += source.attempted_reconnection_count;
    target.accepted_reconnection_count += source.accepted_reconnection_count;
    target.rejected_reconnection_count += source.rejected_reconnection_count;
    for (reason, count) in source.rejected_by_reason {
        *target.rejected_by_reason.entry(reason).or_default() += count;
    }
}
