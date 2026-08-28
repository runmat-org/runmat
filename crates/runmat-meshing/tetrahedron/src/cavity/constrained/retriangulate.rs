use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{
        point_in_closed_triangle_surface, tetrahedron_centroid, Point3, PointInClosedSurface,
    },
    quality::tolerance::MeshingTolerance,
    MeshingCancellationSignal,
};

use super::{
    boundary_nodes::{
        boundary_node_coordinates, candidate_respects_protected_boundary_distance,
        cavity_boundary_node_ids, cavity_boundary_triangles,
    },
    exact_cover::{on_demand_interior_mate_faces_for_trace, BoundaryExactCoverSearch},
    refill_tetrahedra::{raw_refill_tetrahedron_with_rejection_reason, refill_from_tetrahedra},
    topology::{sorted_face, sorted_tetrahedron_nodes},
    validation::{validate_constrained_cavity, validate_refill_options},
    ConstrainedCavity, ConstrainedCavityNode, ConstrainedCavityRefill,
    ConstrainedCavityRefillBudget, ConstrainedCavityRefillError, ConstrainedCavityRefillOptions,
    ConstrainedCavityRefillTetrahedron,
};

pub fn retriangulate_constrained_cavity_from_nodes(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
    budget: ConstrainedCavityRefillBudget,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<ConstrainedCavityRefill>, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_budget(budget)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    if nodes.len() as u64 > budget.maximum_nodes {
        return Err(resource("constrained cavity node limit exceeded"));
    }
    if cavity.boundary_faces.len() as u64 > budget.maximum_boundary_faces {
        return Err(resource("constrained cavity boundary-face limit exceeded"));
    }
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let tolerance = MeshingTolerance::default();
    let mut node_map = BTreeMap::<u32, Point3>::new();
    for node in nodes {
        if node_map.insert(node.node_id, node.coordinates_m).is_some() {
            return Err(ConstrainedCavityRefillError::DuplicateInteriorNode {
                node_id: node.node_id,
            });
        }
    }
    let mut candidate_nodes = Vec::<ConstrainedCavityNode>::new();
    for node in nodes {
        if boundary_node_ids.contains(&node.node_id) {
            candidate_nodes.push(node.clone());
            continue;
        }
        if !candidate_respects_protected_boundary_distance(
            cavity,
            &boundary_node_map,
            node.coordinates_m,
            options,
        ) {
            continue;
        }
        if point_in_closed_triangle_surface(node.coordinates_m, &boundary_triangles, tolerance)
            == PointInClosedSurface::Inside
        {
            candidate_nodes.push(node.clone());
        }
    }
    if candidate_nodes.len() < 4 {
        return Ok(None);
    }
    candidate_nodes.sort_by_key(|node| node.node_id);
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut pending_faces = boundary_faces.clone();
    let generator = CandidateGenerator {
        candidate_nodes: &candidate_nodes,
        node_map: &node_map,
        boundary_triangles: &boundary_triangles,
        options,
        budget,
        cancellation,
    };
    let mut inventory = CandidateInventory {
        work: 0,
        seen: BTreeSet::new(),
        candidates: Vec::new(),
    };
    let mut remaining_search_attempts = budget.maximum_search_attempts;

    for _ in 0..budget.maximum_expansion_rounds {
        let candidate_count_before = inventory.candidates.len();
        generator.append(&pending_faces, &mut inventory)?;
        if inventory.candidates.is_empty() {
            return Ok(None);
        }
        if remaining_search_attempts == 0 {
            return Err(resource(
                "constrained cavity exact-cover search limit exceeded",
            ));
        }
        let attempt_limit = usize::try_from(remaining_search_attempts).unwrap_or(usize::MAX);
        let mut search = BoundaryExactCoverSearch::with_attempt_limit(
            cavity,
            &inventory.candidates,
            options.volume_relative_tolerance,
            attempt_limit,
        );
        let (selected, trace) = search
            .search_with_trace_controlled(cancellation, budget.cancellation_check_interval)
            .map_err(|()| ConstrainedCavityRefillError::Cancelled)?;
        remaining_search_attempts =
            remaining_search_attempts.saturating_sub(search.attempts as u64);
        if let Some(selected) = selected {
            let selected = selected
                .into_iter()
                .map(|index| inventory.candidates[index].clone())
                .collect::<Vec<_>>();
            let used_nodes = selected
                .iter()
                .flat_map(|tetrahedron| tetrahedron.node_ids)
                .collect::<BTreeSet<_>>();
            let mut refill =
                refill_from_tetrahedra(cavity, selected, options.volume_relative_tolerance)
                    .map_err(ConstrainedCavityRefillError::Validation)?;
            refill.inserted_nodes = candidate_nodes
                .iter()
                .filter(|node| !boundary_node_ids.contains(&node.node_id))
                .filter(|node| used_nodes.contains(&node.node_id))
                .cloned()
                .collect();
            return Ok(Some(refill));
        }
        if trace
            .dead_end_reason_counts
            .get("attempt_limit")
            .copied()
            .unwrap_or(0)
            > 0
        {
            return Err(resource(
                "constrained cavity exact-cover search limit exceeded",
            ));
        }
        let Some(next_faces) = on_demand_interior_mate_faces_for_trace(
            cavity,
            &inventory.candidates,
            options,
            &boundary_faces,
            &trace,
        ) else {
            return Ok(None);
        };
        if inventory.candidates.len() == candidate_count_before {
            return Ok(None);
        }
        pending_faces = next_faces;
    }
    Err(resource(
        "constrained cavity candidate expansion round limit exceeded",
    ))
}

struct CandidateGenerator<'a> {
    candidate_nodes: &'a [ConstrainedCavityNode],
    node_map: &'a BTreeMap<u32, Point3>,
    boundary_triangles: &'a [[[f64; 3]; 3]],
    options: ConstrainedCavityRefillOptions,
    budget: ConstrainedCavityRefillBudget,
    cancellation: &'a dyn MeshingCancellationSignal,
}

struct CandidateInventory {
    work: u64,
    seen: BTreeSet<[u32; 4]>,
    candidates: Vec<ConstrainedCavityRefillTetrahedron>,
}

impl CandidateGenerator<'_> {
    fn append(
        &self,
        faces: &BTreeSet<[u32; 3]>,
        inventory: &mut CandidateInventory,
    ) -> Result<(), ConstrainedCavityRefillError> {
        for face in faces {
            for node in self.candidate_nodes {
                inventory.work = inventory
                    .work
                    .checked_add(1)
                    .ok_or_else(|| resource("constrained cavity candidate counter overflowed"))?;
                if inventory.work > self.budget.maximum_candidate_evaluations {
                    return Err(resource(
                        "constrained cavity candidate-evaluation limit exceeded",
                    ));
                }
                if inventory
                    .work
                    .is_multiple_of(self.budget.cancellation_check_interval)
                    && self.cancellation.is_cancelled()
                {
                    return Err(ConstrainedCavityRefillError::Cancelled);
                }
                if face.contains(&node.node_id) {
                    continue;
                }
                let node_ids = sorted_tetrahedron_nodes([face[0], face[1], face[2], node.node_id]);
                if !inventory.seen.insert(node_ids) {
                    continue;
                }
                let points = node_ids.map(|node_id| self.node_map[&node_id]);
                if point_in_closed_triangle_surface(
                    tetrahedron_centroid(points),
                    self.boundary_triangles,
                    MeshingTolerance::default(),
                ) != PointInClosedSurface::Inside
                {
                    continue;
                }
                if let Ok(candidate) =
                    raw_refill_tetrahedron_with_rejection_reason(node_ids, points, self.options)
                {
                    inventory.candidates.push(candidate);
                    if inventory.candidates.len() as u64 > self.budget.maximum_candidate_tetrahedra
                    {
                        return Err(resource(
                            "constrained cavity candidate-tetrahedron limit exceeded",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

fn validate_budget(
    budget: ConstrainedCavityRefillBudget,
) -> Result<(), ConstrainedCavityRefillError> {
    if budget.maximum_nodes == 0
        || budget.maximum_boundary_faces == 0
        || budget.maximum_candidate_tetrahedra == 0
        || budget.maximum_candidate_evaluations == 0
        || budget.maximum_search_attempts == 0
        || budget.maximum_expansion_rounds == 0
        || budget.cancellation_check_interval == 0
    {
        return Err(ConstrainedCavityRefillError::InvalidOptions);
    }
    Ok(())
}

fn resource(reason: impl Into<String>) -> ConstrainedCavityRefillError {
    ConstrainedCavityRefillError::ResourceLimit {
        reason: reason.into(),
    }
}
