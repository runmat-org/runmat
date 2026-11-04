use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{tetrahedron_centroid, Point3, PointInClosedSurface},
    quality::tolerance::MeshingTolerance,
    MeshingCancellationSignal,
};

use super::{
    boundary_nodes::{
        boundary_node_coordinates, candidate_respects_protected_boundary_distance,
        cavity_boundary_node_ids, cavity_boundary_triangles, classify_cavity_point,
        resolved_cavity_tolerance,
    },
    exact_cover::{on_demand_interior_mate_faces_for_trace, BoundaryExactCoverSearch},
    refill_tetrahedra::{
        raw_refill_tetrahedron_with_rejection_reason, refill_from_tetrahedra,
        RefillTetrahedronError,
    },
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
    retriangulate_constrained_cavity_from_nodes_with_diagnostics(
        cavity,
        nodes,
        options,
        budget,
        cancellation,
    )
    .map(|result| result.refill)
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ConstrainedCavityRetriangulation {
    pub refill: Option<ConstrainedCavityRefill>,
    pub accepted_node_count: usize,
    pub candidate_tetrahedron_count: usize,
    pub terminal_reason: &'static str,
    pub dead_end_reason_counts: BTreeMap<&'static str, usize>,
    pub candidate_rejection_counts: BTreeMap<&'static str, usize>,
    pub first_dead_end: Option<String>,
    pub target_volume_m3: f64,
    pub sampled_candidate_volumes_m3: Vec<f64>,
    pub sampled_boundary_faces: Vec<[u32; 3]>,
    pub sampled_candidate_tetrahedra: Vec<[u32; 4]>,
    pub sampled_accepted_nodes: Vec<(u32, Point3)>,
}

struct RetriangulationOutcome {
    refill: Option<ConstrainedCavityRefill>,
    terminal_reason: &'static str,
    dead_end_reason_counts: BTreeMap<&'static str, usize>,
    candidate_rejection_counts: BTreeMap<&'static str, usize>,
    first_dead_end: Option<String>,
}

pub(crate) fn retriangulate_constrained_cavity_from_nodes_with_diagnostics(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
    budget: ConstrainedCavityRefillBudget,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<ConstrainedCavityRetriangulation, ConstrainedCavityRefillError> {
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
    let tolerance = resolved_cavity_tolerance(&boundary_node_map, options.classification_tolerance);
    let mut node_map = BTreeMap::<u32, Point3>::new();
    for node in nodes {
        if node_map
            .insert(node.node_id, node.coordinates_m.canonical_value())
            .is_some()
        {
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
            node.coordinates_m.canonical_value(),
            options,
        ) {
            continue;
        }
        if classify_cavity_point(
            cavity,
            &boundary_node_map,
            &boundary_triangles,
            node.coordinates_m.canonical_value(),
            tolerance,
        ) == PointInClosedSurface::Inside
        {
            candidate_nodes.push(node.clone());
        }
    }
    if candidate_nodes.len() < 4 {
        return Ok(result(
            RetriangulationOutcome {
                refill: None,
                terminal_reason: "fewer_than_four_accepted_nodes",
                dead_end_reason_counts: BTreeMap::new(),
                candidate_rejection_counts: BTreeMap::new(),
                first_dead_end: None,
            },
            &[],
            cavity,
            &candidate_nodes,
        ));
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
        cavity,
        boundary_nodes: &boundary_node_map,
        tolerance,
        maximum_candidate_volume_m3: cavity.target_volume_m3
            + cavity.target_volume_m3.max(1.0e-18) * options.volume_relative_tolerance,
        options,
        budget,
        cancellation,
    };
    let mut inventory = CandidateInventory {
        work: 0,
        seen: BTreeSet::new(),
        candidates: Vec::new(),
        rejection_counts: BTreeMap::new(),
    };
    let mut remaining_search_attempts = budget.maximum_search_attempts;

    for _ in 0..budget.maximum_expansion_rounds {
        let candidate_count_before = inventory.candidates.len();
        generator.append(&pending_faces, &mut inventory)?;
        if inventory.candidates.is_empty() {
            return Ok(result(
                RetriangulationOutcome {
                    refill: None,
                    terminal_reason: "no_candidate_tetrahedra",
                    dead_end_reason_counts: BTreeMap::new(),
                    candidate_rejection_counts: inventory.rejection_counts.clone(),
                    first_dead_end: None,
                },
                &[],
                cavity,
                &candidate_nodes,
            ));
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
            return Ok(result(
                RetriangulationOutcome {
                    refill: Some(refill),
                    terminal_reason: "recovered",
                    dead_end_reason_counts: BTreeMap::new(),
                    candidate_rejection_counts: inventory.rejection_counts.clone(),
                    first_dead_end: None,
                },
                &inventory.candidates,
                cavity,
                &candidate_nodes,
            ));
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
            return Ok(result(
                RetriangulationOutcome {
                    refill: None,
                    terminal_reason: "exact_cover_has_no_expandable_mate_faces",
                    dead_end_reason_counts: trace.dead_end_reason_counts,
                    candidate_rejection_counts: inventory.rejection_counts.clone(),
                    first_dead_end: trace.dead_end.map(|dead_end| format!("{dead_end:?}")),
                },
                &inventory.candidates,
                cavity,
                &candidate_nodes,
            ));
        };
        if inventory.candidates.len() == candidate_count_before {
            return Ok(result(
                RetriangulationOutcome {
                    refill: None,
                    terminal_reason: "candidate_expansion_made_no_progress",
                    dead_end_reason_counts: trace.dead_end_reason_counts,
                    candidate_rejection_counts: inventory.rejection_counts.clone(),
                    first_dead_end: trace.dead_end.map(|dead_end| format!("{dead_end:?}")),
                },
                &inventory.candidates,
                cavity,
                &candidate_nodes,
            ));
        }
        pending_faces = next_faces;
    }
    Err(resource(
        "constrained cavity candidate expansion round limit exceeded",
    ))
}

fn result(
    outcome: RetriangulationOutcome,
    candidates: &[ConstrainedCavityRefillTetrahedron],
    cavity: &ConstrainedCavity,
    accepted_nodes: &[ConstrainedCavityNode],
) -> ConstrainedCavityRetriangulation {
    ConstrainedCavityRetriangulation {
        refill: outcome.refill,
        accepted_node_count: accepted_nodes.len(),
        candidate_tetrahedron_count: candidates.len(),
        terminal_reason: outcome.terminal_reason,
        dead_end_reason_counts: outcome.dead_end_reason_counts,
        candidate_rejection_counts: outcome.candidate_rejection_counts,
        first_dead_end: outcome.first_dead_end,
        target_volume_m3: cavity.target_volume_m3,
        sampled_candidate_volumes_m3: candidates
            .iter()
            .take(64)
            .map(|candidate| candidate.volume_m3)
            .collect(),
        sampled_boundary_faces: cavity
            .boundary_faces
            .iter()
            .take(64)
            .map(|face| sorted_face(face.node_ids))
            .collect(),
        sampled_candidate_tetrahedra: candidates
            .iter()
            .take(64)
            .map(|candidate| candidate.node_ids)
            .collect(),
        sampled_accepted_nodes: accepted_nodes
            .iter()
            .take(64)
            .map(|node| (node.node_id, node.coordinates_m.canonical_value()))
            .collect(),
    }
}

struct CandidateGenerator<'a> {
    candidate_nodes: &'a [ConstrainedCavityNode],
    node_map: &'a BTreeMap<u32, Point3>,
    boundary_triangles: &'a [[[f64; 3]; 3]],
    cavity: &'a ConstrainedCavity,
    boundary_nodes: &'a BTreeMap<u32, Point3>,
    tolerance: MeshingTolerance,
    maximum_candidate_volume_m3: f64,
    options: ConstrainedCavityRefillOptions,
    budget: ConstrainedCavityRefillBudget,
    cancellation: &'a dyn MeshingCancellationSignal,
}

struct CandidateInventory {
    work: u64,
    seen: BTreeSet<[u32; 4]>,
    candidates: Vec<ConstrainedCavityRefillTetrahedron>,
    rejection_counts: BTreeMap<&'static str, usize>,
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
                let classification = classify_cavity_point(
                    self.cavity,
                    self.boundary_nodes,
                    self.boundary_triangles,
                    tetrahedron_centroid(points),
                    self.tolerance,
                );
                match classification {
                    PointInClosedSurface::Inside => {}
                    PointInClosedSurface::OnBoundary => {
                        *inventory
                            .rejection_counts
                            .entry("centroid_on_boundary")
                            .or_default() += 1;
                        continue;
                    }
                    PointInClosedSurface::Outside => {
                        *inventory
                            .rejection_counts
                            .entry("centroid_outside")
                            .or_default() += 1;
                        continue;
                    }
                }
                let candidate = match raw_refill_tetrahedron_with_rejection_reason(
                    node_ids,
                    points,
                    self.options,
                ) {
                    Ok(candidate) => candidate,
                    Err(RefillTetrahedronError::Rejected(reason)) => {
                        *inventory.rejection_counts.entry(reason).or_default() += 1;
                        continue;
                    }
                    Err(RefillTetrahedronError::Accounting(cause)) => {
                        return Err(ConstrainedCavityRefillError::Accounting(cause));
                    }
                };
                if candidate.volume_m3 > self.maximum_candidate_volume_m3 {
                    *inventory
                        .rejection_counts
                        .entry("candidate_exceeds_cavity_volume")
                        .or_default() += 1;
                    continue;
                }
                inventory.candidates.push(candidate);
                if inventory.candidates.len() as u64 > self.budget.maximum_candidate_tetrahedra {
                    return Err(resource(
                        "constrained cavity candidate-tetrahedron limit exceeded",
                    ));
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
