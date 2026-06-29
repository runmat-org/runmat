use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet},
};

use serde::{Deserialize, Serialize};

use crate::{
    predicate::{
        distance_squared, point_in_closed_triangle_surface, tet_centroid, tet_edge_aspect_ratio,
        tet_scaled_jacobian, tet_signed_volume, Point3, PointInClosedSurface, Triangle3,
    },
    tet_candidate::{tetrahedralize_points, ConnectivityPoint, TetCandidate},
    tolerance::MeshingTolerance,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavity {
    pub removed_tet_ids: Vec<u32>,
    pub boundary_faces: Vec<ConstrainedCavityBoundaryFace>,
    #[serde(default)]
    pub protected_node_ids: Vec<u32>,
    pub target_volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityBoundaryFace {
    pub node_ids: [u32; 3],
    pub source_face_id: Option<u32>,
    #[serde(default)]
    pub source_edge_ids: [Option<u32>; 3],
    #[serde(default)]
    pub region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityValidationReport {
    pub boundary_face_count: usize,
    pub boundary_edge_count: usize,
    pub boundary_node_count: usize,
    pub protected_node_count: usize,
    pub target_volume_m3: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityRefillOptions {
    pub min_volume_m3: f64,
    pub max_aspect_ratio: f64,
    pub min_scaled_jacobian: f64,
    pub volume_relative_tolerance: f64,
    pub min_protected_node_distance_m: f64,
}

impl Default for ConstrainedCavityRefillOptions {
    fn default() -> Self {
        Self {
            min_volume_m3: 1.0e-18,
            max_aspect_ratio: 1.0e6,
            min_scaled_jacobian: 0.15,
            volume_relative_tolerance: 1.0e-9,
            min_protected_node_distance_m: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityNode {
    pub node_id: u32,
    pub coordinates_m: Point3,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityRefillTet {
    pub node_ids: [u32; 4],
    pub volume_m3: f64,
    pub aspect_ratio: f64,
    pub exact_scaled_jacobian: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityRefill {
    pub tets: Vec<ConstrainedCavityRefillTet>,
    pub boundary_faces: Vec<ConstrainedCavityBoundaryFace>,
    pub total_volume_m3: f64,
}

const MAX_ANCHOR_TRIM_STATES: usize = 128;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityRefillEvaluation {
    pub refill: Option<ConstrainedCavityRefill>,
    #[serde(default)]
    pub rejected_by_reason: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityValidationError {
    EmptyRemovedTetSet,
    InvalidTargetVolume {
        target_volume_m3: f64,
    },
    TooFewBoundaryFaces {
        boundary_face_count: usize,
    },
    DegenerateBoundaryFace {
        face_index: usize,
        node_ids: [u32; 3],
    },
    DuplicateBoundaryFace {
        node_ids: [u32; 3],
    },
    NonManifoldBoundaryEdge {
        node_ids: [u32; 2],
        face_count: usize,
    },
    ProtectedNodeOutsideBoundary {
        node_id: u32,
    },
    InvalidRefillVolume {
        target_volume_m3: f64,
        candidate_volume_m3: f64,
        tolerance_m3: f64,
    },
    BoundaryFaceCountMismatch {
        expected_count: usize,
        candidate_count: usize,
    },
    MissingBoundaryFace {
        node_ids: [u32; 3],
    },
    UnexpectedBoundaryFace {
        node_ids: [u32; 3],
    },
    BoundarySourceFaceMismatch {
        node_ids: [u32; 3],
        expected_source_face_id: Option<u32>,
        candidate_source_face_id: Option<u32>,
    },
    BoundarySourceEdgeMismatch {
        node_ids: [u32; 2],
        expected_source_edge_id: Option<u32>,
        candidate_source_edge_id: Option<u32>,
    },
    BoundaryRegionMismatch {
        node_ids: [u32; 3],
        expected_region_ids: Vec<String>,
        candidate_region_ids: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityBoundarySplitError {
    SplitNodeReusesFaceNode { node_id: u32 },
    MissingBoundaryFace { node_ids: [u32; 3] },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityExtractionError {
    EmptySelection,
    SelectedTetIndexOutOfBounds { tet_index: usize, tet_count: usize },
    DuplicateSelectedTetIndex { tet_index: usize },
    Validation(ConstrainedCavityValidationError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityRefillError {
    InvalidOptions,
    Validation(ConstrainedCavityValidationError),
    MissingBoundaryNode {
        node_id: u32,
    },
    DuplicateInteriorNode {
        node_id: u32,
    },
    InteriorNodeReusesBoundaryNode {
        node_id: u32,
    },
    InteriorPointOutsideCavity {
        node_id: u32,
    },
    NoValidCandidate {
        rejected_by_reason: BTreeMap<String, usize>,
    },
}

pub fn constrained_cavity_from_selected_tets(
    tets: &[TetCandidate],
    selected_tet_indices: &[usize],
    protected_node_ids: Vec<u32>,
) -> Result<ConstrainedCavity, ConstrainedCavityExtractionError> {
    let selected = selected_tet_index_set(tets, selected_tet_indices)?;
    let cavity = build_constrained_cavity_from_index_set(tets, &selected, protected_node_ids);
    validate_constrained_cavity(&cavity).map_err(ConstrainedCavityExtractionError::Validation)?;
    Ok(cavity)
}

pub fn constrained_cavity_from_selected_tets_with_anchor_trim(
    tets: &[TetCandidate],
    selected_tet_indices: &[usize],
    anchor_tet_index: usize,
    protected_node_ids: Vec<u32>,
) -> Result<Option<ConstrainedCavity>, ConstrainedCavityExtractionError> {
    if anchor_tet_index >= tets.len() {
        return Err(
            ConstrainedCavityExtractionError::SelectedTetIndexOutOfBounds {
                tet_index: anchor_tet_index,
                tet_count: tets.len(),
            },
        );
    }
    let selected = selected_tet_index_set(tets, selected_tet_indices)?;
    if !selected.contains(&anchor_tet_index) {
        return Ok(None);
    }

    anchor_trimmed_constrained_cavity(tets, selected, anchor_tet_index, protected_node_ids)
}

fn anchor_trimmed_constrained_cavity(
    tets: &[TetCandidate],
    selected: BTreeSet<usize>,
    anchor_tet_index: usize,
    protected_node_ids: Vec<u32>,
) -> Result<Option<ConstrainedCavity>, ConstrainedCavityExtractionError> {
    let Some(selected) = anchor_connected_tet_subset(tets, &selected, anchor_tet_index) else {
        return Ok(None);
    };
    let mut pending = vec![selected.clone()];
    let mut visited = BTreeSet::<BTreeSet<usize>>::from([selected]);
    let mut evaluated = 0_usize;

    while !pending.is_empty() && evaluated < MAX_ANCHOR_TRIM_STATES {
        let best_index = pending
            .iter()
            .enumerate()
            .min_by_key(|(_, candidate)| {
                (
                    boundary_edge_defect_score(tets, candidate),
                    Reverse(candidate.len()),
                )
            })
            .map(|(index, _)| index)
            .expect("pending should be non-empty");
        let selected = pending.swap_remove(best_index);
        evaluated += 1;
        let cavity =
            build_constrained_cavity_from_index_set(tets, &selected, protected_node_ids.clone());
        match validate_constrained_cavity(&cavity) {
            Ok(_) => return Ok(Some(cavity)),
            Err(ConstrainedCavityValidationError::NonManifoldBoundaryEdge { .. }) => {
                for edge in non_manifold_boundary_edges(tets, &selected) {
                    for owner in boundary_face_owner_indices_for_edge(tets, &selected, edge) {
                        if owner == anchor_tet_index {
                            continue;
                        }
                        let mut candidate = selected.clone();
                        candidate.remove(&owner);
                        let Some(connected) =
                            anchor_connected_tet_subset(tets, &candidate, anchor_tet_index)
                        else {
                            continue;
                        };
                        if visited.insert(connected.clone()) {
                            pending.push(connected);
                        }
                    }
                }
            }
            Err(ConstrainedCavityValidationError::TooFewBoundaryFaces { .. }) => continue,
            Err(err) => return Err(ConstrainedCavityExtractionError::Validation(err)),
        }
    }
    Ok(None)
}

fn selected_tet_index_set(
    tets: &[TetCandidate],
    selected_tet_indices: &[usize],
) -> Result<BTreeSet<usize>, ConstrainedCavityExtractionError> {
    if selected_tet_indices.is_empty() {
        return Err(ConstrainedCavityExtractionError::EmptySelection);
    }

    let mut selected = BTreeSet::<usize>::new();
    for tet_index in selected_tet_indices {
        if *tet_index >= tets.len() {
            return Err(
                ConstrainedCavityExtractionError::SelectedTetIndexOutOfBounds {
                    tet_index: *tet_index,
                    tet_count: tets.len(),
                },
            );
        }
        if !selected.insert(*tet_index) {
            return Err(
                ConstrainedCavityExtractionError::DuplicateSelectedTetIndex {
                    tet_index: *tet_index,
                },
            );
        }
    }
    Ok(selected)
}

fn build_constrained_cavity_from_index_set(
    tets: &[TetCandidate],
    selected: &BTreeSet<usize>,
    protected_node_ids: Vec<u32>,
) -> ConstrainedCavity {
    let mut target_volume_m3 = 0.0_f64;
    let mut face_owners = BTreeMap::<[u32; 3], Vec<(usize, [u32; 3])>>::new();
    for tet_index in selected {
        let tet = &tets[*tet_index];
        target_volume_m3 += tet.volume_m3;
        for face in tet_faces(tet.node_ids) {
            face_owners
                .entry(sorted_face(face))
                .or_default()
                .push((*tet_index, face));
        }
    }

    let mut boundary_faces = Vec::<ConstrainedCavityBoundaryFace>::new();
    for owners in face_owners.values() {
        if owners.len() != 1 {
            continue;
        }
        let (tet_index, oriented_face) = owners[0];
        boundary_faces.push(ConstrainedCavityBoundaryFace {
            node_ids: oriented_face,
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: tets[tet_index].region_ids.clone(),
        });
    }

    ConstrainedCavity {
        removed_tet_ids: selected
            .iter()
            .map(|tet_index| tets[*tet_index].tet_id)
            .collect(),
        boundary_faces,
        protected_node_ids,
        target_volume_m3,
    }
}

fn boundary_face_owner_indices_for_edge(
    tets: &[TetCandidate],
    selected: &BTreeSet<usize>,
    edge: [u32; 2],
) -> Vec<usize> {
    let target_edge = sorted_edge(edge);
    boundary_face_owners(tets, selected)
        .into_iter()
        .filter_map(|(_, owners)| (owners.len() == 1).then_some(owners[0]))
        .filter_map(|(tet_index, face)| {
            face_edges(face)
                .into_iter()
                .any(|face_edge| sorted_edge(face_edge) == target_edge)
                .then_some(tet_index)
        })
        .collect()
}

fn non_manifold_boundary_edges(tets: &[TetCandidate], selected: &BTreeSet<usize>) -> Vec<[u32; 2]> {
    let mut edge_counts = BTreeMap::<[u32; 2], usize>::new();
    for (_, owners) in boundary_face_owners(tets, selected) {
        if owners.len() != 1 {
            continue;
        }
        for edge in face_edges(owners[0].1) {
            *edge_counts.entry(sorted_edge(edge)).or_default() += 1;
        }
    }
    edge_counts
        .into_iter()
        .filter_map(|(edge, count)| (count != 2).then_some(edge))
        .collect()
}

fn boundary_edge_defect_score(tets: &[TetCandidate], selected: &BTreeSet<usize>) -> usize {
    let mut edge_counts = BTreeMap::<[u32; 2], usize>::new();
    for (_, owners) in boundary_face_owners(tets, selected) {
        if owners.len() != 1 {
            continue;
        }
        for edge in face_edges(owners[0].1) {
            *edge_counts.entry(sorted_edge(edge)).or_default() += 1;
        }
    }
    edge_counts
        .values()
        .map(|count| count.abs_diff(2))
        .sum::<usize>()
}

fn boundary_face_owners(
    tets: &[TetCandidate],
    selected: &BTreeSet<usize>,
) -> BTreeMap<[u32; 3], Vec<(usize, [u32; 3])>> {
    let mut face_owners = BTreeMap::<[u32; 3], Vec<(usize, [u32; 3])>>::new();
    for tet_index in selected {
        for face in tet_faces(tets[*tet_index].node_ids) {
            face_owners
                .entry(sorted_face(face))
                .or_default()
                .push((*tet_index, face));
        }
    }
    face_owners
}

fn anchor_connected_tet_subset(
    tets: &[TetCandidate],
    selected: &BTreeSet<usize>,
    anchor_tet_index: usize,
) -> Option<BTreeSet<usize>> {
    if !selected.contains(&anchor_tet_index) {
        return None;
    }
    let mut face_to_tets = BTreeMap::<[u32; 3], Vec<usize>>::new();
    for tet_index in selected {
        for face in tet_faces(tets[*tet_index].node_ids) {
            face_to_tets
                .entry(sorted_face(face))
                .or_default()
                .push(*tet_index);
        }
    }
    let mut connected = BTreeSet::<usize>::new();
    let mut pending = vec![anchor_tet_index];
    while let Some(tet_index) = pending.pop() {
        if !connected.insert(tet_index) {
            continue;
        }
        for face in tet_faces(tets[tet_index].node_ids) {
            if let Some(neighbors) = face_to_tets.get(&sorted_face(face)) {
                for neighbor in neighbors {
                    if selected.contains(neighbor) && !connected.contains(neighbor) {
                        pending.push(*neighbor);
                    }
                }
            }
        }
    }
    Some(connected)
}

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

    if interior_candidate_nodes.is_empty() {
        if boundary_node_ids.len() == 4 {
            let Some(refill) = single_tet_refill_candidate(cavity, &boundary_node_map, options)
                .map_err(ConstrainedCavityRefillError::Validation)?
            else {
                record_refill_rejection(&mut rejected_by_reason, "single_tet_candidate_rejected");
                return Ok(ConstrainedCavityRefillEvaluation {
                    refill: None,
                    rejected_by_reason,
                });
            };
            return Ok(ConstrainedCavityRefillEvaluation {
                refill: Some(refill),
                rejected_by_reason,
            });
        };
        match boundary_node_refill_candidate(cavity, &boundary_node_map, options) {
            Ok(Ok(refill)) => {
                return Ok(ConstrainedCavityRefillEvaluation {
                    refill: Some(refill),
                    rejected_by_reason,
                });
            }
            Ok(Err(reason)) => record_refill_rejection(&mut rejected_by_reason, reason),
            Err(err) => {
                record_refill_rejection(&mut rejected_by_reason, refill_validation_reason(&err))
            }
        }
        return Ok(ConstrainedCavityRefillEvaluation {
            refill: None,
            rejected_by_reason,
        });
    }

    let mut seen_interior_nodes = BTreeSet::<u32>::new();
    let tolerance = MeshingTolerance::default();
    let mut best = None::<ConstrainedCavityRefill>;
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
    if best.is_none() && boundary_node_ids.len() > 4 {
        match boundary_node_refill_candidate(cavity, &boundary_node_map, options) {
            Ok(Ok(refill)) => best = Some(refill),
            Ok(Err(reason)) => record_refill_rejection(&mut rejected_by_reason, reason),
            Err(err) => {
                record_refill_rejection(&mut rejected_by_reason, refill_validation_reason(&err))
            }
        }
    }

    Ok(ConstrainedCavityRefillEvaluation {
        refill: best,
        rejected_by_reason,
    })
}

pub fn validate_constrained_cavity(
    cavity: &ConstrainedCavity,
) -> Result<ConstrainedCavityValidationReport, ConstrainedCavityValidationError> {
    if cavity.removed_tet_ids.is_empty() {
        return Err(ConstrainedCavityValidationError::EmptyRemovedTetSet);
    }
    if !cavity.target_volume_m3.is_finite() || cavity.target_volume_m3 <= 0.0 {
        return Err(ConstrainedCavityValidationError::InvalidTargetVolume {
            target_volume_m3: cavity.target_volume_m3,
        });
    }
    if cavity.boundary_faces.len() < 4 {
        return Err(ConstrainedCavityValidationError::TooFewBoundaryFaces {
            boundary_face_count: cavity.boundary_faces.len(),
        });
    }

    let mut boundary_faces = BTreeSet::<[u32; 3]>::new();
    let mut boundary_edges = BTreeMap::<[u32; 2], usize>::new();
    let mut boundary_nodes = BTreeSet::<u32>::new();
    for (face_index, face) in cavity.boundary_faces.iter().enumerate() {
        if face.node_ids[0] == face.node_ids[1]
            || face.node_ids[0] == face.node_ids[2]
            || face.node_ids[1] == face.node_ids[2]
        {
            return Err(ConstrainedCavityValidationError::DegenerateBoundaryFace {
                face_index,
                node_ids: face.node_ids,
            });
        }
        let sorted_face = sorted_face(face.node_ids);
        if !boundary_faces.insert(sorted_face) {
            return Err(ConstrainedCavityValidationError::DuplicateBoundaryFace {
                node_ids: sorted_face,
            });
        }
        for node_id in face.node_ids {
            boundary_nodes.insert(node_id);
        }
        for edge in face_edges(face.node_ids) {
            *boundary_edges.entry(sorted_edge(edge)).or_default() += 1;
        }
    }

    for (edge, face_count) in &boundary_edges {
        if *face_count != 2 {
            return Err(ConstrainedCavityValidationError::NonManifoldBoundaryEdge {
                node_ids: *edge,
                face_count: *face_count,
            });
        }
    }

    for node_id in &cavity.protected_node_ids {
        if !boundary_nodes.contains(node_id) {
            return Err(
                ConstrainedCavityValidationError::ProtectedNodeOutsideBoundary {
                    node_id: *node_id,
                },
            );
        }
    }

    Ok(ConstrainedCavityValidationReport {
        boundary_face_count: cavity.boundary_faces.len(),
        boundary_edge_count: boundary_edges.len(),
        boundary_node_count: boundary_nodes.len(),
        protected_node_count: cavity.protected_node_ids.len(),
        target_volume_m3: cavity.target_volume_m3,
    })
}

pub fn validate_constrained_cavity_refill_volume(
    target_volume_m3: f64,
    candidate_volume_m3: f64,
    relative_tolerance: f64,
) -> Result<(), ConstrainedCavityValidationError> {
    if !target_volume_m3.is_finite() || target_volume_m3 <= 0.0 {
        return Err(ConstrainedCavityValidationError::InvalidTargetVolume { target_volume_m3 });
    }
    let tolerance_m3 = target_volume_m3.max(1.0e-18) * relative_tolerance.max(0.0);
    if !candidate_volume_m3.is_finite()
        || candidate_volume_m3 <= 0.0
        || (candidate_volume_m3 - target_volume_m3).abs() > tolerance_m3
    {
        return Err(ConstrainedCavityValidationError::InvalidRefillVolume {
            target_volume_m3,
            candidate_volume_m3,
            tolerance_m3,
        });
    }
    Ok(())
}

pub fn validate_constrained_cavity_boundary_preserved(
    cavity: &ConstrainedCavity,
    candidate_boundary_faces: &[ConstrainedCavityBoundaryFace],
) -> Result<(), ConstrainedCavityValidationError> {
    if cavity.boundary_faces.len() != candidate_boundary_faces.len() {
        return Err(
            ConstrainedCavityValidationError::BoundaryFaceCountMismatch {
                expected_count: cavity.boundary_faces.len(),
                candidate_count: candidate_boundary_faces.len(),
            },
        );
    }

    let expected_faces = boundary_face_map(&cavity.boundary_faces)?;
    let candidate_faces = boundary_face_map(candidate_boundary_faces)?;

    for expected_face in expected_faces.keys() {
        if !candidate_faces.contains_key(expected_face) {
            return Err(ConstrainedCavityValidationError::MissingBoundaryFace {
                node_ids: *expected_face,
            });
        }
    }
    for candidate_face in candidate_faces.keys() {
        if !expected_faces.contains_key(candidate_face) {
            return Err(ConstrainedCavityValidationError::UnexpectedBoundaryFace {
                node_ids: *candidate_face,
            });
        }
    }

    for (face_key, expected) in &expected_faces {
        let candidate = candidate_faces
            .get(face_key)
            .expect("candidate face should exist after key comparison");
        if expected.source_face_id != candidate.source_face_id {
            return Err(
                ConstrainedCavityValidationError::BoundarySourceFaceMismatch {
                    node_ids: *face_key,
                    expected_source_face_id: expected.source_face_id,
                    candidate_source_face_id: candidate.source_face_id,
                },
            );
        }
        let expected_edges = boundary_face_source_edges(expected)?;
        let candidate_edges = boundary_face_source_edges(candidate)?;
        for (edge_key, expected_source_edge_id) in expected_edges {
            let candidate_source_edge_id = candidate_edges.get(&edge_key).copied().flatten();
            if expected_source_edge_id != candidate_source_edge_id {
                return Err(
                    ConstrainedCavityValidationError::BoundarySourceEdgeMismatch {
                        node_ids: edge_key,
                        expected_source_edge_id,
                        candidate_source_edge_id,
                    },
                );
            }
        }
        let expected_regions = sorted_region_ids(&expected.region_ids);
        let candidate_regions = sorted_region_ids(&candidate.region_ids);
        if expected_regions != candidate_regions {
            return Err(ConstrainedCavityValidationError::BoundaryRegionMismatch {
                node_ids: *face_key,
                expected_region_ids: expected_regions,
                candidate_region_ids: candidate_regions,
            });
        }
    }

    Ok(())
}

pub fn split_constrained_cavity_boundary_face(
    face: &ConstrainedCavityBoundaryFace,
    split_node_id: u32,
) -> Result<[ConstrainedCavityBoundaryFace; 3], ConstrainedCavityBoundarySplitError> {
    if face.node_ids.contains(&split_node_id) {
        return Err(
            ConstrainedCavityBoundarySplitError::SplitNodeReusesFaceNode {
                node_id: split_node_id,
            },
        );
    }
    let perimeter_source_edges = boundary_face_source_edges(face).map_err(|_| {
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        }
    })?;
    let [a, b, c] = face.node_ids;
    Ok([
        split_child_boundary_face(face, [a, b, split_node_id], &perimeter_source_edges),
        split_child_boundary_face(face, [b, c, split_node_id], &perimeter_source_edges),
        split_child_boundary_face(face, [c, a, split_node_id], &perimeter_source_edges),
    ])
}

pub fn split_constrained_cavity_boundary_faces(
    faces: &[ConstrainedCavityBoundaryFace],
    face_node_ids: [u32; 3],
    split_node_id: u32,
) -> Result<Vec<ConstrainedCavityBoundaryFace>, ConstrainedCavityBoundarySplitError> {
    let target = sorted_face(face_node_ids);
    let mut split_faces = Vec::<ConstrainedCavityBoundaryFace>::with_capacity(faces.len() + 2);
    let mut found = false;
    for face in faces {
        if sorted_face(face.node_ids) == target {
            found = true;
            split_faces.extend(split_constrained_cavity_boundary_face(face, split_node_id)?);
        } else {
            split_faces.push(face.clone());
        }
    }
    if !found {
        return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace { node_ids: target });
    }
    Ok(split_faces)
}

fn split_child_boundary_face(
    parent: &ConstrainedCavityBoundaryFace,
    node_ids: [u32; 3],
    perimeter_source_edges: &BTreeMap<[u32; 2], Option<u32>>,
) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        source_face_id: parent.source_face_id,
        source_edge_ids: face_edges(node_ids).map(|edge| {
            perimeter_source_edges
                .get(&sorted_edge(edge))
                .copied()
                .flatten()
        }),
        region_ids: parent.region_ids.clone(),
    }
}

fn validate_refill_options(
    options: ConstrainedCavityRefillOptions,
) -> Result<(), ConstrainedCavityRefillError> {
    if !options.min_volume_m3.is_finite()
        || options.min_volume_m3 <= 0.0
        || !options.max_aspect_ratio.is_finite()
        || options.max_aspect_ratio <= 0.0
        || !options.min_scaled_jacobian.is_finite()
        || options.min_scaled_jacobian < 0.0
        || !options.volume_relative_tolerance.is_finite()
        || options.volume_relative_tolerance < 0.0
        || !options.min_protected_node_distance_m.is_finite()
        || options.min_protected_node_distance_m < 0.0
    {
        return Err(ConstrainedCavityRefillError::InvalidOptions);
    }
    Ok(())
}

fn boundary_node_coordinates(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
) -> Result<BTreeMap<u32, Point3>, ConstrainedCavityRefillError> {
    let coordinates = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for face in &cavity.boundary_faces {
        for node_id in face.node_ids {
            if !coordinates.contains_key(&node_id) {
                return Err(ConstrainedCavityRefillError::MissingBoundaryNode { node_id });
            }
        }
    }
    Ok(coordinates)
}

fn cavity_boundary_node_ids(cavity: &ConstrainedCavity) -> BTreeSet<u32> {
    cavity
        .boundary_faces
        .iter()
        .flat_map(|face| face.node_ids)
        .collect()
}

fn candidate_respects_protected_boundary_distance(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    point: Point3,
    options: ConstrainedCavityRefillOptions,
) -> bool {
    if options.min_protected_node_distance_m <= 0.0 || cavity.protected_node_ids.is_empty() {
        return true;
    }
    let min_distance_squared = options.min_protected_node_distance_m.powi(2);
    cavity.protected_node_ids.iter().all(|node_id| {
        boundary_nodes.get(node_id).is_none_or(|protected_point| {
            distance_squared(point, *protected_point) > min_distance_squared
        })
    })
}

fn cavity_boundary_triangles(
    cavity: &ConstrainedCavity,
    nodes: &BTreeMap<u32, Point3>,
) -> Result<Vec<Triangle3>, ConstrainedCavityRefillError> {
    cavity
        .boundary_faces
        .iter()
        .map(|face| {
            Ok([
                *nodes.get(&face.node_ids[0]).ok_or(
                    ConstrainedCavityRefillError::MissingBoundaryNode {
                        node_id: face.node_ids[0],
                    },
                )?,
                *nodes.get(&face.node_ids[1]).ok_or(
                    ConstrainedCavityRefillError::MissingBoundaryNode {
                        node_id: face.node_ids[1],
                    },
                )?,
                *nodes.get(&face.node_ids[2]).ok_or(
                    ConstrainedCavityRefillError::MissingBoundaryNode {
                        node_id: face.node_ids[2],
                    },
                )?,
            ])
        })
        .collect()
}

fn single_tet_refill_candidate(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefill>, ConstrainedCavityValidationError> {
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    if node_ids.len() != 4 {
        return Ok(None);
    }
    let points = [
        boundary_nodes[&node_ids[0]],
        boundary_nodes[&node_ids[1]],
        boundary_nodes[&node_ids[2]],
        boundary_nodes[&node_ids[3]],
    ];
    let Some(tet) = raw_refill_tet(
        [node_ids[0], node_ids[1], node_ids[2], node_ids[3]],
        points,
        options,
    ) else {
        return Ok(None);
    };
    let refill = refill_from_tets(cavity, vec![tet], options.volume_relative_tolerance)?;
    Ok(Some(refill))
}

fn boundary_node_refill_candidate(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Result<Result<ConstrainedCavityRefill, &'static str>, ConstrainedCavityValidationError> {
    let boundary_triangles = cavity
        .boundary_faces
        .iter()
        .map(|face| {
            [
                boundary_nodes[&face.node_ids[0]],
                boundary_nodes[&face.node_ids[1]],
                boundary_nodes[&face.node_ids[2]],
            ]
        })
        .collect::<Vec<_>>();
    let points = cavity_boundary_node_ids(cavity)
        .into_iter()
        .map(|node_id| ConnectivityPoint {
            node_id,
            coordinates_m: boundary_nodes[&node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut refill_tets = Vec::<ConstrainedCavityRefillTet>::new();
    let mut first_rejection = None::<&'static str>;
    for tet in tetrahedralize_points(&points) {
        let node_ids = tet.vertices.map(|index| points[index].node_id);
        let tet_points = tet.vertices.map(|index| points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tet_centroid(tet_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        match raw_refill_tet_with_rejection_reason(node_ids, tet_points, options) {
            Ok(tet) => refill_tets.push(tet),
            Err(reason) => {
                if first_rejection.is_none() {
                    first_rejection = Some(boundary_node_refill_rejection_reason(reason));
                }
            }
        }
    }
    if refill_tets.is_empty() {
        return Ok(Err(
            first_rejection.unwrap_or("boundary_node_delaunay_empty")
        ));
    }
    match refill_from_tets(
        cavity,
        refill_tets.clone(),
        options.volume_relative_tolerance,
    ) {
        Ok(refill) => Ok(Ok(refill)),
        Err(err) => {
            let Some(completed_tets) = complete_missing_boundary_face_tets(
                cavity,
                boundary_nodes,
                refill_tets,
                &boundary_triangles,
                options,
            )?
            else {
                return Ok(Err(boundary_node_refill_validation_reason(&err)));
            };
            let refill =
                match refill_from_tets(cavity, completed_tets, options.volume_relative_tolerance) {
                    Ok(refill) => refill,
                    Err(err) => return Ok(Err(boundary_node_refill_validation_reason(&err))),
                };
            Ok(Ok(refill))
        }
    }
}

fn boundary_node_refill_rejection_reason(reason: &'static str) -> &'static str {
    match reason {
        "star_tet_min_volume" => "boundary_node_tet_min_volume",
        "star_tet_aspect_ratio" => "boundary_node_tet_aspect_ratio",
        "star_tet_scaled_jacobian" => "boundary_node_tet_scaled_jacobian",
        other => other,
    }
}

fn boundary_node_refill_validation_reason(
    error: &ConstrainedCavityValidationError,
) -> &'static str {
    match refill_validation_reason(error) {
        "boundary_face_count_mismatch" => "boundary_node_boundary_face_count_mismatch",
        "missing_boundary_face" => "boundary_node_missing_boundary_face",
        "unexpected_boundary_face" => "boundary_node_unexpected_boundary_face",
        "volume_mismatch" => "boundary_node_volume_mismatch",
        "boundary_source_face_mismatch" => "boundary_node_boundary_source_face_mismatch",
        "boundary_source_edge_mismatch" => "boundary_node_boundary_source_edge_mismatch",
        "boundary_region_mismatch" => "boundary_node_boundary_region_mismatch",
        "invalid_cavity" => "boundary_node_invalid_cavity",
        other => other,
    }
}

fn complete_missing_boundary_face_tets(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    mut refill_tets: Vec<ConstrainedCavityRefillTet>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<Vec<ConstrainedCavityRefillTet>>, ConstrainedCavityValidationError> {
    let mut changed = false;
    loop {
        let missing_faces = missing_refill_boundary_faces(cavity, &refill_tets)?;
        let Some(missing_face) = missing_faces.into_iter().next() else {
            break;
        };
        let Some(tet) = best_boundary_face_completion_tet(
            missing_face,
            cavity,
            boundary_nodes,
            boundary_triangles,
            options,
        ) else {
            return Ok(None);
        };
        if refill_tets
            .iter()
            .any(|existing| sorted_tet_nodes(existing.node_ids) == sorted_tet_nodes(tet.node_ids))
        {
            return Ok(None);
        }
        refill_tets.push(tet);
        changed = true;
    }
    Ok(changed.then_some(refill_tets))
}

fn missing_refill_boundary_faces(
    cavity: &ConstrainedCavity,
    refill_tets: &[ConstrainedCavityRefillTet],
) -> Result<Vec<[u32; 3]>, ConstrainedCavityValidationError> {
    let expected = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let actual = boundary_faces_from_refill_tets(cavity, refill_tets)?
        .into_iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    Ok(expected.difference(&actual).copied().collect())
}

fn best_boundary_face_completion_tet(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Option<ConstrainedCavityRefillTet> {
    cavity_boundary_node_ids(cavity)
        .into_iter()
        .filter(|node_id| !face.contains(node_id))
        .filter_map(|node_id| {
            let node_ids = [face[0], face[1], face[2], node_id];
            let points = node_ids.map(|id| boundary_nodes[&id]);
            if point_in_closed_triangle_surface(
                tet_centroid(points),
                boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
            {
                return None;
            }
            raw_refill_tet_with_rejection_reason(node_ids, points, options).ok()
        })
        .max_by(|left, right| {
            left.exact_scaled_jacobian
                .total_cmp(&right.exact_scaled_jacobian)
                .then_with(|| right.aspect_ratio.total_cmp(&left.aspect_ratio))
        })
}

fn star_refill_candidate_with_rejection_reason(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    interior_node: ConstrainedCavityNode,
    options: ConstrainedCavityRefillOptions,
) -> Result<Result<ConstrainedCavityRefill, &'static str>, ConstrainedCavityValidationError> {
    let mut tets = Vec::<ConstrainedCavityRefillTet>::with_capacity(cavity.boundary_faces.len());
    for face in &cavity.boundary_faces {
        let node_ids = [
            face.node_ids[0],
            face.node_ids[1],
            face.node_ids[2],
            interior_node.node_id,
        ];
        let points = [
            boundary_nodes[&face.node_ids[0]],
            boundary_nodes[&face.node_ids[1]],
            boundary_nodes[&face.node_ids[2]],
            interior_node.coordinates_m,
        ];
        let tet = match raw_refill_tet_with_rejection_reason(node_ids, points, options) {
            Ok(tet) => tet,
            Err(reason) => return Ok(Err(reason)),
        };
        tets.push(tet);
    }
    let refill = refill_from_tets(cavity, tets, options.volume_relative_tolerance)?;
    Ok(Ok(refill))
}

fn raw_refill_tet(
    node_ids: [u32; 4],
    points: [Point3; 4],
    options: ConstrainedCavityRefillOptions,
) -> Option<ConstrainedCavityRefillTet> {
    raw_refill_tet_with_rejection_reason(node_ids, points, options).ok()
}

fn raw_refill_tet_with_rejection_reason(
    mut node_ids: [u32; 4],
    points: [Point3; 4],
    options: ConstrainedCavityRefillOptions,
) -> Result<ConstrainedCavityRefillTet, &'static str> {
    let mut signed_volume_m3 = tet_signed_volume(points);
    if signed_volume_m3 < 0.0 {
        node_ids.swap(1, 2);
        signed_volume_m3 = -signed_volume_m3;
    }
    let volume_m3 = signed_volume_m3.abs();
    if volume_m3 < options.min_volume_m3 {
        return Err("star_tet_min_volume");
    }
    let aspect_ratio = tet_edge_aspect_ratio(points);
    if !aspect_ratio.is_finite() || aspect_ratio > options.max_aspect_ratio {
        return Err("star_tet_aspect_ratio");
    }
    let exact_scaled_jacobian = tet_scaled_jacobian(points);
    if !exact_scaled_jacobian.is_finite() || exact_scaled_jacobian < options.min_scaled_jacobian {
        return Err("star_tet_scaled_jacobian");
    }
    Ok(ConstrainedCavityRefillTet {
        node_ids,
        volume_m3,
        aspect_ratio,
        exact_scaled_jacobian,
    })
}

fn refill_from_tets(
    cavity: &ConstrainedCavity,
    tets: Vec<ConstrainedCavityRefillTet>,
    volume_relative_tolerance: f64,
) -> Result<ConstrainedCavityRefill, ConstrainedCavityValidationError> {
    let boundary_faces = boundary_faces_from_refill_tets(cavity, &tets)?;
    validate_constrained_cavity_boundary_preserved(cavity, &boundary_faces)?;
    let total_volume_m3 = tets.iter().map(|tet| tet.volume_m3).sum::<f64>();
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        total_volume_m3,
        volume_relative_tolerance,
    )?;
    Ok(ConstrainedCavityRefill {
        tets,
        boundary_faces,
        total_volume_m3,
    })
}

fn boundary_faces_from_refill_tets(
    cavity: &ConstrainedCavity,
    tets: &[ConstrainedCavityRefillTet],
) -> Result<Vec<ConstrainedCavityBoundaryFace>, ConstrainedCavityValidationError> {
    let cavity_faces = boundary_face_map(&cavity.boundary_faces)?;
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for tet in tets {
        for face in tet_faces(tet.node_ids) {
            *face_counts.entry(sorted_face(face)).or_default() += 1;
        }
    }
    let boundary_faces = face_counts
        .into_iter()
        .filter_map(|(face, count)| {
            (count == 1).then(|| {
                cavity_faces
                    .get(&face)
                    .map(|source| (*source).clone())
                    .unwrap_or(ConstrainedCavityBoundaryFace {
                        node_ids: face,
                        source_face_id: None,
                        source_edge_ids: [None, None, None],
                        region_ids: Vec::new(),
                    })
            })
        })
        .collect::<Vec<_>>();
    Ok(boundary_faces)
}

fn refill_is_better(
    candidate: &ConstrainedCavityRefill,
    current: &ConstrainedCavityRefill,
) -> bool {
    let candidate_min = candidate
        .tets
        .iter()
        .map(|tet| tet.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    let current_min = current
        .tets
        .iter()
        .map(|tet| tet.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    candidate_min > current_min + 1.0e-12
        || ((candidate_min - current_min).abs() <= 1.0e-12
            && candidate.tets.len() < current.tets.len())
}

fn record_refill_rejection(rejected_by_reason: &mut BTreeMap<String, usize>, reason: &str) {
    *rejected_by_reason.entry(reason.to_string()).or_default() += 1;
}

fn refill_validation_reason(error: &ConstrainedCavityValidationError) -> &'static str {
    match error {
        ConstrainedCavityValidationError::InvalidRefillVolume { .. } => "volume_mismatch",
        ConstrainedCavityValidationError::BoundaryFaceCountMismatch { .. } => {
            "boundary_face_count_mismatch"
        }
        ConstrainedCavityValidationError::MissingBoundaryFace { .. } => "missing_boundary_face",
        ConstrainedCavityValidationError::UnexpectedBoundaryFace { .. } => {
            "unexpected_boundary_face"
        }
        ConstrainedCavityValidationError::BoundarySourceFaceMismatch { .. } => {
            "boundary_source_face_mismatch"
        }
        ConstrainedCavityValidationError::BoundarySourceEdgeMismatch { .. } => {
            "boundary_source_edge_mismatch"
        }
        ConstrainedCavityValidationError::BoundaryRegionMismatch { .. } => {
            "boundary_region_mismatch"
        }
        ConstrainedCavityValidationError::EmptyRemovedTetSet
        | ConstrainedCavityValidationError::InvalidTargetVolume { .. }
        | ConstrainedCavityValidationError::TooFewBoundaryFaces { .. }
        | ConstrainedCavityValidationError::DegenerateBoundaryFace { .. }
        | ConstrainedCavityValidationError::DuplicateBoundaryFace { .. }
        | ConstrainedCavityValidationError::NonManifoldBoundaryEdge { .. }
        | ConstrainedCavityValidationError::ProtectedNodeOutsideBoundary { .. } => "invalid_cavity",
    }
}

fn sorted_face(mut node_ids: [u32; 3]) -> [u32; 3] {
    node_ids.sort();
    node_ids
}

fn sorted_tet_nodes(mut node_ids: [u32; 4]) -> [u32; 4] {
    node_ids.sort();
    node_ids
}

fn sorted_edge(mut node_ids: [u32; 2]) -> [u32; 2] {
    node_ids.sort();
    node_ids
}

fn face_edges(node_ids: [u32; 3]) -> [[u32; 2]; 3] {
    [
        [node_ids[0], node_ids[1]],
        [node_ids[1], node_ids[2]],
        [node_ids[2], node_ids[0]],
    ]
}

fn boundary_face_map(
    faces: &[ConstrainedCavityBoundaryFace],
) -> Result<BTreeMap<[u32; 3], &ConstrainedCavityBoundaryFace>, ConstrainedCavityValidationError> {
    let mut map = BTreeMap::<[u32; 3], &ConstrainedCavityBoundaryFace>::new();
    for (face_index, face) in faces.iter().enumerate() {
        if face.node_ids[0] == face.node_ids[1]
            || face.node_ids[0] == face.node_ids[2]
            || face.node_ids[1] == face.node_ids[2]
        {
            return Err(ConstrainedCavityValidationError::DegenerateBoundaryFace {
                face_index,
                node_ids: face.node_ids,
            });
        }
        let key = sorted_face(face.node_ids);
        if map.insert(key, face).is_some() {
            return Err(ConstrainedCavityValidationError::DuplicateBoundaryFace { node_ids: key });
        }
    }
    Ok(map)
}

fn boundary_face_source_edges(
    face: &ConstrainedCavityBoundaryFace,
) -> Result<BTreeMap<[u32; 2], Option<u32>>, ConstrainedCavityValidationError> {
    let mut edge_sources = BTreeMap::<[u32; 2], Option<u32>>::new();
    for (edge, source_edge_id) in face_edges(face.node_ids)
        .into_iter()
        .zip(face.source_edge_ids)
    {
        let key = sorted_edge(edge);
        if edge_sources.insert(key, source_edge_id).is_some() {
            return Err(ConstrainedCavityValidationError::DegenerateBoundaryFace {
                face_index: 0,
                node_ids: face.node_ids,
            });
        }
    }
    Ok(edge_sources)
}

fn sorted_region_ids(region_ids: &[String]) -> Vec<String> {
    let mut sorted = region_ids.to_vec();
    sorted.sort();
    sorted.dedup();
    sorted
}

fn tet_faces(node_ids: [u32; 4]) -> [[u32; 3]; 4] {
    [
        [node_ids[0], node_ids[2], node_ids[1]],
        [node_ids[0], node_ids[1], node_ids[3]],
        [node_ids[1], node_ids[2], node_ids[3]],
        [node_ids[2], node_ids[0], node_ids[3]],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_single_tet_cavity_from_selected_tets() {
        let tets = vec![candidate_tet(7, [0, 1, 2, 3], 0.25, &["body"])];

        let cavity = constrained_cavity_from_selected_tets(&tets, &[0], vec![0, 1])
            .expect("single tet cavity should extract");

        assert_eq!(cavity.removed_tet_ids, vec![7]);
        assert_eq!(cavity.boundary_faces.len(), 4);
        assert_eq!(cavity.protected_node_ids, vec![0, 1]);
        assert_eq!(cavity.target_volume_m3, 0.25);
        assert!(cavity
            .boundary_faces
            .iter()
            .all(|face| face.region_ids == ["body"]));
    }

    #[test]
    fn extracts_boundary_faces_from_two_tet_cavity() {
        let tets = vec![
            candidate_tet(4, [0, 1, 2, 3], 0.25, &["left"]),
            candidate_tet(9, [0, 2, 1, 4], 0.35, &["right"]),
        ];

        let cavity = constrained_cavity_from_selected_tets(&tets, &[1, 0], vec![])
            .expect("two tet cavity should extract");

        let boundary_faces = cavity
            .boundary_faces
            .iter()
            .map(|face| sorted_face(face.node_ids))
            .collect::<BTreeSet<_>>();

        assert_eq!(cavity.removed_tet_ids, vec![4, 9]);
        assert_eq!(cavity.boundary_faces.len(), 6);
        assert!(!boundary_faces.contains(&[0, 1, 2]));
        assert_eq!(cavity.target_volume_m3, 0.60);
        validate_constrained_cavity(&cavity).expect("extracted cavity should validate");
    }

    #[test]
    fn rejects_duplicate_selected_tet_indices() {
        let tets = vec![candidate_tet(7, [0, 1, 2, 3], 0.25, &[])];

        let err = constrained_cavity_from_selected_tets(&tets, &[0, 0], vec![])
            .expect_err("duplicate selection should fail");

        assert_eq!(
            err,
            ConstrainedCavityExtractionError::DuplicateSelectedTetIndex { tet_index: 0 }
        );
    }

    #[test]
    fn rejects_selected_tet_indices_out_of_bounds() {
        let tets = vec![candidate_tet(7, [0, 1, 2, 3], 0.25, &[])];

        let err = constrained_cavity_from_selected_tets(&tets, &[1], vec![])
            .expect_err("out of bounds selection should fail");

        assert_eq!(
            err,
            ConstrainedCavityExtractionError::SelectedTetIndexOutOfBounds {
                tet_index: 1,
                tet_count: 1
            }
        );
    }

    #[test]
    fn rejects_selected_tets_with_open_boundary() {
        let tets = vec![
            candidate_tet(4, [0, 1, 2, 3], 0.25, &[]),
            candidate_tet(9, [0, 1, 4, 5], 0.35, &[]),
        ];

        let err = constrained_cavity_from_selected_tets(&tets, &[0, 1], vec![])
            .expect_err("nonmanifold selected cavity should fail");

        assert_eq!(
            err,
            ConstrainedCavityExtractionError::Validation(
                ConstrainedCavityValidationError::NonManifoldBoundaryEdge {
                    node_ids: [0, 1],
                    face_count: 4
                }
            )
        );
    }

    #[test]
    fn anchor_trim_removes_non_manifold_dangling_tet() {
        let tets = vec![
            candidate_tet(4, [0, 1, 2, 3], 0.25, &["anchor"]),
            candidate_tet(9, [0, 1, 4, 5], 0.35, &["dangling"]),
        ];

        let cavity =
            constrained_cavity_from_selected_tets_with_anchor_trim(&tets, &[0, 1], 0, vec![0, 1])
                .expect("trim should evaluate")
                .expect("trim should recover the anchor tet cavity");

        assert_eq!(cavity.removed_tet_ids, vec![4]);
        assert_eq!(cavity.target_volume_m3, 0.25);
        assert_eq!(cavity.protected_node_ids, vec![0, 1]);
        assert!(cavity
            .boundary_faces
            .iter()
            .all(|face| face.region_ids == ["anchor"]));
        validate_constrained_cavity(&cavity).expect("trimmed cavity should be manifold");
    }

    #[test]
    fn anchor_trim_preserves_requested_anchor() {
        let tets = vec![
            candidate_tet(4, [0, 1, 2, 3], 0.25, &["left"]),
            candidate_tet(9, [0, 1, 4, 5], 0.35, &["right"]),
        ];

        let cavity =
            constrained_cavity_from_selected_tets_with_anchor_trim(&tets, &[0, 1], 1, vec![])
                .expect("trim should evaluate")
                .expect("trim should keep the requested anchor tet");

        assert_eq!(cavity.removed_tet_ids, vec![9]);
        assert_eq!(cavity.target_volume_m3, 0.35);
        assert!(cavity
            .boundary_faces
            .iter()
            .all(|face| face.region_ids == ["right"]));
        validate_constrained_cavity(&cavity).expect("trimmed cavity should be manifold");
    }

    #[test]
    fn anchor_trim_searches_past_first_defective_edge() {
        let tets = vec![
            candidate_tet(4, [0, 1, 2, 3], 0.25, &["anchor"]),
            candidate_tet(9, [0, 1, 2, 4], 0.35, &["trimmed"]),
            candidate_tet(11, [0, 1, 2, 5], 0.45, &["kept"]),
            candidate_tet(13, [0, 1, 4, 5], 0.55, &["kept"]),
        ];

        let cavity =
            constrained_cavity_from_selected_tets_with_anchor_trim(&tets, &[0, 1, 2, 3], 0, vec![])
                .expect("trim should evaluate")
                .expect("trim should find an anchor-containing manifold subset");

        assert_eq!(cavity.removed_tet_ids, vec![4, 11, 13]);
        assert_eq!(cavity.target_volume_m3, 1.25);
        assert!(cavity
            .boundary_faces
            .iter()
            .all(|face| face.region_ids != ["trimmed"]));
        validate_constrained_cavity(&cavity).expect("trimmed cavity should be manifold");
    }

    #[test]
    fn anchor_trim_returns_none_when_anchor_not_selected() {
        let tets = vec![
            candidate_tet(4, [0, 1, 2, 3], 0.25, &[]),
            candidate_tet(9, [0, 1, 4, 5], 0.35, &[]),
        ];

        let cavity =
            constrained_cavity_from_selected_tets_with_anchor_trim(&tets, &[0], 1, Vec::new())
                .expect("trim should evaluate");

        assert!(cavity.is_none());
    }

    #[test]
    fn boundary_preservation_accepts_reoriented_faces_with_same_provenance() {
        let cavity = provenance_cavity();
        let candidate_faces = cavity
            .boundary_faces
            .iter()
            .map(|face| {
                let mut reoriented = face.clone();
                reoriented.node_ids = [face.node_ids[2], face.node_ids[1], face.node_ids[0]];
                reoriented.source_edge_ids = [
                    source_edge_for(face, [reoriented.node_ids[0], reoriented.node_ids[1]]),
                    source_edge_for(face, [reoriented.node_ids[1], reoriented.node_ids[2]]),
                    source_edge_for(face, [reoriented.node_ids[2], reoriented.node_ids[0]]),
                ];
                reoriented.region_ids.reverse();
                reoriented
            })
            .collect::<Vec<_>>();

        validate_constrained_cavity_boundary_preserved(&cavity, &candidate_faces)
            .expect("same boundary and provenance should validate");
    }

    #[test]
    fn boundary_preservation_rejects_missing_boundary_face() {
        let cavity = provenance_cavity();
        let mut candidate_faces = cavity.boundary_faces.clone();
        candidate_faces[0].node_ids = [10, 11, 12];

        let err = validate_constrained_cavity_boundary_preserved(&cavity, &candidate_faces)
            .expect_err("missing boundary face should fail");

        assert_eq!(
            err,
            ConstrainedCavityValidationError::MissingBoundaryFace {
                node_ids: [0, 1, 2]
            }
        );
    }

    #[test]
    fn boundary_preservation_rejects_source_face_mismatch() {
        let cavity = provenance_cavity();
        let mut candidate_faces = cavity.boundary_faces.clone();
        candidate_faces[0].source_face_id = Some(99);

        let err = validate_constrained_cavity_boundary_preserved(&cavity, &candidate_faces)
            .expect_err("source face mismatch should fail");

        assert_eq!(
            err,
            ConstrainedCavityValidationError::BoundarySourceFaceMismatch {
                node_ids: [0, 1, 2],
                expected_source_face_id: Some(10),
                candidate_source_face_id: Some(99)
            }
        );
    }

    #[test]
    fn boundary_preservation_rejects_source_edge_mismatch() {
        let cavity = provenance_cavity();
        let mut candidate_faces = cavity.boundary_faces.clone();
        candidate_faces[0].source_edge_ids[0] = Some(99);

        let err = validate_constrained_cavity_boundary_preserved(&cavity, &candidate_faces)
            .expect_err("source edge mismatch should fail");

        assert_eq!(
            err,
            ConstrainedCavityValidationError::BoundarySourceEdgeMismatch {
                node_ids: [0, 1],
                expected_source_edge_id: Some(100),
                candidate_source_edge_id: Some(99)
            }
        );
    }

    #[test]
    fn boundary_preservation_rejects_region_mismatch() {
        let cavity = provenance_cavity();
        let mut candidate_faces = cavity.boundary_faces.clone();
        candidate_faces[0].region_ids = vec!["other".to_string()];

        let err = validate_constrained_cavity_boundary_preserved(&cavity, &candidate_faces)
            .expect_err("region mismatch should fail");

        assert_eq!(
            err,
            ConstrainedCavityValidationError::BoundaryRegionMismatch {
                node_ids: [0, 1, 2],
                expected_region_ids: vec!["fixed".to_string(), "loaded".to_string()],
                candidate_region_ids: vec!["other".to_string()]
            }
        );
    }

    #[test]
    fn boundary_face_split_preserves_source_face_regions_and_perimeter_edges() {
        let face = face_with_provenance(
            [0, 1, 2],
            10,
            [Some(100), Some(101), Some(102)],
            &["fixed", "loaded"],
        );

        let children = split_constrained_cavity_boundary_face(&face, 9).expect("face should split");

        assert_eq!(children.len(), 3);
        assert_eq!(children[0].node_ids, [0, 1, 9]);
        assert_eq!(children[1].node_ids, [1, 2, 9]);
        assert_eq!(children[2].node_ids, [2, 0, 9]);
        for child in &children {
            assert_eq!(child.source_face_id, Some(10));
            assert_eq!(
                sorted_region_ids(&child.region_ids),
                vec!["fixed".to_string(), "loaded".to_string()]
            );
        }
        assert_eq!(children[0].source_edge_ids, [Some(100), None, None]);
        assert_eq!(children[1].source_edge_ids, [Some(101), None, None]);
        assert_eq!(children[2].source_edge_ids, [Some(102), None, None]);
    }

    #[test]
    fn boundary_face_list_split_replaces_only_target_face() {
        let cavity = provenance_cavity();

        let split_faces =
            split_constrained_cavity_boundary_faces(&cavity.boundary_faces, [2, 1, 0], 9)
                .expect("target face should split");

        assert_eq!(split_faces.len(), cavity.boundary_faces.len() + 2);
        assert!(!split_faces
            .iter()
            .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
        assert_eq!(
            split_faces
                .iter()
                .filter(|face| face.node_ids.contains(&9))
                .count(),
            3
        );
        for untouched in cavity.boundary_faces.iter().skip(1) {
            assert!(split_faces
                .iter()
                .any(|face| sorted_face(face.node_ids) == sorted_face(untouched.node_ids)));
        }
    }

    #[test]
    fn boundary_face_split_rejects_reused_or_missing_split_targets() {
        let cavity = provenance_cavity();
        let face = &cavity.boundary_faces[0];

        let reused = split_constrained_cavity_boundary_face(face, face.node_ids[0])
            .expect_err("split node cannot reuse an existing face node");
        assert_eq!(
            reused,
            ConstrainedCavityBoundarySplitError::SplitNodeReusesFaceNode {
                node_id: face.node_ids[0]
            }
        );

        let missing =
            split_constrained_cavity_boundary_faces(&cavity.boundary_faces, [10, 11, 12], 9)
                .expect_err("missing target face should fail");
        assert_eq!(
            missing,
            ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
                node_ids: [10, 11, 12]
            }
        );
    }

    #[test]
    fn refill_candidates_preserve_split_boundary_face() {
        let mut cavity = unit_tet_cavity();
        cavity.boundary_faces =
            split_constrained_cavity_boundary_faces(&cavity.boundary_faces, [0, 1, 2], 4)
                .expect("fixture face should split");
        let mut nodes = unit_tet_nodes();
        nodes.push(ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 0.0],
        });

        let refill =
            generate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
                .expect("split boundary cavity should refill");

        validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
            .expect("refill should preserve split boundary faces");
        validate_constrained_cavity_refill_volume(
            cavity.target_volume_m3,
            refill.total_volume_m3,
            1.0e-12,
        )
        .expect("split boundary refill should preserve volume");
        assert!(
            refill
                .boundary_faces
                .iter()
                .filter(|face| face.node_ids.contains(&4))
                .count()
                >= 3
        );
    }

    #[test]
    fn refill_candidates_preserve_single_tet_cavity_boundary_and_volume() {
        let cavity = unit_tet_cavity();
        let nodes = unit_tet_nodes();

        let refill =
            generate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
                .expect("single tet cavity should refill");

        assert_eq!(refill.tets.len(), 1);
        validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
            .expect("refill boundary should match cavity boundary");
        assert!((refill.total_volume_m3 - cavity.target_volume_m3).abs() < 1.0e-12);
    }

    #[test]
    fn single_tet_refill_ignores_non_boundary_nodes_in_coordinate_table() {
        let cavity = unit_tet_cavity();
        let mut nodes = unit_tet_nodes();
        nodes.push(ConstrainedCavityNode {
            node_id: 99,
            coordinates_m: [4.0, 4.0, 4.0],
        });

        let refill =
            generate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
                .expect("coordinate table may contain nodes outside the cavity boundary");

        assert_eq!(refill.tets.len(), 1);
        assert!((refill.total_volume_m3 - cavity.target_volume_m3).abs() < 1.0e-12);
    }

    #[test]
    fn star_refill_candidates_preserve_cavity_boundary_and_volume() {
        let cavity = unit_tet_cavity();
        let nodes = unit_tet_nodes();
        let interior = [ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.25, 0.25, 0.25],
        }];

        let refill = generate_constrained_cavity_refill_candidates(
            &cavity,
            &nodes,
            &interior,
            refill_options(),
        )
        .expect("interior star refill should generate");

        assert_eq!(refill.tets.len(), 4);
        validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
            .expect("star refill boundary should match cavity boundary");
        validate_constrained_cavity_refill_volume(
            cavity.target_volume_m3,
            refill.total_volume_m3,
            1.0e-12,
        )
        .expect("star refill should preserve cavity volume");
    }

    #[test]
    fn refill_candidates_reject_missing_boundary_nodes() {
        let cavity = unit_tet_cavity();
        let mut nodes = unit_tet_nodes();
        nodes.pop();

        let err =
            generate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
                .expect_err("missing boundary node should fail");

        assert_eq!(
            err,
            ConstrainedCavityRefillError::MissingBoundaryNode { node_id: 3 }
        );
    }

    #[test]
    fn star_refill_candidates_reject_exterior_interior_points() {
        let cavity = unit_tet_cavity();
        let nodes = unit_tet_nodes();
        let exterior = [ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [2.0, 2.0, 2.0],
        }];

        let err = generate_constrained_cavity_refill_candidates(
            &cavity,
            &nodes,
            &exterior,
            refill_options(),
        )
        .expect_err("exterior interior candidate should fail");

        assert_eq!(
            err,
            ConstrainedCavityRefillError::NoValidCandidate {
                rejected_by_reason: BTreeMap::from([(
                    "interior_point_outside_cavity".to_string(),
                    1
                )])
            }
        );
    }

    #[test]
    fn star_refill_evaluation_reports_scaled_jacobian_rejections() {
        let cavity = unit_tet_cavity();
        let nodes = unit_tet_nodes();
        let near_corner = [ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [1.0e-4, 1.0e-4, 1.0e-4],
        }];

        let evaluation = evaluate_constrained_cavity_refill_candidates(
            &cavity,
            &nodes,
            &near_corner,
            ConstrainedCavityRefillOptions {
                min_scaled_jacobian: 0.5,
                volume_relative_tolerance: 1.0e-12,
                ..ConstrainedCavityRefillOptions::default()
            },
        )
        .expect("evaluation should classify a low-quality star candidate");

        assert!(evaluation.refill.is_none());
        assert_eq!(
            evaluation.rejected_by_reason,
            BTreeMap::from([("star_tet_scaled_jacobian".to_string(), 1)])
        );
    }

    #[test]
    fn boundary_node_refill_evaluation_reports_contextual_scaled_jacobian_rejections() {
        let cavity = octahedron_cavity();
        let nodes = octahedron_nodes();

        let evaluation = evaluate_constrained_cavity_refill_candidates(
            &cavity,
            &nodes,
            &[],
            ConstrainedCavityRefillOptions {
                min_scaled_jacobian: 0.95,
                volume_relative_tolerance: 1.0e-12,
                ..ConstrainedCavityRefillOptions::default()
            },
        )
        .expect("boundary-node evaluation should classify low-quality candidates");

        assert!(evaluation.refill.is_none());
        assert_eq!(
            evaluation.rejected_by_reason,
            BTreeMap::from([("boundary_node_tet_scaled_jacobian".to_string(), 1)])
        );
    }

    #[test]
    fn refill_evaluation_uses_boundary_nodes_for_multi_face_cavity_without_interior_point() {
        let cavity = octahedron_cavity();
        let nodes = octahedron_nodes();

        let evaluation =
            evaluate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
                .expect("evaluation should complete");

        let refill = evaluation
            .refill
            .expect("boundary-node refill should support closed multi-face cavities");
        assert!(evaluation.rejected_by_reason.is_empty());
        validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
            .expect("boundary-node refill should preserve the cavity boundary");
        validate_constrained_cavity_refill_volume(
            cavity.target_volume_m3,
            refill.total_volume_m3,
            1.0e-12,
        )
        .expect("boundary-node refill should preserve volume");
    }

    #[test]
    fn boundary_node_completion_repairs_missing_cavity_boundary_faces() {
        let cavity = two_tet_bipyramid_cavity();
        let nodes = two_tet_bipyramid_nodes();
        let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
            .expect("fixture nodes should cover cavity boundary");
        let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
            .expect("fixture boundary should build triangles");
        let options = refill_options();
        let points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
        let incomplete_tet = raw_refill_tet_with_rejection_reason([0, 1, 2, 3], points, options)
            .expect("fixture tet should pass quality gates");

        assert!(refill_from_tets(
            &cavity,
            vec![incomplete_tet.clone()],
            options.volume_relative_tolerance
        )
        .is_err());

        let completed = complete_missing_boundary_face_tets(
            &cavity,
            &boundary_nodes,
            vec![incomplete_tet],
            &boundary_triangles,
            options,
        )
        .expect("completion should evaluate")
        .expect("completion should add the missing tet");
        let refill = refill_from_tets(&cavity, completed, options.volume_relative_tolerance)
            .expect("completed refill should validate");

        assert_eq!(refill.tets.len(), 2);
        validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
            .expect("completed refill should preserve the cavity boundary");
        validate_constrained_cavity_refill_volume(
            cavity.target_volume_m3,
            refill.total_volume_m3,
            options.volume_relative_tolerance,
        )
        .expect("completed refill should preserve volume");
    }

    #[test]
    fn refill_evaluation_skips_exterior_points_and_accepts_valid_candidate() {
        let cavity = unit_tet_cavity();
        let nodes = unit_tet_nodes();
        let candidates = [
            ConstrainedCavityNode {
                node_id: 10,
                coordinates_m: [2.0, 2.0, 2.0],
            },
            ConstrainedCavityNode {
                node_id: 11,
                coordinates_m: [0.25, 0.25, 0.25],
            },
        ];

        let evaluation = evaluate_constrained_cavity_refill_candidates(
            &cavity,
            &nodes,
            &candidates,
            refill_options(),
        )
        .expect("evaluation should complete");

        assert!(evaluation.refill.is_some());
        assert_eq!(
            evaluation.rejected_by_reason,
            BTreeMap::from([("interior_point_outside_cavity".to_string(), 1)])
        );
    }

    #[test]
    fn refill_evaluation_skips_points_too_close_to_protected_boundary_nodes() {
        let mut cavity = unit_tet_cavity();
        cavity.protected_node_ids = vec![0];
        let nodes = unit_tet_nodes();
        let candidates = [
            ConstrainedCavityNode {
                node_id: 10,
                coordinates_m: [0.01, 0.01, 0.01],
            },
            ConstrainedCavityNode {
                node_id: 11,
                coordinates_m: [0.25, 0.25, 0.25],
            },
        ];

        let evaluation = evaluate_constrained_cavity_refill_candidates(
            &cavity,
            &nodes,
            &candidates,
            protected_refill_options(),
        )
        .expect("evaluation should continue after protected-distance rejection");

        assert!(evaluation.refill.is_some());
        assert_eq!(
            evaluation.rejected_by_reason,
            BTreeMap::from([("protected_boundary_distance".to_string(), 1)])
        );
    }

    #[test]
    fn refill_generation_reports_protected_boundary_distance_rejections() {
        let mut cavity = unit_tet_cavity();
        cavity.protected_node_ids = vec![0];
        let nodes = unit_tet_nodes();
        let candidates = [ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.01, 0.01, 0.01],
        }];

        let err = generate_constrained_cavity_refill_candidates(
            &cavity,
            &nodes,
            &candidates,
            protected_refill_options(),
        )
        .expect_err("all candidates too close to protected nodes should fail");

        assert_eq!(
            err,
            ConstrainedCavityRefillError::NoValidCandidate {
                rejected_by_reason: BTreeMap::from([(
                    "protected_boundary_distance".to_string(),
                    1
                )])
            }
        );
    }

    #[test]
    fn star_refill_candidates_reject_boundary_node_reuse() {
        let cavity = unit_tet_cavity();
        let nodes = unit_tet_nodes();
        let reused = [ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.25, 0.25, 0.25],
        }];

        let err = generate_constrained_cavity_refill_candidates(
            &cavity,
            &nodes,
            &reused,
            refill_options(),
        )
        .expect_err("interior candidate cannot reuse a boundary node");

        assert_eq!(
            err,
            ConstrainedCavityRefillError::InteriorNodeReusesBoundaryNode { node_id: 0 }
        );
    }

    #[test]
    fn validates_closed_tet_cavity_boundary() {
        let cavity = tet_cavity();

        let report = validate_constrained_cavity(&cavity).expect("closed cavity should validate");

        assert_eq!(report.boundary_face_count, 4);
        assert_eq!(report.boundary_edge_count, 6);
        assert_eq!(report.boundary_node_count, 4);
        assert_eq!(report.protected_node_count, 2);
        assert_eq!(report.target_volume_m3, 1.0);
    }

    #[test]
    fn rejects_duplicate_boundary_faces() {
        let mut cavity = tet_cavity();
        cavity.boundary_faces[1].node_ids = cavity.boundary_faces[0].node_ids;

        let err =
            validate_constrained_cavity(&cavity).expect_err("duplicate boundary face should fail");

        assert_eq!(
            err,
            ConstrainedCavityValidationError::DuplicateBoundaryFace {
                node_ids: [0, 1, 2]
            }
        );
    }

    #[test]
    fn rejects_open_boundary_edges() {
        let mut cavity = tet_cavity();
        cavity.boundary_faces.pop();

        let err = validate_constrained_cavity(&cavity).expect_err("open boundary should fail");

        assert_eq!(
            err,
            ConstrainedCavityValidationError::TooFewBoundaryFaces {
                boundary_face_count: 3
            }
        );
    }

    #[test]
    fn rejects_protected_nodes_outside_boundary() {
        let mut cavity = tet_cavity();
        cavity.protected_node_ids.push(99);

        let err =
            validate_constrained_cavity(&cavity).expect_err("outside protected node should fail");

        assert_eq!(
            err,
            ConstrainedCavityValidationError::ProtectedNodeOutsideBoundary { node_id: 99 }
        );
    }

    #[test]
    fn rejects_refill_volume_mismatch() {
        let err = validate_constrained_cavity_refill_volume(1.0, 1.2, 1.0e-9)
            .expect_err("volume mismatch should fail");

        assert_eq!(
            err,
            ConstrainedCavityValidationError::InvalidRefillVolume {
                target_volume_m3: 1.0,
                candidate_volume_m3: 1.2,
                tolerance_m3: 1.0e-9
            }
        );
    }

    fn tet_cavity() -> ConstrainedCavity {
        ConstrainedCavity {
            removed_tet_ids: vec![7],
            boundary_faces: vec![
                face([0, 1, 2]),
                face([0, 3, 1]),
                face([1, 3, 2]),
                face([2, 3, 0]),
            ],
            protected_node_ids: vec![0, 1],
            target_volume_m3: 1.0,
        }
    }

    fn face(node_ids: [u32; 3]) -> ConstrainedCavityBoundaryFace {
        ConstrainedCavityBoundaryFace {
            node_ids,
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: Vec::new(),
        }
    }

    fn provenance_cavity() -> ConstrainedCavity {
        ConstrainedCavity {
            removed_tet_ids: vec![7],
            boundary_faces: vec![
                face_with_provenance(
                    [0, 1, 2],
                    10,
                    [Some(100), Some(101), Some(102)],
                    &["loaded", "fixed"],
                ),
                face_with_provenance([0, 3, 1], 11, [Some(103), Some(104), Some(100)], &["fixed"]),
                face_with_provenance([1, 3, 2], 12, [Some(104), Some(105), Some(101)], &["solid"]),
                face_with_provenance([2, 3, 0], 13, [Some(105), Some(103), Some(102)], &["solid"]),
            ],
            protected_node_ids: vec![0, 1],
            target_volume_m3: 1.0,
        }
    }

    fn unit_tet_cavity() -> ConstrainedCavity {
        ConstrainedCavity {
            removed_tet_ids: vec![1],
            boundary_faces: tet_faces([0, 1, 2, 3])
                .into_iter()
                .map(|node_ids| ConstrainedCavityBoundaryFace {
                    node_ids,
                    source_face_id: None,
                    source_edge_ids: [None, None, None],
                    region_ids: vec!["body".to_string()],
                })
                .collect(),
            protected_node_ids: Vec::new(),
            target_volume_m3: 1.0 / 6.0,
        }
    }

    fn unit_tet_nodes() -> Vec<ConstrainedCavityNode> {
        vec![
            ConstrainedCavityNode {
                node_id: 0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 1,
                coordinates_m: [1.0, 0.0, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 2,
                coordinates_m: [0.0, 1.0, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 3,
                coordinates_m: [0.0, 0.0, 1.0],
            },
        ]
    }

    fn octahedron_cavity() -> ConstrainedCavity {
        ConstrainedCavity {
            removed_tet_ids: vec![1, 2],
            boundary_faces: [
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
                [1, 0, 5],
                [2, 1, 5],
                [3, 2, 5],
                [0, 3, 5],
            ]
            .into_iter()
            .map(|node_ids| ConstrainedCavityBoundaryFace {
                node_ids,
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: vec!["body".to_string()],
            })
            .collect(),
            protected_node_ids: Vec::new(),
            target_volume_m3: 4.0 / 3.0,
        }
    }

    fn octahedron_nodes() -> Vec<ConstrainedCavityNode> {
        vec![
            ConstrainedCavityNode {
                node_id: 0,
                coordinates_m: [1.0, 0.0, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 1,
                coordinates_m: [0.0, 1.0, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 2,
                coordinates_m: [-1.0, 0.0, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 3,
                coordinates_m: [0.0, -1.0, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 4,
                coordinates_m: [0.0, 0.0, 1.0],
            },
            ConstrainedCavityNode {
                node_id: 5,
                coordinates_m: [0.0, 0.0, -1.0],
            },
        ]
    }

    fn two_tet_bipyramid_cavity() -> ConstrainedCavity {
        ConstrainedCavity {
            removed_tet_ids: vec![1, 2],
            boundary_faces: [
                [0, 1, 3],
                [1, 2, 3],
                [0, 2, 3],
                [0, 2, 4],
                [1, 2, 4],
                [0, 1, 4],
            ]
            .into_iter()
            .map(|node_ids| ConstrainedCavityBoundaryFace {
                node_ids,
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: vec!["body".to_string()],
            })
            .collect(),
            protected_node_ids: Vec::new(),
            target_volume_m3: 1.0 / 3.0,
        }
    }

    fn two_tet_bipyramid_nodes() -> Vec<ConstrainedCavityNode> {
        vec![
            ConstrainedCavityNode {
                node_id: 0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 1,
                coordinates_m: [1.0, 0.0, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 2,
                coordinates_m: [0.0, 1.0, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 3,
                coordinates_m: [0.0, 0.0, 1.0],
            },
            ConstrainedCavityNode {
                node_id: 4,
                coordinates_m: [0.0, 0.0, -1.0],
            },
        ]
    }

    fn refill_options() -> ConstrainedCavityRefillOptions {
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.0,
            volume_relative_tolerance: 1.0e-12,
            ..ConstrainedCavityRefillOptions::default()
        }
    }

    fn protected_refill_options() -> ConstrainedCavityRefillOptions {
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.0,
            volume_relative_tolerance: 1.0e-12,
            min_protected_node_distance_m: 0.10,
            ..ConstrainedCavityRefillOptions::default()
        }
    }

    fn face_with_provenance(
        node_ids: [u32; 3],
        source_face_id: u32,
        source_edge_ids: [Option<u32>; 3],
        region_ids: &[&str],
    ) -> ConstrainedCavityBoundaryFace {
        ConstrainedCavityBoundaryFace {
            node_ids,
            source_face_id: Some(source_face_id),
            source_edge_ids,
            region_ids: region_ids.iter().map(|region| region.to_string()).collect(),
        }
    }

    fn source_edge_for(face: &ConstrainedCavityBoundaryFace, edge: [u32; 2]) -> Option<u32> {
        face_edges(face.node_ids)
            .into_iter()
            .zip(face.source_edge_ids)
            .find_map(|(candidate_edge, source_edge_id)| {
                (sorted_edge(candidate_edge) == sorted_edge(edge)).then_some(source_edge_id)
            })
            .flatten()
    }

    fn candidate_tet(
        tet_id: u32,
        node_ids: [u32; 4],
        volume_m3: f64,
        region_ids: &[&str],
    ) -> TetCandidate {
        TetCandidate {
            tet_id,
            component_id: 0,
            node_ids,
            source_surface_element_id: 0,
            region_ids: region_ids.iter().map(|region| region.to_string()).collect(),
            volume_m3,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 1.0,
        }
    }
}
