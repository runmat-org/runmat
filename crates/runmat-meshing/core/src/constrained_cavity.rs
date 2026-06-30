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
    #[serde(default)]
    pub inserted_nodes: Vec<ConstrainedCavityNode>,
    pub total_volume_m3: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryNodeCompletionDiagnostic {
    pub reason: &'static str,
    pub missing_face_count: usize,
    pub cap_candidate_count: usize,
    pub outside_candidate_count: usize,
    pub duplicate_candidate_count: usize,
    pub max_rejected_scaled_jacobian: f64,
    pub rejected_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub max_rejected_cap_height_ratio: f64,
    pub rejected_cap_height_ratio_bins: BTreeMap<String, usize>,
    pub rejected_scaled_jacobian_worst_corner_bins: BTreeMap<&'static str, usize>,
    pub rejected_cap_node_ids: BTreeMap<u32, usize>,
    pub split_cap_candidate_count: usize,
    pub split_cap_pass_count: usize,
    pub max_split_cap_scaled_jacobian: f64,
    pub split_cap_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub split_cap_scaled_jacobian_worst_corner_bins: BTreeMap<&'static str, usize>,
    pub split_cap_apex_limited_node_ids: BTreeMap<u32, usize>,
    pub edge_split_cap_candidate_count: usize,
    pub edge_split_cap_pass_count: usize,
    pub max_edge_split_cap_scaled_jacobian: f64,
    pub edge_split_cap_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap<&'static str, usize>,
    pub edge_split_cap_apex_limited_node_ids: BTreeMap<u32, usize>,
    pub three_edge_split_cap_candidate_count: usize,
    pub three_edge_split_cap_pass_count: usize,
    pub max_three_edge_split_cap_scaled_jacobian: f64,
    pub three_edge_split_cap_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub three_edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap<&'static str, usize>,
    pub three_edge_split_cap_apex_limited_node_ids: BTreeMap<u32, usize>,
    pub rejected_by_reason: BTreeMap<&'static str, usize>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryExactCoverDiagnostic {
    pub boundary_node_count: usize,
    pub boundary_face_count: usize,
    pub candidate_count: usize,
    pub zero_candidate_boundary_face_count: usize,
    pub min_boundary_face_candidate_count: usize,
    pub max_boundary_face_candidate_count: usize,
    pub selected_tet_count: usize,
    pub search_attempt_count: usize,
    pub found_cover: bool,
    pub reason: &'static str,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundarySteinerExactCoverDiagnostic {
    pub boundary_node_count: usize,
    pub boundary_face_count: usize,
    pub candidate_count: usize,
    pub zero_candidate_boundary_face_count: usize,
    pub min_boundary_face_candidate_count: usize,
    pub max_boundary_face_candidate_count: usize,
    pub selected_tet_count: usize,
    pub search_attempt_count: usize,
    pub found_cover: bool,
    pub reason: &'static str,
    pub max_min_scaled_jacobian: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryPatchSteinerExactCoverDiagnostic {
    pub boundary_node_count: usize,
    pub boundary_face_count: usize,
    pub missing_face_count: usize,
    pub patch_count: usize,
    pub steiner_node_count: usize,
    pub candidate_count: usize,
    pub zero_candidate_boundary_face_count: usize,
    pub min_boundary_face_candidate_count: usize,
    pub max_boundary_face_candidate_count: usize,
    pub selected_tet_count: usize,
    pub search_attempt_count: usize,
    pub found_cover: bool,
    pub reason: &'static str,
    pub max_min_scaled_jacobian: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MissingFaceLocalCapQualityDiagnostic {
    pub missing_face_count: usize,
    pub pass_face_count: usize,
    pub candidate_count: usize,
    pub max_scaled_jacobian: f64,
    pub rejected_by_reason: BTreeMap<&'static str, usize>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MissingFaceLocalCapStitchDiagnostic {
    pub missing_face_count: usize,
    pub capped_face_count: usize,
    pub inserted_node_count: usize,
    pub side_connector_candidate_count: usize,
    pub candidate_tet_count: usize,
    pub cap_side_face_count: usize,
    pub zero_mate_cap_side_face_count: usize,
    pub min_cap_side_face_mate_count: usize,
    pub max_cap_side_face_mate_count: usize,
    pub open_interior_face_count: usize,
    pub open_interior_component_count: usize,
    pub open_interior_component_size_histogram: BTreeMap<usize, usize>,
    pub selected_tet_count: usize,
    pub search_attempt_count: usize,
    pub found_cover: bool,
    pub reason: &'static str,
    pub max_min_scaled_jacobian: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryMissingFaceClusterDiagnostic {
    pub missing_face_count: usize,
    pub edge_component_count: usize,
    pub edge_component_size_histogram: BTreeMap<usize, usize>,
    pub node_component_count: usize,
    pub node_component_size_histogram: BTreeMap<usize, usize>,
    pub node_component_common_node_count_histogram: BTreeMap<usize, usize>,
    pub node_component_common_node_ids: BTreeMap<u32, usize>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InteriorStarQualityDiagnostic {
    pub candidate_count: usize,
    pub pass_count: usize,
    pub max_min_scaled_jacobian: f64,
    pub min_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub min_scaled_jacobian_worst_corner_bins: BTreeMap<&'static str, usize>,
    pub rejected_by_reason: BTreeMap<&'static str, usize>,
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
        match centroid_interior_refill_candidate(
            cavity,
            &boundary_node_map,
            &boundary_triangles,
            options,
        ) {
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

fn split_constrained_cavity_boundary_face_on_edge(
    face: &ConstrainedCavityBoundaryFace,
    edge: [u32; 2],
    split_node_id: u32,
) -> Result<[ConstrainedCavityBoundaryFace; 2], ConstrainedCavityBoundarySplitError> {
    if face.node_ids.contains(&split_node_id) {
        return Err(
            ConstrainedCavityBoundarySplitError::SplitNodeReusesFaceNode {
                node_id: split_node_id,
            },
        );
    }
    let sorted_split_edge = sorted_edge(edge);
    if !face_edges(face.node_ids)
        .into_iter()
        .any(|candidate| sorted_edge(candidate) == sorted_split_edge)
    {
        return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        });
    }
    let source_edges = boundary_face_source_edges(face).map_err(|_| {
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        }
    })?;
    let source_edge_id = source_edges.get(&sorted_split_edge).copied().flatten();
    let [a, b] = edge;
    let c = face
        .node_ids
        .into_iter()
        .find(|node_id| *node_id != a && *node_id != b)
        .ok_or(ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        })?;
    Ok([
        edge_split_child_boundary_face(
            face,
            [a, split_node_id, c],
            split_node_id,
            sorted_split_edge,
            source_edge_id,
            &source_edges,
        ),
        edge_split_child_boundary_face(
            face,
            [split_node_id, b, c],
            split_node_id,
            sorted_split_edge,
            source_edge_id,
            &source_edges,
        ),
    ])
}

fn split_constrained_cavity_boundary_faces_on_edge(
    faces: &[ConstrainedCavityBoundaryFace],
    face_node_ids: [u32; 3],
    edge: [u32; 2],
    split_node_id: u32,
) -> Result<Vec<ConstrainedCavityBoundaryFace>, ConstrainedCavityBoundarySplitError> {
    let target = sorted_face(face_node_ids);
    let split_edge = sorted_edge(edge);
    let mut split_faces = Vec::<ConstrainedCavityBoundaryFace>::with_capacity(faces.len() + 1);
    let mut found = false;
    for face in faces {
        if sorted_face(face.node_ids) == target {
            found = true;
        }
        if face_edges(face.node_ids)
            .into_iter()
            .any(|candidate| sorted_edge(candidate) == split_edge)
        {
            split_faces.extend(split_constrained_cavity_boundary_face_on_edge(
                face,
                edge,
                split_node_id,
            )?);
        } else {
            split_faces.push(face.clone());
        }
    }
    if !found {
        return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace { node_ids: target });
    }
    Ok(split_faces)
}

fn split_constrained_cavity_boundary_faces_on_three_edges(
    faces: &[ConstrainedCavityBoundaryFace],
    face_node_ids: [u32; 3],
    edge_split_node_ids: BTreeMap<[u32; 2], u32>,
) -> Result<Vec<ConstrainedCavityBoundaryFace>, ConstrainedCavityBoundarySplitError> {
    let target = sorted_face(face_node_ids);
    let mut split_faces = Vec::<ConstrainedCavityBoundaryFace>::with_capacity(faces.len() + 6);
    let mut found = false;
    for face in faces {
        if sorted_face(face.node_ids) == target {
            found = true;
            split_faces.extend(split_constrained_cavity_boundary_face_on_three_edges(
                face,
                &edge_split_node_ids,
            )?);
            continue;
        }
        let split_edges = face_edges(face.node_ids)
            .into_iter()
            .filter_map(|edge| {
                edge_split_node_ids
                    .get(&sorted_edge(edge))
                    .copied()
                    .map(|node_id| (edge, node_id))
            })
            .collect::<Vec<_>>();
        if split_edges.is_empty() {
            split_faces.push(face.clone());
            continue;
        }
        if split_edges.len() > 1 {
            return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
                node_ids: sorted_face(face.node_ids),
            });
        }
        let (edge, split_node_id) = split_edges[0];
        split_faces.extend(split_constrained_cavity_boundary_face_on_edge(
            face,
            edge,
            split_node_id,
        )?);
    }
    if !found {
        return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace { node_ids: target });
    }
    Ok(split_faces)
}

fn split_constrained_cavity_boundary_face_on_three_edges(
    face: &ConstrainedCavityBoundaryFace,
    edge_split_node_ids: &BTreeMap<[u32; 2], u32>,
) -> Result<[ConstrainedCavityBoundaryFace; 4], ConstrainedCavityBoundarySplitError> {
    let [a, b, c] = face.node_ids;
    let ab = *edge_split_node_ids.get(&sorted_edge([a, b])).ok_or(
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        },
    )?;
    let bc = *edge_split_node_ids.get(&sorted_edge([b, c])).ok_or(
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        },
    )?;
    let ca = *edge_split_node_ids.get(&sorted_edge([c, a])).ok_or(
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        },
    )?;
    let perimeter_source_edges = boundary_face_source_edges(face).map_err(|_| {
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        }
    })?;
    Ok([
        three_edge_split_child_boundary_face(
            face,
            [a, ab, ca],
            &perimeter_source_edges,
            edge_split_node_ids,
        ),
        three_edge_split_child_boundary_face(
            face,
            [ab, b, bc],
            &perimeter_source_edges,
            edge_split_node_ids,
        ),
        three_edge_split_child_boundary_face(
            face,
            [ca, bc, c],
            &perimeter_source_edges,
            edge_split_node_ids,
        ),
        three_edge_split_child_boundary_face(
            face,
            [ab, bc, ca],
            &perimeter_source_edges,
            edge_split_node_ids,
        ),
    ])
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

fn three_edge_split_child_boundary_face(
    parent: &ConstrainedCavityBoundaryFace,
    node_ids: [u32; 3],
    perimeter_source_edges: &BTreeMap<[u32; 2], Option<u32>>,
    edge_split_node_ids: &BTreeMap<[u32; 2], u32>,
) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        source_face_id: parent.source_face_id,
        source_edge_ids: face_edges(node_ids).map(|edge| {
            let original_edge =
                edge_split_node_ids
                    .iter()
                    .find_map(|(split_edge, split_node_id)| {
                        if edge.contains(split_node_id)
                            && edge.into_iter().any(|node_id| {
                                node_id != *split_node_id && split_edge.contains(&node_id)
                            })
                        {
                            Some(*split_edge)
                        } else {
                            None
                        }
                    })?;
            perimeter_source_edges
                .get(&original_edge)
                .copied()
                .flatten()
        }),
        region_ids: parent.region_ids.clone(),
    }
}

fn edge_split_child_boundary_face(
    parent: &ConstrainedCavityBoundaryFace,
    node_ids: [u32; 3],
    split_node_id: u32,
    split_edge: [u32; 2],
    split_edge_source_id: Option<u32>,
    perimeter_source_edges: &BTreeMap<[u32; 2], Option<u32>>,
) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        source_face_id: parent.source_face_id,
        source_edge_ids: face_edges(node_ids).map(|edge| {
            let sorted = sorted_edge(edge);
            if edge.contains(&split_node_id)
                && edge
                    .into_iter()
                    .any(|node_id| node_id != split_node_id && split_edge.contains(&node_id))
            {
                split_edge_source_id
            } else {
                perimeter_source_edges.get(&sorted).copied().flatten()
            }
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
        Err(_) => {
            if let Some(refill) = boundary_node_exact_cover_refill_candidate(
                cavity,
                boundary_nodes,
                &boundary_triangles,
                options,
            )? {
                return Ok(Ok(refill));
            }
            let (completed_cavity, completed_tets, inserted_nodes) =
                match complete_missing_boundary_face_tets(
                    cavity,
                    boundary_nodes,
                    refill_tets,
                    &boundary_triangles,
                    options,
                )? {
                    Ok(completed_tets) => completed_tets,
                    Err(reason) => return Ok(Err(reason)),
                };
            let mut refill = match refill_from_tets(
                &completed_cavity,
                completed_tets,
                options.volume_relative_tolerance,
            ) {
                Ok(refill) => refill,
                Err(err) => return Ok(Err(boundary_node_refill_validation_reason(&err))),
            };
            refill.inserted_nodes = inserted_nodes;
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

fn centroid_interior_refill_candidate(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Result<Result<ConstrainedCavityRefill, &'static str>, ConstrainedCavityValidationError> {
    let Some(coordinates_m) = cavity_boundary_node_centroid(cavity, boundary_nodes) else {
        return Ok(Err("centroid_interior_refill_empty_boundary"));
    };
    if point_in_closed_triangle_surface(
        coordinates_m,
        boundary_triangles,
        MeshingTolerance::default(),
    ) != PointInClosedSurface::Inside
    {
        return Ok(Err("centroid_interior_refill_outside_cavity"));
    }
    let node = ConstrainedCavityNode {
        node_id: next_cavity_node_id(cavity),
        coordinates_m,
    };
    match star_refill_candidate_with_rejection_reason(cavity, boundary_nodes, node.clone(), options)
    {
        Ok(Ok(mut refill)) => {
            refill.inserted_nodes.push(node);
            Ok(Ok(refill))
        }
        Ok(Err(reason)) => Ok(Err(centroid_interior_refill_rejection_reason(reason))),
        Err(err) => Err(err),
    }
}

fn cavity_boundary_node_centroid(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
) -> Option<Point3> {
    let node_ids = cavity_boundary_node_ids(cavity);
    if node_ids.is_empty() {
        return None;
    }
    let mut centroid = [0.0_f64; 3];
    for node_id in &node_ids {
        let point = boundary_nodes.get(node_id)?;
        centroid[0] += point[0];
        centroid[1] += point[1];
        centroid[2] += point[2];
    }
    let scale = 1.0 / node_ids.len() as f64;
    Some([
        centroid[0] * scale,
        centroid[1] * scale,
        centroid[2] * scale,
    ])
}

fn next_cavity_node_id(cavity: &ConstrainedCavity) -> u32 {
    cavity_boundary_node_ids(cavity)
        .into_iter()
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

fn centroid_interior_refill_rejection_reason(reason: &'static str) -> &'static str {
    match reason {
        "star_tet_min_volume" => "centroid_interior_refill_tet_min_volume",
        "star_tet_aspect_ratio" => "centroid_interior_refill_tet_aspect_ratio",
        "star_tet_scaled_jacobian" => "centroid_interior_refill_tet_scaled_jacobian",
        other => other,
    }
}

fn two_interior_node_refill_candidate(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    interior_candidates: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<Result<ConstrainedCavityRefill, &'static str>, ConstrainedCavityValidationError> {
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    let mut best = None::<ConstrainedCavityRefill>;
    let mut first_rejection = None::<&'static str>;
    for left in 0..interior_candidates.len() {
        for right in (left + 1)..interior_candidates.len() {
            let pair = [
                interior_candidates[left].clone(),
                interior_candidates[right].clone(),
            ];
            let mut points = boundary_node_ids
                .iter()
                .map(|node_id| ConnectivityPoint {
                    node_id: *node_id,
                    coordinates_m: boundary_nodes[node_id],
                    is_super: false,
                })
                .collect::<Vec<_>>();
            points.extend(pair.iter().map(|node| ConnectivityPoint {
                node_id: node.node_id,
                coordinates_m: node.coordinates_m,
                is_super: false,
            }));
            let mut refill_tets = Vec::<ConstrainedCavityRefillTet>::new();
            for tet in tetrahedralize_points(&points) {
                let node_ids = tet.vertices.map(|index| points[index].node_id);
                let tet_points = tet.vertices.map(|index| points[index].coordinates_m);
                if point_in_closed_triangle_surface(
                    tet_centroid(tet_points),
                    boundary_triangles,
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
                if first_rejection.is_none() {
                    first_rejection = Some("two_interior_delaunay_empty");
                }
                continue;
            }
            match refill_from_tets(cavity, refill_tets, options.volume_relative_tolerance) {
                Ok(mut refill) => {
                    refill.inserted_nodes = pair.to_vec();
                    if best
                        .as_ref()
                        .is_none_or(|current| refill_is_better(&refill, current))
                    {
                        best = Some(refill);
                    }
                }
                Err(err) => {
                    if first_rejection.is_none() {
                        first_rejection = Some(boundary_node_refill_validation_reason(&err));
                    }
                }
            }
        }
    }
    Ok(best
        .map(Ok)
        .unwrap_or_else(|| Err(first_rejection.unwrap_or("two_interior_no_candidate"))))
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_node_completion(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryNodeCompletionDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let points = cavity_boundary_node_ids(cavity)
        .into_iter()
        .map(|node_id| ConnectivityPoint {
            node_id,
            coordinates_m: boundary_node_map[&node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut refill_tets = Vec::<ConstrainedCavityRefillTet>::new();
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
        if let Ok(tet) = raw_refill_tet_with_rejection_reason(node_ids, tet_points, options) {
            refill_tets.push(tet);
        }
    }
    let mut aggregate = BoundaryNodeCompletionDiagnostic {
        reason: "boundary_node_completion_no_missing_faces",
        missing_face_count: 0,
        cap_candidate_count: 0,
        outside_candidate_count: 0,
        duplicate_candidate_count: 0,
        max_rejected_scaled_jacobian: 0.0,
        rejected_scaled_jacobian_bins: BTreeMap::new(),
        max_rejected_cap_height_ratio: 0.0,
        rejected_cap_height_ratio_bins: BTreeMap::new(),
        rejected_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        rejected_cap_node_ids: BTreeMap::new(),
        split_cap_candidate_count: 0,
        split_cap_pass_count: 0,
        max_split_cap_scaled_jacobian: 0.0,
        split_cap_scaled_jacobian_bins: BTreeMap::new(),
        split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        split_cap_apex_limited_node_ids: BTreeMap::new(),
        edge_split_cap_candidate_count: 0,
        edge_split_cap_pass_count: 0,
        max_edge_split_cap_scaled_jacobian: 0.0,
        edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
        edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
        three_edge_split_cap_candidate_count: 0,
        three_edge_split_cap_pass_count: 0,
        max_three_edge_split_cap_scaled_jacobian: 0.0,
        three_edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
        three_edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        three_edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
        rejected_by_reason: BTreeMap::new(),
    };
    loop {
        let missing_faces = missing_refill_boundary_faces(cavity, &refill_tets)
            .map_err(ConstrainedCavityRefillError::Validation)?;
        let Some(missing_face) = missing_faces.first().copied() else {
            break;
        };
        aggregate.missing_face_count = missing_faces.len();
        let diagnostic = diagnostic_boundary_face_completion(
            missing_face,
            cavity,
            &boundary_node_map,
            &refill_tets,
            &boundary_triangles,
            options,
            missing_faces.len(),
        );
        aggregate.cap_candidate_count += diagnostic.cap_candidate_count;
        aggregate.outside_candidate_count += diagnostic.outside_candidate_count;
        aggregate.duplicate_candidate_count += diagnostic.duplicate_candidate_count;
        aggregate.max_rejected_scaled_jacobian = aggregate
            .max_rejected_scaled_jacobian
            .max(diagnostic.max_rejected_scaled_jacobian);
        aggregate.max_rejected_cap_height_ratio = aggregate
            .max_rejected_cap_height_ratio
            .max(diagnostic.max_rejected_cap_height_ratio);
        for (bin, count) in diagnostic.rejected_scaled_jacobian_bins {
            *aggregate
                .rejected_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.rejected_cap_height_ratio_bins {
            *aggregate
                .rejected_cap_height_ratio_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.rejected_scaled_jacobian_worst_corner_bins {
            *aggregate
                .rejected_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.rejected_cap_node_ids {
            *aggregate.rejected_cap_node_ids.entry(node_id).or_default() += count;
        }
        aggregate.split_cap_candidate_count += diagnostic.split_cap_candidate_count;
        aggregate.split_cap_pass_count += diagnostic.split_cap_pass_count;
        aggregate.max_split_cap_scaled_jacobian = aggregate
            .max_split_cap_scaled_jacobian
            .max(diagnostic.max_split_cap_scaled_jacobian);
        for (bin, count) in diagnostic.split_cap_scaled_jacobian_bins {
            *aggregate
                .split_cap_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.split_cap_scaled_jacobian_worst_corner_bins {
            *aggregate
                .split_cap_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.split_cap_apex_limited_node_ids {
            *aggregate
                .split_cap_apex_limited_node_ids
                .entry(node_id)
                .or_default() += count;
        }
        aggregate.edge_split_cap_candidate_count += diagnostic.edge_split_cap_candidate_count;
        aggregate.edge_split_cap_pass_count += diagnostic.edge_split_cap_pass_count;
        aggregate.max_edge_split_cap_scaled_jacobian = aggregate
            .max_edge_split_cap_scaled_jacobian
            .max(diagnostic.max_edge_split_cap_scaled_jacobian);
        for (bin, count) in diagnostic.edge_split_cap_scaled_jacobian_bins {
            *aggregate
                .edge_split_cap_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.edge_split_cap_scaled_jacobian_worst_corner_bins {
            *aggregate
                .edge_split_cap_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.edge_split_cap_apex_limited_node_ids {
            *aggregate
                .edge_split_cap_apex_limited_node_ids
                .entry(node_id)
                .or_default() += count;
        }
        aggregate.three_edge_split_cap_candidate_count +=
            diagnostic.three_edge_split_cap_candidate_count;
        aggregate.three_edge_split_cap_pass_count += diagnostic.three_edge_split_cap_pass_count;
        aggregate.max_three_edge_split_cap_scaled_jacobian = aggregate
            .max_three_edge_split_cap_scaled_jacobian
            .max(diagnostic.max_three_edge_split_cap_scaled_jacobian);
        for (bin, count) in diagnostic.three_edge_split_cap_scaled_jacobian_bins {
            *aggregate
                .three_edge_split_cap_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.three_edge_split_cap_scaled_jacobian_worst_corner_bins {
            *aggregate
                .three_edge_split_cap_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.three_edge_split_cap_apex_limited_node_ids {
            *aggregate
                .three_edge_split_cap_apex_limited_node_ids
                .entry(node_id)
                .or_default() += count;
        }
        for (reason, count) in diagnostic.rejected_by_reason {
            *aggregate.rejected_by_reason.entry(reason).or_default() += count;
        }
        let Some(tet) = best_boundary_face_completion_tet(
            missing_face,
            cavity,
            &boundary_node_map,
            &refill_tets,
            &boundary_triangles,
            options,
        ) else {
            aggregate.reason = "boundary_node_completion_no_candidate";
            return Ok(aggregate);
        };
        refill_tets.push(tet);
    }
    if aggregate.missing_face_count == 0 {
        return Ok(BoundaryNodeCompletionDiagnostic {
            reason: "boundary_node_completion_no_missing_faces",
            missing_face_count: 0,
            cap_candidate_count: 0,
            outside_candidate_count: 0,
            duplicate_candidate_count: 0,
            max_rejected_scaled_jacobian: 0.0,
            rejected_scaled_jacobian_bins: BTreeMap::new(),
            max_rejected_cap_height_ratio: 0.0,
            rejected_cap_height_ratio_bins: BTreeMap::new(),
            rejected_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            rejected_cap_node_ids: BTreeMap::new(),
            split_cap_candidate_count: 0,
            split_cap_pass_count: 0,
            max_split_cap_scaled_jacobian: 0.0,
            split_cap_scaled_jacobian_bins: BTreeMap::new(),
            split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            split_cap_apex_limited_node_ids: BTreeMap::new(),
            edge_split_cap_candidate_count: 0,
            edge_split_cap_pass_count: 0,
            max_edge_split_cap_scaled_jacobian: 0.0,
            edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
            edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
            three_edge_split_cap_candidate_count: 0,
            three_edge_split_cap_pass_count: 0,
            max_three_edge_split_cap_scaled_jacobian: 0.0,
            three_edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
            three_edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            three_edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
            rejected_by_reason: BTreeMap::new(),
        });
    }
    aggregate.reason = "boundary_node_completion_completed";
    Ok(aggregate)
}

#[cfg(test)]
pub(crate) fn diagnostic_interior_star_quality(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    interior_candidates: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<InteriorStarQualityDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let mut diagnostic = InteriorStarQualityDiagnostic {
        candidate_count: 0,
        pass_count: 0,
        max_min_scaled_jacobian: 0.0,
        min_scaled_jacobian_bins: BTreeMap::new(),
        min_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        rejected_by_reason: BTreeMap::new(),
    };
    let mut seen_interior_nodes = BTreeSet::<u32>::new();
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    for node in interior_candidates {
        if !seen_interior_nodes.insert(node.node_id) {
            *diagnostic
                .rejected_by_reason
                .entry("duplicate_interior_node")
                .or_default() += 1;
            continue;
        }
        if boundary_node_ids.contains(&node.node_id) {
            *diagnostic
                .rejected_by_reason
                .entry("interior_node_reuses_boundary_node")
                .or_default() += 1;
            continue;
        }
        if !candidate_respects_protected_boundary_distance(
            cavity,
            &boundary_node_map,
            node.coordinates_m,
            options,
        ) {
            *diagnostic
                .rejected_by_reason
                .entry("protected_boundary_distance")
                .or_default() += 1;
            continue;
        }
        if point_in_closed_triangle_surface(
            node.coordinates_m,
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            *diagnostic
                .rejected_by_reason
                .entry("interior_point_outside_cavity")
                .or_default() += 1;
            continue;
        }
        diagnostic.candidate_count += 1;
        match star_refill_candidate_with_rejection_reason(
            cavity,
            &boundary_node_map,
            node.clone(),
            diagnostic_options,
        ) {
            Ok(Ok(refill)) => {
                let min_quality = refill
                    .tets
                    .iter()
                    .map(|tet| tet.exact_scaled_jacobian)
                    .fold(f64::INFINITY, f64::min);
                if min_quality.is_finite() {
                    diagnostic.max_min_scaled_jacobian =
                        diagnostic.max_min_scaled_jacobian.max(min_quality);
                    *diagnostic
                        .min_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(min_quality))
                        .or_default() += 1;
                    if let Some(worst_tet) = refill.tets.iter().min_by(|left, right| {
                        left.exact_scaled_jacobian
                            .total_cmp(&right.exact_scaled_jacobian)
                    }) {
                        let points = worst_tet.node_ids.map(|node_id| {
                            if node_id == node.node_id {
                                node.coordinates_m
                            } else {
                                boundary_node_map[&node_id]
                            }
                        });
                        *diagnostic
                            .min_scaled_jacobian_worst_corner_bins
                            .entry(diagnostic_scaled_jacobian_worst_corner_label(points))
                            .or_default() += 1;
                    }
                    if min_quality >= options.min_scaled_jacobian {
                        diagnostic.pass_count += 1;
                    }
                }
            }
            Ok(Err(reason)) => {
                *diagnostic
                    .rejected_by_reason
                    .entry(boundary_node_refill_rejection_reason(reason))
                    .or_default() += 1;
            }
            Err(err) => {
                *diagnostic
                    .rejected_by_reason
                    .entry(boundary_node_refill_validation_reason(&err))
                    .or_default() += 1;
            }
        }
    }
    Ok(diagnostic)
}

#[cfg(test)]
fn diagnostic_boundary_face_completion(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    refill_tets: &[ConstrainedCavityRefillTet],
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
    missing_face_count: usize,
) -> BoundaryNodeCompletionDiagnostic {
    let mut cap_candidate_count = 0_usize;
    let mut outside_candidate_count = 0_usize;
    let mut duplicate_candidate_count = 0_usize;
    let mut max_rejected_scaled_jacobian = 0.0_f64;
    let mut rejected_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut max_rejected_cap_height_ratio = 0.0_f64;
    let mut rejected_cap_height_ratio_bins = BTreeMap::<String, usize>::new();
    let mut rejected_scaled_jacobian_worst_corner_bins = BTreeMap::<&'static str, usize>::new();
    let mut rejected_cap_node_ids = BTreeMap::<u32, usize>::new();
    let mut split_cap_candidate_count = 0_usize;
    let mut split_cap_pass_count = 0_usize;
    let mut max_split_cap_scaled_jacobian = 0.0_f64;
    let mut split_cap_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut split_cap_scaled_jacobian_worst_corner_bins = BTreeMap::<&'static str, usize>::new();
    let mut split_cap_apex_limited_node_ids = BTreeMap::<u32, usize>::new();
    let mut edge_split_cap_candidate_count = 0_usize;
    let mut edge_split_cap_pass_count = 0_usize;
    let mut max_edge_split_cap_scaled_jacobian = 0.0_f64;
    let mut edge_split_cap_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut edge_split_cap_scaled_jacobian_worst_corner_bins =
        BTreeMap::<&'static str, usize>::new();
    let mut edge_split_cap_apex_limited_node_ids = BTreeMap::<u32, usize>::new();
    let mut three_edge_split_cap_candidate_count = 0_usize;
    let mut three_edge_split_cap_pass_count = 0_usize;
    let mut max_three_edge_split_cap_scaled_jacobian = 0.0_f64;
    let mut three_edge_split_cap_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut three_edge_split_cap_scaled_jacobian_worst_corner_bins =
        BTreeMap::<&'static str, usize>::new();
    let mut three_edge_split_cap_apex_limited_node_ids = BTreeMap::<u32, usize>::new();
    let mut rejected_by_reason = BTreeMap::<&'static str, usize>::new();
    let mut saw_non_duplicate = false;
    for node_id in cavity_boundary_node_ids(cavity) {
        if face.contains(&node_id) {
            continue;
        }
        let node_ids = [face[0], face[1], face[2], node_id];
        let points = node_ids.map(|id| boundary_nodes[&id]);
        if point_in_closed_triangle_surface(
            tet_centroid(points),
            boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            outside_candidate_count += 1;
            continue;
        }
        match raw_refill_tet_with_rejection_reason(node_ids, points, options) {
            Ok(tet) => {
                cap_candidate_count += 1;
                if refill_tets.iter().any(|existing| {
                    sorted_tet_nodes(existing.node_ids) == sorted_tet_nodes(tet.node_ids)
                }) {
                    duplicate_candidate_count += 1;
                } else {
                    saw_non_duplicate = true;
                }
            }
            Err(reason) => {
                *rejected_cap_node_ids.entry(node_id).or_default() += 1;
                let exact_scaled_jacobian = tet_scaled_jacobian(points);
                if exact_scaled_jacobian.is_finite() {
                    max_rejected_scaled_jacobian =
                        max_rejected_scaled_jacobian.max(exact_scaled_jacobian);
                    *rejected_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(exact_scaled_jacobian))
                        .or_default() += 1;
                    *rejected_scaled_jacobian_worst_corner_bins
                        .entry(diagnostic_scaled_jacobian_worst_corner_label(points))
                        .or_default() += 1;
                }
                let cap_height_ratio =
                    diagnostic_face_apex_height_ratio(face, node_id, boundary_nodes);
                if cap_height_ratio.is_finite() {
                    max_rejected_cap_height_ratio =
                        max_rejected_cap_height_ratio.max(cap_height_ratio);
                    *rejected_cap_height_ratio_bins
                        .entry(diagnostic_height_ratio_bin(cap_height_ratio))
                        .or_default() += 1;
                }
                if let Some((split_min_quality, split_worst_corner)) =
                    diagnostic_split_cap_min_scaled_jacobian(face, node_id, boundary_nodes, options)
                {
                    split_cap_candidate_count += 1;
                    max_split_cap_scaled_jacobian =
                        max_split_cap_scaled_jacobian.max(split_min_quality);
                    *split_cap_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(split_min_quality))
                        .or_default() += 1;
                    *split_cap_scaled_jacobian_worst_corner_bins
                        .entry(split_worst_corner)
                        .or_default() += 1;
                    if split_worst_corner == "apex" {
                        *split_cap_apex_limited_node_ids.entry(node_id).or_default() += 1;
                    }
                    if split_min_quality >= options.min_scaled_jacobian {
                        split_cap_pass_count += 1;
                    }
                }
                if let Some((edge_split_min_quality, edge_split_worst_corner)) =
                    diagnostic_edge_split_cap_min_scaled_jacobian(
                        face,
                        node_id,
                        boundary_nodes,
                        options,
                    )
                {
                    edge_split_cap_candidate_count += 1;
                    max_edge_split_cap_scaled_jacobian =
                        max_edge_split_cap_scaled_jacobian.max(edge_split_min_quality);
                    *edge_split_cap_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(edge_split_min_quality))
                        .or_default() += 1;
                    *edge_split_cap_scaled_jacobian_worst_corner_bins
                        .entry(edge_split_worst_corner)
                        .or_default() += 1;
                    if edge_split_worst_corner == "apex" {
                        *edge_split_cap_apex_limited_node_ids
                            .entry(node_id)
                            .or_default() += 1;
                    }
                    if edge_split_min_quality >= options.min_scaled_jacobian {
                        edge_split_cap_pass_count += 1;
                    }
                }
                if let Some((three_edge_split_min_quality, three_edge_split_worst_corner)) =
                    diagnostic_three_edge_split_cap_min_scaled_jacobian(
                        face,
                        node_id,
                        boundary_nodes,
                        options,
                    )
                {
                    three_edge_split_cap_candidate_count += 1;
                    max_three_edge_split_cap_scaled_jacobian =
                        max_three_edge_split_cap_scaled_jacobian.max(three_edge_split_min_quality);
                    *three_edge_split_cap_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(three_edge_split_min_quality))
                        .or_default() += 1;
                    *three_edge_split_cap_scaled_jacobian_worst_corner_bins
                        .entry(three_edge_split_worst_corner)
                        .or_default() += 1;
                    if three_edge_split_worst_corner == "apex" {
                        *three_edge_split_cap_apex_limited_node_ids
                            .entry(node_id)
                            .or_default() += 1;
                    }
                    if three_edge_split_min_quality >= options.min_scaled_jacobian {
                        three_edge_split_cap_pass_count += 1;
                    }
                }
                *rejected_by_reason
                    .entry(boundary_node_refill_rejection_reason(reason))
                    .or_default() += 1;
            }
        }
    }
    let reason = if saw_non_duplicate {
        "boundary_node_completion_has_candidate"
    } else if duplicate_candidate_count > 0 {
        "boundary_node_completion_duplicate_tet"
    } else {
        "boundary_node_completion_no_candidate"
    };
    BoundaryNodeCompletionDiagnostic {
        reason,
        missing_face_count,
        cap_candidate_count,
        outside_candidate_count,
        duplicate_candidate_count,
        max_rejected_scaled_jacobian,
        rejected_scaled_jacobian_bins,
        max_rejected_cap_height_ratio,
        rejected_cap_height_ratio_bins,
        rejected_scaled_jacobian_worst_corner_bins,
        rejected_cap_node_ids,
        split_cap_candidate_count,
        split_cap_pass_count,
        max_split_cap_scaled_jacobian,
        split_cap_scaled_jacobian_bins,
        split_cap_scaled_jacobian_worst_corner_bins,
        split_cap_apex_limited_node_ids,
        edge_split_cap_candidate_count,
        edge_split_cap_pass_count,
        max_edge_split_cap_scaled_jacobian,
        edge_split_cap_scaled_jacobian_bins,
        edge_split_cap_scaled_jacobian_worst_corner_bins,
        edge_split_cap_apex_limited_node_ids,
        three_edge_split_cap_candidate_count,
        three_edge_split_cap_pass_count,
        max_three_edge_split_cap_scaled_jacobian,
        three_edge_split_cap_scaled_jacobian_bins,
        three_edge_split_cap_scaled_jacobian_worst_corner_bins,
        three_edge_split_cap_apex_limited_node_ids,
        rejected_by_reason,
    }
}

#[cfg(test)]
fn diagnostic_split_cap_min_scaled_jacobian(
    face: [u32; 3],
    cap_node_id: u32,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Option<(f64, &'static str)> {
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    boundary_face_split_node_candidates(face, boundary_nodes)
        .into_iter()
        .filter_map(|split_node| {
            split_completion_tets_for_node(
                face,
                cap_node_id,
                &split_node,
                boundary_nodes,
                diagnostic_options,
            )
            .map(|tets| {
                tets.iter()
                    .map(|tet| {
                        let points = tet.node_ids.map(|node_id| {
                            if node_id == split_node.node_id {
                                split_node.coordinates_m
                            } else {
                                boundary_nodes[&node_id]
                            }
                        });
                        (
                            tet.exact_scaled_jacobian,
                            diagnostic_scaled_jacobian_worst_corner_label(points),
                        )
                    })
                    .min_by(|left, right| left.0.total_cmp(&right.0))
                    .unwrap_or((f64::INFINITY, "face_vertex"))
            })
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
}

#[cfg(test)]
fn diagnostic_edge_split_cap_min_scaled_jacobian(
    face: [u32; 3],
    cap_node_id: u32,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Option<(f64, &'static str)> {
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    boundary_face_edge_split_node_candidates(face, boundary_nodes)
        .into_iter()
        .filter_map(|(edge, split_node)| {
            edge_split_completion_tets_for_node(
                face,
                edge,
                cap_node_id,
                &split_node,
                boundary_nodes,
                diagnostic_options,
            )
            .map(|tets| {
                tets.iter()
                    .map(|tet| {
                        let points = tet.node_ids.map(|node_id| {
                            if node_id == split_node.node_id {
                                split_node.coordinates_m
                            } else {
                                boundary_nodes[&node_id]
                            }
                        });
                        (
                            tet.exact_scaled_jacobian,
                            diagnostic_scaled_jacobian_worst_corner_label(points),
                        )
                    })
                    .min_by(|left, right| left.0.total_cmp(&right.0))
                    .unwrap_or((f64::INFINITY, "face_vertex"))
            })
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
}

#[cfg(test)]
fn diagnostic_three_edge_split_cap_min_scaled_jacobian(
    face: [u32; 3],
    cap_node_id: u32,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Option<(f64, &'static str)> {
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let split_nodes = boundary_face_mid_edge_split_nodes(face, boundary_nodes);
    let split_node_by_edge = face_edges(face)
        .into_iter()
        .zip(split_nodes.iter())
        .map(|(edge, node)| (sorted_edge(edge), node.node_id))
        .collect::<BTreeMap<_, _>>();
    let split_node_coordinates = split_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    three_edge_split_completion_tets_for_node(
        face,
        cap_node_id,
        &split_node_by_edge,
        &split_node_coordinates,
        boundary_nodes,
        diagnostic_options,
    )
    .map(|tets| {
        tets.iter()
            .map(|tet| {
                let points = tet.node_ids.map(|node_id| {
                    split_node_coordinates
                        .get(&node_id)
                        .copied()
                        .unwrap_or_else(|| boundary_nodes[&node_id])
                });
                (
                    tet.exact_scaled_jacobian,
                    diagnostic_scaled_jacobian_worst_corner_label(points),
                )
            })
            .min_by(|left, right| left.0.total_cmp(&right.0))
            .unwrap_or((f64::INFINITY, "face_vertex"))
    })
}

#[cfg(test)]
fn diagnostic_scaled_jacobian_bin(value: f64) -> String {
    if value < 0.01 {
        "lt_0_01".to_string()
    } else if value < 0.05 {
        "lt_0_05".to_string()
    } else if value < 0.10 {
        "lt_0_10".to_string()
    } else if value < 0.15 {
        "lt_0_15".to_string()
    } else {
        "gte_0_15".to_string()
    }
}

#[cfg(test)]
fn diagnostic_face_apex_height_ratio(
    face: [u32; 3],
    apex_node_id: u32,
    boundary_nodes: &BTreeMap<u32, Point3>,
) -> f64 {
    let triangle = face.map(|node_id| boundary_nodes[&node_id]);
    let apex = boundary_nodes[&apex_node_id];
    let longest_edge = crate::predicate::distance(triangle[0], triangle[1])
        .max(crate::predicate::distance(triangle[1], triangle[2]))
        .max(crate::predicate::distance(triangle[2], triangle[0]));
    if !longest_edge.is_finite() || longest_edge <= f64::EPSILON {
        return 0.0;
    }
    let edge_ab = [
        triangle[1][0] - triangle[0][0],
        triangle[1][1] - triangle[0][1],
        triangle[1][2] - triangle[0][2],
    ];
    let edge_ac = [
        triangle[2][0] - triangle[0][0],
        triangle[2][1] - triangle[0][1],
        triangle[2][2] - triangle[0][2],
    ];
    let normal = [
        edge_ab[1] * edge_ac[2] - edge_ab[2] * edge_ac[1],
        edge_ab[2] * edge_ac[0] - edge_ab[0] * edge_ac[2],
        edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0],
    ];
    let normal_length =
        (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if !normal_length.is_finite() || normal_length <= f64::EPSILON {
        return 0.0;
    }
    let apex_delta = [
        apex[0] - triangle[0][0],
        apex[1] - triangle[0][1],
        apex[2] - triangle[0][2],
    ];
    let signed_height =
        (apex_delta[0] * normal[0] + apex_delta[1] * normal[1] + apex_delta[2] * normal[2])
            / normal_length;
    signed_height.abs() / longest_edge
}

#[cfg(test)]
fn diagnostic_height_ratio_bin(value: f64) -> String {
    if value < 0.01 {
        "lt_0_01".to_string()
    } else if value < 0.05 {
        "lt_0_05".to_string()
    } else if value < 0.10 {
        "lt_0_10".to_string()
    } else if value < 0.25 {
        "lt_0_25".to_string()
    } else {
        "gte_0_25".to_string()
    }
}

#[cfg(test)]
fn diagnostic_scaled_jacobian_worst_corner_label(points: [Point3; 4]) -> &'static str {
    let corners = [
        (0_usize, points[0], points[1], points[2], points[3]),
        (1_usize, points[1], points[0], points[3], points[2]),
        (2_usize, points[2], points[0], points[1], points[3]),
        (3_usize, points[3], points[0], points[2], points[1]),
    ];
    let worst_corner = corners
        .into_iter()
        .map(|(index, origin, first, second, third)| {
            let first = crate::predicate::sub(first, origin);
            let second = crate::predicate::sub(second, origin);
            let third = crate::predicate::sub(third, origin);
            let denominator = crate::predicate::norm(first)
                * crate::predicate::norm(second)
                * crate::predicate::norm(third);
            let scaled_jacobian = if denominator <= f64::EPSILON {
                0.0
            } else {
                (2.0_f64.sqrt()
                    * crate::predicate::dot(first, crate::predicate::cross(second, third))
                    / denominator)
                    .abs()
            };
            (index, scaled_jacobian)
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(index, _)| index)
        .unwrap_or(3);
    if worst_corner == 3 {
        "apex"
    } else {
        "face_vertex"
    }
}

fn complete_missing_boundary_face_tets(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    mut refill_tets: Vec<ConstrainedCavityRefillTet>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    Result<
        (
            ConstrainedCavity,
            Vec<ConstrainedCavityRefillTet>,
            Vec<ConstrainedCavityNode>,
        ),
        &'static str,
    >,
    ConstrainedCavityValidationError,
> {
    let mut refined_cavity = cavity.clone();
    let mut refined_boundary_nodes = boundary_nodes.clone();
    let mut inserted_nodes = Vec::<ConstrainedCavityNode>::new();
    let mut changed = false;
    loop {
        let boundary_delta = refill_boundary_face_delta(&refined_cavity, &refill_tets)?;
        if boundary_delta.missing.is_empty() {
            if boundary_delta.unexpected.is_empty() {
                break;
            }
            let Some((_, tet)) = best_boundary_face_completion_tet_for_faces(
                &boundary_delta.unexpected,
                &refined_cavity,
                &refined_boundary_nodes,
                &refill_tets,
                boundary_triangles,
                options,
            )?
            else {
                return Ok(Err("boundary_node_completion_no_candidate"));
            };
            refill_tets.push(tet);
            changed = true;
            continue;
        }
        if let Some((_, tet)) = best_boundary_face_completion_tet_for_faces(
            &boundary_delta.missing,
            &refined_cavity,
            &refined_boundary_nodes,
            &refill_tets,
            boundary_triangles,
            options,
        )? {
            refill_tets.push(tet);
            changed = true;
            continue;
        }

        let split_completion = if let Some((split_cavity, split_node, split_tets)) =
            best_boundary_face_edge_split_completion_for_faces(
                &boundary_delta.missing,
                &refined_cavity,
                &refined_boundary_nodes,
                boundary_triangles,
                &refill_tets,
                options,
            )? {
            Some((split_cavity, vec![split_node], split_tets))
        } else if let Some((split_cavity, split_node, split_tets)) =
            best_boundary_face_split_completion_for_faces(
                &boundary_delta.missing,
                &refined_cavity,
                &refined_boundary_nodes,
                boundary_triangles,
                &refill_tets,
                options,
            )?
        {
            Some((split_cavity, vec![split_node], split_tets))
        } else {
            best_boundary_face_three_edge_split_completion_for_faces(
                &boundary_delta.missing,
                &refined_cavity,
                &refined_boundary_nodes,
                boundary_triangles,
                &refill_tets,
                options,
            )?
        };
        let Some((split_cavity, split_nodes, split_tets)) = split_completion else {
            return Ok(Err("boundary_node_completion_no_candidate"));
        };
        for split_node in split_nodes {
            refined_boundary_nodes.insert(split_node.node_id, split_node.coordinates_m);
            inserted_nodes.push(split_node);
        }
        refined_cavity = split_cavity;
        refill_tets.extend(split_tets);
        changed = true;
    }
    if changed {
        Ok(Ok((refined_cavity, refill_tets, inserted_nodes)))
    } else {
        Ok(Err("boundary_node_completion_no_missing_faces"))
    }
}

fn best_boundary_face_completion_tet_for_faces(
    faces: &[[u32; 3]],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    refill_tets: &[ConstrainedCavityRefillTet],
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<([u32; 3], ConstrainedCavityRefillTet)>, ConstrainedCavityValidationError> {
    let current_delta = refill_boundary_face_delta(cavity, refill_tets)?;
    let current_delta_count = current_delta.missing.len() + current_delta.unexpected.len();
    let mut best = None::<([u32; 3], ConstrainedCavityRefillTet, usize)>;
    for face in faces {
        let Some(tet) = best_boundary_face_completion_tet(
            *face,
            cavity,
            boundary_nodes,
            refill_tets,
            boundary_triangles,
            options,
        ) else {
            continue;
        };
        let mut candidate_tets = refill_tets.to_vec();
        candidate_tets.push(tet.clone());
        let candidate_delta = refill_boundary_face_delta(cavity, &candidate_tets)?;
        let candidate_delta_count =
            candidate_delta.missing.len() + candidate_delta.unexpected.len();
        if candidate_delta_count >= current_delta_count {
            continue;
        }
        if best.as_ref().is_none_or(|(_, best_tet, best_delta)| {
            candidate_delta_count < *best_delta
                || (candidate_delta_count == *best_delta
                    && tet.exact_scaled_jacobian > best_tet.exact_scaled_jacobian)
        }) {
            best = Some((*face, tet, candidate_delta_count));
        }
    }
    Ok(best.map(|(face, tet, _)| (face, tet)))
}

fn best_boundary_face_split_completion_for_faces(
    faces: &[[u32; 3]],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    refill_tets: &[ConstrainedCavityRefillTet],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    Option<(
        ConstrainedCavity,
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTet>,
    )>,
    ConstrainedCavityValidationError,
> {
    let current_delta = refill_boundary_face_delta(cavity, refill_tets)?;
    let current_delta_count = current_delta.missing.len() + current_delta.unexpected.len();
    let mut best = None::<(
        ConstrainedCavity,
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTet>,
        f64,
    )>;
    for face in faces {
        let Some((split_cavity, split_node, split_tets)) = best_boundary_face_split_completion(
            *face,
            cavity,
            boundary_nodes,
            boundary_triangles,
            refill_tets,
            options,
        )?
        else {
            continue;
        };
        let mut candidate_tets = refill_tets.to_vec();
        candidate_tets.extend(split_tets.clone());
        let candidate_delta = refill_boundary_face_delta(&split_cavity, &candidate_tets)?;
        let candidate_delta_count =
            candidate_delta.missing.len() + candidate_delta.unexpected.len();
        if candidate_delta_count >= current_delta_count {
            continue;
        }
        let min_quality = split_tets
            .iter()
            .map(|tet| tet.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        if best
            .as_ref()
            .is_none_or(|(_, _, _, best_quality)| min_quality > *best_quality)
        {
            best = Some((split_cavity, split_node, split_tets, min_quality));
        }
    }
    Ok(
        best.map(|(split_cavity, split_node, split_tets, _)| {
            (split_cavity, split_node, split_tets)
        }),
    )
}

fn best_boundary_face_edge_split_completion_for_faces(
    faces: &[[u32; 3]],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    refill_tets: &[ConstrainedCavityRefillTet],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    Option<(
        ConstrainedCavity,
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTet>,
    )>,
    ConstrainedCavityValidationError,
> {
    let current_delta = refill_boundary_face_delta(cavity, refill_tets)?;
    let current_delta_count = current_delta.missing.len() + current_delta.unexpected.len();
    let mut best = None::<(
        ConstrainedCavity,
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTet>,
        f64,
        usize,
    )>;
    for face in faces {
        let Some((split_cavity, split_node, split_tets)) =
            best_boundary_face_edge_split_completion(
                *face,
                cavity,
                boundary_nodes,
                boundary_triangles,
                refill_tets,
                options,
            )?
        else {
            continue;
        };
        let mut candidate_tets = refill_tets.to_vec();
        candidate_tets.extend(split_tets.clone());
        let candidate_delta = refill_boundary_face_delta(&split_cavity, &candidate_tets)?;
        let candidate_delta_count =
            candidate_delta.missing.len() + candidate_delta.unexpected.len();
        if candidate_delta_count >= current_delta_count {
            continue;
        }
        let min_quality = split_tets
            .iter()
            .map(|tet| tet.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        if best
            .as_ref()
            .is_none_or(|(_, _, _, best_quality, best_delta_count)| {
                candidate_delta_count < *best_delta_count
                    || (candidate_delta_count == *best_delta_count && min_quality > *best_quality)
            })
        {
            best = Some((
                split_cavity,
                split_node,
                split_tets,
                min_quality,
                candidate_delta_count,
            ));
        }
    }
    Ok(best
        .map(|(split_cavity, split_node, split_tets, _, _)| (split_cavity, split_node, split_tets)))
}

fn best_boundary_face_three_edge_split_completion_for_faces(
    faces: &[[u32; 3]],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    refill_tets: &[ConstrainedCavityRefillTet],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    Option<(
        ConstrainedCavity,
        Vec<ConstrainedCavityNode>,
        Vec<ConstrainedCavityRefillTet>,
    )>,
    ConstrainedCavityValidationError,
> {
    let current_delta = refill_boundary_face_delta(cavity, refill_tets)?;
    let current_delta_count = current_delta.missing.len() + current_delta.unexpected.len();
    let mut best = None::<(
        ConstrainedCavity,
        Vec<ConstrainedCavityNode>,
        Vec<ConstrainedCavityRefillTet>,
        f64,
        usize,
    )>;
    for face in faces {
        let Some((split_cavity, split_nodes, split_tets)) =
            best_boundary_face_three_edge_split_completion(
                *face,
                cavity,
                boundary_nodes,
                boundary_triangles,
                refill_tets,
                options,
            )?
        else {
            continue;
        };
        let mut candidate_tets = refill_tets.to_vec();
        candidate_tets.extend(split_tets.clone());
        let candidate_delta = refill_boundary_face_delta(&split_cavity, &candidate_tets)?;
        let candidate_delta_count =
            candidate_delta.missing.len() + candidate_delta.unexpected.len();
        if candidate_delta_count >= current_delta_count {
            continue;
        }
        let min_quality = split_tets
            .iter()
            .map(|tet| tet.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        if best
            .as_ref()
            .is_none_or(|(_, _, _, best_quality, best_delta_count)| {
                candidate_delta_count < *best_delta_count
                    || (candidate_delta_count == *best_delta_count && min_quality > *best_quality)
            })
        {
            best = Some((
                split_cavity,
                split_nodes,
                split_tets,
                min_quality,
                candidate_delta_count,
            ));
        }
    }
    Ok(best.map(|(split_cavity, split_nodes, split_tets, _, _)| {
        (split_cavity, split_nodes, split_tets)
    }))
}

fn best_boundary_face_edge_split_completion(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    refill_tets: &[ConstrainedCavityRefillTet],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    Option<(
        ConstrainedCavity,
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTet>,
    )>,
    ConstrainedCavityValidationError,
> {
    let split_candidates = boundary_face_edge_split_node_candidates(face, boundary_nodes);
    let mut best = None::<(
        [u32; 2],
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTet>,
        f64,
    )>;
    for cap_node_id in cavity_boundary_node_ids(cavity) {
        if face.contains(&cap_node_id) {
            continue;
        }
        for (edge, split_node) in &split_candidates {
            let Some(child_tets) = edge_split_completion_tets_for_node(
                face,
                *edge,
                cap_node_id,
                split_node,
                boundary_nodes,
                options,
            ) else {
                continue;
            };
            if child_tets.iter().any(|tet| {
                let tet_points = tet.node_ids.map(|node_id| {
                    if node_id == split_node.node_id {
                        split_node.coordinates_m
                    } else {
                        boundary_nodes[&node_id]
                    }
                });
                point_in_closed_triangle_surface(
                    tet_centroid(tet_points),
                    boundary_triangles,
                    MeshingTolerance::default(),
                ) != PointInClosedSurface::Inside
            }) {
                continue;
            }
            if child_tets.iter().any(|tet| {
                refill_tets.iter().any(|existing| {
                    sorted_tet_nodes(existing.node_ids) == sorted_tet_nodes(tet.node_ids)
                })
            }) {
                continue;
            }
            let min_quality = child_tets
                .iter()
                .map(|tet| tet.exact_scaled_jacobian)
                .fold(f64::INFINITY, f64::min);
            if best
                .as_ref()
                .is_none_or(|(_, _, _, best_quality)| min_quality > *best_quality)
            {
                best = Some((*edge, split_node.clone(), child_tets, min_quality));
            }
        }
    }
    let Some((edge, split_node, split_tets, _)) = best else {
        return Ok(None);
    };
    let split_faces = split_constrained_cavity_boundary_faces_on_edge(
        &cavity.boundary_faces,
        face,
        edge,
        split_node.node_id,
    )
    .map_err(|_| ConstrainedCavityValidationError::MissingBoundaryFace {
        node_ids: sorted_face(face),
    })?;
    let mut split_cavity = cavity.clone();
    split_cavity.boundary_faces = split_faces;
    validate_constrained_cavity(&split_cavity)?;
    Ok(Some((split_cavity, split_node, split_tets)))
}

fn best_boundary_face_three_edge_split_completion(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    refill_tets: &[ConstrainedCavityRefillTet],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    Option<(
        ConstrainedCavity,
        Vec<ConstrainedCavityNode>,
        Vec<ConstrainedCavityRefillTet>,
    )>,
    ConstrainedCavityValidationError,
> {
    let split_nodes = boundary_face_mid_edge_split_nodes(face, boundary_nodes);
    let split_node_by_edge = face_edges(face)
        .into_iter()
        .zip(split_nodes.iter())
        .map(|(edge, node)| (sorted_edge(edge), node.node_id))
        .collect::<BTreeMap<_, _>>();
    let split_node_coordinates = split_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut best = None::<(Vec<ConstrainedCavityRefillTet>, f64)>;
    for cap_node_id in cavity_boundary_node_ids(cavity) {
        if face.contains(&cap_node_id) {
            continue;
        }
        let Some(child_tets) = three_edge_split_completion_tets_for_node(
            face,
            cap_node_id,
            &split_node_by_edge,
            &split_node_coordinates,
            boundary_nodes,
            options,
        ) else {
            continue;
        };
        if child_tets.iter().any(|tet| {
            let tet_points = tet.node_ids.map(|node_id| {
                split_node_coordinates
                    .get(&node_id)
                    .copied()
                    .unwrap_or_else(|| boundary_nodes[&node_id])
            });
            point_in_closed_triangle_surface(
                tet_centroid(tet_points),
                boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
        }) {
            continue;
        }
        if child_tets.iter().any(|tet| {
            refill_tets.iter().any(|existing| {
                sorted_tet_nodes(existing.node_ids) == sorted_tet_nodes(tet.node_ids)
            })
        }) {
            continue;
        }
        let min_quality = child_tets
            .iter()
            .map(|tet| tet.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        if best
            .as_ref()
            .is_none_or(|(_, best_quality)| min_quality > *best_quality)
        {
            best = Some((child_tets, min_quality));
        }
    }
    let Some((split_tets, _)) = best else {
        return Ok(None);
    };
    let split_faces = split_constrained_cavity_boundary_faces_on_three_edges(
        &cavity.boundary_faces,
        face,
        split_node_by_edge,
    )
    .map_err(|_| ConstrainedCavityValidationError::MissingBoundaryFace {
        node_ids: sorted_face(face),
    })?;
    let mut split_cavity = cavity.clone();
    split_cavity.boundary_faces = split_faces;
    validate_constrained_cavity(&split_cavity)?;
    Ok(Some((split_cavity, split_nodes, split_tets)))
}

fn best_boundary_face_split_completion(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    refill_tets: &[ConstrainedCavityRefillTet],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    Option<(
        ConstrainedCavity,
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTet>,
    )>,
    ConstrainedCavityValidationError,
> {
    let split_candidates = boundary_face_split_node_candidates(face, boundary_nodes);
    let mut best = None::<(ConstrainedCavityNode, Vec<ConstrainedCavityRefillTet>, f64)>;
    for cap_node_id in cavity_boundary_node_ids(cavity) {
        if face.contains(&cap_node_id) {
            continue;
        }
        for split_node in &split_candidates {
            let Some(child_tets) = split_completion_tets_for_node(
                face,
                cap_node_id,
                split_node,
                boundary_nodes,
                options,
            ) else {
                continue;
            };
            if child_tets.iter().any(|tet| {
                let tet_points = tet.node_ids.map(|node_id| {
                    if node_id == split_node.node_id {
                        split_node.coordinates_m
                    } else {
                        boundary_nodes[&node_id]
                    }
                });
                point_in_closed_triangle_surface(
                    tet_centroid(tet_points),
                    boundary_triangles,
                    MeshingTolerance::default(),
                ) != PointInClosedSurface::Inside
            }) {
                continue;
            }
            if child_tets.iter().any(|tet| {
                refill_tets.iter().any(|existing| {
                    sorted_tet_nodes(existing.node_ids) == sorted_tet_nodes(tet.node_ids)
                })
            }) {
                continue;
            }
            let min_quality = child_tets
                .iter()
                .map(|tet| tet.exact_scaled_jacobian)
                .fold(f64::INFINITY, f64::min);
            if best
                .as_ref()
                .is_none_or(|(_, _, best_quality)| min_quality > *best_quality)
            {
                best = Some((split_node.clone(), child_tets, min_quality));
            }
        }
    }
    let Some((split_node, split_tets, _)) = best else {
        return Ok(None);
    };
    let split_faces =
        split_constrained_cavity_boundary_faces(&cavity.boundary_faces, face, split_node.node_id)
            .map_err(|_| ConstrainedCavityValidationError::MissingBoundaryFace {
            node_ids: sorted_face(face),
        })?;
    let mut split_cavity = cavity.clone();
    split_cavity.boundary_faces = split_faces;
    validate_constrained_cavity(&split_cavity)?;
    Ok(Some((split_cavity, split_node, split_tets)))
}

fn boundary_face_centroid_node(
    face: [u32; 3],
    boundary_nodes: &BTreeMap<u32, Point3>,
) -> ConstrainedCavityNode {
    boundary_face_split_node(face, boundary_nodes, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
}

fn boundary_face_split_node_candidates(
    face: [u32; 3],
    boundary_nodes: &BTreeMap<u32, Point3>,
) -> Vec<ConstrainedCavityNode> {
    let mut barycentric_candidates = [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5],
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],
        [0.70, 0.05, 0.25],
        [0.70, 0.25, 0.05],
        [0.05, 0.70, 0.25],
        [0.25, 0.70, 0.05],
        [0.05, 0.25, 0.70],
        [0.25, 0.05, 0.70],
    ]
    .into_iter()
    .collect::<Vec<_>>();
    for first in 1..10 {
        for second in 1..(10 - first) {
            let third = 10 - first - second;
            if third == 0 {
                continue;
            }
            let barycentric = [
                first as f64 / 10.0,
                second as f64 / 10.0,
                third as f64 / 10.0,
            ];
            if !barycentric_candidates.iter().any(|candidate| {
                candidate
                    .iter()
                    .zip(barycentric)
                    .all(|(left, right)| (*left - right).abs() <= 1.0e-12)
            }) {
                barycentric_candidates.push(barycentric);
            }
        }
    }
    barycentric_candidates
        .into_iter()
        .map(|barycentric| boundary_face_split_node(face, boundary_nodes, barycentric))
        .collect()
}

fn boundary_face_edge_split_node_candidates(
    face: [u32; 3],
    boundary_nodes: &BTreeMap<u32, Point3>,
) -> Vec<([u32; 2], ConstrainedCavityNode)> {
    face_edges(face)
        .into_iter()
        .flat_map(|edge| {
            [0.5, 0.25, 0.75].into_iter().map(move |fraction| {
                (
                    edge,
                    boundary_edge_split_node(edge, boundary_nodes, fraction),
                )
            })
        })
        .collect()
}

fn boundary_face_mid_edge_split_nodes(
    face: [u32; 3],
    boundary_nodes: &BTreeMap<u32, Point3>,
) -> Vec<ConstrainedCavityNode> {
    let mut next_node_id = boundary_nodes
        .keys()
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    face_edges(face)
        .into_iter()
        .map(|edge| {
            while boundary_nodes.contains_key(&next_node_id) {
                next_node_id = next_node_id.saturating_add(1);
            }
            let mut node = boundary_edge_split_node(edge, boundary_nodes, 0.5);
            node.node_id = next_node_id;
            next_node_id = next_node_id.saturating_add(1);
            node
        })
        .collect()
}

fn boundary_edge_split_node(
    edge: [u32; 2],
    boundary_nodes: &BTreeMap<u32, Point3>,
    fraction: f64,
) -> ConstrainedCavityNode {
    let points = edge.map(|node_id| boundary_nodes[&node_id]);
    let mut node_id = boundary_nodes
        .keys()
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    while boundary_nodes.contains_key(&node_id) {
        node_id = node_id.saturating_add(1);
    }
    ConstrainedCavityNode {
        node_id,
        coordinates_m: [
            points[0][0] * (1.0 - fraction) + points[1][0] * fraction,
            points[0][1] * (1.0 - fraction) + points[1][1] * fraction,
            points[0][2] * (1.0 - fraction) + points[1][2] * fraction,
        ],
    }
}

fn boundary_face_split_node(
    face: [u32; 3],
    boundary_nodes: &BTreeMap<u32, Point3>,
    barycentric: [f64; 3],
) -> ConstrainedCavityNode {
    let points = face.map(|node_id| boundary_nodes[&node_id]);
    let mut node_id = boundary_nodes
        .keys()
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    while boundary_nodes.contains_key(&node_id) {
        node_id = node_id.saturating_add(1);
    }
    ConstrainedCavityNode {
        node_id,
        coordinates_m: [
            points[0][0] * barycentric[0]
                + points[1][0] * barycentric[1]
                + points[2][0] * barycentric[2],
            points[0][1] * barycentric[0]
                + points[1][1] * barycentric[1]
                + points[2][1] * barycentric[2],
            points[0][2] * barycentric[0]
                + points[1][2] * barycentric[1]
                + points[2][2] * barycentric[2],
        ],
    }
}

fn split_completion_tets_for_node(
    face: [u32; 3],
    cap_node_id: u32,
    split_node: &ConstrainedCavityNode,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Option<Vec<ConstrainedCavityRefillTet>> {
    let child_specs = [
        [face[0], face[1], split_node.node_id, cap_node_id],
        [face[1], face[2], split_node.node_id, cap_node_id],
        [face[2], face[0], split_node.node_id, cap_node_id],
    ];
    let mut child_tets = Vec::<ConstrainedCavityRefillTet>::with_capacity(3);
    for node_ids in child_specs {
        let points = [
            boundary_nodes[&node_ids[0]],
            boundary_nodes[&node_ids[1]],
            split_node.coordinates_m,
            boundary_nodes[&cap_node_id],
        ];
        let tet = raw_refill_tet_with_rejection_reason(node_ids, points, options).ok()?;
        if child_tets
            .iter()
            .any(|existing| sorted_tet_nodes(existing.node_ids) == sorted_tet_nodes(tet.node_ids))
        {
            return None;
        }
        child_tets.push(tet);
    }
    Some(child_tets)
}

fn edge_split_completion_tets_for_node(
    face: [u32; 3],
    edge: [u32; 2],
    cap_node_id: u32,
    split_node: &ConstrainedCavityNode,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Option<Vec<ConstrainedCavityRefillTet>> {
    let [a, b] = edge;
    let c = face
        .into_iter()
        .find(|node_id| *node_id != a && *node_id != b)?;
    let child_specs = [
        [a, split_node.node_id, c, cap_node_id],
        [split_node.node_id, b, c, cap_node_id],
    ];
    let mut child_tets = Vec::<ConstrainedCavityRefillTet>::with_capacity(2);
    for node_ids in child_specs {
        let points = node_ids.map(|node_id| {
            if node_id == split_node.node_id {
                split_node.coordinates_m
            } else {
                boundary_nodes[&node_id]
            }
        });
        let tet = raw_refill_tet_with_rejection_reason(node_ids, points, options).ok()?;
        if child_tets
            .iter()
            .any(|existing| sorted_tet_nodes(existing.node_ids) == sorted_tet_nodes(tet.node_ids))
        {
            return None;
        }
        child_tets.push(tet);
    }
    Some(child_tets)
}

fn three_edge_split_completion_tets_for_node(
    face: [u32; 3],
    cap_node_id: u32,
    split_node_by_edge: &BTreeMap<[u32; 2], u32>,
    split_node_coordinates: &BTreeMap<u32, Point3>,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Option<Vec<ConstrainedCavityRefillTet>> {
    let [a, b, c] = face;
    let ab = *split_node_by_edge.get(&sorted_edge([a, b]))?;
    let bc = *split_node_by_edge.get(&sorted_edge([b, c]))?;
    let ca = *split_node_by_edge.get(&sorted_edge([c, a]))?;
    let child_specs = [
        [a, ab, ca, cap_node_id],
        [ab, b, bc, cap_node_id],
        [ca, bc, c, cap_node_id],
        [ab, bc, ca, cap_node_id],
    ];
    let mut child_tets = Vec::<ConstrainedCavityRefillTet>::with_capacity(4);
    for node_ids in child_specs {
        let points = node_ids.map(|node_id| {
            split_node_coordinates
                .get(&node_id)
                .copied()
                .unwrap_or_else(|| boundary_nodes[&node_id])
        });
        let tet = raw_refill_tet_with_rejection_reason(node_ids, points, options).ok()?;
        if child_tets
            .iter()
            .any(|existing| sorted_tet_nodes(existing.node_ids) == sorted_tet_nodes(tet.node_ids))
        {
            return None;
        }
        child_tets.push(tet);
    }
    Some(child_tets)
}

fn missing_refill_boundary_faces(
    cavity: &ConstrainedCavity,
    refill_tets: &[ConstrainedCavityRefillTet],
) -> Result<Vec<[u32; 3]>, ConstrainedCavityValidationError> {
    Ok(refill_boundary_face_delta(cavity, refill_tets)?.missing)
}

#[cfg(test)]
fn open_interior_refill_faces(
    cavity: &ConstrainedCavity,
    refill_tets: &[ConstrainedCavityRefillTet],
) -> Vec<[u32; 3]> {
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for tet in refill_tets {
        for face in tet_faces(tet.node_ids).map(sorted_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    face_counts
        .into_iter()
        .filter_map(|(face, count)| (!boundary_faces.contains(&face) && count == 1).then_some(face))
        .collect()
}

#[cfg(test)]
fn cap_side_face_mate_counts(
    cap_tets: &[ConstrainedCavityRefillTet],
    candidate_tets: &[ConstrainedCavityRefillTet],
    inserted_node_ids: &BTreeSet<u32>,
) -> Vec<usize> {
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for tet in candidate_tets {
        for face in tet_faces(tet.node_ids).map(sorted_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }

    let mut mate_counts = Vec::<usize>::new();
    for cap_tet in cap_tets {
        for face in tet_faces(cap_tet.node_ids).map(sorted_face) {
            if !face
                .iter()
                .any(|node_id| inserted_node_ids.contains(node_id))
            {
                continue;
            }
            mate_counts.push(
                face_counts
                    .get(&face)
                    .copied()
                    .unwrap_or(0)
                    .saturating_sub(1),
            );
        }
    }
    mate_counts
}

struct RefillBoundaryFaceDelta {
    missing: Vec<[u32; 3]>,
    unexpected: Vec<[u32; 3]>,
}

struct BoundaryExactCoverSearch<'a> {
    candidates: &'a [ConstrainedCavityRefillTet],
    candidate_faces: Vec<[[u32; 3]; 4]>,
    boundary_faces: BTreeSet<[u32; 3]>,
    target_volume_m3: f64,
    volume_tolerance_m3: f64,
    max_attempt_count: usize,
    attempts: usize,
}

impl<'a> BoundaryExactCoverSearch<'a> {
    fn new(
        cavity: &ConstrainedCavity,
        candidates: &'a [ConstrainedCavityRefillTet],
        volume_relative_tolerance: f64,
    ) -> Self {
        Self::with_attempt_limit(cavity, candidates, volume_relative_tolerance, 5_000)
    }

    fn with_attempt_limit(
        cavity: &ConstrainedCavity,
        candidates: &'a [ConstrainedCavityRefillTet],
        volume_relative_tolerance: f64,
        max_attempt_count: usize,
    ) -> Self {
        Self {
            candidates,
            candidate_faces: candidates
                .iter()
                .map(|candidate| tet_faces(candidate.node_ids).map(sorted_face))
                .collect(),
            boundary_faces: cavity
                .boundary_faces
                .iter()
                .map(|face| sorted_face(face.node_ids))
                .collect(),
            target_volume_m3: cavity.target_volume_m3,
            volume_tolerance_m3: cavity.target_volume_m3.max(1.0e-18) * volume_relative_tolerance,
            max_attempt_count,
            attempts: 0,
        }
    }

    fn search(&mut self) -> Option<Vec<usize>> {
        self.search_from(0.0, &mut BTreeMap::new(), &mut Vec::new())
    }

    fn search_from(
        &mut self,
        current_volume_m3: f64,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
    ) -> Option<Vec<usize>> {
        self.attempts += 1;
        if self.attempts > self.max_attempt_count
            || current_volume_m3 > self.target_volume_m3 + self.volume_tolerance_m3
        {
            return None;
        }
        let Some(candidate_indices) = self.next_cover_candidates(face_counts, selected) else {
            let boundary_ok = self
                .boundary_faces
                .iter()
                .all(|face| face_counts.get(face).copied().unwrap_or(0) == 1);
            let interior_ok = face_counts
                .iter()
                .all(|(face, count)| self.boundary_faces.contains(face) || *count == 2);
            if boundary_ok
                && interior_ok
                && (current_volume_m3 - self.target_volume_m3).abs() <= self.volume_tolerance_m3
            {
                return Some(selected.clone());
            }
            return None;
        };
        for candidate_index in candidate_indices {
            if selected.contains(&candidate_index) {
                continue;
            }
            for face in self.candidate_faces[candidate_index] {
                *face_counts.entry(face).or_default() += 1;
            }
            selected.push(candidate_index);
            if let Some(result) = self.search_from(
                current_volume_m3 + self.candidates[candidate_index].volume_m3,
                face_counts,
                selected,
            ) {
                return Some(result);
            }
            selected.pop();
            for face in self.candidate_faces[candidate_index] {
                if let Some(count) = face_counts.get_mut(&face) {
                    *count -= 1;
                    if *count == 0 {
                        face_counts.remove(&face);
                    }
                }
            }
        }
        None
    }

    fn next_cover_candidates(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> Option<Vec<usize>> {
        let mut best = None::<Vec<usize>>;
        for face in self
            .boundary_faces
            .iter()
            .filter(|face| face_counts.get(*face).copied().unwrap_or(0) == 0)
        {
            let mut candidates = (0..self.candidates.len())
                .filter(|candidate_index| {
                    !selected.contains(candidate_index)
                        && self.candidate_can_be_added_for_face(
                            *candidate_index,
                            *face,
                            face_counts,
                        )
                })
                .collect::<Vec<_>>();
            self.sort_cover_candidates(&mut candidates);
            if best
                .as_ref()
                .is_none_or(|current| candidates.len() < current.len())
            {
                best = Some(candidates);
            }
        }
        if best.is_some() {
            return best;
        }

        for face in face_counts.iter().filter_map(|(face, count)| {
            (!self.boundary_faces.contains(face) && *count == 1).then_some(*face)
        }) {
            let mut candidates = (0..self.candidates.len())
                .filter(|candidate_index| {
                    !selected.contains(candidate_index)
                        && self.candidate_can_be_added_for_face(*candidate_index, face, face_counts)
                })
                .collect::<Vec<_>>();
            self.sort_cover_candidates(&mut candidates);
            if best
                .as_ref()
                .is_none_or(|current| candidates.len() < current.len())
            {
                best = Some(candidates);
            }
        }
        best
    }

    fn candidate_can_be_added_for_face(
        &self,
        candidate_index: usize,
        face: [u32; 3],
        face_counts: &BTreeMap<[u32; 3], usize>,
    ) -> bool {
        self.candidate_faces[candidate_index].contains(&face)
            && self.candidate_faces[candidate_index]
                .iter()
                .all(|candidate_face| {
                    let count = face_counts.get(candidate_face).copied().unwrap_or(0);
                    if self.boundary_faces.contains(candidate_face) {
                        count == 0
                    } else {
                        count < 2
                    }
                })
    }

    fn sort_cover_candidates(&self, candidates: &mut [usize]) {
        candidates.sort_by(|left, right| {
            self.candidates[*right]
                .exact_scaled_jacobian
                .total_cmp(&self.candidates[*left].exact_scaled_jacobian)
        });
    }
}

fn boundary_node_exact_cover_refill_candidate(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefill>, ConstrainedCavityValidationError> {
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    if node_ids.len() < 4 || node_ids.len() > 8 || cavity.boundary_faces.len() > 16 {
        return Ok(None);
    }
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut candidates = Vec::<ConstrainedCavityRefillTet>::new();
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tet_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    if !tet_faces(tet_node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tet_node_ids.map(|node_id| boundary_nodes[&node_id]);
                    if point_in_closed_triangle_surface(
                        tet_centroid(points),
                        boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tet) =
                        raw_refill_tet_with_rejection_reason(tet_node_ids, points, options)
                    {
                        candidates.push(tet);
                    }
                }
            }
        }
    }
    if candidates.is_empty() || candidates.len() > 80 {
        return Ok(None);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let Some(selected_indices) = search.search() else {
        return Ok(None);
    };
    let selected_tets = selected_indices
        .into_iter()
        .map(|index| candidates[index].clone())
        .collect::<Vec<_>>();
    refill_from_tets(cavity, selected_tets, options.volume_relative_tolerance).map(Some)
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let mut diagnostic = BoundaryExactCoverDiagnostic {
        boundary_node_count: node_ids.len(),
        boundary_face_count: cavity.boundary_faces.len(),
        candidate_count: 0,
        zero_candidate_boundary_face_count: 0,
        min_boundary_face_candidate_count: 0,
        max_boundary_face_candidate_count: 0,
        selected_tet_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
    };
    if node_ids.len() < 4 {
        diagnostic.reason = "too_few_boundary_nodes";
        return Ok(diagnostic);
    }
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut candidates = Vec::<ConstrainedCavityRefillTet>::new();
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tet_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    if !tet_faces(tet_node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tet_node_ids.map(|node_id| boundary_node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tet_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tet) =
                        raw_refill_tet_with_rejection_reason(tet_node_ids, points, relaxed_options)
                    {
                        candidates.push(tet);
                    }
                }
            }
        }
    }
    diagnostic.candidate_count = candidates.len();
    let face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            candidates
                .iter()
                .filter(|candidate| {
                    tet_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect::<Vec<_>>();
    diagnostic.zero_candidate_boundary_face_count = face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.max_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().max().unwrap_or(0);
    if candidates.is_empty() {
        diagnostic.reason = "no_candidate_tets";
        return Ok(diagnostic);
    }
    if candidates.len() > 512 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let selected = search.search();
    diagnostic.search_attempt_count = search.attempts;
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 5_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.selected_tet_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_steiner_exact_cover(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundarySteinerExactCoverDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let mut diagnostic = BoundarySteinerExactCoverDiagnostic {
        boundary_node_count: boundary_node_ids.len(),
        boundary_face_count: cavity.boundary_faces.len(),
        candidate_count: 0,
        zero_candidate_boundary_face_count: 0,
        min_boundary_face_candidate_count: 0,
        max_boundary_face_candidate_count: 0,
        selected_tet_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        max_min_scaled_jacobian: 0.0,
    };
    if boundary_node_ids.len() < 4 {
        diagnostic.reason = "too_few_boundary_nodes";
        return Ok(diagnostic);
    }
    let Some(centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        diagnostic.reason = "empty_boundary";
        return Ok(diagnostic);
    };
    if point_in_closed_triangle_surface(centroid, &boundary_triangles, MeshingTolerance::default())
        != PointInClosedSurface::Inside
    {
        diagnostic.reason = "steiner_point_outside_cavity";
        return Ok(diagnostic);
    }
    let steiner_node_id = next_cavity_node_id(cavity);
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut node_points = boundary_node_ids
        .iter()
        .map(|node_id| (*node_id, boundary_node_map[node_id]))
        .collect::<BTreeMap<_, _>>();
    node_points.insert(steiner_node_id, centroid);
    let mut node_ids = boundary_node_ids.clone();
    node_ids.push(steiner_node_id);
    let mut candidates = Vec::<ConstrainedCavityRefillTet>::new();
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tet_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    if !tet_faces(tet_node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tet_node_ids.map(|node_id| node_points[&node_id]);
                    if point_in_closed_triangle_surface(
                        tet_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tet) =
                        raw_refill_tet_with_rejection_reason(tet_node_ids, points, options)
                    {
                        candidates.push(tet);
                    }
                }
            }
        }
    }
    diagnostic.candidate_count = candidates.len();
    let face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            candidates
                .iter()
                .filter(|candidate| {
                    tet_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect::<Vec<_>>();
    diagnostic.zero_candidate_boundary_face_count = face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.max_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().max().unwrap_or(0);
    if candidates.is_empty() {
        diagnostic.reason = "no_candidate_tets";
        return Ok(diagnostic);
    }
    if candidates.len() > 512 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let selected = search.search();
    diagnostic.search_attempt_count = search.attempts;
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 5_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.max_min_scaled_jacobian = selected
        .iter()
        .map(|index| candidates[*index].exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    diagnostic.selected_tet_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_patch_steiner_exact_cover(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryPatchSteinerExactCoverDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut diagnostic = BoundaryPatchSteinerExactCoverDiagnostic {
        boundary_node_count: boundary_node_ids.len(),
        boundary_face_count: cavity.boundary_faces.len(),
        missing_face_count: 0,
        patch_count: 0,
        steiner_node_count: 0,
        candidate_count: 0,
        zero_candidate_boundary_face_count: 0,
        min_boundary_face_candidate_count: 0,
        max_boundary_face_candidate_count: 0,
        selected_tet_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        max_min_scaled_jacobian: 0.0,
    };
    if boundary_node_ids.len() < 4 {
        diagnostic.reason = "too_few_boundary_nodes";
        return Ok(diagnostic);
    }

    let points = boundary_node_ids
        .iter()
        .map(|node_id| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: boundary_node_map[node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut boundary_refill_tets = Vec::<ConstrainedCavityRefillTet>::new();
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
        if let Ok(tet) = raw_refill_tet_with_rejection_reason(node_ids, tet_points, options) {
            boundary_refill_tets.push(tet);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &boundary_refill_tets)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    diagnostic.missing_face_count = missing_faces.len();
    if missing_faces.is_empty() {
        diagnostic.reason = "no_missing_faces";
        return Ok(diagnostic);
    }
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        diagnostic.reason = "empty_boundary";
        return Ok(diagnostic);
    };

    let components = missing_face_components(&missing_faces, MissingFaceLink::Node);
    diagnostic.patch_count = components.len();
    let mut node_points = boundary_node_ids
        .iter()
        .map(|node_id| (*node_id, boundary_node_map[node_id]))
        .collect::<BTreeMap<_, _>>();
    let mut node_ids = boundary_node_ids.clone();
    let mut next_node_id = next_cavity_node_id(cavity);
    for component in components {
        let mut patch_node_ids = BTreeSet::<u32>::new();
        for face_index in component {
            patch_node_ids.extend(missing_faces[face_index]);
        }
        let Some(surface_point) = centroid_of_node_set(&patch_node_ids, &boundary_node_map) else {
            continue;
        };
        let Some(point) =
            patch_steiner_point_inside_cavity(surface_point, cavity_centroid, &boundary_triangles)
        else {
            continue;
        };
        while node_points.contains_key(&next_node_id) {
            next_node_id = next_node_id.saturating_add(1);
        }
        node_points.insert(next_node_id, point);
        node_ids.push(next_node_id);
        diagnostic.steiner_node_count += 1;
        next_node_id = next_node_id.saturating_add(1);
    }
    if diagnostic.steiner_node_count == 0 {
        diagnostic.reason = "no_valid_patch_steiner_points";
        return Ok(diagnostic);
    }

    let mut candidates = Vec::<ConstrainedCavityRefillTet>::new();
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tet_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    let points = tet_node_ids.map(|node_id| node_points[&node_id]);
                    if point_in_closed_triangle_surface(
                        tet_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tet) =
                        raw_refill_tet_with_rejection_reason(tet_node_ids, points, options)
                    {
                        candidates.push(tet);
                    }
                }
            }
        }
    }
    diagnostic.candidate_count = candidates.len();
    let face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            candidates
                .iter()
                .filter(|candidate| {
                    tet_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect::<Vec<_>>();
    diagnostic.zero_candidate_boundary_face_count = face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.max_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().max().unwrap_or(0);
    if candidates.is_empty() {
        diagnostic.reason = "no_candidate_tets";
        return Ok(diagnostic);
    }
    if candidates.len() > 1_024 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let selected = search.search();
    diagnostic.search_attempt_count = search.attempts;
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 5_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.max_min_scaled_jacobian = selected
        .iter()
        .map(|index| candidates[*index].exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    diagnostic.selected_tet_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_local_cap_quality(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<MissingFaceLocalCapQualityDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let points = boundary_node_ids
        .iter()
        .map(|node_id| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: boundary_node_map[node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut boundary_refill_tets = Vec::<ConstrainedCavityRefillTet>::new();
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
        if let Ok(tet) = raw_refill_tet_with_rejection_reason(node_ids, tet_points, options) {
            boundary_refill_tets.push(tet);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &boundary_refill_tets)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    let mut diagnostic = MissingFaceLocalCapQualityDiagnostic {
        missing_face_count: missing_faces.len(),
        pass_face_count: 0,
        candidate_count: 0,
        max_scaled_jacobian: 0.0,
        rejected_by_reason: BTreeMap::new(),
    };
    if missing_faces.is_empty() {
        return Ok(diagnostic);
    }
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        return Ok(diagnostic);
    };
    let mut next_node_id = next_cavity_node_id(cavity);
    for face in missing_faces {
        let Some(surface_point) = face_centroid(face, &boundary_node_map) else {
            continue;
        };
        let mut face_passed = false;
        for apex in
            local_cap_apex_candidates(face, surface_point, cavity_centroid, &boundary_node_map)
        {
            let tet_points = [
                boundary_node_map[&face[0]],
                boundary_node_map[&face[1]],
                boundary_node_map[&face[2]],
                apex,
            ];
            if point_in_closed_triangle_surface(
                tet_centroid(tet_points),
                &boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
            {
                *diagnostic
                    .rejected_by_reason
                    .entry("cap_centroid_outside_cavity")
                    .or_default() += 1;
                continue;
            }
            while boundary_node_map.contains_key(&next_node_id) {
                next_node_id = next_node_id.saturating_add(1);
            }
            diagnostic.candidate_count += 1;
            match raw_refill_tet_with_rejection_reason(
                [face[0], face[1], face[2], next_node_id],
                tet_points,
                options,
            ) {
                Ok(tet) => {
                    diagnostic.max_scaled_jacobian = diagnostic
                        .max_scaled_jacobian
                        .max(tet.exact_scaled_jacobian);
                    face_passed = true;
                }
                Err(reason) => {
                    *diagnostic.rejected_by_reason.entry(reason).or_default() += 1;
                }
            }
            next_node_id = next_node_id.saturating_add(1);
        }
        diagnostic.pass_face_count += usize::from(face_passed);
    }
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_local_cap_stitch(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<MissingFaceLocalCapStitchDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let points = boundary_node_ids
        .iter()
        .map(|node_id| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: boundary_node_map[node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut boundary_refill_tets = Vec::<ConstrainedCavityRefillTet>::new();
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
        if let Ok(tet) = raw_refill_tet_with_rejection_reason(node_ids, tet_points, options) {
            boundary_refill_tets.push(tet);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &boundary_refill_tets)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    let mut diagnostic = MissingFaceLocalCapStitchDiagnostic {
        missing_face_count: missing_faces.len(),
        capped_face_count: 0,
        inserted_node_count: 0,
        side_connector_candidate_count: 0,
        candidate_tet_count: 0,
        cap_side_face_count: 0,
        zero_mate_cap_side_face_count: 0,
        min_cap_side_face_mate_count: 0,
        max_cap_side_face_mate_count: 0,
        open_interior_face_count: 0,
        open_interior_component_count: 0,
        open_interior_component_size_histogram: BTreeMap::new(),
        selected_tet_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        max_min_scaled_jacobian: 0.0,
    };
    if missing_faces.is_empty() {
        diagnostic.reason = "no_missing_faces";
        return Ok(diagnostic);
    }
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        diagnostic.reason = "empty_boundary";
        return Ok(diagnostic);
    };

    let mut node_points = boundary_node_ids
        .iter()
        .map(|node_id| (*node_id, boundary_node_map[node_id]))
        .collect::<BTreeMap<_, _>>();
    let mut candidate_tets = boundary_refill_tets;
    let mut inserted_nodes = Vec::<ConstrainedCavityNode>::new();
    let mut next_node_id = next_cavity_node_id(cavity);
    let cap_tet_start = candidate_tets.len();
    for face in &missing_faces {
        let Some(surface_point) = face_centroid(*face, &boundary_node_map) else {
            continue;
        };
        let Some((coordinates_m, cap_tet)) = best_local_cap_for_face(
            *face,
            surface_point,
            cavity_centroid,
            next_node_id,
            &boundary_node_map,
            &boundary_triangles,
            options,
        ) else {
            continue;
        };
        while node_points.contains_key(&next_node_id) {
            next_node_id = next_node_id.saturating_add(1);
        }
        node_points.insert(next_node_id, coordinates_m);
        inserted_nodes.push(ConstrainedCavityNode {
            node_id: next_node_id,
            coordinates_m,
        });
        candidate_tets.push(cap_tet);
        diagnostic.capped_face_count += 1;
        next_node_id = next_node_id.saturating_add(1);
    }
    diagnostic.inserted_node_count = inserted_nodes.len();
    if diagnostic.capped_face_count < diagnostic.missing_face_count {
        diagnostic.reason = "incomplete_local_caps";
        diagnostic.candidate_tet_count = candidate_tets.len();
        return Ok(diagnostic);
    }
    let cap_tet_count = candidate_tets.len() - cap_tet_start;

    let connector_points = node_points
        .iter()
        .map(|(node_id, coordinates_m)| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: *coordinates_m,
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut seen_tets = candidate_tets
        .iter()
        .map(|tet| sorted_tet_nodes(tet.node_ids))
        .collect::<BTreeSet<_>>();
    for tet in tetrahedralize_points(&connector_points) {
        let node_ids = tet.vertices.map(|index| connector_points[index].node_id);
        if !seen_tets.insert(sorted_tet_nodes(node_ids)) {
            continue;
        }
        let tet_points = tet
            .vertices
            .map(|index| connector_points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tet_centroid(tet_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tet) = raw_refill_tet_with_rejection_reason(node_ids, tet_points, options) {
            candidate_tets.push(tet);
        }
    }
    diagnostic.side_connector_candidate_count = append_cap_side_connector_tets(
        cap_tet_start,
        cap_tet_count,
        &mut candidate_tets,
        &mut seen_tets,
        &node_points,
        &inserted_nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<BTreeSet<_>>(),
        &boundary_triangles,
        options,
    );
    diagnostic.candidate_tet_count = candidate_tets.len();
    let cap_side_mate_counts = cap_side_face_mate_counts(
        &candidate_tets[cap_tet_start..cap_tet_start + cap_tet_count],
        &candidate_tets,
        &inserted_nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<BTreeSet<_>>(),
    );
    diagnostic.cap_side_face_count = cap_side_mate_counts.len();
    diagnostic.zero_mate_cap_side_face_count = cap_side_mate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_cap_side_face_mate_count =
        cap_side_mate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.max_cap_side_face_mate_count =
        cap_side_mate_counts.iter().copied().max().unwrap_or(0);
    let open_interior_faces = open_interior_refill_faces(cavity, &candidate_tets);
    diagnostic.open_interior_face_count = open_interior_faces.len();
    diagnostic.open_interior_component_count =
        missing_face_components(&open_interior_faces, MissingFaceLink::Node).len();
    diagnostic.open_interior_component_size_histogram = component_size_histogram(
        missing_face_component_sizes(&open_interior_faces, MissingFaceLink::Node),
    );
    if candidate_tets.is_empty() {
        diagnostic.reason = "no_candidate_tets";
        return Ok(diagnostic);
    }
    if candidate_tets.len() > 4_096 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search = BoundaryExactCoverSearch::with_attempt_limit(
        cavity,
        &candidate_tets,
        options.volume_relative_tolerance,
        25_000,
    );
    let selected = search.search();
    diagnostic.search_attempt_count = search.attempts;
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 25_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.max_min_scaled_jacobian = selected
        .iter()
        .map(|index| candidate_tets[*index].exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    diagnostic.selected_tet_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_missing_face_clusters(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryMissingFaceClusterDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let points = cavity_boundary_node_ids(cavity)
        .into_iter()
        .map(|node_id| ConnectivityPoint {
            node_id,
            coordinates_m: boundary_node_map[&node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut refill_tets = Vec::<ConstrainedCavityRefillTet>::new();
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
        if let Ok(tet) = raw_refill_tet_with_rejection_reason(node_ids, tet_points, options) {
            refill_tets.push(tet);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &refill_tets)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    let edge_component_sizes = missing_face_component_sizes(&missing_faces, MissingFaceLink::Edge);
    let node_components = missing_face_components(&missing_faces, MissingFaceLink::Node);
    let node_component_sizes = node_components.iter().map(Vec::len).collect::<Vec<_>>();
    let mut node_component_common_node_count_histogram = BTreeMap::<usize, usize>::new();
    let mut node_component_common_node_ids = BTreeMap::<u32, usize>::new();
    for component in &node_components {
        let common_node_ids = missing_face_component_common_node_ids(&missing_faces, component);
        *node_component_common_node_count_histogram
            .entry(common_node_ids.len())
            .or_default() += 1;
        for node_id in common_node_ids {
            *node_component_common_node_ids.entry(node_id).or_default() += 1;
        }
    }
    Ok(BoundaryMissingFaceClusterDiagnostic {
        missing_face_count: missing_faces.len(),
        edge_component_count: edge_component_sizes.len(),
        edge_component_size_histogram: component_size_histogram(edge_component_sizes),
        node_component_count: node_component_sizes.len(),
        node_component_size_histogram: component_size_histogram(node_component_sizes),
        node_component_common_node_count_histogram,
        node_component_common_node_ids,
    })
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MissingFaceLink {
    Edge,
    Node,
}

#[cfg(test)]
fn missing_face_component_sizes(faces: &[[u32; 3]], link: MissingFaceLink) -> Vec<usize> {
    missing_face_components(faces, link)
        .into_iter()
        .map(|component| component.len())
        .collect()
}

#[cfg(test)]
fn missing_face_components(faces: &[[u32; 3]], link: MissingFaceLink) -> Vec<Vec<usize>> {
    let mut visited = BTreeSet::<usize>::new();
    let mut components = Vec::<Vec<usize>>::new();
    for start in 0..faces.len() {
        if !visited.insert(start) {
            continue;
        }
        let mut component = Vec::<usize>::new();
        let mut pending = vec![start];
        while let Some(index) = pending.pop() {
            component.push(index);
            for neighbor in 0..faces.len() {
                if visited.contains(&neighbor)
                    || !missing_faces_connected(faces[index], faces[neighbor], link)
                {
                    continue;
                }
                visited.insert(neighbor);
                pending.push(neighbor);
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components.sort();
    components
}

#[cfg(test)]
fn missing_face_component_common_node_ids(faces: &[[u32; 3]], component: &[usize]) -> Vec<u32> {
    let Some(first) = component.first() else {
        return Vec::new();
    };
    let mut common = faces[*first].into_iter().collect::<BTreeSet<_>>();
    for index in component.iter().skip(1) {
        let face_nodes = faces[*index].into_iter().collect::<BTreeSet<_>>();
        common.retain(|node_id| face_nodes.contains(node_id));
    }
    common.into_iter().collect()
}

#[cfg(test)]
fn centroid_of_node_set(
    node_ids: &BTreeSet<u32>,
    node_coordinates: &BTreeMap<u32, [f64; 3]>,
) -> Option<[f64; 3]> {
    if node_ids.is_empty() {
        return None;
    }
    let mut centroid = [0.0; 3];
    for node_id in node_ids {
        let point = node_coordinates.get(node_id)?;
        centroid[0] += point[0];
        centroid[1] += point[1];
        centroid[2] += point[2];
    }
    let scale = 1.0 / node_ids.len() as f64;
    Some([
        centroid[0] * scale,
        centroid[1] * scale,
        centroid[2] * scale,
    ])
}

#[cfg(test)]
fn face_centroid(face: [u32; 3], node_coordinates: &BTreeMap<u32, [f64; 3]>) -> Option<[f64; 3]> {
    let first = node_coordinates.get(&face[0]).copied()?;
    let second = node_coordinates.get(&face[1]).copied()?;
    let third = node_coordinates.get(&face[2]).copied()?;
    Some([
        (first[0] + second[0] + third[0]) / 3.0,
        (first[1] + second[1] + third[1]) / 3.0,
        (first[2] + second[2] + third[2]) / 3.0,
    ])
}

#[cfg(test)]
fn best_local_cap_for_face(
    face: [u32; 3],
    surface_point: [f64; 3],
    cavity_centroid: [f64; 3],
    apex_node_id: u32,
    node_coordinates: &BTreeMap<u32, [f64; 3]>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Option<([f64; 3], ConstrainedCavityRefillTet)> {
    local_cap_apex_candidates(face, surface_point, cavity_centroid, node_coordinates)
        .into_iter()
        .filter_map(|apex| {
            let tet_points = [
                node_coordinates[&face[0]],
                node_coordinates[&face[1]],
                node_coordinates[&face[2]],
                apex,
            ];
            if point_in_closed_triangle_surface(
                tet_centroid(tet_points),
                boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
            {
                return None;
            }
            let tet = raw_refill_tet_with_rejection_reason(
                [face[0], face[1], face[2], apex_node_id],
                tet_points,
                options,
            )
            .ok()?;
            Some((apex, tet))
        })
        .max_by(|left, right| {
            left.1
                .exact_scaled_jacobian
                .total_cmp(&right.1.exact_scaled_jacobian)
                .then_with(|| right.1.aspect_ratio.total_cmp(&left.1.aspect_ratio))
        })
}

#[cfg(test)]
fn append_cap_side_connector_tets(
    cap_tet_start: usize,
    cap_tet_count: usize,
    candidate_tets: &mut Vec<ConstrainedCavityRefillTet>,
    seen_tets: &mut BTreeSet<[u32; 4]>,
    node_points: &BTreeMap<u32, [f64; 3]>,
    inserted_node_ids: &BTreeSet<u32>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> usize {
    let cap_tets = candidate_tets
        .iter()
        .skip(cap_tet_start)
        .take(cap_tet_count)
        .cloned()
        .collect::<Vec<_>>();
    let mut inserted_count = 0_usize;
    for cap_tet in cap_tets {
        for face in tet_faces(cap_tet.node_ids) {
            if !face
                .iter()
                .any(|node_id| inserted_node_ids.contains(node_id))
            {
                continue;
            }
            for node_id in node_points.keys().copied() {
                if face.contains(&node_id) {
                    continue;
                }
                let tet_node_ids = [face[0], face[1], face[2], node_id];
                if !seen_tets.insert(sorted_tet_nodes(tet_node_ids)) {
                    continue;
                }
                let tet_points = tet_node_ids.map(|id| node_points[&id]);
                if point_in_closed_triangle_surface(
                    tet_centroid(tet_points),
                    boundary_triangles,
                    MeshingTolerance::default(),
                ) != PointInClosedSurface::Inside
                {
                    continue;
                }
                let Ok(tet) =
                    raw_refill_tet_with_rejection_reason(tet_node_ids, tet_points, options)
                else {
                    continue;
                };
                candidate_tets.push(tet);
                inserted_count += 1;
            }
        }
    }
    inserted_count
}

#[cfg(test)]
fn local_cap_apex_candidates(
    face: [u32; 3],
    surface_point: [f64; 3],
    cavity_centroid: [f64; 3],
    node_coordinates: &BTreeMap<u32, [f64; 3]>,
) -> Vec<[f64; 3]> {
    let mut candidates = Vec::<[f64; 3]>::new();
    for fraction in [0.03, 0.06, 0.1, 0.16, 0.25, 0.38, 0.55, 0.75] {
        candidates.push([
            surface_point[0] + (cavity_centroid[0] - surface_point[0]) * fraction,
            surface_point[1] + (cavity_centroid[1] - surface_point[1]) * fraction,
            surface_point[2] + (cavity_centroid[2] - surface_point[2]) * fraction,
        ]);
    }

    let Some(first) = node_coordinates.get(&face[0]).copied() else {
        return candidates;
    };
    let Some(second) = node_coordinates.get(&face[1]).copied() else {
        return candidates;
    };
    let Some(third) = node_coordinates.get(&face[2]).copied() else {
        return candidates;
    };
    let first_edge = [
        second[0] - first[0],
        second[1] - first[1],
        second[2] - first[2],
    ];
    let second_edge = [
        third[0] - first[0],
        third[1] - first[1],
        third[2] - first[2],
    ];
    let normal = cross(first_edge, second_edge);
    let Some(unit_normal) = normalize(normal) else {
        return candidates;
    };
    let max_edge_length = distance(first, second)
        .max(distance(second, third))
        .max(distance(third, first));
    if !max_edge_length.is_finite() || max_edge_length <= 0.0 {
        return candidates;
    }
    for direction in [
        unit_normal,
        [-unit_normal[0], -unit_normal[1], -unit_normal[2]],
    ] {
        for scale in [0.08, 0.14, 0.22, 0.35, 0.55, 0.85, 1.25] {
            let distance = max_edge_length * scale;
            candidates.push([
                surface_point[0] + direction[0] * distance,
                surface_point[1] + direction[1] * distance,
                surface_point[2] + direction[2] * distance,
            ]);
        }
    }
    candidates
}

#[cfg(test)]
fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

#[cfg(test)]
fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let norm = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return None;
    }
    Some([vector[0] / norm, vector[1] / norm, vector[2] / norm])
}

#[cfg(test)]
fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    let delta = [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
    (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt()
}

#[cfg(test)]
fn patch_steiner_point_inside_cavity(
    surface_point: [f64; 3],
    cavity_centroid: [f64; 3],
    boundary_triangles: &[Triangle3],
) -> Option<[f64; 3]> {
    if point_in_closed_triangle_surface(
        surface_point,
        boundary_triangles,
        MeshingTolerance::default(),
    ) == PointInClosedSurface::Inside
    {
        return Some(surface_point);
    }
    [0.05, 0.1, 0.2, 0.35, 0.5]
        .into_iter()
        .map(|fraction| {
            [
                surface_point[0] + (cavity_centroid[0] - surface_point[0]) * fraction,
                surface_point[1] + (cavity_centroid[1] - surface_point[1]) * fraction,
                surface_point[2] + (cavity_centroid[2] - surface_point[2]) * fraction,
            ]
        })
        .find(|point| {
            point_in_closed_triangle_surface(
                *point,
                boundary_triangles,
                MeshingTolerance::default(),
            ) == PointInClosedSurface::Inside
        })
}

#[cfg(test)]
fn missing_faces_connected(left: [u32; 3], right: [u32; 3], link: MissingFaceLink) -> bool {
    if left == right {
        return true;
    }
    let shared_count = left
        .into_iter()
        .filter(|node_id| right.contains(node_id))
        .count();
    match link {
        MissingFaceLink::Edge => shared_count >= 2,
        MissingFaceLink::Node => shared_count >= 1,
    }
}

#[cfg(test)]
fn component_size_histogram(sizes: Vec<usize>) -> BTreeMap<usize, usize> {
    let mut histogram = BTreeMap::<usize, usize>::new();
    for size in sizes {
        *histogram.entry(size).or_default() += 1;
    }
    histogram
}

fn refill_boundary_face_delta(
    cavity: &ConstrainedCavity,
    refill_tets: &[ConstrainedCavityRefillTet],
) -> Result<RefillBoundaryFaceDelta, ConstrainedCavityValidationError> {
    let expected = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let actual = boundary_faces_from_refill_tets(cavity, refill_tets)?
        .into_iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    Ok(RefillBoundaryFaceDelta {
        missing: expected.difference(&actual).copied().collect(),
        unexpected: actual.difference(&expected).copied().collect(),
    })
}

fn best_boundary_face_completion_tet(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    refill_tets: &[ConstrainedCavityRefillTet],
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
            let tet = raw_refill_tet_with_rejection_reason(node_ids, points, options).ok()?;
            if refill_tets.iter().any(|existing| {
                sorted_tet_nodes(existing.node_ids) == sorted_tet_nodes(tet.node_ids)
            }) {
                return None;
            }
            Some(tet)
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
        inserted_nodes: Vec::new(),
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
    fn boundary_face_edge_split_preserves_source_face_regions_and_split_edge_provenance() {
        let face = face_with_provenance(
            [0, 1, 2],
            10,
            [Some(100), Some(101), Some(102)],
            &["fixed", "loaded"],
        );

        let children = split_constrained_cavity_boundary_face_on_edge(&face, [0, 1], 9)
            .expect("face edge should split");

        assert_eq!(children[0].node_ids, [0, 9, 2]);
        assert_eq!(children[1].node_ids, [9, 1, 2]);
        assert_eq!(children[0].source_edge_ids, [Some(100), None, Some(102)]);
        assert_eq!(children[1].source_edge_ids, [Some(100), Some(101), None]);
        for child in &children {
            assert_eq!(child.source_face_id, Some(10));
            assert_eq!(
                sorted_region_ids(&child.region_ids),
                vec!["fixed".to_string(), "loaded".to_string()]
            );
        }
    }

    #[test]
    fn boundary_face_edge_split_list_replaces_conforming_edge_pair() {
        let cavity = provenance_cavity();

        let split_faces = split_constrained_cavity_boundary_faces_on_edge(
            &cavity.boundary_faces,
            [2, 1, 0],
            [1, 0],
            9,
        )
        .expect("target face edge should split");

        assert_eq!(split_faces.len(), cavity.boundary_faces.len() + 2);
        assert!(!split_faces
            .iter()
            .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
        assert!(!split_faces
            .iter()
            .any(|face| sorted_face(face.node_ids) == [0, 1, 3]));
        assert_eq!(
            split_faces
                .iter()
                .filter(|face| face.node_ids.contains(&9))
                .count(),
            4
        );
        for untouched in cavity.boundary_faces.iter().skip(2) {
            assert!(split_faces
                .iter()
                .any(|face| sorted_face(face.node_ids) == sorted_face(untouched.node_ids)));
        }
    }

    #[test]
    fn boundary_face_three_edge_split_refines_target_and_conforming_neighbors() {
        let cavity = provenance_cavity();
        let split_faces = split_constrained_cavity_boundary_faces_on_three_edges(
            &cavity.boundary_faces,
            [2, 1, 0],
            BTreeMap::from([([0, 1], 9), ([1, 2], 10), ([0, 2], 11)]),
        )
        .expect("target face edges should split");

        assert!(!split_faces
            .iter()
            .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
        assert!(!split_faces
            .iter()
            .any(|face| sorted_face(face.node_ids) == [0, 1, 3]));
        assert_eq!(
            split_faces
                .iter()
                .filter(|face| face.node_ids.contains(&9)
                    || face.node_ids.contains(&10)
                    || face.node_ids.contains(&11))
                .count(),
            10
        );
        let target_children = split_faces
            .iter()
            .filter(|face| {
                [9, 10, 11]
                    .into_iter()
                    .any(|node_id| face.node_ids.contains(&node_id))
                    && face.source_face_id == Some(10)
            })
            .collect::<Vec<_>>();
        assert_eq!(target_children.len(), 4);
        assert_eq!(source_edge_for(target_children[0], [0, 9]), Some(100));
        assert_eq!(source_edge_for(target_children[1], [1, 9]), Some(100));
        assert_eq!(source_edge_for(target_children[1], [1, 10]), Some(101));
        assert_eq!(source_edge_for(target_children[2], [2, 10]), Some(101));
        assert_eq!(source_edge_for(target_children[2], [2, 11]), Some(102));
        assert_eq!(source_edge_for(target_children[0], [0, 11]), Some(102));
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

        let (_, completed, inserted_nodes) = complete_missing_boundary_face_tets(
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

        assert!(inserted_nodes.is_empty());
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
    fn boundary_node_completion_reports_when_no_cap_tet_passes_quality() {
        let cavity = two_tet_bipyramid_cavity();
        let nodes = two_tet_bipyramid_nodes();
        let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
            .expect("fixture nodes should cover cavity boundary");
        let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
            .expect("fixture boundary should build triangles");
        let initial_options = refill_options();
        let points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
        let incomplete_tet =
            raw_refill_tet_with_rejection_reason([0, 1, 2, 3], points, initial_options)
                .expect("fixture tet should pass initial quality gates");

        let rejected = complete_missing_boundary_face_tets(
            &cavity,
            &boundary_nodes,
            vec![incomplete_tet],
            &boundary_triangles,
            ConstrainedCavityRefillOptions {
                min_scaled_jacobian: 0.95,
                ..initial_options
            },
        )
        .expect("completion should evaluate")
        .expect_err("strict quality should reject every cap tet");

        assert_eq!(rejected, "boundary_node_completion_no_candidate");
    }

    #[test]
    fn boundary_node_exact_cover_recovers_bipyramid_cavity() {
        let cavity = two_tet_bipyramid_cavity();
        let nodes = two_tet_bipyramid_nodes();
        let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
            .expect("fixture nodes should cover cavity boundary");
        let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
            .expect("fixture boundary should build triangles");
        let options = refill_options();

        let refill = boundary_node_exact_cover_refill_candidate(
            &cavity,
            &boundary_nodes,
            &boundary_triangles,
            options,
        )
        .expect("exact cover should evaluate")
        .expect("exact cover should recover the cavity");

        assert_eq!(refill.tets.len(), 2);
        validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
            .expect("exact cover should preserve boundary");
        validate_constrained_cavity_refill_volume(
            cavity.target_volume_m3,
            refill.total_volume_m3,
            options.volume_relative_tolerance,
        )
        .expect("exact cover should preserve volume");
    }

    #[test]
    fn boundary_exact_cover_diagnostic_reports_relaxed_cover_feasibility() {
        let cavity = two_tet_bipyramid_cavity();
        let nodes = two_tet_bipyramid_nodes();
        let diagnostic = diagnostic_boundary_exact_cover(&cavity, &nodes, refill_options())
            .expect("diagnostic should evaluate");

        assert_eq!(diagnostic.boundary_node_count, 5);
        assert_eq!(diagnostic.boundary_face_count, 6);
        assert!(diagnostic.candidate_count > 0);
        assert_eq!(diagnostic.zero_candidate_boundary_face_count, 0);
        assert!(diagnostic.min_boundary_face_candidate_count > 0);
        assert!(
            diagnostic.max_boundary_face_candidate_count
                >= diagnostic.min_boundary_face_candidate_count
        );
        assert!(diagnostic.search_attempt_count > 0);
        assert!(diagnostic.found_cover);
        assert_eq!(diagnostic.reason, "cover_found");
        assert_eq!(diagnostic.selected_tet_count, 2);
    }

    #[test]
    fn exact_cover_search_targets_unpaired_interior_faces_after_boundary_faces() {
        let cavity = ConstrainedCavity {
            removed_tet_ids: vec![0],
            boundary_faces: vec![
                ConstrainedCavityBoundaryFace {
                    node_ids: [0, 1, 2],
                    source_face_id: None,
                    source_edge_ids: [None, None, None],
                    region_ids: Vec::new(),
                },
                ConstrainedCavityBoundaryFace {
                    node_ids: [3, 4, 5],
                    source_face_id: None,
                    source_edge_ids: [None, None, None],
                    region_ids: Vec::new(),
                },
            ],
            protected_node_ids: Vec::new(),
            target_volume_m3: 1.0,
        };
        let candidates = vec![
            ConstrainedCavityRefillTet {
                node_ids: [0, 1, 2, 6],
                volume_m3: 0.2,
                aspect_ratio: 1.0,
                exact_scaled_jacobian: 0.4,
            },
            ConstrainedCavityRefillTet {
                node_ids: [3, 4, 5, 7],
                volume_m3: 0.2,
                aspect_ratio: 1.0,
                exact_scaled_jacobian: 0.4,
            },
            ConstrainedCavityRefillTet {
                node_ids: [0, 1, 6, 8],
                volume_m3: 0.2,
                aspect_ratio: 1.0,
                exact_scaled_jacobian: 0.3,
            },
        ];
        let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);
        let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
        for face in [[0, 1, 2], [3, 4, 5], sorted_face([0, 1, 6])] {
            face_counts.insert(face, 1);
        }

        let candidates = search
            .next_cover_candidates(&face_counts, &[0, 1])
            .expect("unpaired interior face should request connector candidates");

        assert_eq!(candidates, vec![2]);
    }

    #[test]
    fn exact_cover_search_uses_configured_attempt_limit() {
        let cavity = two_tet_bipyramid_cavity();
        let candidates = vec![
            ConstrainedCavityRefillTet {
                node_ids: [0, 1, 2, 3],
                volume_m3: 1.0 / 6.0,
                aspect_ratio: 1.0,
                exact_scaled_jacobian: 0.4,
            },
            ConstrainedCavityRefillTet {
                node_ids: [0, 2, 1, 4],
                volume_m3: 1.0 / 6.0,
                aspect_ratio: 1.0,
                exact_scaled_jacobian: 0.4,
            },
        ];
        let mut low_limit_search =
            BoundaryExactCoverSearch::with_attempt_limit(&cavity, &candidates, 1.0e-9, 2);

        assert!(low_limit_search.search().is_none());
        assert!(low_limit_search.attempts > 2);

        let mut sufficient_limit_search =
            BoundaryExactCoverSearch::with_attempt_limit(&cavity, &candidates, 1.0e-9, 3);

        assert_eq!(sufficient_limit_search.search(), Some(vec![0, 1]));
        assert_eq!(sufficient_limit_search.attempts, 3);
    }

    #[test]
    fn boundary_steiner_exact_cover_diagnostic_reports_centroid_candidate_coverage() {
        let mut cavity = unit_tet_cavity();
        let split_specs = [
            ([0, 2, 1], 4),
            ([0, 1, 3], 5),
            ([1, 2, 3], 6),
            ([2, 0, 3], 7),
        ];
        for (face, split_node_id) in split_specs {
            cavity.boundary_faces = split_constrained_cavity_boundary_faces(
                &cavity.boundary_faces,
                face,
                split_node_id,
            )
            .expect("fixture face should split");
        }
        let mut nodes = unit_tet_nodes();
        nodes.extend([
            ConstrainedCavityNode {
                node_id: 4,
                coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 5,
                coordinates_m: [1.0 / 3.0, 0.0, 1.0 / 3.0],
            },
            ConstrainedCavityNode {
                node_id: 6,
                coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            },
            ConstrainedCavityNode {
                node_id: 7,
                coordinates_m: [0.0, 1.0 / 3.0, 1.0 / 3.0],
            },
        ]);

        let diagnostic = diagnostic_boundary_steiner_exact_cover(&cavity, &nodes, refill_options())
            .expect("Steiner exact-cover diagnostic should evaluate");

        assert!(diagnostic.candidate_count > 0);
        assert_eq!(diagnostic.zero_candidate_boundary_face_count, 0);
        assert!(diagnostic.search_attempt_count > 0);
        assert_eq!(diagnostic.reason, "cover_not_found");
        assert_eq!(diagnostic.selected_tet_count, 0);
    }

    #[test]
    fn boundary_patch_steiner_exact_cover_diagnostic_reports_boundary_complete_fixture() {
        let mut cavity = unit_tet_cavity();
        let split_specs = [
            ([0, 2, 1], 4),
            ([0, 1, 3], 5),
            ([1, 2, 3], 6),
            ([2, 0, 3], 7),
        ];
        for (face, split_node_id) in split_specs {
            cavity.boundary_faces = split_constrained_cavity_boundary_faces(
                &cavity.boundary_faces,
                face,
                split_node_id,
            )
            .expect("fixture face should split");
        }
        let mut nodes = unit_tet_nodes();
        nodes.extend([
            ConstrainedCavityNode {
                node_id: 4,
                coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 5,
                coordinates_m: [1.0 / 3.0, 0.0, 1.0 / 3.0],
            },
            ConstrainedCavityNode {
                node_id: 6,
                coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            },
            ConstrainedCavityNode {
                node_id: 7,
                coordinates_m: [0.0, 1.0 / 3.0, 1.0 / 3.0],
            },
        ]);

        let diagnostic =
            diagnostic_boundary_patch_steiner_exact_cover(&cavity, &nodes, refill_options())
                .expect("patch Steiner exact-cover diagnostic should evaluate");

        assert_eq!(diagnostic.boundary_node_count, 8);
        assert_eq!(diagnostic.boundary_face_count, 12);
        assert_eq!(diagnostic.missing_face_count, 0);
        assert_eq!(diagnostic.patch_count, 0);
        assert_eq!(diagnostic.steiner_node_count, 0);
        assert_eq!(diagnostic.candidate_count, 0);
        assert_eq!(diagnostic.search_attempt_count, 0);
        assert_eq!(diagnostic.reason, "no_missing_faces");
    }

    #[test]
    fn missing_face_local_cap_quality_reports_boundary_complete_fixture() {
        let cavity = two_tet_bipyramid_cavity();
        let nodes = two_tet_bipyramid_nodes();
        let diagnostic =
            diagnostic_missing_face_local_cap_quality(&cavity, &nodes, refill_options())
                .expect("local cap diagnostic should evaluate");

        assert_eq!(diagnostic.missing_face_count, 0);
        assert_eq!(diagnostic.pass_face_count, 0);
        assert_eq!(diagnostic.candidate_count, 0);
        assert_eq!(diagnostic.max_scaled_jacobian, 0.0);
        assert!(diagnostic.rejected_by_reason.is_empty());
    }

    #[test]
    fn missing_face_local_cap_stitch_reports_boundary_complete_fixture() {
        let cavity = two_tet_bipyramid_cavity();
        let nodes = two_tet_bipyramid_nodes();
        let diagnostic =
            diagnostic_missing_face_local_cap_stitch(&cavity, &nodes, refill_options())
                .expect("local cap stitch diagnostic should evaluate");

        assert_eq!(diagnostic.missing_face_count, 0);
        assert_eq!(diagnostic.capped_face_count, 0);
        assert_eq!(diagnostic.inserted_node_count, 0);
        assert_eq!(diagnostic.side_connector_candidate_count, 0);
        assert_eq!(diagnostic.candidate_tet_count, 0);
        assert_eq!(diagnostic.cap_side_face_count, 0);
        assert_eq!(diagnostic.zero_mate_cap_side_face_count, 0);
        assert_eq!(diagnostic.min_cap_side_face_mate_count, 0);
        assert_eq!(diagnostic.max_cap_side_face_mate_count, 0);
        assert_eq!(diagnostic.open_interior_face_count, 0);
        assert_eq!(diagnostic.open_interior_component_count, 0);
        assert!(diagnostic.open_interior_component_size_histogram.is_empty());
        assert_eq!(diagnostic.selected_tet_count, 0);
        assert_eq!(diagnostic.search_attempt_count, 0);
        assert!(!diagnostic.found_cover);
        assert_eq!(diagnostic.reason, "no_missing_faces");
    }

    #[test]
    fn missing_face_components_separate_edge_and_node_connected_patches() {
        let faces = [[0, 1, 2], [2, 1, 3], [3, 4, 5], [3, 6, 7]];

        let edge_histogram =
            component_size_histogram(missing_face_component_sizes(&faces, MissingFaceLink::Edge));
        let node_histogram =
            component_size_histogram(missing_face_component_sizes(&faces, MissingFaceLink::Node));
        let node_components = missing_face_components(&faces, MissingFaceLink::Node);
        let common_node_ids =
            missing_face_component_common_node_ids(&faces, node_components.first().unwrap());

        assert_eq!(edge_histogram, BTreeMap::from([(1, 2), (2, 1)]));
        assert_eq!(node_histogram, BTreeMap::from([(4, 1)]));
        assert_eq!(common_node_ids, Vec::<u32>::new());

        let fan_faces = [[9, 1, 2], [9, 2, 3], [9, 3, 4]];
        let fan_components = missing_face_components(&fan_faces, MissingFaceLink::Node);
        assert_eq!(
            missing_face_component_common_node_ids(&fan_faces, fan_components.first().unwrap()),
            vec![9]
        );
    }

    #[test]
    fn open_interior_refill_faces_reports_unpaired_non_boundary_faces() {
        let cavity = two_tet_bipyramid_cavity();
        let lower = ConstrainedCavityRefillTet {
            node_ids: [0, 1, 2, 3],
            volume_m3: 1.0 / 6.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        };
        let upper = ConstrainedCavityRefillTet {
            node_ids: [0, 2, 1, 4],
            volume_m3: 1.0 / 6.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        };

        assert_eq!(
            open_interior_refill_faces(&cavity, &[lower.clone()]),
            vec![[0, 1, 2]]
        );
        assert!(open_interior_refill_faces(&cavity, &[lower, upper]).is_empty());
    }

    #[test]
    fn cap_side_face_mate_counts_report_connector_coverage() {
        let cap_tet = ConstrainedCavityRefillTet {
            node_ids: [0, 1, 2, 4],
            volume_m3: 1.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.5,
        };
        let mate_tet = ConstrainedCavityRefillTet {
            node_ids: [0, 1, 4, 5],
            volume_m3: 1.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.5,
        };

        assert_eq!(
            cap_side_face_mate_counts(
                &[cap_tet.clone()],
                &[cap_tet, mate_tet],
                &BTreeSet::from([4])
            ),
            vec![1, 0, 0]
        );
    }

    #[test]
    fn centroid_interior_refill_candidate_recovers_split_boundary_tet_cavity() {
        let mut cavity = unit_tet_cavity();
        let split_specs = [
            ([0, 2, 1], 4),
            ([0, 1, 3], 5),
            ([1, 2, 3], 6),
            ([2, 0, 3], 7),
        ];
        for (face, split_node_id) in split_specs {
            cavity.boundary_faces = split_constrained_cavity_boundary_faces(
                &cavity.boundary_faces,
                face,
                split_node_id,
            )
            .expect("fixture face should split");
        }
        validate_constrained_cavity(&cavity).expect("split boundary fixture should be valid");
        let mut nodes = unit_tet_nodes();
        nodes.extend([
            ConstrainedCavityNode {
                node_id: 4,
                coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 5,
                coordinates_m: [1.0 / 3.0, 0.0, 1.0 / 3.0],
            },
            ConstrainedCavityNode {
                node_id: 6,
                coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            },
            ConstrainedCavityNode {
                node_id: 7,
                coordinates_m: [0.0, 1.0 / 3.0, 1.0 / 3.0],
            },
        ]);

        let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
            .expect("fixture nodes should cover cavity boundary");
        let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
            .expect("fixture boundary should build triangles");
        let refill = centroid_interior_refill_candidate(
            &cavity,
            &boundary_nodes,
            &boundary_triangles,
            refill_options(),
        )
        .expect("centroid interior refill should evaluate")
        .expect("centroid interior refill should recover the split boundary cavity");

        assert_eq!(refill.inserted_nodes.len(), 1);
        assert_eq!(refill.inserted_nodes[0].node_id, 8);
        assert_eq!(refill.tets.len(), cavity.boundary_faces.len());
        validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
            .expect("centroid interior refill should preserve boundary");
        validate_constrained_cavity_refill_volume(
            cavity.target_volume_m3,
            refill.total_volume_m3,
            refill_options().volume_relative_tolerance,
        )
        .expect("centroid interior refill should preserve volume");
    }

    #[test]
    fn interior_star_quality_diagnostic_bins_candidate_quality() {
        let cavity = two_tet_bipyramid_cavity();
        let nodes = two_tet_bipyramid_nodes();
        let candidates = vec![
            ConstrainedCavityNode {
                node_id: 10,
                coordinates_m: [0.25, 0.25, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 11,
                coordinates_m: [3.0, 3.0, 3.0],
            },
        ];

        let diagnostic = diagnostic_interior_star_quality(
            &cavity,
            &nodes,
            &candidates,
            ConstrainedCavityRefillOptions {
                min_scaled_jacobian: 0.01,
                volume_relative_tolerance: 1.0e-12,
                ..ConstrainedCavityRefillOptions::default()
            },
        )
        .expect("interior star diagnostic should evaluate");

        assert_eq!(diagnostic.candidate_count, 1);
        assert_eq!(diagnostic.pass_count, 1);
        assert!(diagnostic.max_min_scaled_jacobian >= 0.01);
        assert!(!diagnostic.min_scaled_jacobian_bins.is_empty());
        assert_eq!(
            diagnostic.rejected_by_reason,
            BTreeMap::from([("interior_point_outside_cavity", 1)])
        );
    }

    #[test]
    fn two_interior_node_refill_preserves_bipyramid_cavity() {
        let cavity = two_tet_bipyramid_cavity();
        let nodes = two_tet_bipyramid_nodes();
        let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
            .expect("fixture nodes should cover cavity boundary");
        let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
            .expect("fixture boundary should build triangles");
        let interior_candidates = [
            ConstrainedCavityNode {
                node_id: 10,
                coordinates_m: [0.25, 0.25, 0.25],
            },
            ConstrainedCavityNode {
                node_id: 11,
                coordinates_m: [0.25, 0.25, -0.25],
            },
        ];
        let options = refill_options();

        let refill = two_interior_node_refill_candidate(
            &cavity,
            &boundary_nodes,
            &boundary_triangles,
            &interior_candidates,
            options,
        )
        .expect("two-interior refill should evaluate")
        .expect("two-interior refill should recover the cavity");

        assert_eq!(refill.inserted_nodes, interior_candidates);
        validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
            .expect("two-interior refill should preserve boundary");
        validate_constrained_cavity_refill_volume(
            cavity.target_volume_m3,
            refill.total_volume_m3,
            options.volume_relative_tolerance,
        )
        .expect("two-interior refill should preserve volume");
    }

    #[test]
    fn boundary_face_completion_skips_duplicate_cap_tets() {
        let cavity = unit_tet_cavity();
        let nodes = unit_tet_nodes();
        let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
            .expect("fixture nodes should cover cavity boundary");
        let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
            .expect("fixture boundary should build triangles");
        let options = refill_options();
        let points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
        let duplicate_cap = raw_refill_tet_with_rejection_reason([0, 1, 2, 3], points, options)
            .expect("fixture cap should pass quality gates");

        let candidate = best_boundary_face_completion_tet(
            [0, 1, 2],
            &cavity,
            &boundary_nodes,
            &[duplicate_cap],
            &boundary_triangles,
            options,
        );

        assert!(candidate.is_none());
    }

    #[test]
    fn boundary_face_completion_selector_reduces_boundary_delta() {
        let cavity = two_tet_bipyramid_cavity();
        let nodes = two_tet_bipyramid_nodes();
        let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
            .expect("fixture nodes should cover cavity boundary");
        let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
            .expect("fixture boundary should build triangles");
        let options = refill_options();
        let duplicate_points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
        let duplicate_tet =
            raw_refill_tet_with_rejection_reason([0, 1, 2, 3], duplicate_points, options)
                .expect("fixture duplicate should pass quality gates");
        let blocked_face = [0, 1, 2];
        let fillable_face = [0, 2, 4];

        let (selected_face, selected_tet) = best_boundary_face_completion_tet_for_faces(
            &[blocked_face, fillable_face],
            &cavity,
            &boundary_nodes,
            &[duplicate_tet.clone()],
            &boundary_triangles,
            options,
        )
        .expect("completion search should evaluate")
        .expect("completion search should find a delta-reducing face");

        let initial_delta = refill_boundary_face_delta(&cavity, &[duplicate_tet.clone()])
            .expect("initial delta should evaluate");
        let next_delta =
            refill_boundary_face_delta(&cavity, &[duplicate_tet, selected_tet.clone()])
                .expect("next delta should evaluate");
        assert!(
            next_delta.missing.len() + next_delta.unexpected.len()
                < initial_delta.missing.len() + initial_delta.unexpected.len()
        );
        assert!(tet_faces(selected_tet.node_ids)
            .map(sorted_face)
            .contains(&sorted_face(selected_face)));
    }

    #[test]
    fn refill_boundary_delta_reports_unexpected_faces() {
        let cavity = unit_tet_cavity();
        let refill_tets = vec![ConstrainedCavityRefillTet {
            node_ids: [0, 1, 2, 4],
            volume_m3: 1.0 / 6.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 1.0,
        }];

        let delta = refill_boundary_face_delta(&cavity, &refill_tets)
            .expect("boundary delta should evaluate");

        assert!(delta.missing.contains(&[0, 1, 3]));
        assert!(delta.unexpected.contains(&[0, 1, 4]));
    }

    #[test]
    fn boundary_face_split_completion_reports_inserted_node_and_refined_boundary() {
        let cavity = unit_tet_cavity();
        let nodes = unit_tet_nodes();
        let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
            .expect("fixture nodes should cover cavity boundary");
        let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
            .expect("fixture boundary should build triangles");
        let options = refill_options();

        let (refined_cavity, inserted_node, split_tets) = best_boundary_face_split_completion(
            [0, 1, 2],
            &cavity,
            &boundary_nodes,
            &boundary_triangles,
            &[],
            options,
        )
        .expect("split completion should evaluate")
        .expect("split completion should generate child cap tets");

        assert_eq!(inserted_node.node_id, 4);
        assert!(inserted_node.coordinates_m[0] > 0.0);
        assert!(inserted_node.coordinates_m[1] > 0.0);
        assert_eq!(inserted_node.coordinates_m[2], 0.0);
        assert!(inserted_node.coordinates_m[0] + inserted_node.coordinates_m[1] < 1.0);
        assert_eq!(split_tets.len(), 3);
        assert!(split_tets
            .iter()
            .all(|tet| tet.node_ids.contains(&inserted_node.node_id)));
        assert!(!refined_cavity
            .boundary_faces
            .iter()
            .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
        assert_eq!(
            refined_cavity
                .boundary_faces
                .iter()
                .filter(|face| face.node_ids.contains(&inserted_node.node_id))
                .count(),
            3
        );
        let refill = refill_from_tets(
            &refined_cavity,
            split_tets,
            options.volume_relative_tolerance,
        )
        .expect("split child tets should preserve the refined boundary");
        validate_constrained_cavity_refill_volume(
            cavity.target_volume_m3,
            refill.total_volume_m3,
            options.volume_relative_tolerance,
        )
        .expect("split completion should preserve the original target volume");
    }

    #[test]
    fn boundary_face_edge_split_completion_reports_inserted_node_and_refined_boundary() {
        let cavity = unit_tet_cavity();
        let nodes = unit_tet_nodes();
        let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
            .expect("fixture nodes should cover cavity boundary");
        let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
            .expect("fixture boundary should build triangles");
        let options = refill_options();

        let (refined_cavity, inserted_node, split_tets) = best_boundary_face_edge_split_completion(
            [0, 1, 2],
            &cavity,
            &boundary_nodes,
            &boundary_triangles,
            &[],
            options,
        )
        .expect("edge-split completion should evaluate")
        .expect("edge-split completion should generate child cap tets");

        assert_eq!(inserted_node.node_id, 4);
        assert_eq!(inserted_node.coordinates_m[2], 0.0);
        assert!(
            (inserted_node.coordinates_m[0] == 0.0 && inserted_node.coordinates_m[1] > 0.0)
                || (inserted_node.coordinates_m[1] == 0.0 && inserted_node.coordinates_m[0] > 0.0)
                || (inserted_node.coordinates_m[0] + inserted_node.coordinates_m[1] - 1.0).abs()
                    <= 1.0e-12
        );
        assert_eq!(split_tets.len(), 2);
        assert!(split_tets
            .iter()
            .all(|tet| tet.node_ids.contains(&inserted_node.node_id)));
        assert!(!refined_cavity
            .boundary_faces
            .iter()
            .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
        assert_eq!(
            refined_cavity
                .boundary_faces
                .iter()
                .filter(|face| face.node_ids.contains(&inserted_node.node_id))
                .count(),
            4
        );
        let refill = refill_from_tets(
            &refined_cavity,
            split_tets,
            options.volume_relative_tolerance,
        )
        .expect("edge-split child tets should preserve the refined boundary");
        validate_constrained_cavity_refill_volume(
            cavity.target_volume_m3,
            refill.total_volume_m3,
            options.volume_relative_tolerance,
        )
        .expect("edge-split completion should preserve the original target volume");
    }

    #[test]
    fn boundary_face_three_edge_split_completion_reports_inserted_nodes_and_refined_boundary() {
        let cavity = unit_tet_cavity();
        let nodes = unit_tet_nodes();
        let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
            .expect("fixture nodes should cover cavity boundary");
        let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
            .expect("fixture boundary should build triangles");
        let options = refill_options();

        let (refined_cavity, inserted_nodes, split_tets) =
            best_boundary_face_three_edge_split_completion(
                [0, 1, 2],
                &cavity,
                &boundary_nodes,
                &boundary_triangles,
                &[],
                options,
            )
            .expect("three-edge completion should evaluate")
            .expect("three-edge completion should generate child cap tets");

        assert_eq!(inserted_nodes.len(), 3);
        assert_eq!(
            inserted_nodes
                .iter()
                .map(|node| node.node_id)
                .collect::<Vec<_>>(),
            vec![4, 5, 6]
        );
        assert!(inserted_nodes
            .iter()
            .all(|node| node.coordinates_m[2].abs() <= 1.0e-12));
        assert_eq!(split_tets.len(), 4);
        assert!(split_tets.iter().all(|tet| {
            inserted_nodes
                .iter()
                .any(|node| tet.node_ids.contains(&node.node_id))
        }));
        assert!(!refined_cavity
            .boundary_faces
            .iter()
            .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
        assert_eq!(
            refined_cavity
                .boundary_faces
                .iter()
                .filter(|face| inserted_nodes
                    .iter()
                    .any(|node| face.node_ids.contains(&node.node_id)))
                .count(),
            10
        );

        let refill = refill_from_tets(
            &refined_cavity,
            split_tets,
            options.volume_relative_tolerance,
        )
        .expect("three-edge child tets should preserve the refined boundary");
        validate_constrained_cavity_refill_volume(
            cavity.target_volume_m3,
            refill.total_volume_m3,
            options.volume_relative_tolerance,
        )
        .expect("three-edge completion should preserve the original target volume");
    }

    #[test]
    fn boundary_face_split_completion_prefers_higher_quality_split_point() {
        let cavity = ConstrainedCavity {
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
            target_volume_m3: 2.0 / 3.0,
        };
        let nodes = vec![
            ConstrainedCavityNode {
                node_id: 0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 1,
                coordinates_m: [1.649331064611886, 0.0, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 2,
                coordinates_m: [0.10383330216927095, 0.5285988568010986, 0.0],
            },
            ConstrainedCavityNode {
                node_id: 3,
                coordinates_m: [1.583996624105325, 0.04591313203731445, 1.25490017426856],
            },
        ];
        let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
            .expect("fixture nodes should cover cavity boundary");
        let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
            .expect("fixture boundary should build triangles");
        let options = refill_options();

        let centroid_node = boundary_face_centroid_node([0, 1, 2], &boundary_nodes);
        let centroid_tets =
            split_completion_tets_for_node([0, 1, 2], 3, &centroid_node, &boundary_nodes, options)
                .expect("centroid split should generate child cap tets");
        let centroid_min_quality = centroid_tets
            .iter()
            .map(|tet| tet.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);

        let (_, inserted_node, split_tets) = best_boundary_face_split_completion(
            [0, 1, 2],
            &cavity,
            &boundary_nodes,
            &boundary_triangles,
            &[],
            options,
        )
        .expect("split completion should evaluate")
        .expect("split completion should generate child cap tets");
        let selected_min_quality = split_tets
            .iter()
            .map(|tet| tet.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);

        assert!(
            selected_min_quality > centroid_min_quality + 1.0e-9,
            "split search should improve on the centroid split: selected={selected_min_quality} centroid={centroid_min_quality}"
        );
        assert_ne!(inserted_node.coordinates_m, centroid_node.coordinates_m);
    }

    #[test]
    fn boundary_face_split_candidates_include_bounded_interior_lattice() {
        let cavity = unit_tet_cavity();
        let nodes = unit_tet_nodes();
        let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
            .expect("fixture nodes should cover cavity boundary");

        let candidates = boundary_face_split_node_candidates([0, 1, 2], &boundary_nodes);

        assert!(candidates.len() >= 40);
        assert!(candidates.len() <= 64);
        assert!(candidates.iter().all(|node| node.node_id == 4));
        assert!(candidates.iter().all(|node| {
            node.coordinates_m[0] > 0.0
                && node.coordinates_m[1] > 0.0
                && node.coordinates_m[2] == 0.0
                && node.coordinates_m[0] + node.coordinates_m[1] < 1.0
        }));
        assert!(candidates.iter().any(|node| {
            (node.coordinates_m[0] - 0.1).abs() <= 1.0e-12
                && (node.coordinates_m[1] - 0.1).abs() <= 1.0e-12
        }));
    }

    #[test]
    fn boundary_node_completion_diagnostic_classifies_no_cap_candidate() {
        let cavity = two_tet_bipyramid_cavity();
        let nodes = two_tet_bipyramid_nodes();

        let diagnostic = diagnostic_boundary_node_completion(
            &cavity,
            &nodes,
            ConstrainedCavityRefillOptions {
                min_scaled_jacobian: 0.95,
                volume_relative_tolerance: 1.0e-12,
                ..ConstrainedCavityRefillOptions::default()
            },
        )
        .expect("diagnostic should evaluate");

        assert_eq!(diagnostic.reason, "boundary_node_completion_no_candidate");
        assert!(diagnostic.missing_face_count > 0);
        assert_eq!(diagnostic.cap_candidate_count, 0);
        assert!(diagnostic.max_rejected_scaled_jacobian < 0.95);
        assert!(!diagnostic.rejected_scaled_jacobian_bins.is_empty());
        assert!(diagnostic.max_rejected_cap_height_ratio > 0.0);
        assert!(!diagnostic.rejected_cap_height_ratio_bins.is_empty());
        assert!(!diagnostic
            .rejected_scaled_jacobian_worst_corner_bins
            .is_empty());
        assert!(!diagnostic.rejected_cap_node_ids.is_empty());
        assert!(diagnostic.split_cap_candidate_count > 0);
        assert_eq!(diagnostic.split_cap_pass_count, 0);
        assert!(diagnostic.max_split_cap_scaled_jacobian < 0.95);
        assert!(!diagnostic.split_cap_scaled_jacobian_bins.is_empty());
        assert!(!diagnostic
            .split_cap_scaled_jacobian_worst_corner_bins
            .is_empty());
        assert!(!diagnostic.split_cap_apex_limited_node_ids.is_empty());
        assert!(diagnostic.edge_split_cap_candidate_count > 0);
        assert_eq!(diagnostic.edge_split_cap_pass_count, 0);
        assert!(diagnostic.max_edge_split_cap_scaled_jacobian < 0.95);
        assert!(!diagnostic.edge_split_cap_scaled_jacobian_bins.is_empty());
        assert!(!diagnostic
            .edge_split_cap_scaled_jacobian_worst_corner_bins
            .is_empty());
        assert!(!diagnostic.edge_split_cap_apex_limited_node_ids.is_empty());
        assert!(diagnostic.three_edge_split_cap_candidate_count > 0);
        assert_eq!(diagnostic.three_edge_split_cap_pass_count, 0);
        assert!(diagnostic.max_three_edge_split_cap_scaled_jacobian < 0.95);
        assert!(!diagnostic
            .three_edge_split_cap_scaled_jacobian_bins
            .is_empty());
        assert!(!diagnostic
            .three_edge_split_cap_scaled_jacobian_worst_corner_bins
            .is_empty());
        assert!(!diagnostic
            .three_edge_split_cap_apex_limited_node_ids
            .is_empty());
        assert!(!diagnostic.rejected_by_reason.is_empty());
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
