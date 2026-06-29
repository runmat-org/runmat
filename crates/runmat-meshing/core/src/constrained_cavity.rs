use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::tet_candidate::TetCandidate;

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityExtractionError {
    EmptySelection,
    SelectedTetIndexOutOfBounds { tet_index: usize, tet_count: usize },
    DuplicateSelectedTetIndex { tet_index: usize },
    Validation(ConstrainedCavityValidationError),
}

pub fn constrained_cavity_from_selected_tets(
    tets: &[TetCandidate],
    selected_tet_indices: &[usize],
    protected_node_ids: Vec<u32>,
) -> Result<ConstrainedCavity, ConstrainedCavityExtractionError> {
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

    let mut target_volume_m3 = 0.0_f64;
    let mut face_owners = BTreeMap::<[u32; 3], Vec<(usize, [u32; 3])>>::new();
    for tet_index in &selected {
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

    let cavity = ConstrainedCavity {
        removed_tet_ids: selected
            .iter()
            .map(|tet_index| tets[*tet_index].tet_id)
            .collect(),
        boundary_faces,
        protected_node_ids,
        target_volume_m3,
    };
    validate_constrained_cavity(&cavity).map_err(ConstrainedCavityExtractionError::Validation)?;
    Ok(cavity)
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

fn sorted_face(mut node_ids: [u32; 3]) -> [u32; 3] {
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
