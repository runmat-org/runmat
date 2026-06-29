use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
