use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

mod quality;
pub use quality::{
    evaluate_local_tetrahedron_flip_improvement, evaluate_local_tetrahedron_flip_quality,
};

mod topology;
pub use topology::local_tetrahedron_boundary_faces;
use topology::{
    opposite_node, ring_edges_form_cycle, shared_face, sorted_edge, sorted_face,
    sorted_removed_tetrahedron_ids,
};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalTetrahedron {
    pub tetrahedron_id: u32,
    pub node_ids: [u32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalTetrahedronFlipKind {
    TwoToThreeFace,
    ThreeToTwoEdge,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalTetrahedronFlipCandidate {
    pub kind: LocalTetrahedronFlipKind,
    pub removed_tetrahedron_ids: Vec<u32>,
    pub created_tetrahedra: Vec<[u32; 4]>,
    #[serde(default)]
    pub shared_face: Option<[u32; 3]>,
    #[serde(default)]
    pub shared_edge: Option<[u32; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocalTetrahedronFlipQualityThresholds {
    pub min_volume_m3: f64,
    pub min_scaled_jacobian: f64,
}

impl Default for LocalTetrahedronFlipQualityThresholds {
    fn default() -> Self {
        Self {
            min_volume_m3: 1.0e-18,
            min_scaled_jacobian: 0.15,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalTetrahedronFlipQualityReport {
    pub created_tetrahedron_count: usize,
    pub total_volume_m3: f64,
    pub min_volume_m3: f64,
    pub min_scaled_jacobian: f64,
    pub max_aspect_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalTetrahedronFlipImprovementReport {
    pub removed_tetrahedron_count: usize,
    pub created_tetrahedron_count: usize,
    pub current_min_scaled_jacobian: f64,
    pub candidate_min_scaled_jacobian: f64,
    pub current_total_volume_m3: f64,
    pub candidate_total_volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalTetrahedronFlipError {
    DegenerateTetrahedron {
        tetrahedron_id: u32,
        node_ids: [u32; 4],
    },
    NoSharedFace,
    NoSharedEdge,
    InvalidEdgeRing,
    InvalidQualityThresholds,
    MissingNode {
        node_id: u32,
    },
    NonPositiveVolume {
        node_ids: [u32; 4],
    },
    VolumeBelowThreshold {
        node_ids: [u32; 4],
        volume_m3: String,
    },
    ScaledJacobianBelowThreshold {
        node_ids: [u32; 4],
        scaled_jacobian: String,
    },
    QualityDoesNotImprove,
}

pub fn two_to_three_face_flip_candidate(
    left: LocalTetrahedron,
    right: LocalTetrahedron,
) -> Result<LocalTetrahedronFlipCandidate, LocalTetrahedronFlipError> {
    validate_tetrahedron(left)?;
    validate_tetrahedron(right)?;
    let Some(shared_face) = shared_face(left.node_ids, right.node_ids) else {
        return Err(LocalTetrahedronFlipError::NoSharedFace);
    };
    let Some(left_apex) = opposite_node(left.node_ids, &shared_face) else {
        return Err(LocalTetrahedronFlipError::NoSharedFace);
    };
    let Some(right_apex) = opposite_node(right.node_ids, &shared_face) else {
        return Err(LocalTetrahedronFlipError::NoSharedFace);
    };

    Ok(LocalTetrahedronFlipCandidate {
        kind: LocalTetrahedronFlipKind::TwoToThreeFace,
        removed_tetrahedron_ids: sorted_removed_tetrahedron_ids([
            left.tetrahedron_id,
            right.tetrahedron_id,
        ]),
        created_tetrahedra: vec![
            [left_apex, right_apex, shared_face[0], shared_face[1]],
            [left_apex, right_apex, shared_face[1], shared_face[2]],
            [left_apex, right_apex, shared_face[2], shared_face[0]],
        ],
        shared_face: Some(shared_face),
        shared_edge: Some(sorted_edge([left_apex, right_apex])),
    })
}

pub fn three_to_two_edge_flip_candidate(
    tetrahedra: [LocalTetrahedron; 3],
    edge: [u32; 2],
) -> Result<LocalTetrahedronFlipCandidate, LocalTetrahedronFlipError> {
    for tetrahedron in tetrahedra {
        validate_tetrahedron(tetrahedron)?;
    }
    let edge = sorted_edge(edge);
    let mut ring_edges = BTreeSet::<[u32; 2]>::new();
    let mut ring_nodes = BTreeSet::<u32>::new();
    for tetrahedron in tetrahedra {
        if !tetrahedron.node_ids.contains(&edge[0]) || !tetrahedron.node_ids.contains(&edge[1]) {
            return Err(LocalTetrahedronFlipError::NoSharedEdge);
        }
        let opposite = tetrahedron
            .node_ids
            .into_iter()
            .filter(|node_id| !edge.contains(node_id))
            .collect::<Vec<_>>();
        if opposite.len() != 2 {
            return Err(LocalTetrahedronFlipError::InvalidEdgeRing);
        }
        ring_nodes.insert(opposite[0]);
        ring_nodes.insert(opposite[1]);
        ring_edges.insert(sorted_edge([opposite[0], opposite[1]]));
    }
    if ring_nodes.len() != 3
        || ring_edges.len() != 3
        || !ring_edges_form_cycle(&ring_nodes, &ring_edges)
    {
        return Err(LocalTetrahedronFlipError::InvalidEdgeRing);
    }
    let ring = ring_nodes.into_iter().collect::<Vec<_>>();
    Ok(LocalTetrahedronFlipCandidate {
        kind: LocalTetrahedronFlipKind::ThreeToTwoEdge,
        removed_tetrahedron_ids: sorted_removed_tetrahedron_ids([
            tetrahedra[0].tetrahedron_id,
            tetrahedra[1].tetrahedron_id,
            tetrahedra[2].tetrahedron_id,
        ]),
        created_tetrahedra: vec![
            [edge[0], ring[0], ring[1], ring[2]],
            [edge[1], ring[0], ring[2], ring[1]],
        ],
        shared_face: Some(sorted_face([ring[0], ring[1], ring[2]])),
        shared_edge: Some(edge),
    })
}

fn validate_tetrahedron(tetrahedron: LocalTetrahedron) -> Result<(), LocalTetrahedronFlipError> {
    let unique = tetrahedron.node_ids.into_iter().collect::<BTreeSet<_>>();
    if unique.len() != 4 {
        return Err(LocalTetrahedronFlipError::DegenerateTetrahedron {
            tetrahedron_id: tetrahedron.tetrahedron_id,
            node_ids: tetrahedron.node_ids,
        });
    }
    Ok(())
}
