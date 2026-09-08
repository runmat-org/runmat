use std::collections::BTreeMap;

use runmat_execution::Digest;
use runmat_meshing_core::{
    BoundaryFaceRole, BoundaryTriangleOrder, ElementOrder, PersistentEntityId, StableDigest,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct AnalysisProjection {
    pub schema_version: u16,
    pub mesh_digest: StableDigest,
    pub projection_digest: Digest,
    pub policy: AnalysisProjectionPolicy,
    pub nodes: Vec<ProjectedNode>,
    pub elements: Vec<ProjectedElement>,
    pub boundary_faces: Vec<ProjectedBoundaryFace>,
    pub node_neighbors: Vec<Vec<usize>>,
    pub bounds_m: [[f64; 2]; 3],
    pub characteristic_length_m: f64,
    pub(super) node_by_id: BTreeMap<u64, usize>,
    pub(super) element_by_id: BTreeMap<u64, usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisProjectionPolicy {
    pub topology_geometry_revision: u16,
}

impl Default for AnalysisProjectionPolicy {
    fn default() -> Self {
        Self {
            topology_geometry_revision: 1,
        }
    }
}

impl AnalysisProjection {
    pub fn node_index(&self, node_id: u64) -> Option<usize> {
        self.node_by_id.get(&node_id).copied()
    }

    pub fn element_index(&self, element_id: u64) -> Option<usize> {
        self.element_by_id.get(&element_id).copied()
    }

    pub fn boundary_faces_matching<'a>(
        &'a self,
        selector: &'a str,
    ) -> impl Iterator<Item = &'a ProjectedBoundaryFace> + 'a {
        self.boundary_faces.iter().filter(move |face| {
            face.provenance
                .iter()
                .any(|entity| entity.matches_analysis_selection(selector))
        })
    }

    pub fn visit_element_quadrature(
        &self,
        element: &ProjectedElement,
        visit: impl FnMut(VolumeQuadraturePoint),
    ) -> Result<(), AnalysisProjectionError> {
        let expected_node_count = match element.order {
            ElementOrder::Tet4 => 4,
            ElementOrder::Tet10 => 10,
        };
        if element.node_indices.len() != expected_node_count {
            return Err(AnalysisProjectionError::InvalidElementGeometry {
                element_id: element.element_id,
            });
        }
        let mut coordinates = [[0.0; 3]; 10];
        for (local_node, &node_index) in element.node_indices.iter().enumerate() {
            let coordinate = coordinates.get_mut(local_node).ok_or(
                AnalysisProjectionError::InvalidElementGeometry {
                    element_id: element.element_id,
                },
            )?;
            *coordinate = self
                .nodes
                .get(node_index)
                .ok_or(AnalysisProjectionError::InvalidElementGeometry {
                    element_id: element.element_id,
                })?
                .coordinates_m;
        }
        super::quadrature::visit_volume_quadrature(
            element.order,
            &coordinates[..element.node_indices.len()],
            visit,
        )
        .ok_or(AnalysisProjectionError::InvalidElementGeometry {
            element_id: element.element_id,
        })
    }

    pub fn visit_boundary_quadrature(
        &self,
        face: &ProjectedBoundaryFace,
        mut visit: impl FnMut(BoundaryQuadraturePoint),
    ) -> Result<(), AnalysisProjectionError> {
        let expected_node_count = match face.order {
            BoundaryTriangleOrder::Tri3 => 3,
            BoundaryTriangleOrder::Tri6 => 6,
        };
        if face.node_indices.len() != expected_node_count {
            return Err(AnalysisProjectionError::InvalidBoundaryGeometry {
                face_id: face.face_id,
            });
        }
        let mut coordinates = [[0.0; 3]; 6];
        for (local_node, &node_index) in face.node_indices.iter().enumerate() {
            let coordinate = coordinates.get_mut(local_node).ok_or(
                AnalysisProjectionError::InvalidBoundaryGeometry {
                    face_id: face.face_id,
                },
            )?;
            *coordinate = self
                .nodes
                .get(node_index)
                .ok_or(AnalysisProjectionError::InvalidBoundaryGeometry {
                    face_id: face.face_id,
                })?
                .coordinates_m;
        }
        let raw_normal = super::element::cross(
            super::element::subtract(coordinates[1], coordinates[0]),
            super::element::subtract(coordinates[2], coordinates[0]),
        );
        let orientation = if super::element::dot(raw_normal, face.oriented_area_vector_m2) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        super::quadrature::visit_boundary_quadrature(
            face.order,
            &coordinates[..face.node_indices.len()],
            |mut point| {
                point.area_vector_weight_m2 = point
                    .area_vector_weight_m2
                    .map(|component| component * orientation);
                visit(point);
            },
        )
        .ok_or(AnalysisProjectionError::InvalidBoundaryGeometry {
            face_id: face.face_id,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ProjectedNode {
    pub node_id: u64,
    pub stable_identity: StableDigest,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct ProjectedElement {
    pub element_id: u64,
    pub stable_identity: StableDigest,
    pub order: ElementOrder,
    pub node_indices: Vec<usize>,
    pub region_id: PersistentEntityId,
    pub signed_volume_m3: f64,
    pub centroid_m: [f64; 3],
    pub characteristic_length_m: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct VolumeQuadraturePoint {
    pub position_m: [f64; 3],
    pub volume_weight_m3: f64,
    pub(super) shape_values: [f64; 10],
    pub(super) node_count: usize,
}

impl VolumeQuadraturePoint {
    pub fn shape_values(&self) -> &[f64] {
        &self.shape_values[..self.node_count]
    }
}

#[derive(Debug, Clone)]
pub struct ProjectedBoundaryFace {
    pub face_id: u64,
    pub stable_identity: StableDigest,
    pub order: BoundaryTriangleOrder,
    pub role: BoundaryFaceRole,
    pub node_indices: Vec<usize>,
    pub adjacent_element_indices: Vec<usize>,
    pub provenance: Vec<PersistentEntityId>,
    pub centroid_m: [f64; 3],
    pub oriented_area_vector_m2: [f64; 3],
    pub normal_orientation: BoundaryNormalOrientation,
    pub area_m2: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct BoundaryQuadraturePoint {
    pub position_m: [f64; 3],
    pub area_vector_weight_m2: [f64; 3],
    pub area_weight_m2: f64,
    pub(super) shape_values: [f64; 6],
    pub(super) node_count: usize,
}

impl BoundaryQuadraturePoint {
    pub fn shape_values(&self) -> &[f64] {
        &self.shape_values[..self.node_count]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryNormalOrientation {
    ExteriorOutward,
    FirstAdjacentElementOutward,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum AnalysisProjectionError {
    #[error("canonical solver mesh is invalid: {0}")]
    InvalidMesh(String),
    #[error("analysis projection identity encoding failed: {0}")]
    IdentityEncoding(String),
    #[error("{entity_kind} {entity_id} references unknown node {node_id}")]
    UnknownNode {
        entity_kind: &'static str,
        entity_id: u64,
        node_id: u64,
    },
    #[error("boundary face {face_id} references unknown adjacent element {element_id}")]
    UnknownAdjacentElement { face_id: u64, element_id: u64 },
    #[error("volume element {element_id} has unsupported or degenerate geometry")]
    InvalidElementGeometry { element_id: u64 },
    #[error("boundary face {face_id} has unsupported, degenerate, or unoriented geometry")]
    InvalidBoundaryGeometry { face_id: u64 },
}
