use super::storage::SortedValues;
use runmat_execution::Digest;
use runmat_meshing_core::{
    BoundaryFaceRole, BoundaryTriangleOrder, ElementOrder, PersistentEntityId, StableDigest,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
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
    pub(super) node_by_id: SortedValues<u64, usize>,
    pub(super) element_by_id: SortedValues<u64, usize>,
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
    /// High-order boundary integration for nonlinear forms. Unlike the default
    /// geometry-moment rule, this does not certify integrand convergence.
    pub(crate) fn visit_boundary_tensor_rule(
        &self,
        face: &ProjectedBoundaryFace,
        rule: &[(f64, f64)],
        mut visit: impl FnMut(BoundaryQuadraturePoint),
    ) -> Result<(), AnalysisProjectionError> {
        let (coordinates, count, orientation) = self.boundary_coordinates_and_orientation(face)?;
        super::quadrature::visit_boundary_tensor_rule(
            face.order,
            &coordinates[..count],
            rule,
            |mut point| {
                point.area_vector_weight_m2 = point.area_vector_weight_m2.map(|v| v * orientation);
                visit(point);
            },
        )
        .ok_or(AnalysisProjectionError::InvalidBoundaryGeometry {
            face_id: face.face_id,
        })
    }

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

    /// Match the resolved entity inventory when present, not the authored alias.
    pub fn boundary_faces_for_target<'a>(
        &'a self,
        target: &'a runmat_analysis_core::RegionTarget,
    ) -> impl Iterator<Item = &'a ProjectedBoundaryFace> + 'a {
        self.boundary_faces.iter().filter(move |face| {
            face.provenance
                .iter()
                .any(|entity| target.matches_persistent_entity(entity))
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
                .coordinates_m
                .canonical_value();
        }
        super::quadrature::visit_volume_quadrature(
            element.order,
            &coordinates[..element.node_indices.len()],
            visit,
        )
        .map_err(AnalysisProjectionError::Accounting)?
        .ok_or(AnalysisProjectionError::InvalidElementGeometry {
            element_id: element.element_id,
        })
    }

    pub fn visit_boundary_quadrature(
        &self,
        face: &ProjectedBoundaryFace,
        mut visit: impl FnMut(BoundaryQuadraturePoint),
    ) -> Result<(), AnalysisProjectionError> {
        let (coordinates, count, orientation) = self.boundary_coordinates_and_orientation(face)?;
        super::quadrature::visit_boundary_quadrature(
            face.order,
            &coordinates[..count],
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

    pub(crate) fn boundary_nodal_area_vectors(
        &self,
        face: &ProjectedBoundaryFace,
    ) -> Result<Vec<[f64; 3]>, AnalysisProjectionError> {
        let (coordinates, count, orientation) = self.boundary_coordinates_and_orientation(face)?;
        super::quadrature::boundary_nodal_area_vectors(face.order, &coordinates[..count])
            .map(|vectors| {
                vectors
                    .into_iter()
                    .map(|vector| vector.map(|component| orientation * component))
                    .collect()
            })
            .ok_or(AnalysisProjectionError::InvalidBoundaryGeometry {
                face_id: face.face_id,
            })
    }

    /// Sign that turns the corner-ordered parametric normal `x,ξ × x,η` into the face's
    /// oriented area vector, shared by reference and deformed-surface integration.
    pub(crate) fn boundary_parametric_orientation(
        &self,
        face: &ProjectedBoundaryFace,
    ) -> Result<f64, AnalysisProjectionError> {
        self.boundary_coordinates_and_orientation(face)
            .map(|(_, _, orientation)| orientation)
    }

    fn boundary_coordinates_and_orientation(
        &self,
        face: &ProjectedBoundaryFace,
    ) -> Result<([[f64; 3]; 6], usize, f64), AnalysisProjectionError> {
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
        for (coordinate, &index) in coordinates.iter_mut().zip(&face.node_indices) {
            *coordinate = self
                .nodes
                .get(index)
                .ok_or(AnalysisProjectionError::InvalidBoundaryGeometry {
                    face_id: face.face_id,
                })?
                .coordinates_m
                .canonical_value();
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
        Ok((coordinates, expected_node_count, orientation))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedNode {
    pub node_id: u64,
    pub stable_identity: StableDigest,
    pub coordinates_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedElement {
    pub element_id: u64,
    pub stable_identity: StableDigest,
    pub order: ElementOrder,
    pub node_indices: Vec<usize>,
    pub region_id: PersistentEntityId,
    /// Original CAD ownership retained by the admitted solver element.
    pub provenance: Vec<PersistentEntityId>,
    pub signed_volume_m3: f64,
    pub centroid_m: [f64; 3],
    pub characteristic_length_m: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct VolumeQuadraturePoint {
    pub position_m: [f64; 3],
    pub volume_weight_m3: f64,
    pub(super) barycentric: [f64; 4],
    pub(super) physical_barycentric_gradients: [[f64; 3]; 4],
    pub(super) shape_values: [f64; 10],
    pub(super) physical_shape_gradients: [[f64; 3]; 10],
    pub(super) node_count: usize,
}

/// Canonical finite-element coordinates of a physical point inside the solid projection.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumePointLocation {
    pub element_index: usize,
    pub element_identity: StableDigest,
    pub position_m: [f64; 3],
    pub node_indices: Vec<usize>,
    pub node_identities: Vec<StableDigest>,
    pub shape_values: Vec<f64>,
}

impl VolumeQuadraturePoint {
    pub fn shape_values(&self) -> &[f64] {
        &self.shape_values[..self.node_count]
    }

    /// Corner (P1) barycentric gradients at this point; constant on affine elements.
    pub fn physical_barycentric_gradients(&self) -> &[[f64; 3]; 4] {
        &self.physical_barycentric_gradients
    }

    pub fn physical_shape_gradients(&self) -> &[[f64; 3]] {
        &self.physical_shape_gradients[..self.node_count]
    }
}

#[derive(Debug, Clone, PartialEq)]
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
    pub area_m2: runmat_types::physical::Area,
}

#[derive(Debug, Clone, Copy)]
pub struct BoundaryQuadraturePoint {
    pub position_m: [f64; 3],
    pub area_vector_weight_m2: [f64; 3],
    pub area_weight_m2: f64,
    pub(super) shape_values: [f64; 6],
    pub(super) node_count: usize,
    pub(super) reference_barycentric: [f64; 3],
    pub(super) reference_weight: f64,
}

impl BoundaryQuadraturePoint {
    /// Original reference coordinates, never recovered from world position or nodal averages.
    pub fn reference_barycentric(&self) -> [f64; 3] {
        self.reference_barycentric
    }
    pub fn reference_weight(&self) -> f64 {
        self.reference_weight
    }

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
    #[error("analysis projection numerical accounting failed: {0}")]
    Accounting(#[from] runmat_simulation_usage::LedgerError),
    #[error("analysis projection construction cancelled")]
    Cancelled,
    #[error("edge space does not match canonical projection: {0}")]
    InvalidEdgeSpace(&'static str),
    #[error("analysis projection resource estimate failed: {0}")]
    ResourceEstimate(String),
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
