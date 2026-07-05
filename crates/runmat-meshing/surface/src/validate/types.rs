use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SurfaceValidationOptions {
    pub max_projection_error_m: f64,
    pub min_orientation_alignment: f64,
    pub require_source_edge_conformity: bool,
}

impl Default for SurfaceValidationOptions {
    fn default() -> Self {
        Self {
            max_projection_error_m: 1.0e-8,
            min_orientation_alignment: 1.0 - 1.0e-8,
            require_source_edge_conformity: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceValidationReport {
    pub source_face_count: usize,
    pub surface_element_count: usize,
    pub source_edge_loop_count: usize,
    pub closed_source_edge_loop_count: usize,
    pub conforming_source_edge_count: usize,
    pub missing_source_edge_count: usize,
    pub max_projection_error_m: f64,
    pub min_orientation_alignment: f64,
    pub face_coverage_ratio: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceValidationError {
    InvalidOptions,
    EmptySurface,
    MissingSurfaceNode {
        element_id: u32,
        node_id: u32,
    },
    MissingSourceFace {
        source_face_id: u32,
    },
    MissingSourceEdge {
        source_edge_id: u32,
    },
    MissingCadFaceId {
        element_id: u32,
    },
    MissingCadFace {
        cad_face_id: String,
    },
    InvalidCadLoopEdgeId {
        cad_face_id: String,
        loop_edge_id: String,
    },
    SourceEdgeOutsideCadFace {
        cad_face_id: String,
        source_edge_id: u32,
    },
    EdgeConformityFailed {
        source_edge_id: u32,
        source_edge_node_ids: [u32; 2],
        recovered_segment_count: usize,
    },
    OpenSourceLoop {
        source_edge_id: u32,
        endpoint_id: u32,
        incidence_count: usize,
    },
    DegenerateElement {
        element_id: u32,
    },
    InvalidElementNormal {
        element_id: u32,
    },
    InvalidElementArea {
        element_id: u32,
    },
    InvalidProjectionEvidence {
        element_id: u32,
    },
    ProjectionError {
        element_id: u32,
        error_m: f64,
        max_error_m: f64,
    },
    OrientationMismatch {
        element_id: u32,
        source_face_id: u32,
        alignment: f64,
        min_alignment: f64,
    },
    UncoveredSourceFace {
        source_face_id: u32,
    },
    UncoveredCadFace {
        cad_face_id: String,
    },
}

impl std::fmt::Display for SurfaceValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOptions => write!(
                formatter,
                "surface validation options must use finite projection and orientation thresholds"
            ),
            Self::EmptySurface => write!(formatter, "surface validation input has no elements"),
            Self::MissingSurfaceNode {
                element_id,
                node_id,
            } => write!(
                formatter,
                "surface element {element_id} references missing node {node_id}"
            ),
            Self::MissingSourceFace { source_face_id } => {
                write!(formatter, "source face {source_face_id} is missing")
            }
            Self::MissingSourceEdge { source_edge_id } => {
                write!(formatter, "source edge {source_edge_id} is missing")
            }
            Self::MissingCadFaceId { element_id } => write!(
                formatter,
                "surface element {element_id} does not carry CAD face provenance"
            ),
            Self::MissingCadFace { cad_face_id } => {
                write!(formatter, "CAD face {cad_face_id} is missing")
            }
            Self::InvalidCadLoopEdgeId {
                cad_face_id,
                loop_edge_id,
            } => write!(
                formatter,
                "CAD face {cad_face_id} has invalid loop edge id {loop_edge_id}"
            ),
            Self::SourceEdgeOutsideCadFace {
                cad_face_id,
                source_edge_id,
            } => write!(
                formatter,
                "source edge {source_edge_id} is not a loop edge for CAD face {cad_face_id}"
            ),
            Self::EdgeConformityFailed {
                source_edge_id,
                source_edge_node_ids,
                recovered_segment_count,
            } => write!(
                formatter,
                "source edge {source_edge_id} with endpoints {:?} is not represented by a matching surface element edge; recovered surface segment count is {recovered_segment_count}",
                source_edge_node_ids
            ),
            Self::OpenSourceLoop {
                source_edge_id,
                endpoint_id,
                incidence_count,
            } => write!(
                formatter,
                "source edge {source_edge_id} endpoint {endpoint_id} has loop incidence {incidence_count}, expected 2"
            ),
            Self::DegenerateElement { element_id } => {
                write!(formatter, "surface element {element_id} is degenerate")
            }
            Self::InvalidElementNormal { element_id } => write!(
                formatter,
                "surface element {element_id} has invalid unit-normal evidence"
            ),
            Self::InvalidElementArea { element_id } => write!(
                formatter,
                "surface element {element_id} has invalid area evidence"
            ),
            Self::InvalidProjectionEvidence { element_id } => write!(
                formatter,
                "surface element {element_id} has invalid projection evidence"
            ),
            Self::ProjectionError {
                element_id,
                error_m,
                max_error_m,
            } => write!(
                formatter,
                "surface element {element_id} projection error {error_m:.6e} m exceeds {max_error_m:.6e} m"
            ),
            Self::OrientationMismatch {
                element_id,
                source_face_id,
                alignment,
                min_alignment,
            } => write!(
                formatter,
                "surface element {element_id} on source face {source_face_id} orientation alignment {alignment:.6e} is below {min_alignment:.6e}"
            ),
            Self::UncoveredSourceFace { source_face_id } => {
                write!(formatter, "source face {source_face_id} is not covered")
            }
            Self::UncoveredCadFace { cad_face_id } => {
                write!(formatter, "CAD face {cad_face_id} is not covered")
            }
        }
    }
}

impl std::error::Error for SurfaceValidationError {}
