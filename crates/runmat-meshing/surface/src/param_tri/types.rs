use runmat_meshing_curve::{CurveValidationError, CurveValidationReport};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceDiscretizationOptions {
    pub preserve_source_faces: bool,
    pub centroid_subdivision: bool,
    pub max_curve_segments_per_edge: usize,
}

impl Default for SurfaceDiscretizationOptions {
    fn default() -> Self {
        Self {
            preserve_source_faces: true,
            centroid_subdivision: false,
            max_curve_segments_per_edge: 256,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceNode {
    pub node_id: u32,
    pub source_vertex_id: u32,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceElement {
    pub element_id: u32,
    pub source_face_id: u32,
    #[serde(default)]
    pub cad_face_id: Option<String>,
    pub source_edge_ids: [u32; 3],
    pub node_ids: [u32; 3],
    #[serde(default)]
    pub parametric_node_uv: [[f64; 2]; 3],
    #[serde(default)]
    pub max_projection_error_m: f64,
    pub region_ids: Vec<String>,
    pub area_m2: f64,
    pub unit_normal: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceDiscretization {
    pub nodes: Vec<SurfaceNode>,
    pub elements: Vec<SurfaceElement>,
    #[serde(default)]
    pub curve_boundary_validation: Option<CurveValidationReport>,
    #[serde(default)]
    pub loop_coverage: Option<SurfaceLoopCoverageReport>,
    #[serde(default)]
    pub exact_cad_sample_node_count: usize,
    #[serde(default)]
    pub rejected_exact_cad_sample_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceLoopCoverageReport {
    pub source_face_count: usize,
    pub recovered_face_count: usize,
    pub boundary_loop_count: usize,
    pub recovered_source_edge_count: usize,
    pub boundary_segment_count: usize,
    pub max_loops_per_face: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceDiscretizationError {
    InvalidCurveBoundary(CurveValidationError),
    MissingFaceVertex {
        face_id: u32,
        node_id: u32,
    },
    MissingFaceEdge {
        face_id: u32,
        edge_id: u32,
    },
    MissingCadFaceFrame {
        source_face_id: u32,
    },
    CadFaceWithoutSourceFaces {
        cad_face_id: String,
    },
    InvalidCadLoopEdgeId {
        cad_face_id: String,
        loop_edge_id: String,
    },
    MissingCurveEdge {
        source_edge_id: u32,
    },
    InvalidFaceEdgeOrientation {
        face_id: u32,
        edge_id: u32,
    },
    CadProjectionOutsideFaceDomain {
        face_id: u32,
        node_id: u32,
    },
    EmptyFaceLoop {
        face_id: u32,
    },
    InvalidFaceLoopTopology {
        face_id: u32,
        node_id: u32,
        incident_segment_count: usize,
    },
}

impl std::fmt::Display for SurfaceDiscretizationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCurveBoundary(source) => {
                write!(formatter, "invalid recovered curve boundary: {source}")
            }
            Self::MissingFaceVertex { face_id, node_id } => write!(
                formatter,
                "source face {face_id} references missing topology vertex {node_id}"
            ),
            Self::MissingFaceEdge { face_id, edge_id } => write!(
                formatter,
                "source face {face_id} references missing topology edge {edge_id}"
            ),
            Self::MissingCadFaceFrame { source_face_id } => write!(
                formatter,
                "source face {source_face_id} does not have a CAD evaluation frame"
            ),
            Self::CadFaceWithoutSourceFaces { cad_face_id } => write!(
                formatter,
                "CAD face {cad_face_id} does not reference any source faces"
            ),
            Self::InvalidCadLoopEdgeId {
                cad_face_id,
                loop_edge_id,
            } => write!(
                formatter,
                "CAD face {cad_face_id} has invalid loop edge id {loop_edge_id}"
            ),
            Self::MissingCurveEdge { source_edge_id } => write!(
                formatter,
                "source edge {source_edge_id} does not have curve discretization nodes"
            ),
            Self::InvalidFaceEdgeOrientation { face_id, edge_id } => write!(
                formatter,
                "source face {face_id} cannot orient source edge {edge_id} along its boundary"
            ),
            Self::CadProjectionOutsideFaceDomain { face_id, node_id } => write!(
                formatter,
                "source face {face_id} node {node_id} projects outside the CAD face domain"
            ),
            Self::EmptyFaceLoop { face_id } => {
                write!(formatter, "source face {face_id} has an empty boundary loop")
            }
            Self::InvalidFaceLoopTopology {
                face_id,
                node_id,
                incident_segment_count,
            } => write!(
                formatter,
                "source face {face_id} boundary node {node_id} has {incident_segment_count} incident curve segments"
            ),
        }
    }
}

impl std::error::Error for SurfaceDiscretizationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidCurveBoundary(source) => Some(source),
            _ => None,
        }
    }
}
