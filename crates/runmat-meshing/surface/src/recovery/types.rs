use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SurfaceRecoveryOptions {
    pub require_closed: bool,
    pub max_area_relative_error: f64,
    pub min_normal_alignment: f64,
}

impl Default for SurfaceRecoveryOptions {
    fn default() -> Self {
        Self {
            require_closed: true,
            max_area_relative_error: 1.0e-8,
            min_normal_alignment: 1.0 - 1.0e-8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceRecoveryReport {
    pub surface_element_count: usize,
    pub recovered_edge_count: usize,
    pub open_edge_count: usize,
    pub nonmanifold_edge_count: usize,
    pub max_area_relative_error: f64,
    pub min_normal_alignment: f64,
    pub source_face_coverage_ratio: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceRecoveryError {
    EmptySurface,
    InvalidOptions,
    MissingSurfaceNode {
        element_id: u32,
        node_id: u32,
    },
    NonFiniteSurfaceNode {
        node_id: u32,
    },
    DegenerateElement {
        element_id: u32,
    },
    AreaMismatch {
        element_id: u32,
        relative_error: f64,
        max_relative_error: f64,
    },
    SourceFaceAreaMismatch {
        source_face_id: u32,
        relative_error: f64,
        max_relative_error: f64,
    },
    NormalMismatch {
        element_id: u32,
        alignment: f64,
        min_alignment: f64,
    },
    MissingSourceFace {
        source_face_id: u32,
    },
    UncoveredSourceFace {
        source_face_id: u32,
    },
    OpenEdge {
        edge: [u32; 2],
        count: usize,
    },
    NonManifoldEdge {
        edge: [u32; 2],
        count: usize,
    },
}

impl std::fmt::Display for SurfaceRecoveryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySurface => write!(formatter, "surface recovery input has no elements"),
            Self::InvalidOptions => write!(
                formatter,
                "surface recovery options must use finite area and normal thresholds"
            ),
            Self::MissingSurfaceNode {
                element_id,
                node_id,
            } => write!(
                formatter,
                "surface element {element_id} references missing node {node_id}"
            ),
            Self::NonFiniteSurfaceNode { node_id } => {
                write!(formatter, "surface node {node_id} has non-finite coordinates")
            }
            Self::DegenerateElement { element_id } => {
                write!(formatter, "surface element {element_id} is degenerate")
            }
            Self::AreaMismatch {
                element_id,
                relative_error,
                max_relative_error,
            } => write!(
                formatter,
                "surface element {element_id} area relative error {relative_error:.6e} exceeds {max_relative_error:.6e}"
            ),
            Self::SourceFaceAreaMismatch {
                source_face_id,
                relative_error,
                max_relative_error,
            } => write!(
                formatter,
                "source face {source_face_id} recovered area relative error {relative_error:.6e} exceeds {max_relative_error:.6e}"
            ),
            Self::NormalMismatch {
                element_id,
                alignment,
                min_alignment,
            } => write!(
                formatter,
                "surface element {element_id} normal alignment {alignment:.6e} is below {min_alignment:.6e}"
            ),
            Self::MissingSourceFace { source_face_id } => {
                write!(formatter, "source face {source_face_id} is not present in topology")
            }
            Self::UncoveredSourceFace { source_face_id } => {
                write!(formatter, "source face {source_face_id} is not covered by surface mesh")
            }
            Self::OpenEdge { edge, count } => write!(
                formatter,
                "surface edge {}-{} has incidence {count}, expected 2",
                edge[0], edge[1]
            ),
            Self::NonManifoldEdge { edge, count } => write!(
                formatter,
                "surface edge {}-{} has non-manifold incidence {count}, expected 2",
                edge[0], edge[1]
            ),
        }
    }
}

impl std::error::Error for SurfaceRecoveryError {}
