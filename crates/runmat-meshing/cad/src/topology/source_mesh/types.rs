use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceTopologyVertex {
    pub vertex_id: u32,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceTopologyEdge {
    pub edge_id: u32,
    pub node_ids: [u32; 2],
    pub adjacent_face_ids: Vec<u32>,
    pub region_ids: Vec<String>,
    pub length_m: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceTopologyFace {
    pub face_id: u32,
    pub source_triangle_id: u32,
    pub node_ids: [u32; 3],
    pub edge_ids: [u32; 3],
    pub region_ids: Vec<String>,
    pub area_m2: f64,
    pub unit_normal: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceTopologyModel {
    pub mesh_id: String,
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
    pub source_geometry_sha256: Option<String>,
    pub vertices: Vec<SourceTopologyVertex>,
    pub edges: Vec<SourceTopologyEdge>,
    pub faces: Vec<SourceTopologyFace>,
    pub bounds_min_m: [f64; 3],
    pub bounds_max_m: [f64; 3],
    pub region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SourceTopologyTriangle {
    pub triangle_id: u32,
    pub node_ids: [u32; 3],
    #[serde(default)]
    pub region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SourceTopologyInput {
    pub mesh_id: String,
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
    pub source_geometry_sha256: Option<String>,
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<SourceTopologyTriangle>,
    pub bounds_min_m: [f64; 3],
    pub bounds_max_m: [f64; 3],
    #[serde(default)]
    pub region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceTopologyError {
    NoSurfaceMeshes,
    EmptySurfaceMesh {
        mesh_id: String,
    },
    NonFiniteVertex {
        mesh_id: String,
        vertex_index: usize,
    },
    TriangleIndexOutOfBounds {
        mesh_id: String,
        triangle_id: u32,
    },
    DegenerateBounds {
        mesh_id: String,
    },
    OpenBoundaryEdge {
        mesh_id: String,
        edge: [u32; 2],
        count: u32,
    },
}

impl std::fmt::Display for SourceTopologyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSurfaceMeshes => write!(formatter, "geometry has no surface mesh input"),
            Self::EmptySurfaceMesh { mesh_id } => {
                write!(
                    formatter,
                    "surface mesh {mesh_id} has no vertices or triangles"
                )
            }
            Self::NonFiniteVertex {
                mesh_id,
                vertex_index,
            } => write!(
                formatter,
                "surface mesh {mesh_id} has non-finite vertex {vertex_index}"
            ),
            Self::TriangleIndexOutOfBounds {
                mesh_id,
                triangle_id,
            } => write!(
                formatter,
                "surface mesh {mesh_id} triangle {triangle_id} references an unknown vertex"
            ),
            Self::DegenerateBounds { mesh_id } => {
                write!(
                    formatter,
                    "surface mesh {mesh_id} does not span a 3D volume"
                )
            }
            Self::OpenBoundaryEdge {
                mesh_id,
                edge,
                count,
            } => write!(
                formatter,
                "surface mesh {mesh_id} boundary edge {}-{} has incidence {count}, expected 2",
                edge[0], edge[1]
            ),
        }
    }
}

impl std::error::Error for SourceTopologyError {}
