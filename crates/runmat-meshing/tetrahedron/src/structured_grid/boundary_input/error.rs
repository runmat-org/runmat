#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::structured_grid) enum BoundaryMeshInputError {
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

impl std::fmt::Display for BoundaryMeshInputError {
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

impl std::error::Error for BoundaryMeshInputError {}
