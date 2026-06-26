use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{EntityKind, GeometryAsset};
use serde::{Deserialize, Serialize};

use crate::provenance::{MeshEntityProvenance, SourceEntityKind};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryMeshTriangle {
    pub triangle_id: u32,
    pub node_ids: [u32; 3],
    #[serde(default)]
    pub region_ids: Vec<String>,
    #[serde(default)]
    pub provenance: Vec<MeshEntityProvenance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryMeshInput {
    pub mesh_id: String,
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
    pub source_geometry_sha256: Option<String>,
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<BoundaryMeshTriangle>,
    pub bounds_min_m: [f64; 3],
    pub bounds_max_m: [f64; 3],
    #[serde(default)]
    pub region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundaryMeshInputError {
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

impl BoundaryMeshInput {
    pub fn from_geometry(geometry: &GeometryAsset) -> Result<Self, BoundaryMeshInputError> {
        let surface = geometry
            .surface_meshes
            .iter()
            .min_by(|left, right| left.mesh_id.cmp(&right.mesh_id))
            .ok_or(BoundaryMeshInputError::NoSurfaceMeshes)?;
        if surface.vertices.is_empty() || surface.triangles.is_empty() {
            return Err(BoundaryMeshInputError::EmptySurfaceMesh {
                mesh_id: surface.mesh_id.clone(),
            });
        }

        let mut min = surface.vertices[0];
        let mut max = surface.vertices[0];
        for (vertex_index, vertex) in surface.vertices.iter().copied().enumerate() {
            if vertex.iter().any(|coordinate| !coordinate.is_finite()) {
                return Err(BoundaryMeshInputError::NonFiniteVertex {
                    mesh_id: surface.mesh_id.clone(),
                    vertex_index,
                });
            }
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
            }
        }
        if (0..3).any(|axis| max[axis] <= min[axis]) {
            return Err(BoundaryMeshInputError::DegenerateBounds {
                mesh_id: surface.mesh_id.clone(),
            });
        }

        let mut edge_incidence = BTreeMap::<[u32; 2], u32>::new();
        for (triangle_id, triangle) in surface.triangles.iter().copied().enumerate() {
            if triangle
                .iter()
                .any(|node_id| *node_id as usize >= surface.vertices.len())
            {
                return Err(BoundaryMeshInputError::TriangleIndexOutOfBounds {
                    mesh_id: surface.mesh_id.clone(),
                    triangle_id: triangle_id as u32,
                });
            }
            for edge in triangle_edges(triangle) {
                *edge_incidence.entry(edge).or_insert(0) += 1;
            }
        }
        for (edge, count) in edge_incidence {
            if count != 2 {
                return Err(BoundaryMeshInputError::OpenBoundaryEdge {
                    mesh_id: surface.mesh_id.clone(),
                    edge,
                    count,
                });
            }
        }

        let mut all_region_ids = geometry
            .regions
            .iter()
            .map(|region| region.region_id.clone())
            .collect::<Vec<_>>();
        all_region_ids.sort();
        all_region_ids.dedup();

        let mut triangles = Vec::with_capacity(surface.triangles.len());
        for (triangle_id, node_ids) in surface.triangles.iter().copied().enumerate() {
            let mut region_ids = geometry
                .region_entity_mappings
                .iter()
                .filter(|mapping| {
                    mapping.mesh_id == surface.mesh_id
                        && matches!(mapping.entity_kind, EntityKind::Face | EntityKind::Element)
                        && mapping.contains_entity(triangle_id as u64)
                })
                .map(|mapping| mapping.region_id.clone())
                .collect::<Vec<_>>();
            region_ids.sort();
            region_ids.dedup();
            if region_ids.is_empty() {
                region_ids = all_region_ids.clone();
            }
            triangles.push(BoundaryMeshTriangle {
                triangle_id: triangle_id as u32,
                node_ids,
                region_ids: region_ids.clone(),
                provenance: vec![MeshEntityProvenance {
                    source_geometry_id: geometry.geometry_id.clone(),
                    source_geometry_revision: geometry.revision,
                    source_entity_kind: SourceEntityKind::Face,
                    source_entity_id: triangle_id.to_string(),
                    region_ids,
                }],
            });
        }

        let region_ids = triangles
            .iter()
            .flat_map(|triangle| triangle.region_ids.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        Ok(Self {
            mesh_id: surface.mesh_id.clone(),
            source_geometry_id: geometry.geometry_id.clone(),
            source_geometry_revision: geometry.revision,
            source_geometry_sha256: Some(geometry.source.sha256.clone()),
            vertices: surface.vertices.clone(),
            triangles,
            bounds_min_m: min,
            bounds_max_m: max,
            region_ids,
        })
    }
}

fn triangle_edges(triangle: [u32; 3]) -> [[u32; 2]; 3] {
    [
        sorted_edge(triangle[0], triangle[1]),
        sorted_edge(triangle[1], triangle[2]),
        sorted_edge(triangle[2], triangle[0]),
    ]
}

fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    [left.min(right), left.max(right)]
}
