use serde::{Deserialize, Serialize};

use super::provenance::MeshEntityProvenance;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryMeshTriangle {
    pub triangle_id: u32,
    pub node_ids: [u32; 3],
    #[serde(default)]
    pub region_ids: Vec<String>,
    #[serde(default)]
    pub material_region_ids: Vec<String>,
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
    #[serde(default)]
    pub material_region_ids: Vec<String>,
}
