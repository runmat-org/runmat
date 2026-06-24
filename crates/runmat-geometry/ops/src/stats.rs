use runmat_geometry_core::GeometryAsset;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryStats {
    pub mesh_count: usize,
    pub total_vertices: u64,
    pub total_elements: u64,
    pub region_count: usize,
}

pub fn compute_stats(asset: &GeometryAsset) -> GeometryStats {
    GeometryStats {
        mesh_count: asset.meshes.len(),
        total_vertices: asset.meshes.iter().map(|mesh| mesh.vertex_count).sum(),
        total_elements: asset.meshes.iter().map(|mesh| mesh.element_count).sum(),
        region_count: asset.regions.len(),
    }
}
