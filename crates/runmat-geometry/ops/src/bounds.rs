use runmat_geometry_core::GeometryAsset;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisAlignedBounds {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

pub fn compute_axis_aligned_bounds(asset: &GeometryAsset) -> AxisAlignedBounds {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut found = false;
    for vertex in asset
        .surface_meshes
        .iter()
        .flat_map(|surface_mesh| surface_mesh.vertices.iter())
    {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
        found = true;
    }
    if found {
        return AxisAlignedBounds { min, max };
    }
    AxisAlignedBounds {
        min: [0.0, 0.0, 0.0],
        max: [0.0, 0.0, 0.0],
    }
}
