use runmat_geometry_core::GeometryAsset;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisAlignedBounds {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

pub fn compute_axis_aligned_bounds(_asset: &GeometryAsset) -> AxisAlignedBounds {
    AxisAlignedBounds {
        min: [0.0, 0.0, 0.0],
        max: [0.0, 0.0, 0.0],
    }
}
