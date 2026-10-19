//! Request-local matrix execution structure. Scalar controls use exact IEEE
//! bits so signed zeros participate in query identity without lossy JSON floats.
use crate::{MatmulAffine, ScaleOp};
use serde::{Deserialize, Serialize};

/// Frozen launch recipe of the provider's matrix/diagonal composite. These
/// values describe the compiled kernels, not process-wide tuning defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderMatmulGeometry {
    pub tile_size: u32,
    pub diagonal_workgroup_size: u32,
}

impl ProviderMatmulGeometry {
    /// Checked group counts in matrix X/Y and diagonal X order.
    pub fn dispatch_groups(self, rows: u64, columns: u64, diagonal: bool) -> Option<[u32; 3]> {
        if self.tile_size == 0 || self.diagonal_workgroup_size == 0 {
            return None;
        }
        let x = columns.div_ceil(u64::from(self.tile_size));
        let y = rows.div_ceil(u64::from(self.tile_size));
        let d = if diagonal {
            rows.min(columns)
                .div_ceil(u64::from(self.diagonal_workgroup_size))
        } else {
            0
        };
        Some([
            u32::try_from(x).ok()?,
            u32::try_from(y).ok()?,
            u32::try_from(d).ok()?,
        ])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatmulScaleWork {
    pub input: usize,
    pub operation: ScaleOp,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatmulWorkload {
    pub left: usize,
    pub right: usize,
    pub rows: u64,
    pub inner: u64,
    pub columns: u64,
    pub affine: MatmulAffine<u64>,
    pub row: Option<MatmulScaleWork>,
    pub column: Option<MatmulScaleWork>,
    pub clamp_min: Option<u64>,
    pub clamp_max: Option<u64>,
    pub exponent: Option<u64>,
    pub diagonal: bool,
}

#[cfg(test)]
#[path = "matmul/geometry_tests.rs"]
mod geometry_tests;
