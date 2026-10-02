//! Admitted convolution controls and physical launch limits. Placement consumes
//! these facts; public option decoding and scheduling remain with their owners.

use runmat_types::ConvolutionMode;
use serde::{Deserialize, Serialize};

use crate::ProviderConvOrientation;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderConvolutionForm {
    Vector {
        orientation: ProviderConvOrientation,
    },
    Matrix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConvolutionWorkload {
    pub form: ProviderConvolutionForm,
    pub mode: ConvolutionMode,
}

/// Actual conv1d shader limits from the selected WGPU adapter and launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderConvolutionGeometry {
    pub max_buffer_bytes: u64,
    pub max_binding_bytes: u64,
    pub max_workgroups_per_dimension: u32,
    pub workgroup_size: u32,
}
