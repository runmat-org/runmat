//! Exact constructor identity for provider admission. Floating controls are
//! represented by their bits so signatures cannot merge distinct requests.

use serde::{Deserialize, Serialize};

use crate::{FspecialFilter, FspecialRequest};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderFspecialFilter {
    Average {
        rows: u32,
        cols: u32,
    },
    Disk {
        radius_bits: u64,
        size: u32,
    },
    Gaussian {
        rows: u32,
        cols: u32,
        sigma_bits: u64,
    },
    Laplacian {
        alpha_bits: u64,
    },
    Log {
        rows: u32,
        cols: u32,
        sigma_bits: u64,
    },
    Motion {
        length: u32,
        kernel_size: u32,
        angle_bits: u64,
        oversample: u32,
    },
    Prewitt,
    Sobel,
    Unsharp {
        alpha_bits: u64,
    },
}

impl From<&FspecialRequest> for ProviderFspecialFilter {
    fn from(request: &FspecialRequest) -> Self {
        match &request.filter {
            FspecialFilter::Average { rows, cols } => Self::Average {
                rows: *rows,
                cols: *cols,
            },
            FspecialFilter::Disk { radius, size } => Self::Disk {
                radius_bits: radius.to_bits(),
                size: *size,
            },
            FspecialFilter::Gaussian { rows, cols, sigma } => Self::Gaussian {
                rows: *rows,
                cols: *cols,
                sigma_bits: sigma.to_bits(),
            },
            FspecialFilter::Laplacian { alpha } => Self::Laplacian {
                alpha_bits: alpha.to_bits(),
            },
            FspecialFilter::Log { rows, cols, sigma } => Self::Log {
                rows: *rows,
                cols: *cols,
                sigma_bits: sigma.to_bits(),
            },
            FspecialFilter::Motion {
                length,
                kernel_size,
                angle_degrees,
                oversample,
            } => Self::Motion {
                length: *length,
                kernel_size: *kernel_size,
                angle_bits: angle_degrees.to_bits(),
                oversample: *oversample,
            },
            FspecialFilter::Prewitt => Self::Prewitt,
            FspecialFilter::Sobel => Self::Sobel,
            FspecialFilter::Unsharp { alpha } => Self::Unsharp {
                alpha_bits: alpha.to_bits(),
            },
        }
    }
}

impl ProviderFspecialFilter {
    pub fn output_shape(&self) -> Option<[u64; 2]> {
        match *self {
            Self::Average { rows, cols }
            | Self::Gaussian { rows, cols, .. }
            | Self::Log { rows, cols, .. } => Some([u64::from(rows), u64::from(cols)]),
            Self::Disk { size, .. }
            | Self::Motion {
                kernel_size: size, ..
            } => Some([u64::from(size), u64::from(size)]),
            Self::Laplacian { .. } | Self::Prewitt | Self::Sobel | Self::Unsharp { .. } => {
                Some([3, 3])
            }
        }
    }
}
