pub mod bar;
pub mod contour;
pub mod contour_fill;
pub mod histogram;
pub mod line;
pub mod scatter2;
pub mod scatter3;
pub mod stairs;
pub mod surface;
pub mod vertex;

/// Backwards-compatible re-export so existing code can keep referencing
/// `shaders::histogram_counts::F32/F64`.
pub mod histogram_counts {
    pub use super::histogram::counts::{F32, F64};
}

/// Backwards-compatible wrapper around the histogram conversion shader template.
pub mod histogram_convert {
    pub const TEMPLATE: &str = super::histogram::CONVERT;
}
