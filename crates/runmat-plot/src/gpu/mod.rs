pub mod area;
pub mod axis;
pub mod bar;
pub mod contour;
pub mod contour_fill;
pub mod errorbar;
pub mod histogram;
pub mod image;
mod input_buffer;
mod readback_error;
mod scatter_error;
pub use input_buffer::GpuInputBuffer;
pub use readback_error::PlotReadbackError;
pub use scatter_error::ScatterPackError;
pub mod line;
pub mod line3;
pub mod quiver;
pub mod scatter2;
pub mod scatter3;
pub mod shaders;
pub mod stairs;
pub mod stem;
pub mod surface;
#[cfg(test)]
mod test_device;
pub mod tuning;
pub mod util;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarType {
    F32,
    F64,
}

impl ScalarType {
    pub fn from_is_f64(is_f64: bool) -> Self {
        if is_f64 {
            ScalarType::F64
        } else {
            ScalarType::F32
        }
    }
}
