pub mod bar;
pub mod contour;
pub mod contour_fill;
pub mod histogram;
pub mod line;
pub mod scatter2;
pub mod scatter3;
pub mod shaders;
pub mod stairs;
pub mod surface;
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
