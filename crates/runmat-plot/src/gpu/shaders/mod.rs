pub mod area;
pub mod bar;
pub mod contour;
pub mod contour_fill;
pub mod errorbar;
pub mod histogram;
pub mod image;
pub mod line;
pub mod line3;
pub mod quiver;
pub mod scatter2;
pub mod scatter3;
pub mod stairs;
pub mod stem;
pub mod surface;
pub mod vertex;
mod source;
pub(crate) use source::compute_source;

#[cfg(test)]
mod validation_tests;
