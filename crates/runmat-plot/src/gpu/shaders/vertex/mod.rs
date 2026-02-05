pub mod grid_plane;
pub mod image_direct;
pub mod line;
pub mod line_direct;
pub mod point;
pub mod point_direct;
pub mod triangle;

pub use grid_plane::SHADER as GRID_PLANE;
pub use image_direct::SHADER as IMAGE_DIRECT;
pub use line::SHADER as LINE;
pub use line_direct::SHADER as LINE_DIRECT;
pub use point::SHADER as POINT;
pub use point_direct::SHADER as POINT_DIRECT;
pub use triangle::SHADER as TRIANGLE;
