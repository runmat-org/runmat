mod grid;
mod linear;
mod types;
mod uniform;

pub use linear::LinearSpatialIndex;
pub use types::{Aabb3, SpatialEntry};
pub use uniform::UniformGridSpatialIndex;

#[cfg(test)]
mod tests;
