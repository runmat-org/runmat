mod evaluate;
mod types;
mod validate;

pub use evaluate::evaluate_exact_face_geometry;
pub use types::{
    ExactFaceGeometry, ExactFaceGeometryError, ExactFaceGeometryErrorKind, ExactFaceGeometryVertex,
    ExactFaceTriangleGeometry,
};
pub use validate::validate_exact_face_geometry;

#[cfg(test)]
mod tests;
