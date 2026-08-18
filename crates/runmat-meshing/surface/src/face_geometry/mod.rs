mod evaluate;
mod types;
mod validate;

pub use evaluate::{
    evaluate_exact_face_geometry, evaluate_exact_face_geometry_in_parameterization,
};
pub use types::{
    ExactFaceGeometry, ExactFaceGeometryContext, ExactFaceGeometryError,
    ExactFaceGeometryErrorKind, ExactFaceGeometryVertex, ExactFaceTriangleGeometry,
};
pub use validate::{
    validate_exact_face_geometry, validate_exact_face_geometry_in_parameterization,
};

#[cfg(test)]
mod tests;
