mod build;
mod types;
mod validate;

pub use build::build_exact_face_pslg;
pub use types::{
    ExactFacePslg, ExactFacePslgError, ExactFacePslgErrorKind, ExactFacePslgLoop,
    ExactFacePslgSegment, ExactFacePslgVertex,
};
pub use validate::validate_exact_face_pslg;

#[cfg(test)]
mod tests;
