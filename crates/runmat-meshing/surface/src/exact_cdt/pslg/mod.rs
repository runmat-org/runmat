mod build;
mod identity;
mod types;
mod validate;

pub use build::build_exact_face_pslg;
pub use identity::exact_face_interior_node_id;
pub use types::{
    ExactFacePslg, ExactFacePslgError, ExactFacePslgErrorKind, ExactFacePslgLoop,
    ExactFacePslgSegment, ExactFacePslgVertex,
};
pub use validate::validate_exact_face_pslg;

#[cfg(test)]
mod tests;

pub(crate) const MAX_FACE_PSLG_ITEMS: usize = 10_000_000;
