mod sample;
mod types;

pub use sample::{accept_exact_face_mesh, validate_exact_face_acceptance};
pub use types::{
    ExactFaceAcceptanceError, ExactFaceAcceptanceErrorKind, ExactFaceAcceptanceOptions,
    ExactFaceAcceptanceReport, ExactFaceTriangleAcceptance,
};
