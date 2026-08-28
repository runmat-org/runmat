mod chart;
mod sample;
mod types;

pub use chart::{accept_exact_face_chart_mesh, validate_exact_face_chart_acceptance};
pub use sample::{accept_exact_face_mesh, validate_exact_face_acceptance};
pub(super) use sample::{
    accept_exact_face_mesh_in_parameterization, validate_exact_face_acceptance_in_parameterization,
};
pub use types::{
    ExactFaceAcceptanceError, ExactFaceAcceptanceErrorKind, ExactFaceAcceptanceOptions,
    ExactFaceAcceptanceReport, ExactFaceChartAcceptanceReport, ExactFaceTriangleAcceptance,
};
