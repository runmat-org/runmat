mod chart;
mod mesh_view;
mod sample;
mod types;
pub(crate) use mesh_view::ExactFaceRefinedMeshView;

pub use chart::{accept_exact_face_chart_mesh, validate_exact_face_chart_acceptance};
pub use sample::{accept_exact_face_mesh, validate_exact_face_acceptance};
pub(super) use sample::{
    accept_exact_face_mesh_in_parameterization, sample_exact_face_mesh_in_parameterization,
    validate_exact_face_acceptance_in_parameterization,
};
pub use types::{
    ExactFaceAcceptanceError, ExactFaceAcceptanceErrorKind, ExactFaceAcceptanceOptions,
    ExactFaceAcceptanceReport, ExactFaceAcceptanceViolation, ExactFaceChartAcceptanceReport,
    ExactFaceTriangleAcceptance,
};
