mod annulus;
mod domain;
mod parameterization;
mod periodic;
mod triangulation;
mod types;

pub use domain::{recover_exact_face_chart_domains, validate_exact_face_chart_domains};
pub use parameterization::ExactFaceChartParameterization;
pub use periodic::{build_exact_face_charts, validate_exact_face_charts};
pub use triangulation::{triangulate_exact_face_charts, validate_exact_face_chart_delaunay};
pub use types::{
    ExactFaceChart, ExactFaceChartConstrainedDomain, ExactFaceChartDelaunay,
    ExactFaceChartDelaunayContext, ExactFaceChartError, ExactFaceChartErrorKind,
    ExactFaceChartOptions, ExactFaceCharts,
};

#[cfg(test)]
mod tests;
