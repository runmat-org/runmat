mod annulus;
mod periodic;
mod triangulation;
mod types;

pub use periodic::{build_exact_face_charts, validate_exact_face_charts};
pub use triangulation::{triangulate_exact_face_charts, validate_exact_face_chart_delaunay};
pub use types::{
    ExactFaceChart, ExactFaceChartDelaunay, ExactFaceChartDelaunayContext, ExactFaceChartError,
    ExactFaceChartErrorKind, ExactFaceChartOptions, ExactFaceCharts,
};

#[cfg(test)]
mod tests;
