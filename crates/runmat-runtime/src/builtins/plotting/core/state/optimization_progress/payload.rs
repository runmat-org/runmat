//! Shared scene-update payload, independent of any optimizer controller.
pub(crate) enum OptimizationPlotData {
    /// No metric exists yet; initialize axes without inventing a sample.
    Empty,
    Reference(runmat_plot::plots::ReferenceLine),
    Sample {
        point: Vec<f64>,
        iteration: f64,
        value: f64,
    },
    Bars {
        labels: Vec<String>,
        values: Vec<f64>,
    },
    Edges {
        generation: f64,
        x: Vec<f64>,
        y: Vec<f64>,
    },
}
