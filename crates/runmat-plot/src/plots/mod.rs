//! High-level plot type implementations
//!
//! This module contains implementations of specific plot types like
//! line plots, scatter plots, surfaces, etc.

pub mod area;
pub mod bar;
pub mod contour;
pub mod contour_fill;
pub mod errorbar;
pub mod figure;
pub mod image;
pub mod line;
pub mod pie;
pub mod quiver;
pub mod scatter;
pub mod scatter3;
pub mod stairs;
pub mod stem;
pub mod surface;
pub mod volume;

pub use area::AreaPlot;
pub use bar::BarChart;
pub use contour::ContourPlot;
pub use contour_fill::ContourFillPlot;
pub use errorbar::ErrorBar;
pub use figure::{Figure, LegendEntry, PlotElement, PlotType};
pub use image::ImagePlot;
pub use line::{LinePlot, LineStyle};
pub use pie::PieChart;
pub use quiver::QuiverPlot;
pub use scatter::{MarkerStyle, ScatterPlot};
pub use scatter3::Scatter3Plot;
pub use stairs::StairsPlot;
pub use stem::StemPlot;
pub use surface::{ColorMap, ShadingMode, SurfacePlot, SurfaceStatistics};
pub use volume::*;
