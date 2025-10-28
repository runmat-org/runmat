//! High-level plot type implementations
//!
//! This module contains implementations of specific plot types like
//! line plots, scatter plots, surfaces, etc.

pub mod bar;
pub mod figure;
pub mod errorbar;
pub mod stairs;
pub mod stem;
pub mod area;
pub mod quiver;
pub mod pie;
pub mod image;
pub mod line;
pub mod scatter;
pub mod surface;
pub mod volume;

pub use bar::BarChart;
pub use figure::{Figure, LegendEntry, PlotElement, PlotType};
pub use errorbar::ErrorBar;
pub use stairs::StairsPlot;
pub use stem::StemPlot;
pub use area::AreaPlot;
pub use quiver::QuiverPlot;
pub use pie::PieChart;
pub use image::ImagePlot;
pub use line::{LinePlot, LineStyle};
// point_cloud removed; use scatter3 API at runtime level or future 3D scatter in Surface/Scatter
pub use scatter::{MarkerStyle, ScatterPlot};
pub use surface::{ColorMap, ShadingMode, SurfacePlot, SurfaceStatistics};
pub use volume::*;
