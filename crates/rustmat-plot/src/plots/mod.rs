//! High-level plot type implementations
//!
//! This module contains implementations of specific plot types like
//! line plots, scatter plots, surfaces, etc.

pub mod bar;
pub mod figure;
pub mod histogram;
pub mod line;
pub mod point_cloud;
pub mod scatter;
pub mod surface;
pub mod volume;

pub use bar::BarChart;
pub use figure::{Figure, LegendEntry, PlotElement, PlotType};
pub use histogram::Histogram;
pub use line::{LinePlot, LineStyle};
pub use point_cloud::{PointCloudPlot, PointCloudStatistics, PointStyle, SizeMode};
pub use scatter::{MarkerStyle, ScatterPlot};
pub use surface::{ColorMap, ShadingMode, SurfacePlot, SurfaceStatistics};
pub use volume::*;
