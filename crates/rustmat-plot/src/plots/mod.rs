//! High-level plot type implementations
//! 
//! This module contains implementations of specific plot types like
//! line plots, scatter plots, surfaces, etc.

pub mod line;
pub mod scatter;
pub mod bar;
pub mod histogram;
pub mod figure;
pub mod surface;
pub mod point_cloud;
pub mod volume;

pub use line::{LinePlot, LineStyle};
pub use scatter::{ScatterPlot, MarkerStyle};
pub use bar::BarChart;
pub use histogram::Histogram;
pub use figure::{Figure, PlotElement, LegendEntry, PlotType};
pub use surface::{SurfacePlot, ColorMap, ShadingMode, SurfaceStatistics};
pub use point_cloud::{PointCloudPlot, PointStyle, SizeMode, PointCloudStatistics};
pub use volume::*;