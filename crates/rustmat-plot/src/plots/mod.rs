//! High-level plot type implementations
//! 
//! This module contains implementations of specific plot types like
//! line plots, scatter plots, surfaces, etc.

pub mod line;
pub mod scatter;
pub mod surface;
pub mod volume;

pub use line::{LinePlot, LineStyle};
pub use scatter::{ScatterPlot, MarkerStyle};
pub use surface::*;
pub use volume::*;