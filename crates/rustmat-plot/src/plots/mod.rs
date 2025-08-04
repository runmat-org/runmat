//! High-level plot type implementations
//! 
//! This module contains implementations of specific plot types like
//! line plots, scatter plots, surfaces, etc.

pub mod line;
pub mod scatter;
pub mod surface;
pub mod volume;

pub use line::*;
pub use scatter::*;
pub use surface::*;
pub use volume::*;