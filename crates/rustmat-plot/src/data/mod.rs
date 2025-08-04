//! Data processing and optimization for plotting
//! 
//! Handles large datasets, level-of-detail, GPU buffer management,
//! and geometric algorithms.

pub mod buffers;
pub mod geometry;
pub mod lod;

pub use buffers::*;
pub use geometry::*;
pub use lod::*;