//! Core rendering and scene management for RustMat Plot
//! 
//! This module provides the foundational components for GPU-accelerated
//! interactive plotting, including WGPU rendering, scene graphs, and
//! camera systems.

pub mod renderer;
pub mod scene;
pub mod camera;
pub mod interaction;

pub use renderer::*;
pub use scene::*;
pub use camera::{Camera, CameraController, ProjectionType};
pub use interaction::{PlotEvent, KeyCode, EventHandler};