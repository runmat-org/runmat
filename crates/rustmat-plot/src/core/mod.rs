//! Core rendering and scene management for RustMat Plot
//! 
//! This module provides the foundational components for GPU-accelerated
//! interactive plotting, including WGPU rendering, scene graphs, and
//! camera systems.

pub mod renderer;
pub mod scene;
pub mod camera;
// pub mod camera_controller; // Temporarily disabled due to API mismatch
pub mod interaction;

pub use renderer::*;
pub use scene::*;
pub use camera::{Camera, ProjectionType};
// pub use camera_controller::{CameraController, CameraConstraints, InteractionMode};
pub use interaction::{PlotEvent, KeyCode, EventHandler};