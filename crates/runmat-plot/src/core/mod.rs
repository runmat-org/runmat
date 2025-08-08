//! Core rendering and scene management for RunMat Plot
//!
//! This module provides the foundational components for GPU-accelerated
//! interactive plotting, including WGPU rendering, scene graphs, and
//! camera systems.

pub mod camera;
pub mod interaction;
pub mod plot_renderer;
pub mod renderer;
pub mod scene;

pub use camera::{Camera, CameraController, MouseButton, ProjectionType};
pub use interaction::{EventHandler, KeyCode, PlotEvent};
pub use plot_renderer::{plot_utils, PlotRenderConfig, PlotRenderer, RenderResult};
pub use renderer::*;
pub use scene::*;
