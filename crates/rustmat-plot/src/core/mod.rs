//! Core rendering and scene management for RustMat Plot
//! 
//! This module provides the foundational components for GPU-accelerated
//! interactive plotting, including WGPU rendering, scene graphs, and
//! camera systems.

pub mod renderer;
pub mod scene;
pub mod camera;
pub mod interaction;
pub mod plot_renderer;

pub use renderer::*;
pub use scene::*;
pub use camera::{Camera, CameraController, ProjectionType, MouseButton};
pub use interaction::{PlotEvent, KeyCode, EventHandler};
pub use plot_renderer::{PlotRenderer, PlotRenderConfig, RenderResult, plot_utils};