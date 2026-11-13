//! Core rendering and scene management for RunMat Plot
//!
//! This module provides the foundational components for GPU-accelerated
//! interactive plotting, including WGPU rendering, scene graphs, and
//! camera systems.

pub mod camera;
pub(crate) mod axes_layout;
pub mod depth;
pub mod gpu_pack;
pub mod instances;
pub mod interaction;
pub mod plot_renderer;
pub mod renderer;
pub mod scene;
pub mod triad;
pub mod viewport;

pub use camera::{Camera, CameraController, CameraViewPreset, MouseButton, ProjectionType};
mod navigation;
pub use depth::{ClipPolicy, DepthMode};
pub use gpu_pack::GpuPackContext;
pub use instances::{InstanceRaw, NodeInstances};
pub use interaction::{EventHandler, KeyCode, PlotEvent};
pub use navigation::{Antialiasing, NavigationOptions, NavigationPreset, OrbitCenter};
pub use plot_renderer::{plot_utils, PlotRenderConfig, PlotRenderer, RenderResult};
pub use renderer::*;
pub use scene::*;
pub(crate) use viewport::ResolvedViewport;
pub use viewport::{data_units_per_px, data_units_per_px_3d, data_units_per_px_3d_camera};
