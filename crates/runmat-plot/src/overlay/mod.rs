//! Egui overlay rendering helpers.
//!
//! This module is intentionally lightweight: it does **not** depend on winit.
//! It exists so wasm/WebGPU surfaces can render axes/ticks/labels via egui-wgpu.

#[path = "../gui/plot_overlay.rs"]
pub mod plot_overlay;
