//! Egui overlay rendering helpers.
//!
//! This module is intentionally lightweight: it does **not** depend on winit.
//! It exists so wasm/WebGPU surfaces can render axes/ticks/labels via egui-wgpu.

#[path = "../gui/plot_overlay.rs"]
pub mod plot_overlay;

#[path = "../gui/legend_placement.rs"]
pub(crate) mod legend_placement;

#[path = "../gui/cad_overlay.rs"]
pub(crate) mod cad_overlay;

#[path = "../gui/product_ui.rs"]
pub(crate) mod product_ui;

// egui 0.27 painted native rect strokes centered on their bounds. The
// existing browser overlay used inside strokes. Preserve each presentation.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const OUTLINE_STROKE_KIND: egui::StrokeKind = egui::StrokeKind::Middle;
#[cfg(target_arch = "wasm32")]
pub(crate) const OUTLINE_STROKE_KIND: egui::StrokeKind = egui::StrokeKind::Inside;
