use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::path::PathBuf;

pub(super) use super::defaults::default_true;

/// Plotting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlottingConfig {
    /// Plotting mode
    #[serde(default)]
    pub mode: PlotMode,
    /// Force headless mode
    #[serde(default)]
    pub force_headless: bool,
    /// Default plot backend
    #[serde(default)]
    pub backend: PlotBackend,
    /// GUI settings
    pub gui: Option<GuiConfig>,
    /// Export settings
    pub export: Option<ExportConfig>,
    /// Target scatter point budget for GPU decimation overrides
    #[serde(default)]
    pub scatter_target_points: Option<u32>,
    /// Surface vertex budget override for LOD selection
    #[serde(default)]
    pub surface_vertex_budget: Option<u64>,
}

/// GUI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuiConfig {
    /// Window width
    #[serde(default = "default_window_width")]
    pub width: u32,
    /// Window height
    #[serde(default = "default_window_height")]
    pub height: u32,
    /// Enable VSync
    #[serde(default = "default_true")]
    pub vsync: bool,
    /// Enable maximized window
    #[serde(default)]
    pub maximized: bool,
    /// Bounded ingress, independent of the number of already open windows.
    #[serde(default = "default_gui_windows")]
    pub maximum_pending_opens: NonZeroUsize,
    #[serde(default = "default_gui_windows")]
    pub maximum_windows: NonZeroUsize,
    /// Retained scene-batch upper bound; not serialized export, renderer,
    /// GPU-allocation or whole-process memory capacity.
    #[serde(default = "default_gui_snapshot_bytes")]
    pub maximum_snapshot_bytes: NonZeroUsize,
    #[serde(default = "default_gui_callbacks")]
    pub maximum_pending_callbacks: NonZeroUsize,
    #[serde(default = "default_gui_callbacks")]
    pub maximum_callback_registrations: NonZeroUsize,
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportConfig {
    /// Default export format
    #[serde(default)]
    pub format: ExportFormat,
    /// Default DPI for raster exports
    #[serde(default = "default_dpi")]
    pub dpi: u32,
    /// Default output directory
    pub output_dir: Option<PathBuf>,
    /// Maximum serialized figure scene payload size.
    #[serde(default = "default_scene_budget_bytes")]
    pub scene_budget_bytes: usize,
}

/// Plotting mode enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum PlotMode {
    /// Automatic detection based on environment
    Auto,
    /// Force GUI mode
    Gui,
    /// Force headless/static mode
    Headless,
}

/// Plot backend enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum PlotBackend {
    /// Automatic backend selection
    Auto,
    /// WGPU GPU-accelerated backend
    Wgpu,
    /// Static plotters backend
    Static,
    /// Web/browser backend
    Web,
}

/// Export format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    Png,
    Svg,
    Pdf,
    Html,
}

impl Default for PlottingConfig {
    fn default() -> Self {
        Self {
            mode: PlotMode::Auto,
            force_headless: false,
            backend: PlotBackend::Auto,
            gui: Some(GuiConfig::default()),
            export: Some(ExportConfig::default()),
            scatter_target_points: None,
            surface_vertex_budget: None,
        }
    }
}

impl Default for GuiConfig {
    fn default() -> Self {
        Self {
            width: default_window_width(),
            height: default_window_height(),
            vsync: true,
            maximized: false,
            maximum_pending_opens: default_gui_windows(),
            maximum_windows: default_gui_windows(),
            maximum_snapshot_bytes: default_gui_snapshot_bytes(),
            maximum_pending_callbacks: default_gui_callbacks(),
            maximum_callback_registrations: default_gui_callbacks(),
        }
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Png,
            dpi: default_dpi(),
            output_dir: None,
            scene_budget_bytes: default_scene_budget_bytes(),
        }
    }
}

impl Default for PlotMode {
    fn default() -> Self {
        Self::Auto
    }
}

impl Default for PlotBackend {
    fn default() -> Self {
        Self::Auto
    }
}

impl Default for ExportFormat {
    fn default() -> Self {
        Self::Png
    }
}

fn default_window_width() -> u32 {
    1200
}

// Interactive defaults bound ingress/retention, not certified GPU or process
// capacity. Embedders can select smaller/larger explicit live-host limits.
fn default_gui_windows() -> NonZeroUsize {
    NonZeroUsize::new(64).expect("nonzero GUI default")
}
fn default_gui_callbacks() -> NonZeroUsize {
    NonZeroUsize::new(1024).expect("nonzero GUI default")
}
fn default_gui_snapshot_bytes() -> NonZeroUsize {
    NonZeroUsize::new(256 * 1024 * 1024).expect("nonzero GUI default")
}

fn default_window_height() -> u32 {
    800
}

fn default_dpi() -> u32 {
    300
}

fn default_scene_budget_bytes() -> usize {
    8 * 1024 * 1024
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_gui_limits_default_and_reject_zero_without_changing_export_budget() {
        let gui: GuiConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(gui.maximum_windows.get(), 64);
        assert_eq!(gui.maximum_pending_callbacks.get(), 1024);
        assert_eq!(gui.maximum_snapshot_bytes.get(), 256 * 1024 * 1024);
        assert_ne!(
            gui.maximum_snapshot_bytes.get(),
            ExportConfig::default().scene_budget_bytes
        );
        for field in [
            "maximum_pending_opens",
            "maximum_windows",
            "maximum_snapshot_bytes",
            "maximum_pending_callbacks",
            "maximum_callback_registrations",
        ] {
            assert!(
                serde_json::from_value::<GuiConfig>(serde_json::json!({ (field): 0 })).is_err(),
                "{field}"
            );
        }
    }
}
