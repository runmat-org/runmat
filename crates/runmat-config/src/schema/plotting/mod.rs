mod jupyter;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub use jupyter::{
    JupyterConfig, JupyterOutputFormat, JupyterPerformanceConfig, JupyterStaticConfig,
    JupyterWidgetConfig,
};

pub(super) use super::defaults::default_true;

/// Plotting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Default export format
    #[serde(default)]
    pub format: ExportFormat,
    /// Default DPI for raster exports
    #[serde(default = "default_dpi")]
    pub dpi: u32,
    /// Default output directory
    pub output_dir: Option<PathBuf>,
    /// Jupyter notebook configuration
    pub jupyter: Option<JupyterConfig>,
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
    /// Jupyter notebook mode
    Jupyter,
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
        }
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Png,
            dpi: default_dpi(),
            output_dir: None,
            jupyter: Some(JupyterConfig::default()),
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

fn default_window_height() -> u32 {
    800
}

fn default_dpi() -> u32 {
    300
}
