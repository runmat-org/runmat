use serde::{Deserialize, Serialize};

/// Jupyter notebook integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterConfig {
    /// Default output format for Jupyter cells
    #[serde(default)]
    pub output_format: JupyterOutputFormat,
    /// Enable interactive widgets
    #[serde(default = "super::default_true")]
    pub enable_widgets: bool,
    /// Enable static image fallback
    #[serde(default = "super::default_true")]
    pub enable_static_fallback: bool,
    /// Widget configuration
    pub widget: Option<JupyterWidgetConfig>,
    /// Static export configuration
    pub static_export: Option<JupyterStaticConfig>,
    /// Performance settings
    pub performance: Option<JupyterPerformanceConfig>,
}

/// Jupyter widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterWidgetConfig {
    /// Enable client-side rendering (WebAssembly)
    #[serde(default = "super::default_true")]
    pub client_side_rendering: bool,
    /// Enable server-side streaming
    #[serde(default)]
    pub server_side_streaming: bool,
    /// Widget cache size in MB
    #[serde(default = "default_widget_cache_size")]
    pub cache_size_mb: u32,
    /// Update frequency for animations (FPS)
    #[serde(default = "default_widget_fps")]
    pub update_fps: u32,
    /// Enable GPU acceleration in browser
    #[serde(default = "super::default_true")]
    pub gpu_acceleration: bool,
}

/// Jupyter static export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterStaticConfig {
    /// Image width in pixels
    #[serde(default = "default_jupyter_width")]
    pub width: u32,
    /// Image height in pixels
    #[serde(default = "default_jupyter_height")]
    pub height: u32,
    /// Image quality (0.0-1.0)
    #[serde(default = "default_jupyter_quality")]
    pub quality: f32,
    /// Include metadata in exports
    #[serde(default = "super::default_true")]
    pub include_metadata: bool,
    /// Preferred formats in order of preference
    #[serde(default)]
    pub preferred_formats: Vec<JupyterOutputFormat>,
}

/// Jupyter performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterPerformanceConfig {
    /// Maximum render time per frame (ms)
    #[serde(default = "default_max_render_time")]
    pub max_render_time_ms: u32,
    /// Enable progressive rendering
    #[serde(default = "super::default_true")]
    pub progressive_rendering: bool,
    /// LOD (Level of Detail) threshold
    #[serde(default = "default_lod_threshold")]
    pub lod_threshold: u32,
    /// Enable texture compression
    #[serde(default = "super::default_true")]
    pub texture_compression: bool,
}

/// Jupyter-specific output formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JupyterOutputFormat {
    /// Interactive HTML widget with WebAssembly
    Widget,
    /// Static PNG image
    Png,
    /// Static SVG image
    Svg,
    /// Base64-encoded image
    Base64,
    /// Plotly-compatible JSON
    PlotlyJson,
    /// Auto-detect based on environment
    Auto,
}

impl Default for JupyterConfig {
    fn default() -> Self {
        Self {
            output_format: JupyterOutputFormat::default(),
            enable_widgets: true,
            enable_static_fallback: true,
            widget: Some(JupyterWidgetConfig::default()),
            static_export: Some(JupyterStaticConfig::default()),
            performance: Some(JupyterPerformanceConfig::default()),
        }
    }
}

impl Default for JupyterWidgetConfig {
    fn default() -> Self {
        Self {
            client_side_rendering: true,
            server_side_streaming: false,
            cache_size_mb: default_widget_cache_size(),
            update_fps: default_widget_fps(),
            gpu_acceleration: true,
        }
    }
}

impl Default for JupyterStaticConfig {
    fn default() -> Self {
        Self {
            width: default_jupyter_width(),
            height: default_jupyter_height(),
            quality: default_jupyter_quality(),
            include_metadata: true,
            preferred_formats: vec![
                JupyterOutputFormat::Widget,
                JupyterOutputFormat::Png,
                JupyterOutputFormat::Svg,
            ],
        }
    }
}

impl Default for JupyterPerformanceConfig {
    fn default() -> Self {
        Self {
            max_render_time_ms: default_max_render_time(),
            progressive_rendering: true,
            lod_threshold: default_lod_threshold(),
            texture_compression: true,
        }
    }
}

impl Default for JupyterOutputFormat {
    fn default() -> Self {
        Self::Auto
    }
}

fn default_widget_cache_size() -> u32 {
    64 // 64MB cache
}

fn default_widget_fps() -> u32 {
    30 // 30 FPS for smooth animations
}

fn default_jupyter_width() -> u32 {
    800
}

fn default_jupyter_height() -> u32 {
    600
}

fn default_jupyter_quality() -> f32 {
    0.9 // High quality (0.0-1.0)
}

fn default_max_render_time() -> u32 {
    16 // 16ms for 60 FPS
}

fn default_lod_threshold() -> u32 {
    10000 // Points threshold for LOD
}
