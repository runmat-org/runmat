//! Jupyter notebook integration for interactive plotting
//!
//! Provides seamless integration with Jupyter notebooks, enabling interactive
//! plotting output directly in notebook cells with full GPU acceleration.

use runmat_time::unix_timestamp_us;
use std::collections::HashMap;
// use std::path::Path; // Not currently used
use crate::plots::{Figure, LinePlot, ScatterPlot, SurfacePlot};

/// Jupyter notebook output handler
#[derive(Debug)]
pub struct JupyterBackend {
    /// Output format preferences
    pub output_format: OutputFormat,

    /// Interactive mode settings
    interactive_mode: bool,

    /// Export settings
    export_settings: ExportSettings,
}

/// Output format for Jupyter cells
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    /// Static PNG image
    PNG,
    /// Static SVG image
    SVG,
    /// Interactive HTML widget
    HTML,
    /// Base64 encoded image
    Base64,
    /// Plotly-compatible JSON
    PlotlyJSON,
}

/// Widget state for interactive plots
#[derive(Debug, Clone)]
pub struct WidgetState {
    /// Widget ID
    pub widget_id: String,

    /// Current view state
    pub camera_position: [f32; 3],
    pub camera_target: [f32; 3],
    pub zoom_level: f32,

    /// Visibility states
    pub visible_plots: Vec<bool>,

    /// Style overrides
    pub style_overrides: HashMap<String, String>,

    /// Interactive mode
    pub interactive: bool,
}

/// Export settings for different formats
#[derive(Debug, Clone)]
pub struct ExportSettings {
    /// Image resolution for raster formats
    pub width: u32,
    pub height: u32,

    /// DPI for high-resolution displays
    pub dpi: f32,

    /// Background color
    pub background_color: [f32; 4],

    /// Quality settings
    pub quality: Quality,

    /// Include metadata
    pub include_metadata: bool,
}

/// Quality settings for exports
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Quality {
    /// Draft quality (fast)
    Draft,
    /// Standard quality
    Standard,
    /// High quality (slow)
    High,
    /// Print quality (very slow)
    Print,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::HTML
    }
}

impl Default for Quality {
    fn default() -> Self {
        Self::Standard
    }
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            dpi: 96.0,
            background_color: [1.0, 1.0, 1.0, 1.0],
            quality: Quality::default(),
            include_metadata: true,
        }
    }
}

impl JupyterBackend {
    /// Create a new Jupyter backend
    pub fn new() -> Self {
        Self {
            output_format: OutputFormat::default(),
            interactive_mode: true,
            export_settings: ExportSettings::default(),
        }
    }

    /// Create backend with specific output format
    pub fn with_format(format: OutputFormat) -> Self {
        let mut backend = Self::new();
        backend.output_format = format;
        backend
    }

    /// Set interactive mode
    pub fn set_interactive(&mut self, interactive: bool) {
        self.interactive_mode = interactive;
    }

    /// Set export settings
    pub fn set_export_settings(&mut self, settings: ExportSettings) {
        self.export_settings = settings;
    }

    /// Display a figure in Jupyter notebook
    pub fn display_figure(&mut self, figure: &mut Figure) -> Result<String, String> {
        match self.output_format {
            OutputFormat::PNG => self.export_png(figure),
            OutputFormat::SVG => self.export_svg(figure),
            OutputFormat::HTML => self.export_html_widget(figure),
            OutputFormat::Base64 => self.export_base64(figure),
            OutputFormat::PlotlyJSON => self.export_plotly_json(figure),
        }
    }

    /// Display a line plot
    pub fn display_line_plot(&mut self, plot: &LinePlot) -> Result<String, String> {
        let mut figure = Figure::new();
        figure.add_line_plot(plot.clone());
        self.display_figure(&mut figure)
    }

    /// Display a scatter plot
    pub fn display_scatter_plot(&mut self, plot: &ScatterPlot) -> Result<String, String> {
        let mut figure = Figure::new();
        figure.add_scatter_plot(plot.clone());
        self.display_figure(&mut figure)
    }

    /// Display a surface plot
    pub fn display_surface_plot(&mut self, _plot: &SurfacePlot) -> Result<String, String> {
        // TODO: Implement once Figure supports 3D plots
        Ok("<div>3D Surface Plot (not yet integrated with Figure)</div>".to_string())
    }

    // Scatter3 display not yet implemented for the new renderer

    // Session IDs removed as not currently used

    /// Export as PNG image using our GPU-accelerated export system
    fn export_png(&self, figure: &mut Figure) -> Result<String, String> {
        use crate::export::ImageExporter;

        let output_path =
            std::env::temp_dir().join(format!("runmat_plot_{}.png", Self::generate_plot_id()));

        // Use our high-performance GPU export system
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create async runtime: {e}"))?;

        runtime.block_on(async {
            let exporter = ImageExporter::new()
                .await
                .map_err(|e| format!("Failed to create image exporter: {e}"))?;

            exporter
                .export_png(figure, &output_path)
                .await
                .map_err(|e| format!("Failed to export PNG: {e}"))?;

            Ok::<(), String>(())
        })?;

        // Return HTML img tag for Jupyter
        let output_path_str = output_path.to_string_lossy();
        Ok(format!(
            "<img src='{}' alt='RunMat Plot' width='{}' height='{}' />",
            output_path_str, self.export_settings.width, self.export_settings.height
        ))
    }

    /// Export as SVG image using our vector export system
    fn export_svg(&self, figure: &mut Figure) -> Result<String, String> {
        use crate::export::VectorExporter;

        let exporter = VectorExporter::new();
        let svg_content = exporter.render_to_svg(figure)?;

        // Return SVG directly for Jupyter
        Ok(svg_content)
    }

    /// Export as interactive HTML widget using our web export system
    fn export_html_widget(&self, _figure: &mut Figure) -> Result<String, String> {
        use crate::export::WebExporter;

        let mut exporter = WebExporter::new();
        let html_content = exporter.render_to_html()?;

        Ok(html_content)
    }

    /// Export as base64 encoded image using our PNG export system
    fn export_base64(&self, figure: &mut Figure) -> Result<String, String> {
        use crate::export::ImageExporter;

        // Create temporary file for PNG export
        let temp_path =
            std::env::temp_dir().join(format!("runmat_base64_{}.png", Self::generate_plot_id()));

        // Export PNG first
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create async runtime: {e}"))?;

        runtime.block_on(async {
            let exporter = ImageExporter::new()
                .await
                .map_err(|e| format!("Failed to create image exporter: {e}"))?;

            exporter
                .export_png(figure, &temp_path)
                .await
                .map_err(|e| format!("Failed to export PNG: {e}"))?;

            Ok::<(), String>(())
        })?;

        // Read PNG data and encode as base64
        let png_data =
            std::fs::read(&temp_path).map_err(|e| format!("Failed to read PNG file: {e}"))?;

        let base64_data = base64_encode(&png_data);

        // Clean up temporary file
        let _ = std::fs::remove_file(&temp_path);

        // Return data URL for Jupyter
        Ok(format!(
            "<img src='data:image/png;base64,{}' alt='RunMat Plot' width='{}' height='{}' />",
            base64_data, self.export_settings.width, self.export_settings.height
        ))
    }

    /// Export as Plotly-compatible JSON
    fn export_plotly_json(&self, figure: &mut Figure) -> Result<String, String> {
        // Convert Figure to Plotly JSON format
        let plotly_data = self.convert_to_plotly_format(figure)?;

        let html = format!(
            r#"
            <div id="plotly_div_{}" style="width: {}px; height: {}px;"></div>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                Plotly.newPlot('plotly_div_{}', {}, {{}});
            </script>
            "#,
            Self::generate_plot_id(),
            self.export_settings.width,
            self.export_settings.height,
            Self::generate_plot_id(),
            plotly_data
        );

        Ok(html)
    }

    /// Generate unique plot ID
    fn generate_plot_id() -> String {
        let timestamp = unix_timestamp_us();
        format!("{timestamp}")
    }

    /// Serialize figure data for JavaScript
    /// Note: Will be used for WebAssembly widget serialization
    #[allow(dead_code)]
    fn serialize_figure_data(&self, _figure: &Figure) -> Result<String, String> {
        // TODO: Implement proper serialization
        Ok("{}".to_string())
    }

    /// Serialize plot options for JavaScript
    /// Note: Will be used for WebAssembly widget configuration
    #[allow(dead_code)]
    fn serialize_plot_options(&self) -> Result<String, String> {
        // TODO: Implement proper serialization
        Ok("{}".to_string())
    }

    /// Convert figure to Plotly format
    fn convert_to_plotly_format(&self, _figure: &Figure) -> Result<String, String> {
        // TODO: Implement Plotly conversion
        Ok("[]".to_string())
    }
}

impl Default for JupyterBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for Jupyter integration
pub mod utils {
    use super::*;

    /// Check if running in Jupyter environment
    pub fn is_jupyter_environment() -> bool {
        std::env::var("JPY_PARENT_PID").is_ok() || std::env::var("JUPYTER_RUNTIME_DIR").is_ok()
    }

    /// Get Jupyter kernel information
    pub fn get_kernel_info() -> Option<KernelInfo> {
        if !is_jupyter_environment() {
            return None;
        }

        Some(KernelInfo {
            kernel_type: detect_kernel_type(),
            session_id: std::env::var("JPY_SESSION_NAME").ok(),
            runtime_dir: std::env::var("JUPYTER_RUNTIME_DIR").ok(),
        })
    }

    /// Detect the type of Jupyter kernel
    fn detect_kernel_type() -> KernelType {
        if std::env::var("IPYKERNEL").is_ok() {
            KernelType::IPython
        } else if std::env::var("IRUST_JUPYTER").is_ok() {
            KernelType::Rust
        } else {
            KernelType::Unknown
        }
    }

    /// Auto-configure backend for current environment
    pub fn auto_configure_backend() -> JupyterBackend {
        if is_jupyter_environment() {
            JupyterBackend::with_format(OutputFormat::HTML)
        } else {
            JupyterBackend::with_format(OutputFormat::PNG)
        }
    }
}

/// Jupyter kernel information
#[derive(Debug, Clone)]
pub struct KernelInfo {
    pub kernel_type: KernelType,
    pub session_id: Option<String>,
    pub runtime_dir: Option<String>,
}

/// Types of Jupyter kernels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelType {
    IPython,
    Rust,
    Unknown,
}

/// Simple base64 encoding using standard library
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();

    for chunk in data.chunks(3) {
        let mut buf = [0u8; 3];
        for (i, &byte) in chunk.iter().enumerate() {
            buf[i] = byte;
        }

        let b = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | (buf[2] as u32);

        result.push(CHARS[((b >> 18) & 63) as usize] as char);
        result.push(CHARS[((b >> 12) & 63) as usize] as char);
        result.push(if chunk.len() > 1 {
            CHARS[((b >> 6) & 63) as usize] as char
        } else {
            '='
        });
        result.push(if chunk.len() > 2 {
            CHARS[(b & 63) as usize] as char
        } else {
            '='
        });
    }

    result
}

/// Extension trait for easy Jupyter integration
pub trait JupyterDisplay {
    /// Display this object in Jupyter notebook
    fn display(&self) -> Result<String, String>;

    /// Display with specific format
    fn display_as(&self, format: OutputFormat) -> Result<String, String>;
}

// TODO: Implement JupyterDisplay trait once borrowing issues are resolved

// TODO: Implement actual export functions in the main simple_plots module

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plots::LinePlot;

    #[test]
    fn test_jupyter_backend_creation() {
        let backend = JupyterBackend::new();

        assert_eq!(backend.output_format, OutputFormat::HTML);
        assert!(backend.interactive_mode);
    }

    #[test]
    fn test_jupyter_backend_with_format() {
        let backend = JupyterBackend::with_format(OutputFormat::PNG);

        assert_eq!(backend.output_format, OutputFormat::PNG);
    }

    #[test]
    fn test_export_settings() {
        let settings = ExportSettings::default();

        assert_eq!(settings.width, 800);
        assert_eq!(settings.height, 600);
        assert_eq!(settings.quality, Quality::Standard);
        assert!(settings.include_metadata);
    }

    #[test]
    fn test_jupyter_environment_detection() {
        // This will be false in test environment
        assert!(!utils::is_jupyter_environment());
    }

    #[test]
    fn test_auto_configure_backend() {
        let backend = utils::auto_configure_backend();

        // Should default to PNG when not in Jupyter
        assert_eq!(backend.output_format, OutputFormat::PNG);
    }

    #[test]
    fn test_jupyter_backend_functionality() {
        let line_plot = LinePlot::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
        let mut backend = JupyterBackend::new();

        // Should not panic and return some output
        let result = backend.display_line_plot(&line_plot);
        assert!(result.is_ok());
    }

    #[test]
    fn test_widget_state() {
        let state = WidgetState {
            widget_id: "test_widget".to_string(),
            camera_position: [0.0, 0.0, 5.0],
            camera_target: [0.0, 0.0, 0.0],
            zoom_level: 1.0,
            visible_plots: vec![true, false, true],
            style_overrides: HashMap::new(),
            interactive: true,
        };

        assert_eq!(state.widget_id, "test_widget");
        assert_eq!(state.camera_position, [0.0, 0.0, 5.0]);
        assert!(state.interactive);
    }
}
