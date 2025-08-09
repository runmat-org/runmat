//! Jupyter plotting integration for the RunMat kernel
//!
//! Provides seamless integration between the RunMat kernel and the plotting system,
//! enabling automatic plot display in Jupyter notebooks.

#[cfg(feature = "jupyter")]
use crate::{KernelError, Result};
#[cfg(feature = "jupyter")]
use runmat_plot::jupyter::{JupyterBackend, OutputFormat};
#[cfg(feature = "jupyter")]
use runmat_plot::plots::Figure;
#[cfg(feature = "jupyter")]
use serde_json::Value as JsonValue;
#[cfg(feature = "jupyter")]
use std::collections::HashMap;

/// Jupyter plotting manager for the RunMat kernel
#[cfg(feature = "jupyter")]
pub struct JupyterPlottingManager {
    /// Jupyter backend for rendering plots
    backend: JupyterBackend,
    /// Plotting configuration from global config
    config: JupyterPlottingConfig,
    /// Active plots in the current session
    active_plots: HashMap<String, Figure>,
    /// Plot counter for unique IDs
    plot_counter: u64,
}

/// Jupyter plotting configuration
#[cfg(feature = "jupyter")]
#[derive(Debug, Clone)]
pub struct JupyterPlottingConfig {
    /// Default output format
    pub output_format: OutputFormat,
    /// Auto-display plots after creation
    pub auto_display: bool,
    /// Maximum number of plots to keep in memory
    pub max_plots: usize,
    /// Enable inline display
    pub inline_display: bool,
    /// Image width for exports
    pub image_width: u32,
    /// Image height for exports
    pub image_height: u32,
}

/// Display data for Jupyter protocol
#[derive(Debug, Clone)]
pub struct DisplayData {
    /// MIME type -> content mapping
    pub data: HashMap<String, JsonValue>,
    /// Metadata for the display
    pub metadata: HashMap<String, JsonValue>,
    /// Transient data
    pub transient: HashMap<String, JsonValue>,
}

#[cfg(feature = "jupyter")]
impl Default for JupyterPlottingConfig {
    fn default() -> Self {
        Self {
            output_format: OutputFormat::HTML,
            auto_display: true,
            max_plots: 100,
            inline_display: true,
            image_width: 800,
            image_height: 600,
        }
    }
}

#[cfg(feature = "jupyter")]
impl JupyterPlottingManager {
    /// Create a new Jupyter plotting manager
    pub fn new() -> Self {
        Self::with_config(JupyterPlottingConfig::default())
    }

    /// Create manager with specific configuration
    pub fn with_config(config: JupyterPlottingConfig) -> Self {
        let backend = match config.output_format {
            OutputFormat::HTML => JupyterBackend::with_format(OutputFormat::HTML),
            OutputFormat::PNG => JupyterBackend::with_format(OutputFormat::PNG),
            OutputFormat::SVG => JupyterBackend::with_format(OutputFormat::SVG),
            OutputFormat::Base64 => JupyterBackend::with_format(OutputFormat::Base64),
            OutputFormat::PlotlyJSON => JupyterBackend::with_format(OutputFormat::PlotlyJSON),
        };

        Self {
            backend,
            config,
            active_plots: HashMap::new(),
            plot_counter: 0,
        }
    }

    /// Register a new plot and optionally display it
    pub fn register_plot(&mut self, mut figure: Figure) -> Result<Option<DisplayData>> {
        self.plot_counter += 1;
        let plot_id = format!("plot_{}", self.plot_counter);

        // Store the plot
        self.active_plots.insert(plot_id.clone(), figure.clone());

        // Clean up old plots if we exceed the maximum
        if self.active_plots.len() > self.config.max_plots {
            self.cleanup_old_plots();
        }

        // Auto-display if enabled
        if self.config.auto_display && self.config.inline_display {
            let display_data = self.create_display_data(&mut figure)?;
            Ok(Some(display_data))
        } else {
            Ok(None)
        }
    }

    /// Create display data for a figure
    pub fn create_display_data(&mut self, figure: &mut Figure) -> Result<DisplayData> {
        let mut data = HashMap::new();
        let mut metadata = HashMap::new();

        // Generate content based on output format
        match self.config.output_format {
            OutputFormat::HTML => {
                let html_content = self
                    .backend
                    .display_figure(figure)
                    .map_err(|e| KernelError::Execution(format!("HTML generation failed: {e}")))?;

                data.insert("text/html".to_string(), JsonValue::String(html_content));
                metadata.insert(
                    "text/html".to_string(),
                    JsonValue::Object({
                        let mut meta = serde_json::Map::new();
                        meta.insert("isolated".to_string(), JsonValue::Bool(true));
                        meta.insert(
                            "width".to_string(),
                            JsonValue::Number(self.config.image_width.into()),
                        );
                        meta.insert(
                            "height".to_string(),
                            JsonValue::Number(self.config.image_height.into()),
                        );
                        meta
                    }),
                );
            }
            OutputFormat::PNG => {
                let png_content = self
                    .backend
                    .display_figure(figure)
                    .map_err(|e| KernelError::Execution(format!("PNG generation failed: {e}")))?;

                data.insert("text/html".to_string(), JsonValue::String(png_content));
            }
            OutputFormat::SVG => {
                let svg_content = self
                    .backend
                    .display_figure(figure)
                    .map_err(|e| KernelError::Execution(format!("SVG generation failed: {e}")))?;

                data.insert("image/svg+xml".to_string(), JsonValue::String(svg_content));
                metadata.insert(
                    "image/svg+xml".to_string(),
                    JsonValue::Object({
                        let mut meta = serde_json::Map::new();
                        meta.insert("isolated".to_string(), JsonValue::Bool(true));
                        meta
                    }),
                );
            }
            OutputFormat::Base64 => {
                let base64_content = self.backend.display_figure(figure).map_err(|e| {
                    KernelError::Execution(format!("Base64 generation failed: {e}"))
                })?;

                data.insert("text/html".to_string(), JsonValue::String(base64_content));
            }
            OutputFormat::PlotlyJSON => {
                let plotly_content = self.backend.display_figure(figure).map_err(|e| {
                    KernelError::Execution(format!("Plotly generation failed: {e}"))
                })?;

                data.insert("text/html".to_string(), JsonValue::String(plotly_content));
                metadata.insert(
                    "text/html".to_string(),
                    JsonValue::Object({
                        let mut meta = serde_json::Map::new();
                        meta.insert("isolated".to_string(), JsonValue::Bool(true));
                        meta
                    }),
                );
            }
        }

        // Add RunMat metadata
        let mut transient = HashMap::new();
        transient.insert(
            "runmat_plot_id".to_string(),
            JsonValue::String(format!("plot_{}", self.plot_counter)),
        );
        transient.insert(
            "runmat_version".to_string(),
            JsonValue::String("0.0.1".to_string()),
        );

        Ok(DisplayData {
            data,
            metadata,
            transient,
        })
    }

    /// Get a plot by ID
    pub fn get_plot(&self, plot_id: &str) -> Option<&Figure> {
        self.active_plots.get(plot_id)
    }

    /// List all active plots
    pub fn list_plots(&self) -> Vec<String> {
        self.active_plots.keys().cloned().collect()
    }

    /// Clear all plots
    pub fn clear_plots(&mut self) {
        self.active_plots.clear();
        self.plot_counter = 0;
    }

    /// Update configuration
    pub fn update_config(&mut self, config: JupyterPlottingConfig) {
        self.config = config;

        // Update backend format if needed
        self.backend = match self.config.output_format {
            OutputFormat::HTML => JupyterBackend::with_format(OutputFormat::HTML),
            OutputFormat::PNG => JupyterBackend::with_format(OutputFormat::PNG),
            OutputFormat::SVG => JupyterBackend::with_format(OutputFormat::SVG),
            OutputFormat::Base64 => JupyterBackend::with_format(OutputFormat::Base64),
            OutputFormat::PlotlyJSON => JupyterBackend::with_format(OutputFormat::PlotlyJSON),
        };
    }

    /// Get current configuration
    pub fn config(&self) -> &JupyterPlottingConfig {
        &self.config
    }

    /// Clean up old plots to maintain memory limits
    fn cleanup_old_plots(&mut self) {
        // Simple cleanup: remove oldest plots
        let mut plot_ids: Vec<String> = self.active_plots.keys().cloned().collect();
        plot_ids.sort();

        while self.active_plots.len() > self.config.max_plots {
            if let Some(oldest_id) = plot_ids.first() {
                self.active_plots.remove(oldest_id);
                plot_ids.remove(0);
            } else {
                break;
            }
        }
    }

    /// Handle plot function calls from MATLAB code
    pub fn handle_plot_function(
        &mut self,
        function_name: &str,
        args: &[JsonValue],
    ) -> Result<Option<DisplayData>> {
        println!(
            "DEBUG: Handling plot function '{}' with {} args",
            function_name,
            args.len()
        );

        // Create appropriate plot based on function name
        let mut figure = Figure::new();

        match function_name {
            "plot" => {
                if args.len() >= 2 {
                    // Extract x and y data from arguments
                    let x_data = self.extract_numeric_array(&args[0])?;
                    let y_data = self.extract_numeric_array(&args[1])?;

                    if x_data.len() == y_data.len() {
                        let line_plot =
                            runmat_plot::plots::LinePlot::new(x_data, y_data).map_err(|e| {
                                KernelError::Execution(format!("Failed to create line plot: {e}"))
                            })?;
                        figure.add_line_plot(line_plot);
                    } else {
                        return Err(KernelError::Execution(
                            "X and Y data must have the same length".to_string(),
                        ));
                    }
                }
            }
            "scatter" => {
                if args.len() >= 2 {
                    let x_data = self.extract_numeric_array(&args[0])?;
                    let y_data = self.extract_numeric_array(&args[1])?;

                    if x_data.len() == y_data.len() {
                        let scatter_plot = runmat_plot::plots::ScatterPlot::new(x_data, y_data)
                            .map_err(KernelError::Execution)?;
                        figure.add_scatter_plot(scatter_plot);
                    } else {
                        return Err(KernelError::Execution(
                            "X and Y data must have the same length".to_string(),
                        ));
                    }
                }
            }
            "bar" => {
                if !args.is_empty() {
                    let y_data = self.extract_numeric_array(&args[0])?;
                    let x_labels: Vec<String> = (0..y_data.len()).map(|i| format!("{i}")).collect();

                    let bar_chart = runmat_plot::plots::BarChart::new(x_labels, y_data)
                        .map_err(KernelError::Execution)?;
                    figure.add_bar_chart(bar_chart);
                }
            }
            "hist" => {
                if !args.is_empty() {
                    let data = self.extract_numeric_array(&args[0])?;
                    let bins = if args.len() > 1 {
                        self.extract_number(&args[1])? as usize
                    } else {
                        20
                    };

                    let histogram = runmat_plot::plots::Histogram::new(data, bins)
                        .map_err(KernelError::Execution)?;
                    figure.add_histogram(histogram);
                }
            }
            _ => {
                return Err(KernelError::Execution(format!(
                    "Unknown plot function: {function_name}"
                )));
            }
        }

        // Register and potentially display the plot
        self.register_plot(figure)
    }

    /// Extract numeric array from JSON value
    fn extract_numeric_array(&self, value: &JsonValue) -> Result<Vec<f64>> {
        match value {
            JsonValue::Array(arr) => {
                let mut result = Vec::new();
                for item in arr {
                    if let Some(num) = item.as_f64() {
                        result.push(num);
                    } else if let Some(num) = item.as_i64() {
                        result.push(num as f64);
                    } else {
                        return Err(KernelError::Execution(
                            "Array must contain only numbers".to_string(),
                        ));
                    }
                }
                Ok(result)
            }
            JsonValue::Number(num) => {
                if let Some(val) = num.as_f64() {
                    Ok(vec![val])
                } else {
                    Err(KernelError::Execution("Invalid number format".to_string()))
                }
            }
            _ => Err(KernelError::Execution(
                "Expected array or number".to_string(),
            )),
        }
    }

    /// Extract single number from JSON value
    fn extract_number(&self, value: &JsonValue) -> Result<f64> {
        match value {
            JsonValue::Number(num) => num
                .as_f64()
                .ok_or_else(|| KernelError::Execution("Invalid number format".to_string())),
            _ => Err(KernelError::Execution("Expected number".to_string())),
        }
    }
}

impl Default for JupyterPlottingManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for ExecutionEngine to add Jupyter plotting support
pub trait JupyterPlottingExtension {
    /// Handle plot functions with automatic Jupyter display
    fn handle_jupyter_plot(
        &mut self,
        function_name: &str,
        args: &[JsonValue],
    ) -> Result<Option<DisplayData>>;

    /// Get the plotting manager
    fn plotting_manager(&mut self) -> &mut JupyterPlottingManager;
}

// Note: This would need to be implemented on ExecutionEngine in the actual integration
// For now, we provide the trait definition and placeholder implementation

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jupyter_plotting_manager_creation() {
        let manager = JupyterPlottingManager::new();
        assert_eq!(manager.config.output_format, OutputFormat::HTML);
        assert!(manager.config.auto_display);
        assert_eq!(manager.active_plots.len(), 0);
    }

    #[test]
    fn test_config_update() {
        let mut manager = JupyterPlottingManager::new();

        let new_config = JupyterPlottingConfig {
            output_format: OutputFormat::SVG,
            auto_display: false,
            max_plots: 50,
            inline_display: false,
            image_width: 1024,
            image_height: 768,
        };

        manager.update_config(new_config.clone());
        assert_eq!(manager.config.output_format, OutputFormat::SVG);
        assert!(!manager.config.auto_display);
        assert_eq!(manager.config.max_plots, 50);
    }

    #[test]
    fn test_plot_management() {
        let mut manager = JupyterPlottingManager::new();
        let figure = Figure::new().with_title("Test Plot");

        // Register a plot
        let display_data = manager.register_plot(figure).unwrap();
        assert!(display_data.is_some());
        assert_eq!(manager.active_plots.len(), 1);
        assert_eq!(manager.list_plots().len(), 1);

        // Clear plots
        manager.clear_plots();
        assert_eq!(manager.active_plots.len(), 0);
        assert_eq!(manager.plot_counter, 0);
    }

    #[test]
    fn test_extract_numeric_array() {
        let manager = JupyterPlottingManager::new();

        let json_array = JsonValue::Array(vec![
            JsonValue::Number(serde_json::Number::from(1)),
            JsonValue::Number(serde_json::Number::from(2)),
            JsonValue::Number(serde_json::Number::from(3)),
        ]);

        let result = manager.extract_numeric_array(&json_array).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_plot_function_handling() {
        let mut manager = JupyterPlottingManager::new();

        let x_data = JsonValue::Array(vec![
            JsonValue::Number(serde_json::Number::from(1)),
            JsonValue::Number(serde_json::Number::from(2)),
            JsonValue::Number(serde_json::Number::from(3)),
        ]);

        let y_data = JsonValue::Array(vec![
            JsonValue::Number(serde_json::Number::from(2)),
            JsonValue::Number(serde_json::Number::from(4)),
            JsonValue::Number(serde_json::Number::from(6)),
        ]);

        let result = manager
            .handle_plot_function("plot", &[x_data, y_data])
            .unwrap();
        assert!(result.is_some());
        assert_eq!(manager.active_plots.len(), 1);
    }
}
