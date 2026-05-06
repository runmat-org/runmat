//! Configuration system for RunMat
//!
//! Supports multiple configuration sources with proper precedence:
//! 1. Command-line arguments (highest priority)
//! 2. Environment variables
//! 3. Configuration files (.runmat.yaml, .runmat.json, etc.)
//! 4. Built-in defaults (lowest priority)

mod loader;
mod schema;

pub use loader::ConfigLoader;
pub use schema::{
    error_namespace_for_language_compat, AccelPowerPreference, AccelerateConfig,
    AccelerateProviderPreference, AutoOffloadConfig, AutoOffloadLogLevel, ExportConfig,
    ExportFormat, GcConfig, GcPreset, GuiConfig, JitConfig, JitOptLevel, JupyterConfig,
    JupyterOutputFormat, JupyterPerformanceConfig, JupyterStaticConfig, JupyterWidgetConfig,
    KernelConfig, KernelPorts, LanguageCompatMode, LanguageConfig, LogLevel, LoggingConfig,
    PlotBackend, PlotMode, PlottingConfig, RunMatConfig, RuntimeConfig, TelemetryConfig,
    TelemetryDrainMode,
};
