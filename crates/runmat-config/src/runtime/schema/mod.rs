mod accelerate;
mod defaults;
mod execution;
mod language;
mod logging;
mod plotting;
mod root;
mod telemetry;

pub use accelerate::{
    AccelPowerPreference, AccelerateConfig, AccelerateProviderPreference, AutoOffloadConfig,
    AutoOffloadLogLevel,
};
pub use execution::{GcConfig, GcPreset, JitConfig, JitOptLevel, RuntimeConfig};
pub use language::{error_namespace_for_language_compat, LanguageCompatMode, LanguageConfig};
pub use logging::{LogLevel, LoggingConfig};
pub use plotting::{ExportConfig, ExportFormat, GuiConfig, PlotBackend, PlotMode, PlottingConfig};
pub use root::RunMatRuntimeConfig;
pub use telemetry::{TelemetryConfig, TelemetryDrainMode};
