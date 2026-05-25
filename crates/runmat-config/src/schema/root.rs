use serde::{Deserialize, Serialize};

use super::{
    AccelerateConfig, GcConfig, JitConfig, LanguageConfig, LoggingConfig, PlottingConfig,
    RuntimeConfig, TelemetryConfig,
};

/// Main RunMat configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RunMatConfig {
    /// Runtime configuration
    pub runtime: RuntimeConfig,
    /// Acceleration configuration
    #[serde(default)]
    pub accelerate: AccelerateConfig,
    /// Language compatibility configuration
    #[serde(default)]
    pub language: LanguageConfig,
    /// Telemetry configuration
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    /// JIT compiler configuration
    pub jit: JitConfig,
    /// Garbage collector configuration
    pub gc: GcConfig,
    /// Plotting configuration
    pub plotting: PlottingConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
}
