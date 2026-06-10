use serde::{Deserialize, Serialize};

use super::{
    AccelerateConfig, AnalysisConfig, GcConfig, JitConfig, LanguageConfig, LoggingConfig,
    PlottingConfig, RuntimeConfig, TelemetryConfig,
};

/// Main RunMat configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RunMatRuntimeConfig {
    /// Runtime configuration
    #[serde(default)]
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
    #[serde(default)]
    pub jit: JitConfig,
    /// Garbage collector configuration
    #[serde(default)]
    pub gc: GcConfig,
    /// Plotting configuration
    #[serde(default)]
    pub plotting: PlottingConfig,
    /// Analysis and geometry artifact configuration
    #[serde(default)]
    pub analysis: AnalysisConfig,
    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,
}
