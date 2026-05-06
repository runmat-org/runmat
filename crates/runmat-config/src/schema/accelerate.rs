use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::defaults::default_true;

/// Acceleration (GPU) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerateConfig {
    /// Enable acceleration subsystem
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Preferred provider (auto, wgpu, inprocess)
    #[serde(default)]
    pub provider: AccelerateProviderPreference,
    /// Allow automatic fallback to the in-process provider when hardware backend fails
    #[serde(default = "default_true")]
    pub allow_inprocess_fallback: bool,
    /// Preferred WGPU power profile
    #[serde(default)]
    pub wgpu_power_preference: AccelPowerPreference,
    /// Force use of WGPU fallback adapter even if a high-performance adapter exists
    #[serde(default)]
    pub wgpu_force_fallback_adapter: bool,
    /// Auto-offload planner configuration
    #[serde(default)]
    pub auto_offload: AutoOffloadConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum AccelerateProviderPreference {
    Auto,
    Wgpu,
    InProcess,
}

impl Default for AccelerateProviderPreference {
    fn default() -> Self {
        Self::Wgpu
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum AccelPowerPreference {
    Auto,
    HighPerformance,
    LowPower,
}

impl Default for AccelPowerPreference {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum, Default)]
#[serde(rename_all = "kebab-case")]
pub enum AutoOffloadLogLevel {
    Off,
    Info,
    #[default]
    Trace,
}

/// Auto-offload planner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoOffloadConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_true")]
    pub calibrate: bool,
    pub profile_path: Option<PathBuf>,
    #[serde(default)]
    pub log_level: AutoOffloadLogLevel,
}

impl Default for AccelerateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: AccelerateProviderPreference::Wgpu,
            allow_inprocess_fallback: true,
            wgpu_power_preference: AccelPowerPreference::Auto,
            wgpu_force_fallback_adapter: false,
            auto_offload: AutoOffloadConfig::default(),
        }
    }
}

impl Default for AutoOffloadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            calibrate: true,
            profile_path: None,
            log_level: AutoOffloadLogLevel::Trace,
        }
    }
}

#[cfg(feature = "accelerate")]
mod runtime_conversions {
    use super::{
        AccelPowerPreference, AccelerateConfig, AccelerateProviderPreference, AutoOffloadConfig,
        AutoOffloadLogLevel,
    };
    use runmat_accelerate::{
        AccelPowerPreference as RuntimePowerPreference, AccelerateInitOptions,
        AccelerateProviderPreference as RuntimeProviderPreference,
        AutoOffloadLogLevel as RuntimeAutoLogLevel, AutoOffloadOptions,
    };

    impl From<AccelPowerPreference> for RuntimePowerPreference {
        fn from(pref: AccelPowerPreference) -> Self {
            match pref {
                AccelPowerPreference::Auto => RuntimePowerPreference::Auto,
                AccelPowerPreference::HighPerformance => RuntimePowerPreference::HighPerformance,
                AccelPowerPreference::LowPower => RuntimePowerPreference::LowPower,
            }
        }
    }

    impl From<AccelerateProviderPreference> for RuntimeProviderPreference {
        fn from(pref: AccelerateProviderPreference) -> Self {
            match pref {
                AccelerateProviderPreference::Auto => RuntimeProviderPreference::Auto,
                AccelerateProviderPreference::Wgpu => RuntimeProviderPreference::Wgpu,
                AccelerateProviderPreference::InProcess => RuntimeProviderPreference::InProcess,
            }
        }
    }

    impl From<AutoOffloadLogLevel> for RuntimeAutoLogLevel {
        fn from(level: AutoOffloadLogLevel) -> Self {
            match level {
                AutoOffloadLogLevel::Off => RuntimeAutoLogLevel::Off,
                AutoOffloadLogLevel::Info => RuntimeAutoLogLevel::Info,
                AutoOffloadLogLevel::Trace => RuntimeAutoLogLevel::Trace,
            }
        }
    }

    impl From<&AutoOffloadConfig> for AutoOffloadOptions {
        fn from(cfg: &AutoOffloadConfig) -> Self {
            AutoOffloadOptions {
                enabled: cfg.enabled,
                calibrate: cfg.calibrate,
                profile_path: cfg.profile_path.clone(),
                log_level: cfg.log_level.into(),
            }
        }
    }

    impl From<&AccelerateConfig> for AccelerateInitOptions {
        fn from(cfg: &AccelerateConfig) -> Self {
            AccelerateInitOptions {
                enabled: cfg.enabled,
                provider: cfg.provider.into(),
                allow_inprocess_fallback: cfg.allow_inprocess_fallback,
                wgpu_power_preference: cfg.wgpu_power_preference.into(),
                wgpu_force_fallback_adapter: cfg.wgpu_force_fallback_adapter,
                auto_offload: AutoOffloadOptions::from(&cfg.auto_offload),
            }
        }
    }
}
