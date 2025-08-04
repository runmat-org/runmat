//! Snapshot presets for common use cases
//!
//! Provides pre-configured snapshot settings optimized for different
//! deployment scenarios and performance requirements.

use crate::{CompressionAlgorithm, SnapshotConfig};
use std::time::Duration;

/// Snapshot preset configurations
#[derive(Debug, Clone)]
pub enum SnapshotPreset {
    /// Fast development iteration
    Development,

    /// Production deployment
    Production,

    /// High-performance computing
    HighPerformance,

    /// Memory-constrained environments
    LowMemory,

    /// Network-optimized (minimal size)
    NetworkOptimized,

    /// Debug-friendly (maximum validation)
    Debug,

    /// Custom configuration
    Custom(SnapshotConfig),
}

impl SnapshotPreset {
    /// Get configuration for preset
    pub fn config(&self) -> SnapshotConfig {
        match self {
            SnapshotPreset::Development => Self::development_config(),
            SnapshotPreset::Production => Self::production_config(),
            SnapshotPreset::HighPerformance => Self::high_performance_config(),
            SnapshotPreset::LowMemory => Self::low_memory_config(),
            SnapshotPreset::NetworkOptimized => Self::network_optimized_config(),
            SnapshotPreset::Debug => Self::debug_config(),
            SnapshotPreset::Custom(config) => config.clone(),
        }
    }

    /// Development preset - fast build, minimal compression
    fn development_config() -> SnapshotConfig {
        SnapshotConfig {
            compression_enabled: true,
            compression_algorithm: CompressionAlgorithm::Lz4,
            compression_level: 1, // Fast compression
            validation_enabled: true,
            memory_mapping_enabled: true,
            parallel_loading: true,
            progress_reporting: true,         // Helpful during development
            max_cache_size: 64 * 1024 * 1024, // 64MB
            cache_eviction_policy: crate::CacheEvictionPolicy::LeastRecentlyUsed,
        }
    }

    /// Production preset - balanced performance and size
    fn production_config() -> SnapshotConfig {
        SnapshotConfig {
            compression_enabled: true,
            compression_algorithm: CompressionAlgorithm::Auto,
            compression_level: 6, // Balanced compression
            validation_enabled: true,
            memory_mapping_enabled: true,
            parallel_loading: true,
            progress_reporting: false,         // No progress in production
            max_cache_size: 128 * 1024 * 1024, // 128MB
            cache_eviction_policy: crate::CacheEvictionPolicy::Adaptive,
        }
    }

    /// High-performance preset - optimized for speed
    fn high_performance_config() -> SnapshotConfig {
        SnapshotConfig {
            compression_enabled: true,
            compression_algorithm: CompressionAlgorithm::Lz4,
            compression_level: 1,      // Fastest decompression
            validation_enabled: false, // Skip validation for speed
            memory_mapping_enabled: true,
            parallel_loading: true,
            progress_reporting: false,
            max_cache_size: 256 * 1024 * 1024, // 256MB - more cache
            cache_eviction_policy: crate::CacheEvictionPolicy::LeastRecentlyUsed,
        }
    }

    /// Low-memory preset - minimal memory usage
    fn low_memory_config() -> SnapshotConfig {
        SnapshotConfig {
            compression_enabled: true,
            compression_algorithm: CompressionAlgorithm::Zstd,
            compression_level: 9, // Maximum compression
            validation_enabled: true,
            memory_mapping_enabled: false, // Avoid memory mapping
            parallel_loading: false,       // Reduce memory overhead
            progress_reporting: false,
            max_cache_size: 16 * 1024 * 1024, // 16MB - minimal cache
            cache_eviction_policy: crate::CacheEvictionPolicy::TimeToLive(Duration::from_secs(60)),
        }
    }

    /// Network-optimized preset - smallest possible size
    fn network_optimized_config() -> SnapshotConfig {
        SnapshotConfig {
            compression_enabled: true,
            compression_algorithm: CompressionAlgorithm::Zstd,
            compression_level: 9, // Maximum compression
            validation_enabled: true,
            memory_mapping_enabled: true,
            parallel_loading: true,
            progress_reporting: false,
            max_cache_size: 32 * 1024 * 1024, // 32MB
            cache_eviction_policy: crate::CacheEvictionPolicy::LeastFrequentlyUsed,
        }
    }

    /// Debug preset - maximum validation and reporting
    fn debug_config() -> SnapshotConfig {
        SnapshotConfig {
            compression_enabled: false, // No compression for easier debugging
            compression_algorithm: CompressionAlgorithm::None,
            compression_level: 0,
            validation_enabled: true,
            memory_mapping_enabled: false, // Easier to debug without mmap
            parallel_loading: false,       // Sequential for easier debugging
            progress_reporting: true,      // Detailed progress
            max_cache_size: 128 * 1024 * 1024, // 128MB
            cache_eviction_policy: crate::CacheEvictionPolicy::LeastRecentlyUsed,
        }
    }

    /// Get all available presets
    pub fn all_presets() -> Vec<SnapshotPreset> {
        vec![
            SnapshotPreset::Development,
            SnapshotPreset::Production,
            SnapshotPreset::HighPerformance,
            SnapshotPreset::LowMemory,
            SnapshotPreset::NetworkOptimized,
            SnapshotPreset::Debug,
        ]
    }

    /// Get preset by name
    pub fn from_name(name: &str) -> Option<SnapshotPreset> {
        match name.to_lowercase().as_str() {
            "development" | "dev" => Some(SnapshotPreset::Development),
            "production" | "prod" => Some(SnapshotPreset::Production),
            "high-performance" | "highperf" | "performance" => {
                Some(SnapshotPreset::HighPerformance)
            }
            "low-memory" | "lowmem" | "minimal" => Some(SnapshotPreset::LowMemory),
            "network-optimized" | "network" | "small" => Some(SnapshotPreset::NetworkOptimized),
            "debug" => Some(SnapshotPreset::Debug),
            _ => None,
        }
    }

    /// Get preset name
    pub fn name(&self) -> &'static str {
        match self {
            SnapshotPreset::Development => "Development",
            SnapshotPreset::Production => "Production",
            SnapshotPreset::HighPerformance => "High-Performance",
            SnapshotPreset::LowMemory => "Low-Memory",
            SnapshotPreset::NetworkOptimized => "Network-Optimized",
            SnapshotPreset::Debug => "Debug",
            SnapshotPreset::Custom(_) => "Custom",
        }
    }

    /// Get preset description
    pub fn description(&self) -> &'static str {
        match self {
            SnapshotPreset::Development => {
                "Fast build times with minimal compression, ideal for development iteration"
            }
            SnapshotPreset::Production => {
                "Balanced performance and size, recommended for production deployments"
            }
            SnapshotPreset::HighPerformance => {
                "Optimized for fastest loading times, minimal validation overhead"
            }
            SnapshotPreset::LowMemory => {
                "Minimal memory usage with maximum compression for constrained environments"
            }
            SnapshotPreset::NetworkOptimized => {
                "Smallest possible file size for network distribution"
            }
            SnapshotPreset::Debug => "Maximum validation and debugging information, no compression",
            SnapshotPreset::Custom(_) => "Custom configuration with user-specified settings",
        }
    }

    /// Get expected characteristics
    pub fn characteristics(&self) -> PresetCharacteristics {
        match self {
            SnapshotPreset::Development => PresetCharacteristics {
                build_time: BuildTime::Fast,
                load_time: LoadTime::Fast,
                file_size: FileSize::Medium,
                memory_usage: MemoryUsage::Medium,
                validation_level: ValidationLevel::Standard,
                debugging_friendly: true,
            },
            SnapshotPreset::Production => PresetCharacteristics {
                build_time: BuildTime::Medium,
                load_time: LoadTime::Fast,
                file_size: FileSize::Small,
                memory_usage: MemoryUsage::Medium,
                validation_level: ValidationLevel::Standard,
                debugging_friendly: false,
            },
            SnapshotPreset::HighPerformance => PresetCharacteristics {
                build_time: BuildTime::Fast,
                load_time: LoadTime::VeryFast,
                file_size: FileSize::Medium,
                memory_usage: MemoryUsage::High,
                validation_level: ValidationLevel::Minimal,
                debugging_friendly: false,
            },
            SnapshotPreset::LowMemory => PresetCharacteristics {
                build_time: BuildTime::Slow,
                load_time: LoadTime::Medium,
                file_size: FileSize::VerySmall,
                memory_usage: MemoryUsage::VeryLow,
                validation_level: ValidationLevel::Standard,
                debugging_friendly: false,
            },
            SnapshotPreset::NetworkOptimized => PresetCharacteristics {
                build_time: BuildTime::Slow,
                load_time: LoadTime::Medium,
                file_size: FileSize::VerySmall,
                memory_usage: MemoryUsage::Low,
                validation_level: ValidationLevel::Standard,
                debugging_friendly: false,
            },
            SnapshotPreset::Debug => PresetCharacteristics {
                build_time: BuildTime::Medium,
                load_time: LoadTime::Medium,
                file_size: FileSize::Large,
                memory_usage: MemoryUsage::Medium,
                validation_level: ValidationLevel::Comprehensive,
                debugging_friendly: true,
            },
            SnapshotPreset::Custom(_) => PresetCharacteristics {
                build_time: BuildTime::Medium,
                load_time: LoadTime::Medium,
                file_size: FileSize::Medium,
                memory_usage: MemoryUsage::Medium,
                validation_level: ValidationLevel::Standard,
                debugging_friendly: false,
            },
        }
    }
}

/// Preset characteristics for comparison
#[derive(Debug, Clone)]
pub struct PresetCharacteristics {
    pub build_time: BuildTime,
    pub load_time: LoadTime,
    pub file_size: FileSize,
    pub memory_usage: MemoryUsage,
    pub validation_level: ValidationLevel,
    pub debugging_friendly: bool,
}

/// Build time characteristics
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum BuildTime {
    VeryFast,
    Fast,
    Medium,
    Slow,
    VerySlow,
}

/// Load time characteristics
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum LoadTime {
    VeryFast,
    Fast,
    Medium,
    Slow,
    VerySlow,
}

/// File size characteristics
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum FileSize {
    VerySmall,
    Small,
    Medium,
    Large,
    VeryLarge,
}

/// Memory usage characteristics
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryUsage {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Validation level
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationLevel {
    None,
    Minimal,
    Standard,
    Comprehensive,
    Exhaustive,
}

impl std::fmt::Display for BuildTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildTime::VeryFast => write!(f, "Very Fast"),
            BuildTime::Fast => write!(f, "Fast"),
            BuildTime::Medium => write!(f, "Medium"),
            BuildTime::Slow => write!(f, "Slow"),
            BuildTime::VerySlow => write!(f, "Very Slow"),
        }
    }
}

impl std::fmt::Display for LoadTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadTime::VeryFast => write!(f, "Very Fast"),
            LoadTime::Fast => write!(f, "Fast"),
            LoadTime::Medium => write!(f, "Medium"),
            LoadTime::Slow => write!(f, "Slow"),
            LoadTime::VerySlow => write!(f, "Very Slow"),
        }
    }
}

impl std::fmt::Display for FileSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileSize::VerySmall => write!(f, "Very Small"),
            FileSize::Small => write!(f, "Small"),
            FileSize::Medium => write!(f, "Medium"),
            FileSize::Large => write!(f, "Large"),
            FileSize::VeryLarge => write!(f, "Very Large"),
        }
    }
}

impl std::fmt::Display for MemoryUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryUsage::VeryLow => write!(f, "Very Low"),
            MemoryUsage::Low => write!(f, "Low"),
            MemoryUsage::Medium => write!(f, "Medium"),
            MemoryUsage::High => write!(f, "High"),
            MemoryUsage::VeryHigh => write!(f, "Very High"),
        }
    }
}

impl std::fmt::Display for ValidationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationLevel::None => write!(f, "None"),
            ValidationLevel::Minimal => write!(f, "Minimal"),
            ValidationLevel::Standard => write!(f, "Standard"),
            ValidationLevel::Comprehensive => write!(f, "Comprehensive"),
            ValidationLevel::Exhaustive => write!(f, "Exhaustive"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_from_name() {
        assert!(matches!(
            SnapshotPreset::from_name("development"),
            Some(SnapshotPreset::Development)
        ));
        assert!(matches!(
            SnapshotPreset::from_name("prod"),
            Some(SnapshotPreset::Production)
        ));
        assert!(matches!(
            SnapshotPreset::from_name("highperf"),
            Some(SnapshotPreset::HighPerformance)
        ));
        assert!(SnapshotPreset::from_name("invalid").is_none());
    }

    #[test]
    fn test_preset_names() {
        assert_eq!(SnapshotPreset::Development.name(), "Development");
        assert_eq!(SnapshotPreset::Production.name(), "Production");
        assert_eq!(SnapshotPreset::Debug.name(), "Debug");
    }

    #[test]
    fn test_preset_configs() {
        let dev_config = SnapshotPreset::Development.config();
        assert!(dev_config.progress_reporting);
        assert_eq!(dev_config.compression_level, 1);

        let prod_config = SnapshotPreset::Production.config();
        assert!(!prod_config.progress_reporting);
        assert_eq!(prod_config.compression_level, 6);

        let debug_config = SnapshotPreset::Debug.config();
        assert!(!debug_config.compression_enabled);
        assert!(debug_config.validation_enabled);
    }

    #[test]
    fn test_preset_characteristics() {
        let high_perf = SnapshotPreset::HighPerformance;
        let chars = high_perf.characteristics();

        assert_eq!(chars.load_time, LoadTime::VeryFast);
        assert_eq!(chars.memory_usage, MemoryUsage::High);
        assert_eq!(chars.validation_level, ValidationLevel::Minimal);

        let low_mem = SnapshotPreset::LowMemory;
        let chars = low_mem.characteristics();

        assert_eq!(chars.memory_usage, MemoryUsage::VeryLow);
        assert_eq!(chars.file_size, FileSize::VerySmall);
    }

    #[test]
    fn test_all_presets() {
        let presets = SnapshotPreset::all_presets();
        assert_eq!(presets.len(), 6);

        // Ensure all presets have unique names
        let names: std::collections::HashSet<_> = presets.iter().map(|p| p.name()).collect();
        assert_eq!(names.len(), 6);
    }

    #[test]
    fn test_characteristic_ordering() {
        assert!(BuildTime::VeryFast < BuildTime::Fast);
        assert!(LoadTime::Fast < LoadTime::Medium);
        assert!(FileSize::Small < FileSize::Large);
        assert!(MemoryUsage::Low < MemoryUsage::High);
        assert!(ValidationLevel::Minimal < ValidationLevel::Standard);
    }
}
