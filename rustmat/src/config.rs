//! Configuration system for RustMat
//! 
//! Supports multiple configuration sources with proper precedence:
//! 1. Command-line arguments (highest priority)
//! 2. Environment variables  
//! 3. Configuration files (.rustmat.yaml, .rustmat.json, etc.)
//! 4. Built-in defaults (lowest priority)

use std::path::{Path, PathBuf};
use std::fs;
use std::env;
use serde::{Deserialize, Serialize};
use anyhow::{Context, Result};
use log::{debug, info};
use clap::ValueEnum;

/// Main RustMat configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustMatConfig {
    /// Runtime configuration
    pub runtime: RuntimeConfig,
    /// JIT compiler configuration
    pub jit: JitConfig,
    /// Garbage collector configuration
    pub gc: GcConfig,
    /// Plotting configuration
    pub plotting: PlottingConfig,
    /// Kernel configuration
    pub kernel: KernelConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
}

/// Runtime execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Execution timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    /// Enable verbose output
    #[serde(default)]
    pub verbose: bool,
    /// Snapshot file to preload
    pub snapshot_path: Option<PathBuf>,
}

/// JIT compiler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitConfig {
    /// Enable JIT compilation
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// JIT compilation threshold
    #[serde(default = "default_jit_threshold")]
    pub threshold: u32,
    /// JIT optimization level
    #[serde(default)]
    pub optimization_level: JitOptLevel,
}

/// GC configuration 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcConfig {
    /// GC preset
    pub preset: Option<GcPreset>,
    /// Young generation size in MB
    pub young_size_mb: Option<usize>,
    /// Number of GC threads
    pub threads: Option<usize>,
    /// Enable GC statistics collection
    #[serde(default)]
    pub collect_stats: bool,
}

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
}

/// Kernel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Default IP address
    #[serde(default = "default_kernel_ip")]
    pub ip: String,
    /// Authentication key
    pub key: Option<String>,
    /// Port configuration
    pub ports: Option<KernelPorts>,
}

/// Kernel port configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelPorts {
    pub shell: Option<u16>,
    pub iopub: Option<u16>,
    pub stdin: Option<u16>,
    pub control: Option<u16>,
    pub heartbeat: Option<u16>,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    #[serde(default)]
    pub level: LogLevel,
    /// Enable debug logging
    #[serde(default)]
    pub debug: bool,
    /// Log file path
    pub file: Option<PathBuf>,
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

/// JIT optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JitOptLevel {
    None,
    Size,
    Speed,
    Aggressive,
}

/// GC preset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GcPreset {
    LowLatency,
    HighThroughput,
    LowMemory,
    Debug,
}

/// Log level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

// Default value functions
fn default_timeout() -> u64 { 300 }
fn default_true() -> bool { true }
fn default_jit_threshold() -> u32 { 10 }
fn default_window_width() -> u32 { 1200 }
fn default_window_height() -> u32 { 800 }
fn default_dpi() -> u32 { 300 }
fn default_kernel_ip() -> String { "127.0.0.1".to_string() }

impl Default for RustMatConfig {
    fn default() -> Self {
        Self {
            runtime: RuntimeConfig::default(),
            jit: JitConfig::default(),
            gc: GcConfig::default(),
            plotting: PlottingConfig::default(),
            kernel: KernelConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            timeout: default_timeout(),
            verbose: false,
            snapshot_path: None,
        }
    }
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: default_jit_threshold(),
            optimization_level: JitOptLevel::Speed,
        }
    }
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            preset: None,
            young_size_mb: None,
            threads: None,
            collect_stats: false,
        }
    }
}

impl Default for PlottingConfig {
    fn default() -> Self {
        Self {
            mode: PlotMode::Auto,
            force_headless: false,
            backend: PlotBackend::Auto,
            gui: Some(GuiConfig::default()),
            export: Some(ExportConfig::default()),
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
        }
    }
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            ip: default_kernel_ip(),
            key: None,
            ports: None,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            debug: false,
            file: None,
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

impl Default for JitOptLevel {
    fn default() -> Self {
        Self::Speed
    }
}

impl Default for LogLevel {
    fn default() -> Self {
        Self::Info
    }
}

/// Configuration loader with multiple source support
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load configuration from all sources with proper precedence
    pub fn load() -> Result<RustMatConfig> {
        let mut config = Self::load_from_files()?;
        Self::apply_environment_variables(&mut config)?;
        Ok(config)
    }
    
    /// Find and load configuration from files
    fn load_from_files() -> Result<RustMatConfig> {
        // Try to find config file in order of preference
        let config_paths = Self::find_config_files();
        
        for path in config_paths {
            if path.exists() {
                info!("Loading configuration from: {}", path.display());
                return Self::load_from_file(&path);
            }
        }
        
        debug!("No configuration file found, using defaults");
        Ok(RustMatConfig::default())
    }
    
    /// Find potential configuration file paths
    fn find_config_files() -> Vec<PathBuf> {
        let mut paths = Vec::new();
        
        // 1. Environment variable override
        if let Ok(config_path) = env::var("RUSTMAT_CONFIG") {
            paths.push(PathBuf::from(config_path));
        }
        
        // 2. Current directory
        let current_dir_configs = [
            ".rustmat.yaml",
            ".rustmat.yml", 
            ".rustmat.json",
            ".rustmat.toml",
            "rustmat.config.yaml",
            "rustmat.config.yml",
            "rustmat.config.json",
            "rustmat.config.toml",
        ];
        
        for name in &current_dir_configs {
            if let Ok(current_dir) = env::current_dir() {
                paths.push(current_dir.join(name));
            }
        }
        
        // 3. Home directory
        if let Some(home_dir) = dirs::home_dir() {
            paths.push(home_dir.join(".rustmat.yaml"));
            paths.push(home_dir.join(".rustmat.yml"));
            paths.push(home_dir.join(".rustmat.json"));
            paths.push(home_dir.join(".config/rustmat/config.yaml"));
            paths.push(home_dir.join(".config/rustmat/config.yml"));
            paths.push(home_dir.join(".config/rustmat/config.json"));
        }
        
        // 4. System-wide configurations
        #[cfg(unix)]
        {
            paths.push(PathBuf::from("/etc/rustmat/config.yaml"));
            paths.push(PathBuf::from("/etc/rustmat/config.yml"));
            paths.push(PathBuf::from("/etc/rustmat/config.json"));
        }
        
        paths
    }
    
    /// Load configuration from a specific file
    pub fn load_from_file(path: &Path) -> Result<RustMatConfig> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        
        let config = match path.extension().and_then(|ext| ext.to_str()) {
            Some("yaml") | Some("yml") => {
                serde_yaml::from_str(&content)
                    .with_context(|| format!("Failed to parse YAML config: {}", path.display()))?
            },
            Some("json") => {
                serde_json::from_str(&content)
                    .with_context(|| format!("Failed to parse JSON config: {}", path.display()))?
            },
            Some("toml") => {
                toml::from_str(&content)
                    .with_context(|| format!("Failed to parse TOML config: {}", path.display()))?
            },
            _ => {
                // Try to auto-detect format
                if let Ok(config) = serde_yaml::from_str(&content) {
                    config
                } else if let Ok(config) = serde_json::from_str(&content) {
                    config
                } else {
                    return Err(anyhow::anyhow!(
                        "Could not parse config file {} (tried YAML, JSON, TOML)", 
                        path.display()
                    ));
                }
            }
        };
        
        Ok(config)
    }
    
    /// Apply environment variable overrides
    fn apply_environment_variables(config: &mut RustMatConfig) -> Result<()> {
        // Runtime settings
        if let Ok(timeout) = env::var("RUSTMAT_TIMEOUT") {
            if let Ok(timeout) = timeout.parse() {
                config.runtime.timeout = timeout;
            }
        }
        
        if let Ok(verbose) = env::var("RUSTMAT_VERBOSE") {
            config.runtime.verbose = parse_bool(&verbose).unwrap_or(false);
        }
        
        if let Ok(snapshot) = env::var("RUSTMAT_SNAPSHOT_PATH") {
            config.runtime.snapshot_path = Some(PathBuf::from(snapshot));
        }
        
        // JIT settings
        if let Ok(jit_enabled) = env::var("RUSTMAT_JIT_ENABLE") {
            config.jit.enabled = parse_bool(&jit_enabled).unwrap_or(true);
        }
        
        if let Ok(jit_disabled) = env::var("RUSTMAT_JIT_DISABLE") {
            if parse_bool(&jit_disabled).unwrap_or(false) {
                config.jit.enabled = false;
            }
        }
        
        if let Ok(threshold) = env::var("RUSTMAT_JIT_THRESHOLD") {
            if let Ok(threshold) = threshold.parse() {
                config.jit.threshold = threshold;
            }
        }
        
        if let Ok(opt_level) = env::var("RUSTMAT_JIT_OPT_LEVEL") {
            config.jit.optimization_level = match opt_level.to_lowercase().as_str() {
                "none" => JitOptLevel::None,
                "size" => JitOptLevel::Size,
                "speed" => JitOptLevel::Speed,
                "aggressive" => JitOptLevel::Aggressive,
                _ => config.jit.optimization_level,
            };
        }
        
        // GC settings
        if let Ok(preset) = env::var("RUSTMAT_GC_PRESET") {
            config.gc.preset = match preset.to_lowercase().as_str() {
                "low-latency" => Some(GcPreset::LowLatency),
                "high-throughput" => Some(GcPreset::HighThroughput),
                "low-memory" => Some(GcPreset::LowMemory),
                "debug" => Some(GcPreset::Debug),
                _ => config.gc.preset,
            };
        }
        
        if let Ok(young_size) = env::var("RUSTMAT_GC_YOUNG_SIZE") {
            if let Ok(young_size) = young_size.parse() {
                config.gc.young_size_mb = Some(young_size);
            }
        }
        
        if let Ok(threads) = env::var("RUSTMAT_GC_THREADS") {
            if let Ok(threads) = threads.parse() {
                config.gc.threads = Some(threads);
            }
        }
        
        if let Ok(stats) = env::var("RUSTMAT_GC_STATS") {
            config.gc.collect_stats = parse_bool(&stats).unwrap_or(false);
        }
        
        // Plotting settings
        if let Ok(plot_mode) = env::var("RUSTMAT_PLOT_MODE") {
            config.plotting.mode = match plot_mode.to_lowercase().as_str() {
                "auto" => PlotMode::Auto,
                "gui" => PlotMode::Gui,
                "headless" => PlotMode::Headless,
                "jupyter" => PlotMode::Jupyter,
                _ => config.plotting.mode,
            };
        }
        
        if let Ok(headless) = env::var("RUSTMAT_PLOT_HEADLESS") {
            config.plotting.force_headless = parse_bool(&headless).unwrap_or(false);
        }
        
        if let Ok(backend) = env::var("RUSTMAT_PLOT_BACKEND") {
            config.plotting.backend = match backend.to_lowercase().as_str() {
                "auto" => PlotBackend::Auto,
                "wgpu" => PlotBackend::Wgpu,
                "static" => PlotBackend::Static,
                "web" => PlotBackend::Web,
                _ => config.plotting.backend,
            };
        }
        
        // Logging settings
        if let Ok(debug) = env::var("RUSTMAT_DEBUG") {
            config.logging.debug = parse_bool(&debug).unwrap_or(false);
        }
        
        if let Ok(log_level) = env::var("RUSTMAT_LOG_LEVEL") {
            config.logging.level = match log_level.to_lowercase().as_str() {
                "error" => LogLevel::Error,
                "warn" => LogLevel::Warn,
                "info" => LogLevel::Info,
                "debug" => LogLevel::Debug,
                "trace" => LogLevel::Trace,
                _ => config.logging.level,
            };
        }
        
        // Kernel settings
        if let Ok(ip) = env::var("RUSTMAT_KERNEL_IP") {
            config.kernel.ip = ip;
        }
        
        if let Ok(key) = env::var("RUSTMAT_KERNEL_KEY") {
            config.kernel.key = Some(key);
        }
        
        Ok(())
    }
    
    /// Save configuration to a file
    pub fn save_to_file(config: &RustMatConfig, path: &Path) -> Result<()> {
        let content = match path.extension().and_then(|ext| ext.to_str()) {
            Some("yaml") | Some("yml") => {
                serde_yaml::to_string(config)
                    .context("Failed to serialize config to YAML")?
            },
            Some("json") => {
                serde_json::to_string_pretty(config)
                    .context("Failed to serialize config to JSON")?
            },
            Some("toml") => {
                toml::to_string_pretty(config)
                    .context("Failed to serialize config to TOML")?
            },
            _ => {
                // Default to YAML
                serde_yaml::to_string(config)
                    .context("Failed to serialize config to YAML")?
            }
        };
        
        fs::write(path, content)
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;
        
        info!("Configuration saved to: {}", path.display());
        Ok(())
    }
    
    /// Generate a sample configuration file
    pub fn generate_sample_config() -> String {
        let config = RustMatConfig::default();
        serde_yaml::to_string(&config).unwrap_or_else(|_| "# Failed to generate config".to_string())
    }
}

/// Parse a boolean value from string with various formats
fn parse_bool(s: &str) -> Option<bool> {
    match s.to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" | "enable" | "enabled" => Some(true),
        "0" | "false" | "no" | "off" | "disable" | "disabled" => Some(false),
        "" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_config_defaults() {
        let config = RustMatConfig::default();
        assert_eq!(config.runtime.timeout, 300);
        assert_eq!(config.jit.enabled, true);
        assert_eq!(config.jit.threshold, 10);
        assert_eq!(config.plotting.mode, PlotMode::Auto);
    }
    
    #[test]
    fn test_yaml_serialization() {
        let config = RustMatConfig::default();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let parsed: RustMatConfig = serde_yaml::from_str(&yaml).unwrap();
        
        assert_eq!(parsed.runtime.timeout, config.runtime.timeout);
        assert_eq!(parsed.jit.enabled, config.jit.enabled);
    }
    
    #[test]
    fn test_json_serialization() {
        let config = RustMatConfig::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        let parsed: RustMatConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed.runtime.timeout, config.runtime.timeout);
        assert_eq!(parsed.plotting.mode, config.plotting.mode);
    }
    
    #[test]
    fn test_file_loading() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join(".rustmat.yaml");
        
        let mut config = RustMatConfig::default();
        config.runtime.timeout = 600;
        config.jit.threshold = 20;
        
        ConfigLoader::save_to_file(&config, &config_path).unwrap();
        let loaded = ConfigLoader::load_from_file(&config_path).unwrap();
        
        assert_eq!(loaded.runtime.timeout, 600);
        assert_eq!(loaded.jit.threshold, 20);
    }
    
    #[test]
    fn test_bool_parsing() {
        assert_eq!(parse_bool("true"), Some(true));
        assert_eq!(parse_bool("1"), Some(true));
        assert_eq!(parse_bool("yes"), Some(true));
        assert_eq!(parse_bool("false"), Some(false));
        assert_eq!(parse_bool("0"), Some(false));
        assert_eq!(parse_bool("invalid"), None);
    }
}