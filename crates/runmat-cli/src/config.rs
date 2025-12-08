//! Configuration system for RunMat
//!
//! Supports multiple configuration sources with proper precedence:
//! 1. Command-line arguments (highest priority)
//! 2. Environment variables  
//! 3. Configuration files (.runmat.yaml, .runmat.json, etc.)
//! 4. Built-in defaults (lowest priority)

use anyhow::{Context, Result};
use clap::ValueEnum;
use log::{debug, info};
use runmat_accelerate::{
    AccelPowerPreference, AccelerateInitOptions, AccelerateProviderPreference, AutoOffloadLogLevel,
    AutoOffloadOptions,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const MIN_QUEUE_SIZE: usize = 8;

/// Main RunMat configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RunMatConfig {
    /// Runtime configuration
    pub runtime: RuntimeConfig,
    /// Acceleration configuration
    #[serde(default)]
    pub accelerate: AccelerateConfig,
    /// Telemetry configuration
    #[serde(default)]
    pub telemetry: TelemetryConfig,
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
    /// Package manager configuration
    #[serde(default)]
    pub packages: PackagesConfig,
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

/// Telemetry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable runtime telemetry
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Echo each payload to stdout for transparency
    #[serde(default)]
    pub show_payloads: bool,
    /// Optional HTTP endpoint override
    pub http_endpoint: Option<String>,
    /// Optional UDP endpoint override (host:port)
    pub udp_endpoint: Option<String>,
    /// Bounded queue size for async delivery
    #[serde(default = "default_telemetry_queue")]
    pub queue_size: usize,
    /// Require ingestion key (self-built binaries default to false)
    #[serde(default = "default_true")]
    pub require_ingestion_key: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            show_payloads: false,
            http_endpoint: None,
            udp_endpoint: Some("udp.telemetry.runmat.org:7846".to_string()),
            queue_size: default_telemetry_queue(),
            require_ingestion_key: true,
        }
    }
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

fn default_telemetry_queue() -> usize {
    256
}

impl AccelerateConfig {
    pub fn to_init_options(&self) -> AccelerateInitOptions {
        AccelerateInitOptions {
            enabled: self.enabled,
            provider: self.provider,
            allow_inprocess_fallback: self.allow_inprocess_fallback,
            wgpu_power_preference: self.wgpu_power_preference,
            wgpu_force_fallback_adapter: self.wgpu_force_fallback_adapter,
            auto_offload: self.auto_offload.to_options(),
        }
    }
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

impl AutoOffloadConfig {
    fn to_options(&self) -> AutoOffloadOptions {
        AutoOffloadOptions {
            enabled: self.enabled,
            calibrate: self.calibrate,
            profile_path: self.profile_path.clone(),
            log_level: self.log_level,
        }
    }
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
    /// Target scatter point budget for GPU decimation overrides
    #[serde(default)]
    pub scatter_target_points: Option<u32>,
    /// Surface vertex budget override for LOD selection
    #[serde(default)]
    pub surface_vertex_budget: Option<u64>,
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
    /// Jupyter notebook configuration
    pub jupyter: Option<JupyterConfig>,
}

/// Jupyter notebook integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterConfig {
    /// Default output format for Jupyter cells
    #[serde(default)]
    pub output_format: JupyterOutputFormat,
    /// Enable interactive widgets
    #[serde(default = "default_true")]
    pub enable_widgets: bool,
    /// Enable static image fallback
    #[serde(default = "default_true")]
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
    #[serde(default = "default_true")]
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
    #[serde(default = "default_true")]
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
    #[serde(default = "default_true")]
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
    #[serde(default = "default_true")]
    pub progressive_rendering: bool,
    /// LOD (Level of Detail) threshold
    #[serde(default = "default_lod_threshold")]
    pub lod_threshold: u32,
    /// Enable texture compression
    #[serde(default = "default_true")]
    pub texture_compression: bool,
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

/// Package manager configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PackagesConfig {
    /// Enable package manager
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Registries to search for packages (first match wins)
    #[serde(default = "default_registries")]
    pub registries: Vec<Registry>,
    /// Dependencies declared by the workspace (name -> spec)
    #[serde(default)]
    pub dependencies: HashMap<String, PackageSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Registry {
    /// Registry logical name
    pub name: String,
    /// Base URL for index/API (e.g., https://packages.runmat.org)
    pub url: String,
}

/// Package specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "kebab-case")]
pub enum PackageSpec {
    /// Resolve from a registry by name
    Registry {
        /// Semver range (e.g. "^1.2"), or exact version
        version: String,
        /// Optional registry override (defaults to first registry)
        #[serde(default)]
        registry: Option<String>,
        /// Optional feature flags
        #[serde(default)]
        features: Vec<String>,
        /// Optional mark for optional dependency
        #[serde(default)]
        optional: bool,
    },
    /// Git repository
    Git {
        url: String,
        #[serde(default)]
        rev: Option<String>,
        #[serde(default)]
        features: Vec<String>,
        #[serde(default)]
        optional: bool,
    },
    /// Local path dependency (useful for development)
    Path {
        path: String,
        #[serde(default)]
        features: Vec<String>,
        #[serde(default)]
        optional: bool,
    },
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
fn default_timeout() -> u64 {
    300
}
fn default_true() -> bool {
    true
}
fn default_jit_threshold() -> u32 {
    10
}
fn default_window_width() -> u32 {
    1200
}
fn default_window_height() -> u32 {
    800
}
fn default_dpi() -> u32 {
    300
}
fn default_kernel_ip() -> String {
    "127.0.0.1".to_string()
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

fn default_registries() -> Vec<Registry> {
    vec![Registry {
        name: "runmat".to_string(),
        url: "https://packages.runmat.org".to_string(),
    }]
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

impl Default for PlottingConfig {
    fn default() -> Self {
        Self {
            mode: PlotMode::Auto,
            force_headless: false,
            backend: PlotBackend::Auto,
            gui: Some(GuiConfig::default()),
            export: Some(ExportConfig::default()),
            scatter_target_points: None,
            surface_vertex_budget: None,
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
            jupyter: Some(JupyterConfig::default()),
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

impl Default for JupyterOutputFormat {
    fn default() -> Self {
        Self::Auto
    }
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

/// Configuration loader with multiple source support
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load configuration from all sources with proper precedence
    pub fn load() -> Result<RunMatConfig> {
        let mut config = Self::load_from_files()?;
        Self::apply_environment_variables(&mut config)?;
        Ok(config)
    }

    /// Find and load configuration from files
    fn load_from_files() -> Result<RunMatConfig> {
        // Try to find config file in order of preference
        let config_paths = Self::find_config_files();

        for path in config_paths {
            if path.is_dir() {
                info!(
                    "Ignoring config directory path (expected file): {}",
                    path.display()
                );
                continue;
            }
            if path.exists() {
                info!("Loading configuration from: {}", path.display());
                return Self::load_from_file(&path);
            }
        }

        debug!("No configuration file found, using defaults");
        Ok(RunMatConfig::default())
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
            ".runmat", // preferred single-file format
            ".runmat.yaml",
            ".runmat.yml",
            ".runmat.json",
            ".runmat.toml",
            "runmat.config.yaml",
            "runmat.config.yml",
            "runmat.config.json",
            "runmat.config.toml",
        ];

        for name in &current_dir_configs {
            if let Ok(current_dir) = env::current_dir() {
                paths.push(current_dir.join(name));
            }
        }

        // 3. Home directory
        if let Some(home_dir) = dirs::home_dir() {
            paths.push(home_dir.join(".runmat"));
            paths.push(home_dir.join(".runmat.yaml"));
            paths.push(home_dir.join(".runmat.yml"));
            paths.push(home_dir.join(".runmat.json"));
            paths.push(home_dir.join(".config/runmat/config.yaml"));
            paths.push(home_dir.join(".config/runmat/config.yml"));
            paths.push(home_dir.join(".config/runmat/config.json"));
        }

        // 4. System-wide configurations
        #[cfg(unix)]
        {
            paths.push(PathBuf::from("/etc/runmat/config.yaml"));
            paths.push(PathBuf::from("/etc/runmat/config.yml"));
            paths.push(PathBuf::from("/etc/runmat/config.json"));
        }

        paths
    }

    /// Load configuration from a specific file
    pub fn load_from_file(path: &Path) -> Result<RunMatConfig> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let config = match path.extension().and_then(|ext| ext.to_str()) {
            // `.runmat` is a TOML alias by default (single canonical format)
            None if path.file_name().and_then(|n| n.to_str()) == Some(".runmat") => {
                toml::from_str(&content).with_context(|| {
                    format!("Failed to parse .runmat (TOML) config: {}", path.display())
                })?
            }
            Some("runmat") => toml::from_str(&content).with_context(|| {
                format!("Failed to parse .runmat (TOML) config: {}", path.display())
            })?,
            Some("yaml") | Some("yml") => serde_yaml::from_str(&content)
                .with_context(|| format!("Failed to parse YAML config: {}", path.display()))?,
            Some("json") => serde_json::from_str(&content)
                .with_context(|| format!("Failed to parse JSON config: {}", path.display()))?,
            Some("toml") => toml::from_str(&content)
                .with_context(|| format!("Failed to parse TOML config: {}", path.display()))?,
            _ => {
                // Try auto-detect (prefer TOML for unknown/no extension)
                if let Ok(config) = toml::from_str(&content) {
                    config
                } else if let Ok(config) = serde_yaml::from_str(&content) {
                    config
                } else if let Ok(config) = serde_json::from_str(&content) {
                    config
                } else {
                    return Err(anyhow::anyhow!(
                        "Could not parse config file {} (tried TOML, YAML, JSON)",
                        path.display()
                    ));
                }
            }
        };

        Ok(config)
    }

    /// Apply environment variable overrides
    fn apply_environment_variables(config: &mut RunMatConfig) -> Result<()> {
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

        // Telemetry settings
        if let Some(flag) = env::var("RUNMAT_TELEMETRY")
            .ok()
            .and_then(|v| parse_bool(&v))
        {
            config.telemetry.enabled = flag;
        }
        if let Some(flag) = env::var("RUNMAT_NO_TELEMETRY")
            .ok()
            .and_then(|v| parse_bool(&v))
        {
            if flag {
                config.telemetry.enabled = false;
            }
        }
        if let Ok(show) = env::var("RUNMAT_TELEMETRY_SHOW") {
            config.telemetry.show_payloads = parse_bool(&show).unwrap_or(false);
        }
        if let Ok(endpoint) = env::var("RUNMAT_TELEMETRY_ENDPOINT")
            .or_else(|_| env::var("RUNMAT_TELEMETRY_HTTP_ENDPOINT"))
        {
            let trimmed = endpoint.trim();
            if trimmed.is_empty() {
                config.telemetry.http_endpoint = None;
            } else {
                config.telemetry.http_endpoint = Some(trimmed.to_string());
            }
        }
        if let Ok(udp) = env::var("RUNMAT_TELEMETRY_UDP_ENDPOINT") {
            let trimmed = udp.trim();
            if trimmed.is_empty() || trimmed == "0" || trimmed.eq_ignore_ascii_case("off") {
                config.telemetry.udp_endpoint = None;
            } else {
                config.telemetry.udp_endpoint = Some(trimmed.to_string());
            }
        }
        if let Ok(queue) = env::var("RUNMAT_TELEMETRY_QUEUE_SIZE") {
            if let Ok(parsed) = queue.parse::<usize>() {
                config.telemetry.queue_size = parsed.max(MIN_QUEUE_SIZE);
            }
        }

        // Acceleration settings
        if let Ok(accel) =
            env::var("RUSTMAT_ACCEL_ENABLE").or_else(|_| env::var("RUNMAT_ACCEL_ENABLE"))
        {
            if let Some(flag) = parse_bool(&accel) {
                config.accelerate.enabled = flag;
            }
        }

        if let Ok(provider) =
            env::var("RUSTMAT_ACCEL_PROVIDER").or_else(|_| env::var("RUNMAT_ACCEL_PROVIDER"))
        {
            if let Some(pref) = parse_provider_preference(&provider) {
                config.accelerate.provider = pref;
            }
        }

        if let Ok(force_inprocess) = env::var("RUNMAT_ACCEL_FORCE_INPROCESS") {
            if parse_bool(&force_inprocess).unwrap_or(false) {
                config.accelerate.provider = AccelerateProviderPreference::InProcess;
            }
        }

        if let Ok(wgpu_toggle) = env::var("RUNMAT_ACCEL_WGPU") {
            if let Some(enabled) = parse_bool(&wgpu_toggle) {
                config.accelerate.provider = if enabled {
                    AccelerateProviderPreference::Wgpu
                } else {
                    AccelerateProviderPreference::InProcess
                };
            }
        }

        if let Ok(fallback) = env::var("RUSTMAT_ACCEL_DISABLE_FALLBACK") {
            if let Some(disable) = parse_bool(&fallback) {
                config.accelerate.allow_inprocess_fallback = !disable;
            }
        }

        if let Ok(force_fallback) = env::var("RUSTMAT_ACCEL_WGPU_FORCE_FALLBACK")
            .or_else(|_| env::var("RUNMAT_ACCEL_WGPU_FORCE_FALLBACK"))
        {
            if let Some(flag) = parse_bool(&force_fallback) {
                config.accelerate.wgpu_force_fallback_adapter = flag;
            }
        }

        if let Ok(power) =
            env::var("RUSTMAT_ACCEL_WGPU_POWER").or_else(|_| env::var("RUNMAT_ACCEL_WGPU_POWER"))
        {
            if let Some(pref) = parse_power_preference(&power) {
                config.accelerate.wgpu_power_preference = pref;
            }
        }

        if let Ok(auto_enabled) = env::var("RUNMAT_ACCEL_AUTO_OFFLOAD") {
            if let Some(flag) = parse_bool(&auto_enabled) {
                config.accelerate.auto_offload.enabled = flag;
            }
        }

        if let Ok(auto_calibrate) = env::var("RUNMAT_ACCEL_CALIBRATE") {
            if let Some(flag) = parse_bool(&auto_calibrate) {
                config.accelerate.auto_offload.calibrate = flag;
            }
        }

        if let Ok(profile_path) = env::var("RUNMAT_ACCEL_PROFILE") {
            config.accelerate.auto_offload.profile_path = Some(PathBuf::from(profile_path));
        }

        if let Ok(auto_log) = env::var("RUNMAT_ACCEL_AUTO_LOG") {
            if let Some(level) = parse_auto_offload_log_level(&auto_log) {
                config.accelerate.auto_offload.log_level = level;
            }
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
    pub fn save_to_file(config: &RunMatConfig, path: &Path) -> Result<()> {
        let content = match path.extension().and_then(|ext| ext.to_str()) {
            Some("yaml") | Some("yml") => {
                serde_yaml::to_string(config).context("Failed to serialize config to YAML")?
            }
            Some("json") => serde_json::to_string_pretty(config)
                .context("Failed to serialize config to JSON")?,
            Some("toml") => {
                toml::to_string_pretty(config).context("Failed to serialize config to TOML")?
            }
            _ => {
                // Default to YAML
                serde_yaml::to_string(config).context("Failed to serialize config to YAML")?
            }
        };

        fs::write(path, content)
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;

        info!("Configuration saved to: {}", path.display());
        Ok(())
    }

    /// Generate a sample configuration file
    pub fn generate_sample_config() -> String {
        let config = RunMatConfig::default();
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

fn parse_auto_offload_log_level(value: &str) -> Option<AutoOffloadLogLevel> {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" => Some(AutoOffloadLogLevel::Off),
        "info" => Some(AutoOffloadLogLevel::Info),
        "trace" => Some(AutoOffloadLogLevel::Trace),
        _ => None,
    }
}

fn parse_provider_preference(value: &str) -> Option<AccelerateProviderPreference> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Some(AccelerateProviderPreference::Auto),
        "wgpu" => Some(AccelerateProviderPreference::Wgpu),
        "inprocess" | "cpu" | "host" => Some(AccelerateProviderPreference::InProcess),
        _ => None,
    }
}

fn parse_power_preference(value: &str) -> Option<AccelPowerPreference> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Some(AccelPowerPreference::Auto),
        "high" | "highperformance" | "performance" => Some(AccelPowerPreference::HighPerformance),
        "low" | "lowpower" | "battery" => Some(AccelPowerPreference::LowPower),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::sync::Mutex;
    use tempfile::TempDir;

    static ENV_GUARD: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    #[test]
    fn test_config_defaults() {
        let config = RunMatConfig::default();
        assert_eq!(config.runtime.timeout, 300);
        assert!(config.jit.enabled);
        assert_eq!(config.jit.threshold, 10);
        assert_eq!(config.plotting.mode, PlotMode::Auto);
    }

    #[test]
    fn test_yaml_serialization() {
        let config = RunMatConfig::default();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let parsed: RunMatConfig = serde_yaml::from_str(&yaml).unwrap();

        assert_eq!(parsed.runtime.timeout, config.runtime.timeout);
        assert_eq!(parsed.jit.enabled, config.jit.enabled);
        assert_eq!(parsed.accelerate.provider, config.accelerate.provider);
    }

    #[test]
    fn test_json_serialization() {
        let config = RunMatConfig::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        let parsed: RunMatConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.runtime.timeout, config.runtime.timeout);
        assert_eq!(parsed.plotting.mode, config.plotting.mode);
        assert_eq!(parsed.accelerate.enabled, config.accelerate.enabled);
    }

    #[test]
    fn test_parse_auto_offload_log_level_cases() {
        assert_eq!(
            parse_auto_offload_log_level("off"),
            Some(AutoOffloadLogLevel::Off)
        );
        assert_eq!(
            parse_auto_offload_log_level("INFO"),
            Some(AutoOffloadLogLevel::Info)
        );
        assert_eq!(
            parse_auto_offload_log_level("trace"),
            Some(AutoOffloadLogLevel::Trace)
        );
        assert_eq!(parse_auto_offload_log_level("unknown"), None);
    }

    #[test]
    fn accelerate_to_options() {
        let accel = AccelerateConfig::default();
        let opts = accel.to_init_options();
        assert!(opts.enabled);
        assert_eq!(opts.provider, AccelerateProviderPreference::Wgpu);
    }

    #[test]
    fn test_file_loading() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join(".runmat.yaml");

        let mut config = RunMatConfig::default();
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

    #[test]
    fn telemetry_env_overrides_respect_empty_values() {
        let _lock = ENV_GUARD.lock().unwrap();
        std::env::set_var("RUNMAT_TELEMETRY_ENDPOINT", "https://custom.example/ingest");
        std::env::set_var("RUNMAT_TELEMETRY_UDP_ENDPOINT", "off");
        let mut config = RunMatConfig::default();
        ConfigLoader::apply_environment_variables(&mut config).unwrap();
        assert_eq!(
            config.telemetry.http_endpoint.as_deref(),
            Some("https://custom.example/ingest")
        );
        assert!(config.telemetry.udp_endpoint.is_none());
        std::env::remove_var("RUNMAT_TELEMETRY_ENDPOINT");
        std::env::remove_var("RUNMAT_TELEMETRY_UDP_ENDPOINT");
    }
}
