use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::defaults::default_true;

/// Runtime execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Execution timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    /// Maximum number of call stack frames to record
    #[serde(default = "default_callstack_limit")]
    pub callstack_limit: usize,
    /// Namespace prefix for runtime/semantic error identifiers
    #[serde(default = "default_error_namespace")]
    pub error_namespace: String,
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

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            timeout: default_timeout(),
            callstack_limit: default_callstack_limit(),
            error_namespace: default_error_namespace(),
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

impl Default for JitOptLevel {
    fn default() -> Self {
        Self::Speed
    }
}

fn default_timeout() -> u64 {
    300
}

fn default_callstack_limit() -> usize {
    200
}

fn default_error_namespace() -> String {
    "".to_string()
}

fn default_jit_threshold() -> u32 {
    10
}
