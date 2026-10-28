use runmat_config::runtime::{AccelerateConfig, ForeignConfig};
use serde::{Deserialize, Serialize};

/// Resolved execution settings carried over the private native worker protocol.
/// Language semantics belong to the frozen test revision, not these host settings.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeWorkerExecutionConfig {
    pub enable_jit: bool,
    pub accelerate: AccelerateConfig,
    pub foreign: ForeignConfig,
}

impl Default for NativeWorkerExecutionConfig {
    fn default() -> Self {
        Self {
            enable_jit: true,
            // Embedders opt into device initialization explicitly. A generic
            // CPU test worker must not discover GPU hardware as a side effect.
            accelerate: AccelerateConfig {
                enabled: false,
                allow_inprocess_fallback: false,
                ..AccelerateConfig::default()
            },
            foreign: ForeignConfig::default(),
        }
    }
}
