use serde::{Deserialize, Serialize};

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

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            ip: default_kernel_ip(),
            key: None,
            ports: None,
        }
    }
}

fn default_kernel_ip() -> String {
    "127.0.0.1".to_string()
}
