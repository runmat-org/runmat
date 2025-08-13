use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuTensorHandle {
    pub shape: Vec<usize>,
    pub device_id: u32,
    pub buffer_id: u64,
}

/// Device/provider interface that backends implement and register into the runtime layer
pub trait AccelProvider: Send + Sync {
    fn upload(&self, host: &crate::HostTensorView) -> anyhow::Result<GpuTensorHandle>;
    fn download(&self, h: &GpuTensorHandle) -> anyhow::Result<crate::HostTensorOwned>;
    fn free(&self, h: &GpuTensorHandle) -> anyhow::Result<()>;
    fn device_info(&self) -> String;
}

static mut GLOBAL_PROVIDER: Option<&'static dyn AccelProvider> = None;

/// Register a global acceleration provider. The caller must guarantee the provider lives for program lifetime.
pub unsafe fn register_provider(p: &'static dyn AccelProvider) {
    GLOBAL_PROVIDER = Some(p);
}

pub fn provider() -> Option<&'static dyn AccelProvider> {
    unsafe { GLOBAL_PROVIDER }
}

// Minimal host tensor views to avoid depending on runmat-builtins and cycles
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostTensorOwned {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

#[derive(Debug)]
pub struct HostTensorView<'a> {
    pub data: &'a [f64],
    pub shape: &'a [usize],
}


