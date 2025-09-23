//! RunMat Accelerate: GPU Acceleration Abstraction Layer
//!
//! Goals:
//! - Provide a backend-agnostic API surface that maps RunMat operations to GPU kernels.
//! - Support multiple backends via features (CUDA, ROCm, Metal, Vulkan, OpenCL, wgpu).
//! - Allow zero-copy interop with `runmat-builtins::Matrix` where possible.
//! - Defer actual kernel authoring to backend crates/modules; this crate defines traits and wiring.
//!
//! This is scaffolding only; implementations will land after interpreter/JIT semantics are complete.

use once_cell::sync::Lazy;
use runmat_builtins::{Tensor, Value};
use std::path::PathBuf;
use std::sync::RwLock;

pub mod native_auto;
pub mod simple_provider;
#[cfg(feature = "wgpu")]
pub mod wgpu_backend;
pub use native_auto::{
    is_sink, prepare_builtin_args, promote_binary, promote_reduction_args, promote_unary, BinaryOp,
    ReductionOp, UnaryOp,
};
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::AccelProvider;
use serde::{Deserialize, Serialize};
#[cfg(feature = "wgpu")]
use wgpu::PowerPreference;

/// Preferred acceleration provider selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AccelerateProviderPreference {
    Auto,
    Wgpu,
    InProcess,
}

impl Default for AccelerateProviderPreference {
    fn default() -> Self {
        Self::Auto
    }
}

/// Power preference used when initializing a WGPU backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

/// Logging verbosity for auto-offload promotion decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AutoOffloadLogLevel {
    Off,
    Info,
    Trace,
}

impl Default for AutoOffloadLogLevel {
    fn default() -> Self {
        AutoOffloadLogLevel::Trace
    }
}

/// Configuration passed to the native auto-offload planner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoOffloadOptions {
    pub enabled: bool,
    pub calibrate: bool,
    #[serde(default)]
    pub profile_path: Option<PathBuf>,
    #[serde(default)]
    pub log_level: AutoOffloadLogLevel,
}

impl Default for AutoOffloadOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            calibrate: true,
            profile_path: None,
            log_level: AutoOffloadLogLevel::Trace,
        }
    }
}

static AUTO_OFFLOAD_OPTIONS: Lazy<RwLock<AutoOffloadOptions>> =
    Lazy::new(|| RwLock::new(AutoOffloadOptions::default()));

pub fn configure_auto_offload(options: AutoOffloadOptions) {
    if let Ok(mut guard) = AUTO_OFFLOAD_OPTIONS.write() {
        *guard = options;
    }
}

pub(crate) fn auto_offload_options() -> AutoOffloadOptions {
    AUTO_OFFLOAD_OPTIONS
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_default()
}

/// Initialization options for selecting and configuring the acceleration provider.
#[derive(Debug, Clone)]
pub struct AccelerateInitOptions {
    pub enabled: bool,
    pub provider: AccelerateProviderPreference,
    pub allow_inprocess_fallback: bool,
    pub wgpu_power_preference: AccelPowerPreference,
    pub wgpu_force_fallback_adapter: bool,
    pub auto_offload: AutoOffloadOptions,
}

impl Default for AccelerateInitOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: AccelerateProviderPreference::Auto,
            allow_inprocess_fallback: true,
            wgpu_power_preference: AccelPowerPreference::Auto,
            wgpu_force_fallback_adapter: false,
            auto_offload: AutoOffloadOptions::default(),
        }
    }
}

/// Initialize the global acceleration provider using the supplied options.
pub fn initialize_acceleration_provider_with(options: &AccelerateInitOptions) {
    configure_auto_offload(options.auto_offload.clone());

    if runmat_accelerate_api::provider().is_some() {
        return;
    }

    if !options.enabled {
        if options.allow_inprocess_fallback {
            simple_provider::register_inprocess_provider();
            log::info!(
                "RunMat Accelerate: acceleration disabled; using in-process provider for compatibility"
            );
        } else {
            log::info!("RunMat Accelerate: acceleration disabled; no provider registered");
        }
        return;
    }

    #[allow(unused_mut)]
    let mut registered = false;

    #[cfg(feature = "wgpu")]
    {
        if !registered
            && matches!(
                options.provider,
                AccelerateProviderPreference::Auto | AccelerateProviderPreference::Wgpu
            )
        {
            let wgpu_options = wgpu_backend::WgpuProviderOptions {
                power_preference: match options.wgpu_power_preference {
                    AccelPowerPreference::Auto => PowerPreference::HighPerformance,
                    AccelPowerPreference::HighPerformance => PowerPreference::HighPerformance,
                    AccelPowerPreference::LowPower => PowerPreference::LowPower,
                },
                force_fallback_adapter: options.wgpu_force_fallback_adapter,
            };

            match wgpu_backend::register_wgpu_provider(wgpu_options) {
                Ok(provider) => {
                    registered = true;
                    let info = provider.device_info_struct();
                    let backend = info.backend.as_deref().unwrap_or("unknown");
                    log::info!(
                        "RunMat Accelerate: using WGPU provider {} (vendor: {}, backend: {})",
                        info.name,
                        info.vendor,
                        backend
                    );
                }
                Err(err) => {
                    log::warn!(
                        "RunMat Accelerate: failed to initialize WGPU provider, falling back: {err}"
                    );
                }
            }
        }
    }

    #[cfg(not(feature = "wgpu"))]
    {
        if matches!(options.provider, AccelerateProviderPreference::Wgpu) {
            log::warn!(
                "RunMat Accelerate: WGPU provider requested but crate built without 'wgpu' feature"
            );
        }
    }

    if !registered {
        if options.allow_inprocess_fallback
            || matches!(options.provider, AccelerateProviderPreference::InProcess)
        {
            simple_provider::register_inprocess_provider();
            log::info!("RunMat Accelerate: using in-process acceleration provider");
        } else {
            log::warn!("RunMat Accelerate: no acceleration provider registered");
        }
    }
}

/// Initialize the acceleration provider using default options.
pub fn initialize_acceleration_provider() {
    initialize_acceleration_provider_with(&AccelerateInitOptions::default());
}

/// High-level device kind. Concrete selection is provided by backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    OpenCl,
    Wgpu,
}

/// Device descriptor used for selection and capabilities query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub kind: DeviceKind,
    pub name: String,
    pub vendor: String,
    pub memory_bytes: Option<u64>,
    pub compute_capability: Option<String>,
}

/// Abstract buffer that may reside on device or be host-pinned.
pub trait BufferHandle: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Abstract matrix allocated on a device backend.
pub trait DeviceMatrix: Send + Sync {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn as_buffer(&self) -> &dyn BufferHandle;
}

/// Core backend interface that concrete backends must implement.
pub trait AccelerateBackend: Send + Sync {
    fn device_info(&self) -> DeviceInfo;

    // Memory
    fn upload_matrix(&self, host: &Tensor) -> anyhow::Result<Box<dyn DeviceMatrix>>;
    fn download_matrix(&self, dev: &dyn DeviceMatrix) -> anyhow::Result<Tensor>;

    // Elementwise
    fn elem_add(
        &self,
        a: &dyn DeviceMatrix,
        b: &dyn DeviceMatrix,
    ) -> anyhow::Result<Box<dyn DeviceMatrix>>;
    fn elem_sub(
        &self,
        a: &dyn DeviceMatrix,
        b: &dyn DeviceMatrix,
    ) -> anyhow::Result<Box<dyn DeviceMatrix>>;
    fn elem_mul(
        &self,
        a: &dyn DeviceMatrix,
        b: &dyn DeviceMatrix,
    ) -> anyhow::Result<Box<dyn DeviceMatrix>>;
    fn elem_div(
        &self,
        a: &dyn DeviceMatrix,
        b: &dyn DeviceMatrix,
    ) -> anyhow::Result<Box<dyn DeviceMatrix>>;
    fn elem_pow(
        &self,
        a: &dyn DeviceMatrix,
        b: &dyn DeviceMatrix,
    ) -> anyhow::Result<Box<dyn DeviceMatrix>>;

    // Linear algebra (future): matmul, transpose, BLAS/LAPACK analogs
    fn matmul(
        &self,
        a: &dyn DeviceMatrix,
        b: &dyn DeviceMatrix,
    ) -> anyhow::Result<Box<dyn DeviceMatrix>>;
    fn transpose(&self, a: &dyn DeviceMatrix) -> anyhow::Result<Box<dyn DeviceMatrix>>;
}

/// Planner determines whether to execute on CPU or a selected backend.
/// This will eventually consult sizes, heuristics, and device availability.
#[derive(Default)]
pub struct Planner {
    backend: Option<Box<dyn AccelerateBackend>>,
}

impl Planner {
    pub fn new(backend: Option<Box<dyn AccelerateBackend>>) -> Self {
        Self { backend }
    }

    pub fn device(&self) -> Option<&dyn AccelerateBackend> {
        self.backend.as_deref()
    }

    /// Example decision hook: execute elementwise add on GPU if large enough.
    pub fn choose_elem_add(&self, a: &Tensor, b: &Tensor) -> ExecutionTarget {
        if let Some(bk) = &self.backend {
            if a.data.len() >= 1 << 16 && a.rows() == b.rows() && a.cols() == b.cols() {
                return ExecutionTarget::Gpu(bk.device_info());
            }
        }
        ExecutionTarget::Cpu
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionTarget {
    Cpu,
    Gpu(DeviceInfo),
}

/// High-level façade for accelerated operations, falling back to `runmat-runtime`.
pub struct Accelerator {
    planner: Planner,
}

impl Accelerator {
    pub fn new(planner: Planner) -> Self {
        Self { planner }
    }

    pub fn elementwise_add(&self, a: &Value, b: &Value) -> anyhow::Result<Value> {
        match (a, b) {
            (Value::Tensor(ma), Value::Tensor(mb)) => match self.planner.choose_elem_add(ma, mb) {
                ExecutionTarget::Cpu => {
                    runmat_runtime::elementwise_add(a, b).map_err(|e| anyhow::anyhow!(e))
                }
                ExecutionTarget::Gpu(_) => {
                    let bk = self
                        .planner
                        .device()
                        .ok_or_else(|| anyhow::anyhow!("no backend"))?;
                    let da = bk.upload_matrix(ma)?;
                    let db = bk.upload_matrix(mb)?;
                    let dc = bk.elem_add(da.as_ref(), db.as_ref())?;
                    let out = bk.download_matrix(dc.as_ref())?;
                    Ok(Value::Tensor(out))
                }
            },
            (Value::GpuTensor(ga), Value::GpuTensor(gb)) => {
                // Placeholder: assume same device; in practice look up buffers by id
                // Fallback to CPU until device registry is implemented
                let ha = self.gather_handle(ga)?;
                let hb = self.gather_handle(gb)?;
                self.elementwise_add(&ha, &hb)
            }
            (Value::GpuTensor(ga), other) => {
                let ha = self.gather_handle(ga)?;
                self.elementwise_add(&ha, other)
            }
            (other, Value::GpuTensor(gb)) => {
                let hb = self.gather_handle(gb)?;
                self.elementwise_add(other, &hb)
            }
            _ => runmat_runtime::elementwise_add(a, b).map_err(|e| anyhow::anyhow!(e)),
        }
    }

    fn gather_handle(&self, h: &runmat_accelerate_api::GpuTensorHandle) -> anyhow::Result<Value> {
        if let Some(p) = runmat_accelerate_api::provider() {
            let ht = p.download(h).map_err(|e| anyhow::anyhow!(e))?;
            let t = Tensor::new(ht.data, ht.shape).map_err(|e| anyhow::anyhow!(e))?;
            Ok(Value::Tensor(t))
        } else {
            // Fallback to zeros with same shape if no provider is registered
            let shape = h.shape.clone();
            let total: usize = shape.iter().product();
            let zeros = Tensor::new(vec![0.0; total], shape).map_err(|e| anyhow::anyhow!(e))?;
            Ok(Value::Tensor(zeros))
        }
    }
}
