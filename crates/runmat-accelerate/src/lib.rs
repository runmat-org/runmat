//! RunMat Accelerate: GPU Acceleration Abstraction Layer
//!
//! Goals:
//! - Provide a backend-agnostic API surface that maps RunMat operations to GPU kernels.
//! - Support multiple backends via features (CUDA, ROCm, Metal, Vulkan, OpenCL, wgpu).
//! - Allow zero-copy interop with `runmat-builtins::Matrix` where possible.
//! - Defer actual kernel authoring to backend crates/modules; this crate defines traits and wiring.
//!
//! This is scaffolding only; implementations will land after interpreter/JIT semantics are complete.

use runmat_builtins::{Tensor, Value};

pub mod simple_provider;
#[cfg(feature = "wgpu")]
pub mod wgpu_backend;
use serde::{Deserialize, Serialize};

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

// NOTE: No concrete backend is provided in this crate yet. Future crates (or modules enabled via
// features) will implement `AccelerateBackend` for CUDA/ROCm/Metal/Vulkan/OpenCL/etc.
