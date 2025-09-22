use anyhow::{anyhow, Result};
use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use once_cell::sync::OnceCell;
use pollster::block_on;
use runmat_accelerate_api::{
    AccelProvider, ApiDeviceInfo, GpuTensorHandle, HostTensorOwned, HostTensorView, ReduceDimResult,
};
use log::info;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;
const REDUCE_WORKGROUP_SIZE: u32 = 256;
const MATMUL_TILE: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumericPrecision {
    F32,
    F64,
}

#[derive(Clone, Debug)]
pub struct WgpuProviderOptions {
    pub power_preference: wgpu::PowerPreference,
    pub force_fallback_adapter: bool,
}

impl Default for WgpuProviderOptions {
    fn default() -> Self {
        Self {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LenOpParams {
    len: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScalarParamsF64 {
    len: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
    scalar: f64,
    _pad_scalar: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScalarParamsF32 {
    len: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
    scalar: f32,
    _pad_scalar: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TransposeParams {
    rows: u32,
    cols: u32,
    len: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatmulParams {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReduceGlobalParams {
    len: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReduceDimParams {
    rows: u32,
    cols: u32,
    dim: u32,
    op: u32,
}

#[derive(Clone, Copy)]
enum BinaryOpCode {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
}

#[derive(Clone, Copy)]
enum UnaryOpCode {
    Sin = 0,
    Cos = 1,
    Abs = 2,
    Exp = 3,
    Log = 4,
    Sqrt = 5,
}

#[derive(Clone, Copy)]
enum ScalarOpCode {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    RSub = 4,
    RDiv = 5,
}

#[derive(Clone, Copy)]
enum GlobalReduceOp {
    Sum = 0,
    Min = 1,
    Max = 2,
}

#[derive(Clone, Copy)]
enum DimReduceOp {
    Sum = 0,
    Mean = 1,
}

#[derive(Clone, Copy)]
enum DimReduceExtrema {
    Min = 0,
    Max = 1,
}

struct PipelineBundle {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

struct WgpuPipelines {
    binary: PipelineBundle,
    unary: PipelineBundle,
    scalar: PipelineBundle,
    transpose: PipelineBundle,
    matmul: PipelineBundle,
    reduce_global: PipelineBundle,
    reduce_dim_sum_mean: PipelineBundle,
    reduce_dim_minmax: PipelineBundle,
}

impl WgpuPipelines {
    fn new(device: &wgpu::Device, precision: NumericPrecision) -> Self {
        let binary = create_pipeline(
            device,
            "runmat-binary-layout",
            "runmat-binary-shader",
            "runmat-binary-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => BINARY_SHADER_F64,
                NumericPrecision::F32 => BINARY_SHADER_F32,
            },
        );

        let unary = create_pipeline(
            device,
            "runmat-unary-layout",
            "runmat-unary-shader",
            "runmat-unary-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => UNARY_SHADER_F64,
                NumericPrecision::F32 => UNARY_SHADER_F32,
            },
        );

        let scalar = create_pipeline(
            device,
            "runmat-scalar-layout",
            "runmat-scalar-shader",
            "runmat-scalar-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => SCALAR_SHADER_F64,
                NumericPrecision::F32 => SCALAR_SHADER_F32,
            },
        );

        let transpose = create_pipeline(
            device,
            "runmat-transpose-layout",
            "runmat-transpose-shader",
            "runmat-transpose-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => TRANSPOSE_SHADER_F64,
                NumericPrecision::F32 => TRANSPOSE_SHADER_F32,
            },
        );

        let matmul = create_pipeline(
            device,
            "runmat-matmul-layout",
            "runmat-matmul-shader",
            "runmat-matmul-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => MATMUL_SHADER_F64,
                NumericPrecision::F32 => MATMUL_SHADER_F32,
            },
        );

        let reduce_global = create_pipeline(
            device,
            "runmat-reduce-global-layout",
            "runmat-reduce-global-shader",
            "runmat-reduce-global-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => REDUCE_GLOBAL_SHADER_F64,
                NumericPrecision::F32 => REDUCE_GLOBAL_SHADER_F32,
            },
        );

        let reduce_dim_sum_mean = create_pipeline(
            device,
            "runmat-reduce-dim-layout",
            "runmat-reduce-dim-shader",
            "runmat-reduce-dim-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => REDUCE_DIM_SHADER_F64,
                NumericPrecision::F32 => REDUCE_DIM_SHADER_F32,
            },
        );

        let reduce_dim_minmax = create_pipeline(
            device,
            "runmat-reduce-dim-minmax-layout",
            "runmat-reduce-dim-minmax-shader",
            "runmat-reduce-dim-minmax-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => REDUCE_DIM_MINMAX_SHADER_F64,
                NumericPrecision::F32 => REDUCE_DIM_MINMAX_SHADER_F32,
            },
        );

        Self {
            binary,
            unary,
            scalar,
            transpose,
            matmul,
            reduce_global,
            reduce_dim_sum_mean,
            reduce_dim_minmax,
        }
    }
}

struct BufferEntry {
    buffer: Arc<wgpu::Buffer>,
    len: usize,
    shape: Vec<usize>,
    precision: NumericPrecision,
}

pub struct WgpuProvider {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    buffers: Mutex<HashMap<u64, BufferEntry>>,
    next_id: AtomicU64,
    pipelines: WgpuPipelines,
    device_id: u32,
    precision: NumericPrecision,
    element_size: usize,
}

impl WgpuProvider {
    pub fn new(opts: WgpuProviderOptions) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: opts.power_preference,
            force_fallback_adapter: opts.force_fallback_adapter,
            compatible_surface: None,
        }))
        .ok_or_else(|| anyhow!("wgpu: no compatible adapter found"))?;

        let adapter_features = adapter.features();
        let precision = if adapter_features.contains(wgpu::Features::SHADER_F64) {
            NumericPrecision::F64
        } else {
            NumericPrecision::F32
        };

        let required_features = match precision {
            NumericPrecision::F64 => wgpu::Features::SHADER_F64,
            NumericPrecision::F32 => wgpu::Features::empty(),
        };
        let limits = adapter.limits();

        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("RunMat WGPU Device"),
                required_features,
                required_limits: limits.clone(),
            },
            None,
        ))?;

        let pipelines = WgpuPipelines::new(&device, precision);
        let adapter_info = adapter.get_info();
        let device_id = adapter_info.device;
        let element_size = match precision {
            NumericPrecision::F64 => std::mem::size_of::<f64>(),
            NumericPrecision::F32 => std::mem::size_of::<f32>(),
        };

        match precision {
            NumericPrecision::F64 => info!(
                "WGPU adapter '{}' supports shader-f64; using f64 kernels",
                adapter_info.name
            ),
            NumericPrecision::F32 => info!(
                "WGPU adapter '{}' lacks shader-f64; falling back to f32 kernels",
                adapter_info.name
            ),
        }

        Ok(Self {
            device,
            queue,
            adapter_info,
            buffers: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            pipelines,
            device_id,
            precision,
            element_size,
        })
    }

    fn register_existing_buffer(
        &self,
        buffer: Arc<wgpu::Buffer>,
        shape: Vec<usize>,
        len: usize,
    ) -> GpuTensorHandle {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let entry = BufferEntry {
            buffer,
            len,
            shape: shape.clone(),
            precision: self.precision,
        };
        self.buffers
            .lock()
            .expect("buffer mutex poisoned")
            .insert(id, entry);
        GpuTensorHandle {
            shape,
            device_id: self.device_id,
            buffer_id: id,
        }
    }

    fn create_storage_buffer(&self, len: usize, label: &str) -> Arc<wgpu::Buffer> {
        let size_bytes = (len.max(1) as u64) * self.element_size as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Arc::new(buffer)
    }

    fn uniform_buffer<T: Pod>(&self, data: &T, label: &str) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    fn get_entry(&self, handle: &GpuTensorHandle) -> Result<BufferEntry> {
        if handle.device_id != self.device_id {
            return Err(anyhow!(
                "handle device mismatch: expected {}, got {}",
                self.device_id,
                handle.device_id
            ));
        }
        let guard = self.buffers.lock().expect("buffer mutex poisoned");
        guard
            .get(&handle.buffer_id)
            .map(|entry| BufferEntry {
                buffer: entry.buffer.clone(),
                len: entry.len,
                shape: entry.shape.clone(),
                precision: entry.precision,
            })
            .ok_or_else(|| anyhow!("buffer not found: {}", handle.buffer_id))
    }

    fn binary_op(
        &self,
        op: BinaryOpCode,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("shape mismatch for binary op"));
        }
        let len = entry_a.len;
        let out_buffer = self.create_storage_buffer(len, "runmat-binary-out");
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry_a.shape, entry_a.len));
        }
        let params = LenOpParams {
            len: len as u32,
            op: op as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-binary-params");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runmat-binary-bind"),
            layout: &self.pipelines.binary.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: entry_a.buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: entry_b.buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-binary-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-binary-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.binary.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = dispatch_size(len as u32, WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.submit(encoder);
        Ok(self.register_existing_buffer(out_buffer, entry_a.shape, len))
    }

    fn unary_op(&self, op: UnaryOpCode, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let len = entry_a.len;
        let out_buffer = self.create_storage_buffer(len, "runmat-unary-out");
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry_a.shape, entry_a.len));
        }
        let params = LenOpParams {
            len: len as u32,
            op: op as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-unary-params");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runmat-unary-bind"),
            layout: &self.pipelines.unary.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: entry_a.buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-unary-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-unary-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.unary.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = dispatch_size(len as u32, WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.submit(encoder);
        Ok(self.register_existing_buffer(out_buffer, entry_a.shape, len))
    }

    fn scalar_op(
        &self,
        op: ScalarOpCode,
        a: &GpuTensorHandle,
        scalar: f64,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let len = entry_a.len;
        let out_buffer = self.create_storage_buffer(len, "runmat-scalar-out");
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry_a.shape, entry_a.len));
        }
        let params_buffer = match self.precision {
            NumericPrecision::F64 => {
                let params = ScalarParamsF64 {
                    len: len as u32,
                    op: op as u32,
                    _pad0: 0,
                    _pad1: 0,
                    scalar,
                    _pad_scalar: 0.0,
                };
                self.uniform_buffer(&params, "runmat-scalar-params")
            }
            NumericPrecision::F32 => {
                let params = ScalarParamsF32 {
                    len: len as u32,
                    op: op as u32,
                    _pad0: 0,
                    _pad1: 0,
                    scalar: scalar as f32,
                    _pad_scalar: [0.0; 3],
                };
                self.uniform_buffer(&params, "runmat-scalar-params")
            }
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runmat-scalar-bind"),
            layout: &self.pipelines.scalar.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: entry_a.buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-scalar-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-scalar-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.scalar.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = dispatch_size(len as u32, WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.submit(encoder);
        Ok(self.register_existing_buffer(out_buffer, entry_a.shape, len))
    }

    fn transpose_impl(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if entry.shape.len() != 2 {
            return Err(anyhow!("transpose: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let len = entry.len;
        let out_shape = vec![cols, rows];
        let out_buffer = self.create_storage_buffer(len, "runmat-transpose-out");
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, len));
        }
        let params = TransposeParams {
            rows: rows as u32,
            cols: cols as u32,
            len: len as u32,
            _pad: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-transpose-params");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runmat-transpose-bind"),
            layout: &self.pipelines.transpose.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: entry.buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-transpose-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-transpose-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.transpose.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = dispatch_size(len as u32, WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.submit(encoder);
        Ok(self.register_existing_buffer(out_buffer, out_shape, len))
    }

    fn matmul_impl(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape.len() != 2 || entry_b.shape.len() != 2 {
            return Err(anyhow!("matmul: only 2D tensors supported"));
        }
        let (m, k_a) = (entry_a.shape[0], entry_a.shape[1]);
        let (k_b, n) = (entry_b.shape[0], entry_b.shape[1]);
        if k_a != k_b {
            return Err(anyhow!("matmul: inner dimensions must match"));
        }
        let out_shape = vec![m, n];
        let len = m * n;
        let out_buffer = self.create_storage_buffer(len, "runmat-matmul-out");
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, len));
        }
        let params = MatmulParams {
            m: m as u32,
            n: n as u32,
            k: k_a as u32,
            lda: entry_a.shape[0] as u32,
            ldb: entry_b.shape[0] as u32,
            ldc: m as u32,
            _pad: [0, 0],
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-matmul-params");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runmat-matmul-bind"),
            layout: &self.pipelines.matmul.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: entry_a.buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: entry_b.buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-matmul-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-matmul-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.matmul.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups_x = dispatch_size_dim(n as u32, MATMUL_TILE);
            let groups_y = dispatch_size_dim(m as u32, MATMUL_TILE);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.submit(encoder);
        Ok(self.register_existing_buffer(out_buffer, out_shape, len))
    }

    fn reduce_global(&self, a: &GpuTensorHandle, op: GlobalReduceOp) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if entry.len == 0 {
            let default = match op {
                GlobalReduceOp::Sum => 0.0,
                GlobalReduceOp::Min => f64::INFINITY,
                GlobalReduceOp::Max => f64::NEG_INFINITY,
            };
            let data = [default];
            let shape = [1usize, 1usize];
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            return self.upload(&view);
        }
        let mut current = entry.buffer.clone();
        let mut current_len = entry.len;
        while current_len > 1 {
            let output_len = ((current_len + (REDUCE_WORKGROUP_SIZE as usize * 2) - 1)
                / (REDUCE_WORKGROUP_SIZE as usize * 2))
                .max(1);
            let out_buffer = self.create_storage_buffer(output_len, "runmat-reduce-pass");
            let params = ReduceGlobalParams {
                len: current_len as u32,
                op: op as u32,
                _pad0: 0,
                _pad1: 0,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-reduce-global-params");
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-reduce-global-bind"),
                layout: &self.pipelines.reduce_global.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: current.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-reduce-global-encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-reduce-global-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.reduce_global.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let groups = dispatch_size_reduce(current_len as u32, REDUCE_WORKGROUP_SIZE);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            self.submit(encoder);
            current = out_buffer;
            current_len = output_len;
        }
        Ok(self.register_existing_buffer(current, vec![1, 1], 1))
    }

    fn reduce_dim_sum_mean(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        op: DimReduceOp,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if entry.shape.len() != 2 {
            return Err(anyhow!("reduce: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let reduce_dim = if dim <= 1 { 1 } else { 2 };
        let out_len = if reduce_dim == 1 { cols } else { rows };
        let out_shape = if reduce_dim == 1 {
            vec![1, cols]
        } else {
            vec![rows, 1]
        };
        let out_buffer = self.create_storage_buffer(out_len, "runmat-reduce-dim-out");
        if out_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, out_len));
        }
        let params = ReduceDimParams {
            rows: rows as u32,
            cols: cols as u32,
            dim: reduce_dim as u32,
            op: op as u32,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-reduce-dim-params");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runmat-reduce-dim-bind"),
            layout: &self.pipelines.reduce_dim_sum_mean.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: entry.buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-reduce-dim-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-reduce-dim-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.reduce_dim_sum_mean.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = dispatch_size(out_len as u32, WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.submit(encoder);
        Ok(self.register_existing_buffer(out_buffer, out_shape, out_len))
    }

    fn reduce_dim_minmax(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        op: DimReduceExtrema,
    ) -> Result<ReduceDimResult> {
        let entry = self.get_entry(a)?;
        if entry.shape.len() != 2 {
            return Err(anyhow!("reduce: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let reduce_dim = if dim <= 1 { 1 } else { 2 };
        let out_len = if reduce_dim == 1 { cols } else { rows };
        let out_shape = if reduce_dim == 1 {
            vec![1, cols]
        } else {
            vec![rows, 1]
        };
        let values_buffer = self.create_storage_buffer(out_len, "runmat-reduce-dim-ext-values");
        let indices_buffer = self.create_storage_buffer(out_len, "runmat-reduce-dim-ext-indices");
        if out_len == 0 {
            let values_handle =
                self.register_existing_buffer(values_buffer, out_shape.clone(), out_len);
            let indices_handle = self.register_existing_buffer(indices_buffer, out_shape, out_len);
            return Ok(ReduceDimResult {
                values: values_handle,
                indices: indices_handle,
            });
        }
        let params = ReduceDimParams {
            rows: rows as u32,
            cols: cols as u32,
            dim: reduce_dim as u32,
            op: op as u32,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-reduce-dim-ext-params");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runmat-reduce-dim-ext-bind"),
            layout: &self.pipelines.reduce_dim_minmax.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: entry.buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: values_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indices_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-reduce-dim-ext-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-reduce-dim-ext-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.reduce_dim_minmax.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = dispatch_size(out_len as u32, WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.submit(encoder);
        let values_handle =
            self.register_existing_buffer(values_buffer, out_shape.clone(), out_len);
        let indices_handle = self.register_existing_buffer(indices_buffer, out_shape, out_len);
        Ok(ReduceDimResult {
            values: values_handle,
            indices: indices_handle,
        })
    }
}

impl AccelProvider for WgpuProvider {
    fn upload(&self, host: &HostTensorView) -> Result<GpuTensorHandle> {
        let len = host.data.len();
        let shape = host.shape.to_vec();
        let buffer = if len == 0 {
            self.create_storage_buffer(0, "runmat-upload-empty")
        } else {
            match self.precision {
                NumericPrecision::F64 => {
                    let contents = cast_slice(host.data);
                    Arc::new(
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-upload-buffer"),
                                contents,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            }),
                    )
                }
                NumericPrecision::F32 => {
                    let data_f32: Vec<f32> = host.data.iter().map(|v| *v as f32).collect();
                    let contents = cast_slice(&data_f32);
                    Arc::new(
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-upload-buffer"),
                                contents,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            }),
                    )
                }
            }
        };
        Ok(self.register_existing_buffer(buffer, shape, len))
    }

    fn download(&self, h: &GpuTensorHandle) -> Result<HostTensorOwned> {
        let entry = self.get_entry(h)?;
        if entry.len == 0 {
            return Ok(HostTensorOwned {
                data: Vec::new(),
                shape: entry.shape,
            });
        }
        let size_bytes = (entry.len * self.element_size) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-download-staging"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-download-encoder"),
            });
        encoder.copy_buffer_to_buffer(entry.buffer.as_ref(), 0, &staging, 0, size_bytes);
        self.submit(encoder);
        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| anyhow!("map_async callback dropped"))?
            .map_err(|e| anyhow!(e))?;
        let data = slice.get_mapped_range();
        let mut out = vec![0.0f64; entry.len];
        match entry.precision {
            NumericPrecision::F64 => {
                out.copy_from_slice(cast_slice(&data));
            }
            NumericPrecision::F32 => {
                let f32_slice: &[f32] = cast_slice(&data);
                for (dst, src) in out.iter_mut().zip(f32_slice.iter()) {
                    *dst = *src as f64;
                }
            }
        }
        drop(data);
        staging.unmap();
        Ok(HostTensorOwned {
            data: out,
            shape: entry.shape,
        })
    }

    fn free(&self, h: &GpuTensorHandle) -> Result<()> {
        let mut guard = self.buffers.lock().expect("buffer mutex poisoned");
        guard.remove(&h.buffer_id);
        Ok(())
    }

    fn device_info(&self) -> String {
        format!(
            "{} ({:?})",
            self.adapter_info.name, self.adapter_info.backend
        )
    }

    fn device_info_struct(&self) -> ApiDeviceInfo {
        ApiDeviceInfo {
            device_id: self.device_id,
            name: self.adapter_info.name.clone(),
            vendor: self.adapter_info.vendor.to_string(),
            memory_bytes: None,
            backend: Some(format!("{:?}", self.adapter_info.backend)),
        }
    }

    fn elem_add(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op(BinaryOpCode::Add, a, b)
    }

    fn elem_mul(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op(BinaryOpCode::Mul, a, b)
    }

    fn elem_sub(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op(BinaryOpCode::Sub, a, b)
    }

    fn elem_div(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op(BinaryOpCode::Div, a, b)
    }

    fn unary_sin(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op(UnaryOpCode::Sin, a)
    }

    fn unary_cos(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op(UnaryOpCode::Cos, a)
    }

    fn unary_abs(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op(UnaryOpCode::Abs, a)
    }

    fn unary_exp(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op(UnaryOpCode::Exp, a)
    }

    fn unary_log(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op(UnaryOpCode::Log, a)
    }

    fn unary_sqrt(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op(UnaryOpCode::Sqrt, a)
    }

    fn scalar_add(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op(ScalarOpCode::Add, a, scalar)
    }

    fn scalar_sub(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op(ScalarOpCode::Sub, a, scalar)
    }

    fn scalar_mul(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op(ScalarOpCode::Mul, a, scalar)
    }

    fn scalar_div(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op(ScalarOpCode::Div, a, scalar)
    }

    fn scalar_rsub(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op(ScalarOpCode::RSub, a, scalar)
    }

    fn scalar_rdiv(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op(ScalarOpCode::RDiv, a, scalar)
    }

    fn matmul(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.matmul_impl(a, b)
    }

    fn transpose(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.transpose_impl(a)
    }

    fn reduce_sum(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.reduce_global(a, GlobalReduceOp::Sum)
    }

    fn reduce_sum_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<GpuTensorHandle> {
        self.reduce_dim_sum_mean(a, dim, DimReduceOp::Sum)
    }

    fn reduce_mean(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let sum_handle = self.reduce_global(a, GlobalReduceOp::Sum)?;
        let entry = self.get_entry(a)?;
        let denom = entry.len.max(1) as f64;
        self.scalar_op(ScalarOpCode::Div, &sum_handle, denom)
    }

    fn reduce_mean_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<GpuTensorHandle> {
        self.reduce_dim_sum_mean(a, dim, DimReduceOp::Mean)
    }

    fn reduce_min(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.reduce_global(a, GlobalReduceOp::Min)
    }

    fn reduce_min_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<ReduceDimResult> {
        self.reduce_dim_minmax(a, dim, DimReduceExtrema::Min)
    }

    fn reduce_max(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.reduce_global(a, GlobalReduceOp::Max)
    }

    fn reduce_max_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<ReduceDimResult> {
        self.reduce_dim_minmax(a, dim, DimReduceExtrema::Max)
    }
}

pub fn register_wgpu_provider(opts: WgpuProviderOptions) -> Result<&'static WgpuProvider> {
    static INSTANCE: OnceCell<&'static WgpuProvider> = OnceCell::new();
    INSTANCE
        .get_or_try_init(move || {
            let provider = WgpuProvider::new(opts)?;
            let leaked: &'static WgpuProvider = Box::leak(Box::new(provider));
            unsafe { runmat_accelerate_api::register_provider(leaked) };
            Ok(leaked)
        })
        .map(|p| *p)
}

pub fn ensure_wgpu_provider() -> Result<Option<&'static WgpuProvider>> {
    match register_wgpu_provider(WgpuProviderOptions::default()) {
        Ok(p) => Ok(Some(p)),
        Err(e) => {
            log::warn!("RunMat Accelerate: wgpu provider initialization failed: {e}");
            Ok(None)
        }
    }
}

fn dispatch_size(elements: u32, workgroup: u32) -> u32 {
    if elements == 0 {
        0
    } else {
        ((elements + workgroup - 1) / workgroup).max(1)
    }
}

fn dispatch_size_reduce(elements: u32, workgroup: u32) -> u32 {
    if elements == 0 {
        0
    } else {
        ((elements + workgroup * 2 - 1) / (workgroup * 2)).max(1)
    }
}

fn dispatch_size_dim(elements: u32, tile: u32) -> u32 {
    if elements == 0 {
        0
    } else {
        ((elements + tile - 1) / tile).max(1)
    }
}

fn storage_read_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_read_write_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    layout_label: &str,
    shader_label: &str,
    pipeline_label: &str,
    entries: Vec<wgpu::BindGroupLayoutEntry>,
    shader_source: &str,
) -> PipelineBundle {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(layout_label),
        entries: &entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&(String::from(pipeline_label) + "-layout")),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(shader_label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(pipeline_label),
        module: &module,
        layout: Some(&pipeline_layout),
        entry_point: "main",
    });
    PipelineBundle { pipeline, layout }
}

const BINARY_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

fn apply(a: f64, b: f64) -> f64 {
    switch params.op {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        default: { return a; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    Out.data[idx] = apply(A.data[idx], B.data[idx]);
}
"#;

const BINARY_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

fn apply(a: f32, b: f32) -> f32 {
    switch params.op {
        default: { return pow(a, b); }
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    Out.data[idx] = apply(A.data[idx], B.data[idx]);
}
"#;

const UNARY_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn apply(a: f64) -> f64 {
    switch params.op {
        case 0u: { return sin(a); }
        case 1u: { return cos(a); }
        case 2u: { return abs(a); }
        case 3u: { return exp(a); }
        case 4u: { return log(a); }
        case 5u: { return sqrt(a); }
        default: { return a; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    Out.data[idx] = apply(A.data[idx]);
}
"#;

const UNARY_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn apply(a: f32) -> f32 {
    switch params.op {
        case 0u: { return sin(a); }
        case 1u: { return cos(a); }
        case 2u: { return abs(a); }
        case 3u: { return exp(a); }
        case 4u: { return log(a); }
        case 5u: { return sqrt(a); }
        default: { return a; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    Out.data[idx] = apply(A.data[idx]);
}
"#;

const SCALAR_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
    scalar: f64,
    scalar_pad: f64,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    let a = A.data[idx];
    let scalar = params.scalar;
    var result: f64 = a;
    switch params.op {
        case 0u: { result = a + scalar; }
        case 1u: { result = a - scalar; }
        case 2u: { result = a * scalar; }
        case 3u: { result = a / scalar; }
        case 4u: { result = scalar - a; }
        case 5u: { result = scalar / a; }
        default: { result = a; }
    }
    Out.data[idx] = result;
}
"#;

const SCALAR_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
    scalar: f32,
    scalar_pad: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    let a = A.data[idx];
    let s = params.scalar;
    var result: f32 = a;
    switch params.op {
        case 0u: { result = a + s; }
        case 1u: { result = a - s; }
        case 2u: { result = a * s; }
        case 3u: { result = a / s; }
        case 4u: { result = s - a; }
        case 5u: { result = s / a; }
        default: { result = a; }
    }
    Out.data[idx] = result;
}
"#;

const TRANSPOSE_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    rows: u32,
    cols: u32,
    len: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    let rows = params.rows;
    let cols = params.cols;
    let row = idx % rows;
    let col = idx / rows;
    let out_rows = cols;
    let tgt_idx = col + row * out_rows;
    Out.data[tgt_idx] = A.data[idx];
}
"#;

const TRANSPOSE_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    rows: u32,
    cols: u32,
    len: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    let rows = params.rows;
    let cols = params.cols;
    let row = idx % rows;
    let col = idx / rows;
    let out_rows = cols;
    let tgt_idx = col + row * out_rows;
    Out.data[tgt_idx] = A.data[idx];
}
"#;

const MATMUL_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let col = gid.x;
    let row = gid.y;
    if row >= params.m || col >= params.n {
        return;
    }
    var acc: f64 = 0.0;
    let lda = params.lda;
    let ldb = params.ldb;
    let ldc = params.ldc;
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        let a_idx = row + kk * lda;
        let b_idx = kk + col * ldb;
        acc = acc + A.data[a_idx] * B.data[b_idx];
    }
    let out_idx = row + col * ldc;
    Out.data[out_idx] = acc;
}
"#;

const MATMUL_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let col = gid.x;
    let row = gid.y;
    if row >= params.m || col >= params.n {
        return;
    }
    var acc: f32 = 0.0;
    let lda = params.lda;
    let ldb = params.ldb;
    let ldc = params.ldc;
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        let a_idx = row + kk * lda;
        let b_idx = kk + col * ldb;
        acc = acc + A.data[a_idx] * B.data[b_idx];
    }
    let out_idx = row + col * ldc;
    Out.data[out_idx] = acc;
}
"#;

const REDUCE_GLOBAL_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutBuf: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared: array<f64, 256>;

fn combine(a: f64, b: f64, op: u32) -> f64 {
    switch op {
        case 0u: { return a + b; }
        case 1u: {
            if b < a {
                return b;
            }
            return a;
        }
        case 2u: {
            if b > a {
                return b;
            }
            return a;
        }
        default: { return a; }
    }
}

fn identity(op: u32) -> f64 {
    switch op {
        case 0u: { return 0.0; }
        case 1u: { return 1.0 / 0.0; }
        case 2u: { return -1.0 / 0.0; }
        default: { return 0.0; }
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let base = wid.x * 512u;
    let idx = base + lid.x;
    var acc = identity(params.op);
    if idx < params.len {
        acc = InBuf.data[idx];
    }
    if idx + 256u < params.len {
        acc = combine(acc, InBuf.data[idx + 256u], params.op);
    }
    shared[lid.x] = acc;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if stride == 0u {
            break;
        }
        if lid.x < stride {
            shared[lid.x] = combine(shared[lid.x], shared[lid.x + stride], params.op);
        }
        stride = stride / 2u;
        workgroupBarrier();
    }
    if lid.x == 0u {
        OutBuf.data[wid.x] = shared[0u];
    }
}
"#;

const REDUCE_GLOBAL_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutBuf: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared: array<f32, 256>;

fn combine(a: f32, b: f32, op: u32) -> f32 {
    switch op {
        case 0u: { return a + b; }
        case 1u: { return select(a, b, b < a); }
        case 2u: { return select(a, b, b > a); }
        default: { return a; }
    }
}

fn identity(op: u32) -> f32 {
    switch op {
        case 0u: { return 0.0; }
        case 1u: { return 1.0 / 0.0; }
        case 2u: { return -1.0 / 0.0; }
        default: { return 0.0; }
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let base = wid.x * 512u;
    let idx = base + lid.x;
    var acc = identity(params.op);
    if idx < params.len {
        acc = InBuf.data[idx];
    }
    if idx + 256u < params.len {
        acc = combine(acc, InBuf.data[idx + 256u], params.op);
    }
    shared[lid.x] = acc;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if stride == 0u {
            break;
        }
        if lid.x < stride {
            shared[lid.x] = combine(shared[lid.x], shared[lid.x + stride], params.op);
        }
        stride = stride / 2u;
        workgroupBarrier();
    }
    if lid.x == 0u {
        OutBuf.data[wid.x] = shared[0u];
    }
}
"#;

const REDUCE_DIM_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    rows: u32,
    cols: u32,
    dim: u32,
    op: u32,
};

@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutBuf: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if params.dim == 1u {
        if idx >= params.cols {
            return;
        }
        var acc: f64 = 0.0;
        for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
            let linear = r + idx * params.rows;
            acc = acc + InBuf.data[linear];
        }
        if params.op == 1u {
            acc = acc / f64(params.rows);
        }
        OutBuf.data[idx] = acc;
    } else {
        if idx >= params.rows {
            return;
        }
        var acc: f64 = 0.0;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let linear = idx + c * params.rows;
            acc = acc + InBuf.data[linear];
        }
        if params.op == 1u {
            acc = acc / f64(params.cols);
        }
        OutBuf.data[idx] = acc;
    }
}
"#;

const REDUCE_DIM_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    rows: u32,
    cols: u32,
    dim: u32,
    op: u32,
};

@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutBuf: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if params.dim == 1u {
        if idx >= params.cols {
            return;
        }
        var acc: f32 = 0.0;
        for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
            let linear = r + idx * params.rows;
            acc = acc + InBuf.data[linear];
        }
        if params.op == 1u {
            acc = acc / f32(params.rows);
        }
        OutBuf.data[idx] = acc;
    } else {
        if idx >= params.rows {
            return;
        }
        var acc: f32 = 0.0;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let linear = idx + c * params.rows;
            acc = acc + InBuf.data[linear];
        }
        if params.op == 1u {
            acc = acc / f32(params.cols);
        }
        OutBuf.data[idx] = acc;
    }
}
"#;

const REDUCE_DIM_MINMAX_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    rows: u32,
    cols: u32,
    dim: u32,
    op: u32,
};

@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutVals: Tensor;
@group(0) @binding(2) var<storage, read_write> OutIdx: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

fn better(current: f64, candidate: f64, op: u32) -> bool {
    if op == 0u {
        return candidate < current;
    }
    return candidate > current;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if params.dim == 1u {
        if idx >= params.cols {
            return;
        }
        var best: f64;
        if params.op == 0u {
            best = 1.0 / 0.0;
        } else {
            best = -1.0 / 0.0;
        }
        var best_idx: u32 = 1u;
        for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
            let linear = r + idx * params.rows;
            let value = InBuf.data[linear];
            if r == 0u || better(best, value, params.op) {
                best = value;
                best_idx = r + 1u;
            }
        }
        OutVals.data[idx] = best;
        OutIdx.data[idx] = f64(best_idx);
    } else {
        if idx >= params.rows {
            return;
        }
        var best: f64;
        if params.op == 0u {
            best = 1.0 / 0.0;
        } else {
            best = -1.0 / 0.0;
        }
        var best_idx: u32 = 1u;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let linear = idx + c * params.rows;
            let value = InBuf.data[linear];
            if c == 0u || better(best, value, params.op) {
                best = value;
                best_idx = c + 1u;
            }
        }
        OutVals.data[idx] = best;
        OutIdx.data[idx] = f64(best_idx);
    }
}
"#;

const REDUCE_DIM_MINMAX_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    rows: u32,
    cols: u32,
    dim: u32,
    op: u32,
};

@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutVals: Tensor;
@group(0) @binding(2) var<storage, read_write> OutIdx: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

fn better(current: f32, candidate: f32, op: u32) -> bool {
    if op == 0u {
        return candidate < current;
    }
    return candidate > current;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if params.dim == 1u {
        if idx >= params.cols {
            return;
        }
        var best: f32 = if params.op == 0u { 1.0 / 0.0 } else { -1.0 / 0.0 };
        var best_idx: u32 = 1u;
        for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
            let linear = r + idx * params.rows;
            let value = InBuf.data[linear];
            if r == 0u || better(best, value, params.op) {
                best = value;
                best_idx = r + 1u;
            }
        }
        OutVals.data[idx] = best;
        OutIdx.data[idx] = f32(best_idx);
    } else {
        if idx >= params.rows {
            return;
        }
        var best: f32 = if params.op == 0u { 1.0 / 0.0 } else { -1.0 / 0.0 };
        var best_idx: u32 = 1u;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let linear = idx + c * params.rows;
            let value = InBuf.data[linear];
            if c == 0u || better(best, value, params.op) {
                best = value;
                best_idx = c + 1u;
            }
        }
        OutVals.data[idx] = best;
        OutIdx.data[idx] = f32(best_idx);
    }
}
"#;
