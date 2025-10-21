use anyhow::{anyhow, Result};
use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use log::info;
use once_cell::sync::OnceCell;
use pollster::block_on;
use runmat_accelerate_api::{
    AccelProvider, ApiDeviceInfo, GpuTensorHandle, HostTensorOwned, HostTensorView,
    ProviderPrecision, ReduceDimResult,
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
// (no futures/async; synchronous creation with instrumentation)
use std::time::Instant;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;
const REDUCE_WORKGROUP_SIZE: u32 = 256;
const DEFAULT_TWO_PASS_THRESHOLD: usize = 1024;
const DEFAULT_REDUCTION_WG: u32 = 256;
const MATMUL_TILE: u32 = 16;
const MAX_DISPATCH_WORKGROUPS: u32 = 65_535;

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
    offset: u32,
    total: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScalarParamsF64 {
    len: u32,
    op: u32,
    offset: u32,
    total: u32,
    scalar: f64,
    _pad_scalar: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScalarParamsF32 {
    len: u32,
    op: u32,
    offset: u32,
    total: u32,
    scalar: f32,
    _pad_scalar: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FusionParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TransposeParams {
    rows: u32,
    cols: u32,
    len: u32,
    offset: u32,
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
    pub device_id: u32,
    precision: NumericPrecision,
    element_size: usize,
    // Simple pipeline cache for fused kernels keyed by shader hash and layout signature
    fused_pipeline_cache: Mutex<HashMap<u64, Arc<wgpu::ComputePipeline>>>,
    // Metrics counters
    fused_cache_hits: AtomicU64,
    fused_cache_misses: AtomicU64,
    last_warmup_millis: AtomicU64,
    // Tunables
    reduction_two_pass_threshold: usize,
    reduction_workgroup_size_default: u32,
    // Optional on-disk pipeline cache directory (lazy created)
    pipeline_cache_dir: Option<std::path::PathBuf>,
}

#[derive(Serialize, Deserialize)]
struct PipelineMeta {
    label: String,
    layout_tag: Option<String>,
    workgroup_size: Option<u32>,
}

impl WgpuProvider {
    // (threaded pipeline builder removed)

    fn build_bgl_for_layout_tag(&self, tag: &str) -> Option<wgpu::BindGroupLayout> {
        // Elementwise: runmat-fusion-layout-<n_inputs>
        if let Some(rest) = tag.strip_prefix("runmat-fusion-layout-") {
            if let Ok(n_inputs) = rest.parse::<usize>() {
                let mut entries = Vec::with_capacity(n_inputs + 2);
                for i in 0..n_inputs {
                    entries.push(storage_read_entry(i as u32));
                }
                entries.push(storage_read_write_entry(n_inputs as u32));
                entries.push(uniform_entry((n_inputs + 1) as u32));
                return Some(self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-fusion-bgl"),
                    entries: &entries,
                }));
            }
        }
        // Reductions: single pass and two-pass use the same 3-entry layout
        match tag {
            "runmat-reduction-bgl" | "runmat-reduction-p1-bgl" | "runmat-reduction-p2-bgl" => {
                let entries = [storage_read_entry(0), storage_read_write_entry(1), uniform_entry(2)];
                return Some(self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("warmup-reduction-bgl"),
                    entries: &entries,
                }));
            }
            _ => {}
        }
        None
    }

    fn warmup_from_disk(&self) {
        let Some(dir) = &self.pipeline_cache_dir else { return; };
        let Ok(rd) = std::fs::read_dir(dir) else { return; };
        let mut compiled = 0usize;
        for entry in rd.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") { continue; }
            let stem = match path.file_stem().and_then(|s| s.to_str()) { Some(s) => s, None => continue };
            // Read meta
            let meta_bytes = match std::fs::read(&path) { Ok(b) => b, Err(_) => continue };
            let meta: PipelineMeta = match serde_json::from_slice(&meta_bytes) { Ok(m) => m, Err(_) => continue };
            let layout_tag = match meta.layout_tag.as_deref() { Some(t) => t, None => continue };
            // Read WGSL
            let wgsl_path = dir.join(format!("{stem}.wgsl"));
            let wgsl_bytes = match std::fs::read(&wgsl_path) { Ok(b) => b, Err(_) => continue };
            let wgsl_str = match std::str::from_utf8(&wgsl_bytes) { Ok(s) => s, Err(_) => continue };
            // Recreate bind group layout and pipeline layout
            let bgl = match self.build_bgl_for_layout_tag(layout_tag) { Some(b) => b, None => continue };
            let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("warmup-pipeline-layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
            // Shader module
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("warmup-shader-module"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl_str)),
            });
            // Compute key and create pipeline (which inserts into cache and persists meta)
            let key = self.compute_pipeline_hash_bytes(&wgsl_bytes, layout_tag, meta.workgroup_size);
            let pipeline = self.get_or_create_pipeline(
                key,
                &pl,
                &module,
                "warmup-precompiled-pipeline",
                Some(&wgsl_bytes),
                Some(layout_tag),
                meta.workgroup_size,
            );
            // Optional tiny noop pass to nudge driver state (no bindings required)
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("warmup-noop-precompiled-enc"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("warmup-noop-precompiled-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&*pipeline);
            }
            self.submit(enc);
            compiled += 1;
        }
        if compiled > 0 {
            log::info!("warmup: precompiled {} pipelines from on-disk cache", compiled);
        }
    }

    pub fn try_compile_kernel(&self, label: &str, wgsl_src: &str) -> Result<()> {
        let t0 = Instant::now();
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{}-module", label)),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl_src)),
            });
        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{}-pl", label)),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });
        let _pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{}-pipeline", label)),
                layout: Some(&pl),
                module: &module,
                entry_point: "main",
            });
        log::info!(
            "try_compile_kernel: '{}' compiled in {:.3} ms",
            label,
            t0.elapsed().as_secs_f64() * 1000.0
        );
        Ok(())
    }

    pub fn probe_kernel_with_buffers(&self, label: &str, wgsl_src: &str, wg: u32) -> Result<()> {
        let t0 = Instant::now();
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{}-module", label)),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl_src)),
            });
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{}-bgl", label)),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{}-pl", label)),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{}-pipeline", label)),
                layout: Some(&pl),
                module: &module,
                entry_point: "main",
            });
        // tiny buffers
        let in_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}-in", label)),
            contents: bytemuck::cast_slice(&[0.0f32; 4]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let out_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}-out", label)),
            contents: bytemuck::cast_slice(&[0.0f32; 4]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{}-bg", label)),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: in_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
            ],
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(&format!("{}-enc", label)) });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(&format!("{}-pass", label)), timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1.max(wg / wg), 1, 1);
        }
        self.submit(enc);
        log::info!(
            "probe_kernel_with_buffers: '{}' compiled+submitted in {:.3} ms",
            label,
            t0.elapsed().as_secs_f64() * 1000.0
        );
        Ok(())
    }

    /// Get or create a compute pipeline from cache using a caller-provided hash key.
    /// The hash should incorporate shader source and any layout/workgroup parameters
    /// that affect compatibility. This helper unifies cache lookups and creation.
    fn get_or_create_pipeline(
        &self,
        hash_key: u64,
        pipeline_layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        label: &str,
        persist_wgsl_src: Option<&[u8]>,
        persist_layout_tag: Option<&str>,
        persist_workgroup_size: Option<u32>,
    ) -> Arc<wgpu::ComputePipeline> {
        if let Some(p) = self
            .fused_pipeline_cache
            .try_lock()
            .ok()
            .and_then(|guard| guard.get(&hash_key).cloned())
        {
            self.fused_cache_hits.fetch_add(1, Ordering::Relaxed);
            return p;
        }
        self.fused_cache_misses.fetch_add(1, Ordering::Relaxed);
        // Persist WGSL + meta for warmup on next run
        self.persist_pipeline_meta(hash_key, label, persist_layout_tag, persist_workgroup_size, persist_wgsl_src);
        let p = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(pipeline_layout),
            module,
            entry_point: "main",
        }));
        if let Ok(mut guard) = self.fused_pipeline_cache.try_lock() {
            guard.insert(hash_key, p.clone());
        }
        p
    }

    pub fn compute_pipeline_hash_bytes(
        &self,
        shader_bytes: &[u8],
        layout_tag: &str,
        workgroup_size: Option<u32>,
    ) -> u64 {
        let mut hasher = DefaultHasher::new();
        shader_bytes.hash(&mut hasher);
        layout_tag.hash(&mut hasher);
        if let Some(wg) = workgroup_size {
            wg.hash(&mut hasher);
        }
        hasher.finish()
    }

    fn persist_pipeline_meta(
        &self,
        hash_key: u64,
        label: &str,
        layout_tag: Option<&str>,
        workgroup_size: Option<u32>,
        wgsl_src: Option<&[u8]>,
    ) {
        if let Some(dir) = &self.pipeline_cache_dir {
            let _ = std::fs::create_dir_all(dir);
            if let Some(src) = wgsl_src {
                let wgsl_path = dir.join(format!("{hash_key:016x}.wgsl"));
                let _ = std::fs::write(&wgsl_path, src);
            }
            let meta = PipelineMeta {
                label: label.to_string(),
                layout_tag: layout_tag.map(|s| s.to_string()),
                workgroup_size,
            };
            let meta_path = dir.join(format!("{hash_key:016x}.json"));
            if let Ok(json) = serde_json::to_vec_pretty(&meta) {
                let _ = std::fs::write(&meta_path, json);
            }
        }
    }
    // (removed async helper)
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

        // Tunables with env overrides
        let two_pass_threshold = std::env::var("RUNMAT_TWO_PASS_THRESHOLD")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_TWO_PASS_THRESHOLD);
        let reduction_wg_default = std::env::var("RUNMAT_REDUCTION_WG")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(DEFAULT_REDUCTION_WG);

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

        // Choose a cache dir: prefer RUNMAT_PIPELINE_CACHE_DIR, else OS cache dir
        let cache_dir = if let Ok(custom) = std::env::var("RUNMAT_PIPELINE_CACHE_DIR") {
            std::path::PathBuf::from(custom)
        } else if let Some(base) = dirs::cache_dir() {
            base.join("runmat").join("pipelines").join(format!("device-{}", device_id))
        } else {
            // Fallback to local target/tmp
            std::path::PathBuf::from("target").join("tmp").join(format!("wgpu-pipeline-cache-{}", device_id))
        };

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
            fused_pipeline_cache: Mutex::new(HashMap::new()),
            fused_cache_hits: AtomicU64::new(0),
            fused_cache_misses: AtomicU64::new(0),
            reduction_two_pass_threshold: two_pass_threshold,
            reduction_workgroup_size_default: reduction_wg_default,
            pipeline_cache_dir: Some(cache_dir),
            last_warmup_millis: AtomicU64::new(0),
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
        if len == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-binary-out");
            return Ok(self.register_existing_buffer(out_buffer, entry_a.shape, entry_a.len));
        }
        if len > (u32::MAX as usize) {
            return Err(anyhow!("tensor too large for GPU buffer"));
        }

        // Metal workaround: warm up pipeline and poll before first real dispatch to avoid stalls
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-binary-noop"),
                });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-binary-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.binary.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device.poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-binary-flush-gap"),
                });
            self.submit(enc);
        }

        let out_buffer = self.create_storage_buffer(len, "runmat-binary-out");
        let chunk_capacity = (MAX_DISPATCH_WORKGROUPS as usize) * WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < len {
            let remaining = len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = LenOpParams {
                len: chunk_len as u32,
                op: op as u32,
                offset: offset as u32,
                total: len as u32,
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
                let groups = dispatch_size(chunk_len as u32, WORKGROUP_SIZE);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            self.submit(encoder);
            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, entry_a.shape, len))
    }

    fn unary_op(&self, op: UnaryOpCode, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let len = entry_a.len;
        let out_buffer = self.create_storage_buffer(len, "runmat-unary-out");
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry_a.shape, entry_a.len));
        }
        if len > (u32::MAX as usize) {
            return Err(anyhow!("tensor too large for GPU buffer"));
        }

        // Metal workaround: warm up pipeline and poll before first real dispatch to avoid stalls
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-unary-noop"),
                });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-unary-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.unary.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device.poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-unary-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (MAX_DISPATCH_WORKGROUPS as usize) * WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < len {
            let remaining = len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = LenOpParams {
                len: chunk_len as u32,
                op: op as u32,
                offset: offset as u32,
                total: len as u32,
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
                let groups = dispatch_size(chunk_len as u32, WORKGROUP_SIZE);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            self.submit(encoder);
            offset += chunk_len;
        }

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
        if len > (u32::MAX as usize) {
            return Err(anyhow!("tensor too large for GPU buffer"));
        }

        let chunk_capacity = (MAX_DISPATCH_WORKGROUPS as usize) * WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < len {
            let remaining = len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = ScalarParamsF64 {
                        len: chunk_len as u32,
                        op: op as u32,
                        offset: offset as u32,
                        total: len as u32,
                        scalar,
                        _pad_scalar: 0.0,
                    };
                    self.uniform_buffer(&params, "runmat-scalar-params")
                }
                NumericPrecision::F32 => {
                    let params = ScalarParamsF32 {
                        len: chunk_len as u32,
                        op: op as u32,
                        offset: offset as u32,
                        total: len as u32,
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
                let groups = dispatch_size(chunk_len as u32, WORKGROUP_SIZE);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            self.submit(encoder);
            offset += chunk_len;
        }

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
        if len > (u32::MAX as usize) {
            return Err(anyhow!("tensor too large for GPU buffer"));
        }

        // Metal workaround: warm up pipeline and poll before first real dispatch to avoid stalls
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-transpose-noop"),
                });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-transpose-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.transpose.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device.poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-transpose-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (MAX_DISPATCH_WORKGROUPS as usize) * WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < len {
            let remaining = len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = TransposeParams {
                rows: rows as u32,
                cols: cols as u32,
                len: chunk_len as u32,
                offset: offset as u32,
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
                let groups = dispatch_size(chunk_len as u32, WORKGROUP_SIZE);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            self.submit(encoder);
            offset += chunk_len;
        }

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
        // Metal workaround: warm up pipeline and poll before first real dispatch to avoid stalls
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-matmul-noop"),
                });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-matmul-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.matmul.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device.poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-matmul-flush-gap"),
                });
            self.submit(enc);
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
        let reduce_dim = match dim {
            0 => 1,
            1 => 2,
            _ => return Err(anyhow!("reduce_dim: only dims 0 or 1 supported")),
        };
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
        let reduce_dim = match dim {
            0 => 1,
            1 => 2,
            _ => return Err(anyhow!("reduce_dim: only dims 0 or 1 supported")),
        };
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

    fn warmup_internal(&self) {
        // Preload any pipelines from on-disk cache, then synthesize a few tiny kernels
        self.warmup_from_disk();
        let start = std::time::Instant::now();
        // Compile trivial elementwise and reduction templates to populate cache
        // Elementwise: minimal passthrough (len is small)
        let shader = r#"struct Tensor { data: array<f32> };
struct Params { len: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  output.data[idx] = input0.data[idx];
}
"#;
        // Set up a tiny buffer and run fused_elementwise to trigger pipeline creation
        let view = HostTensorView { data: &[0.0f64; 4], shape: &[4usize] };
        if let Ok(h) = self.upload(&view) {
            let _ = self.fused_elementwise(shader, &[h.clone()], &[4], 4);
            let _ = self.free(&h);
        }

        // Reduction warmup: single-pass for small reduce_len
        // Two-pass threshold tuning: keep workgroup 256; vary small nrows
        let red = r#"struct Tensor { data: array<f32> };
struct MParams { nrows:u32, ncols:u32, ld:u32, flags:u32 }
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: MParams;
const OMITNAN: bool = false;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let slice = wid.x; if (slice >= params.ncols) { return; }
  var acc: f32 = 0.0; var r = lid.x; while (r < params.nrows) {
    let v = input0.data[(slice * params.ld) + r]; acc = acc + v; r += 64u; }
  var<workgroup> tile: array<f32,64u>; tile[lid.x] = acc; workgroupBarrier();
  var off = 32u; loop { if (off==0u){break;} if(lid.x<off){tile[lid.x]=tile[lid.x]+tile[lid.x+off];} workgroupBarrier(); off=off/2u; }
  if (lid.x==0u) { output.data[slice] = tile[0u]; }
}
"#;
        let view2 = HostTensorView { data: &[1.0f64; 8], shape: &[4usize, 2usize] };
        if let Ok(h2) = self.upload(&view2) {
            let _ = self.fused_reduction(red, &[h2.clone()], &[2], 4, 2, 64);
            let _ = self.free(&h2);
        }
        // Additional tiny warmup for a different shader hash size
        let view3 = HostTensorView { data: &[2.0f64; 16], shape: &[8usize, 2usize] };
        if let Ok(h3) = self.upload(&view3) {
            let _ = self.fused_reduction(red, &[h3.clone()], &[2], 8, 2, 64);
            let _ = self.free(&h3);
        }
        let elapsed = start.elapsed();
        let (hits, misses) = self.fused_cache_counters();
        log::info!(
            "WGPU warmup completed in {:?}; fused cache hits={}, misses={}",
            elapsed, hits, misses
        );
        self.last_warmup_millis
            .store((elapsed.as_secs_f64() * 1000.0) as u64, Ordering::Relaxed);
    }
}

impl AccelProvider for WgpuProvider {
    fn precision(&self) -> ProviderPrecision {
        match self.precision {
            NumericPrecision::F32 => ProviderPrecision::F32,
            NumericPrecision::F64 => ProviderPrecision::F64,
        }
    }

    fn upload(&self, host: &HostTensorView) -> Result<GpuTensorHandle> {
        let len = host.data.len();
        let shape = host.shape.to_vec();
        let buffer =
            if len == 0 {
                self.create_storage_buffer(0, "runmat-upload-empty")
            } else {
                match self.precision {
                    NumericPrecision::F64 => {
                        let contents = cast_slice(host.data);
                        Arc::new(self.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-upload-buffer"),
                                contents,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            },
                        ))
                    }
                    NumericPrecision::F32 => {
                        let data_f32: Vec<f32> = host.data.iter().map(|v| *v as f32).collect();
                        let contents = cast_slice(&data_f32);
                        Arc::new(self.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-upload-buffer"),
                                contents,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            },
                        ))
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
            .map_err(|e: wgpu::BufferAsyncError| anyhow!(e))?;
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

    fn fused_elementwise(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
    ) -> Result<GpuTensorHandle> {
        if inputs.is_empty() {
            return Err(anyhow!("fused_elementwise: no inputs"));
        }
        if len > u32::MAX as usize {
            return Err(anyhow!("fused_elementwise: tensor too large"));
        }

        let entries = inputs
            .iter()
            .map(|handle| self.get_entry(handle))
            .collect::<Result<Vec<_>>>()?;

        let output_buffer = self.create_storage_buffer(len, "runmat-fusion-output");

        let mut layout_entries = Vec::with_capacity(inputs.len() + 2);
        for idx in 0..inputs.len() {
            layout_entries.push(storage_read_entry(idx as u32));
        }
        layout_entries.push(storage_read_write_entry(inputs.len() as u32));
        layout_entries.push(uniform_entry((inputs.len() + 1) as u32));

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("runmat-fusion-bind-layout"),
                    entries: &layout_entries,
                });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("runmat-fusion-pipeline-layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let layout_tag = {
            let mut tag = String::from("runmat-fusion-layout-");
            tag.push_str(&inputs.len().to_string());
            tag
        };
        let shader_hash = self.compute_pipeline_hash_bytes(shader.as_bytes(), &layout_tag, Some(WORKGROUP_SIZE));
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("runmat-fusion-shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
            });
        let pipeline = self.get_or_create_pipeline(
            shader_hash,
            &pipeline_layout,
            &module,
            "runmat-fusion-pipeline",
            Some(shader.as_bytes()),
            Some(&layout_tag),
            Some(WORKGROUP_SIZE),
        );

        let mut bind_entries = Vec::with_capacity(inputs.len() + 2);
        for (idx, entry) in entries.iter().enumerate() {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: idx as u32,
                resource: entry.buffer.as_ref().as_entire_binding(),
            });
        }
        bind_entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: output_buffer.as_ref().as_entire_binding(),
        });
        let params = FusionParams {
            len: len as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-fusion-params");
        bind_entries.push(wgpu::BindGroupEntry {
            binding: (inputs.len() + 1) as u32,
            resource: params_buffer.as_entire_binding(),
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runmat-fusion-bind-group"),
            layout: &bind_group_layout,
            entries: &bind_entries,
        });

        // Warm-up noop pass to mirror reduction path and avoid driver stalls
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-noop-elementwise"),
                });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-noop-pass-elementwise"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&*pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device.poll(wgpu::Maintain::Poll);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-fusion-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-fusion-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&*pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = dispatch_size(len as u32, WORKGROUP_SIZE);
            if workgroups > 0 {
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }
        self.submit(encoder);

        Ok(self.register_existing_buffer(output_buffer, output_shape.to_vec(), len))
    }

    fn fused_reduction(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
    ) -> Result<GpuTensorHandle> {
        if inputs.is_empty() {
            return Err(anyhow!("fused_reduction: no inputs"));
        }
        if reduce_len == 0 {
            return Err(anyhow!("fused_reduction: zero reduce_len"));
        }
        let out_elems: usize = output_shape.iter().product();
        if out_elems != num_slices.max(1) {
            return Err(anyhow!(
                "fused_reduction: output_shape {:?} inconsistent with num_slices {}",
                output_shape,
                num_slices
            ));
        }

        log::info!(
            "fused_reduction: start reduce_len={} slices={} wg={}",
            reduce_len, num_slices, workgroup_size
        );
        // Allow caller to pass 0 to request provider default WG size
        let workgroup_size = if workgroup_size == 0 {
            self.reduction_workgroup_size_default
        } else {
            workgroup_size
        };
        // Decide path: single-pass if reduce_len <= threshold else two-pass
        let two_pass = reduce_len > self.reduction_two_pass_threshold as usize;
        if !two_pass {
            log::info!("fused_reduction: path single-pass");
            // Single-pass using provided shader
            // Create shader module (label includes hash to aid driver-side caching)
            // hash computed via compute_pipeline_hash_bytes below
            let layout_tag = "runmat-reduction-bgl";
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("runmat-fused-reduction-module"),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader)),
                });
            let bind_group_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("runmat-reduction-bgl"),
                        entries: &[
                            storage_read_entry(0),
                            storage_read_write_entry(1),
                            uniform_entry(2),
                        ],
                    });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("runmat-reduction-pl"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });
            // Optional debug: skip pipeline creation/dispatch entirely
            if std::env::var("RUNMAT_DEBUG_PIPELINE_ONLY").is_ok() {
                log::info!(
                    "fused_reduction: RUNMAT_DEBUG_PIPELINE_ONLY set, skipping pipeline+dispatch (single-pass)"
                );
                let out_len = num_slices.max(1);
                let out_buffer = self.create_storage_buffer(out_len, "runmat-reduction-out");
                let handle = self.register_existing_buffer(out_buffer, output_shape.to_vec(), out_len);
                return Ok(handle);
            }

            // Cache per-shader hash (non-blocking to avoid potential mutex stalls)
            let key = self.compute_pipeline_hash_bytes(shader.as_bytes(), layout_tag, Some(workgroup_size));
            let pipeline = self.get_or_create_pipeline(
                key,
                &pipeline_layout,
                &module,
                "runmat-reduction-pipeline",
                Some(shader.as_bytes()),
                Some(layout_tag),
                Some(workgroup_size),
            );

            if std::env::var("RUNMAT_DEBUG_PIPELINE_ONLY").is_ok() {
                log::info!(
                    "fused_reduction: RUNMAT_DEBUG_PIPELINE_ONLY set, skipping dispatch (single-pass)"
                );
                let out_len = num_slices.max(1);
                let out_buffer = self.create_storage_buffer(out_len, "runmat-reduction-out");
                let handle = self.register_existing_buffer(out_buffer, output_shape.to_vec(), out_len);
                return Ok(handle);
            }

            // Optional tiny noop compute pass using the pipeline to warm up driver state
            log::info!("fused_reduction: submitting noop compute pass (single-pass)");
            {
                let mut enc = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-noop-single-pass"),
                    });
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-noop-pass-single"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&*pipeline);
                drop(pass);
                self.submit(enc);
            }
            self.device.poll(wgpu::Maintain::Poll);

            // Buffers (ensure device is polled/flushed before allocations to avoid backend stalls)
            log::info!("fused_reduction: polling device before buffer allocs (single-pass)");
            self.device.poll(wgpu::Maintain::Poll);
            let flush_enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-flush-single-pass-gap"),
                });
            // Submit empty encoder to force driver to flush any pending work
            self.submit(flush_enc);
            // Buffers
            let input_buf = self.get_entry(&inputs[0])?.buffer.clone();
            let out_len = num_slices.max(1);
            let out_buffer = self.create_storage_buffer(out_len, "runmat-reduction-out");

            #[repr(C)]
            #[derive(Clone, Copy, Pod, Zeroable)]
            struct Params {
                nrows: u32,
                ncols: u32,
                ld: u32,
                flags: u32,
            }
            let flags = if shader.contains("const OMITNAN: bool = true") {
                1u32
            } else {
                0u32
            };
            let params = Params {
                nrows: reduce_len as u32,
                ncols: num_slices as u32,
                ld: reduce_len as u32,
                flags,
            };
            let params_buffer = Arc::new(self.uniform_buffer(&params, "runmat-reduction-params"));

            log::info!("fused_reduction: creating single-pass bind group");
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-reduction-bg"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_ref().as_entire_binding(),
                    },
                ],
            });
            log::info!("fused_reduction: single-pass bind group created");

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-reduction-encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-reduction-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&*pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let groups = (num_slices as u32).max(1);
                log::info!("fused_reduction: dispatch groups=({},1,1)", groups);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            log::info!("fused_reduction: submitting (single-pass)");
            self.submit(encoder);
            log::info!("fused_reduction: single-pass submitted");

            let handle = self.register_existing_buffer(out_buffer, output_shape.to_vec(), out_len);
            return Ok(handle);
        }

        // Two-pass reduction for large reduce_len
        let scalar_ty = match self.precision {
            NumericPrecision::F64 => "f64",
            _ => "f32",
        };
        let flags = if shader.contains("const OMITNAN: bool = true") {
            1u32
        } else {
            0u32
        };
        let chunks = ((reduce_len as u32) + workgroup_size - 1) / workgroup_size;
        log::info!(
            "fused_reduction: two-pass params chunks={} partials_len={}",
            chunks,
            num_slices.max(1) * (chunks as usize)
        );
        let partials_len = num_slices.max(1) * (chunks as usize);

        // Pass 1 shader: each (slice, chunk) reduces up to workgroup_size elements
        let pass1 = format!(
            "struct Tensor {{ data: array<{st}> }};\nstruct P1Params {{ nrows:u32,ncols:u32,ld:u32,flags:u32,chunks:u32 }}\n@group(0) @binding(0) var<storage,read> input0: Tensor;\n@group(0) @binding(1) var<storage,read_write> partials: Tensor;\n@group(0) @binding(2) var<uniform> params: P1Params;\nvar<workgroup> tile: array<f32,{wg}>;\nfn isNan(x: {st}) -> bool {{ return x != x; }}\n@compute @workgroup_size({wg})\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n  let slice = wid.x; let chunk = wid.y;\n  if (slice >= params.ncols || chunk >= params.chunks) {{ return; }}\n  let start = chunk * {wg}u; let end = min(params.nrows, start + {wg}u);\n  var acc: {st} = {zero};\n  var i = start + lid.x;\n  loop {{ if (i >= end) {{ break; }} let v = input0.data[(slice * params.ld) + i]; if ((params.flags & 1u)==1u) {{ if (!isNan(v)) {{ acc = acc + v; }} }} else {{ if (isNan(v)) {{ acc = v; }} else {{ acc = acc + v; }} }} i += {wg}u; }}\n  tile[lid.x] = acc; workgroupBarrier();\n  var off: u32 = {half}u; loop {{ if (off==0u) {{ break; }} if (lid.x < off) {{ let a = tile[lid.x]; let b = tile[lid.x+off]; tile[lid.x] = a + b; }} workgroupBarrier(); off = off/2u; }}\n  if (lid.x==0u) {{ partials.data[(slice * params.chunks) + chunk] = {cast}tile[0u]; }}\n}}",
            st=scalar_ty, wg=workgroup_size, half=workgroup_size/2, zero=if scalar_ty=="f64" { "f64(0.0)" } else { "0.0" }, cast=if scalar_ty=="f64" { "f64(" } else { "" }
        );

        // Pass 2 shader: each slice reduces across chunks
        let pass2 = format!(
            "struct Tensor {{ data: array<{st}> }};\nstruct P2Params {{ ncols:u32,chunks:u32,flags:u32 }}\n@group(0) @binding(0) var<storage,read> partials: Tensor;\n@group(0) @binding(1) var<storage,read_write> output: Tensor;\n@group(0) @binding(2) var<uniform> params: P2Params;\nvar<workgroup> tile: array<f32,{wg}>;\nfn isNan(x: {st}) -> bool {{ return x != x; }}\n@compute @workgroup_size({wg})\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n  let slice = wid.x; if (slice >= params.ncols) {{ return; }}\n  var acc: {st} = {zero}; var c = lid.x;\n  loop {{ if (c >= params.chunks) {{ break; }} let v = partials.data[(slice * params.chunks) + c]; if ((params.flags & 1u)==1u) {{ if (!isNan(v)) {{ acc = acc + v; }} }} else {{ if (isNan(v)) {{ acc = v; }} else {{ acc = acc + v; }} }} c += {wg}u; }}\n  tile[lid.x] = acc; workgroupBarrier();\n  var off: u32 = {half}u; loop {{ if (off==0u) {{ break; }} if (lid.x < off) {{ let a = tile[lid.x]; let b = tile[lid.x+off]; tile[lid.x] = a + b; }} workgroupBarrier(); off = off/2u; }}\n  if (lid.x==0u) {{ output.data[slice] = {cast}tile[0u]; }}\n}}",
            st=scalar_ty, wg=workgroup_size, half=workgroup_size/2, zero=if scalar_ty=="f64" { "f64(0.0)" } else { "0.0" }, cast=if scalar_ty=="f64" { "f64(" } else { "" }
        );

        // Create pipelines
        let m1_t0 = std::time::Instant::now();
        let m1 = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("runmat-reduction-pass1"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(pass1.clone())),
            });
        log::info!(
            "fused_reduction: pass1 module created in {:.3} ms",
            m1_t0.elapsed().as_secs_f64() * 1000.0
        );
        let m2_t0 = std::time::Instant::now();
        let m2 = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("runmat-reduction-pass2"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(pass2.clone())),
            });
        log::info!(
            "fused_reduction: pass2 module created in {:.3} ms",
            m2_t0.elapsed().as_secs_f64() * 1000.0
        );
        log::info!("fused_reduction: creating BGL1");
        let bgl1 = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("runmat-reduction-p1-bgl"),
                entries: &[
                    storage_read_entry(0),
                    storage_read_write_entry(1),
                    uniform_entry(2),
                ],
            });
        log::info!("fused_reduction: creating BGL2");
        let bgl2 = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("runmat-reduction-p2-bgl"),
                entries: &[
                    storage_read_entry(0),
                    storage_read_write_entry(1),
                    uniform_entry(2),
                ],
            });
        log::info!("fused_reduction: creating pipeline layout PL1");
        let pl1 = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("runmat-reduction-p1-pl"),
                bind_group_layouts: &[&bgl1],
                push_constant_ranges: &[],
            });
        log::info!("fused_reduction: creating pipeline layout PL2");
        let pl2 = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("runmat-reduction-p2-pl"),
                bind_group_layouts: &[&bgl2],
                push_constant_ranges: &[],
            });

        // Optional debug: skip pipeline creation/dispatch entirely (two-pass)
        if std::env::var("RUNMAT_DEBUG_PIPELINE_ONLY").is_ok() {
            log::info!(
                "fused_reduction: RUNMAT_DEBUG_PIPELINE_ONLY set, skipping pipeline+dispatch (two-pass)"
            );
            let out_len = num_slices.max(1);
            let out_buffer = self.create_storage_buffer(out_len, "runmat-reduction-out");
            return Ok(self.register_existing_buffer(out_buffer, output_shape.to_vec(), out_len));
        }
        // keys computed via compute_pipeline_hash_bytes below
        // Build pass2 first
        let p2_key = self.compute_pipeline_hash_bytes(pass2.as_bytes(), "runmat-reduction-p2-bgl", Some(workgroup_size));
        let pipeline_p2 = self.get_or_create_pipeline(
            p2_key,
            &pl2,
            &m2,
            "runmat-reduction-pass2",
            Some(pass2.as_bytes()),
            Some("runmat-reduction-p2-bgl"),
            Some(workgroup_size),
        );
        // After pass2 pipeline is ready, poll and flush before pass1 to avoid stalls on some drivers
        log::info!("fused_reduction: polling device before pass1 pipeline");
        self.device.poll(wgpu::Maintain::Poll);
        let flush_enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-flush-before-pass1"),
            });
        self.submit(flush_enc);

        // Insert a tiny noop compute submit to warm up driver state after pass2 pipeline
        log::info!("fused_reduction: submitting noop compute pass after pass2 pipeline");
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-noop-after-pass2"),
                });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&*pipeline_p2);
            drop(pass);
            self.submit(enc);
        }
        self.device.poll(wgpu::Maintain::Poll);

        let p1_key = self.compute_pipeline_hash_bytes(pass1.as_bytes(), "runmat-reduction-p1-bgl", Some(workgroup_size));
        let pipeline_p1 = self.get_or_create_pipeline(
            p1_key,
            &pl1,
            &m1,
            "runmat-reduction-pass1",
            Some(pass1.as_bytes()),
            Some("runmat-reduction-p1-bgl"),
            Some(workgroup_size),
        );

        // (debug guard moved earlier)

        // Buffers (ensure device is polled before allocations to avoid backend stalls)
        log::info!("fused_reduction: polling device before buffer allocs (two-pass)");
        self.device.poll(wgpu::Maintain::Poll);
        // Buffers
        log::info!(
            "fused_reduction: creating two-pass buffers partials_len={} out_len={}",
            partials_len,
            num_slices.max(1)
        );
        log::info!("fused_reduction: retrieving input buffer 0");
        let input_buf = self.get_entry(&inputs[0])?.buffer.clone();
        log::info!("fused_reduction: input buffer retrieved");
        let partials_buffer = self.create_storage_buffer(partials_len, "runmat-reduction-partials");
        log::info!("fused_reduction: partials buffer created (len={})", partials_len);
        let out_buffer = self.create_storage_buffer(num_slices.max(1), "runmat-reduction-out");
        log::info!(
            "fused_reduction: out buffer created (len={})",
            num_slices.max(1)
        );

        // Uniforms
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            nrows: u32,
            ncols: u32,
            ld: u32,
            flags: u32,
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct P1 {
            nrows: u32,
            ncols: u32,
            ld: u32,
            flags: u32,
            chunks: u32,
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct P2 {
            ncols: u32,
            chunks: u32,
            flags: u32,
        }
        let p1 = P1 {
            nrows: reduce_len as u32,
            ncols: num_slices as u32,
            ld: reduce_len as u32,
            flags,
            chunks,
        };
        let p2u = P2 {
            ncols: num_slices as u32,
            chunks,
            flags,
        };
        log::info!("fused_reduction: creating pass1 uniform buffer");
        let p1_buf = Arc::new(self.uniform_buffer(&p1, "runmat-reduction-p1-params"));
        log::info!("fused_reduction: pass1 uniform buffer created");
        log::info!("fused_reduction: creating pass2 uniform buffer");
        let p2_buf = Arc::new(self.uniform_buffer(&p2u, "runmat-reduction-p2-params"));
        log::info!("fused_reduction: pass2 uniform buffer created");

        // Bind groups
        log::info!("fused_reduction: creating pass1 bind group");
        let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runmat-reduction-p1-bg"),
            layout: &bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: partials_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p1_buf.as_ref().as_entire_binding(),
                },
            ],
        });
        log::info!("fused_reduction: pass1 bind group created");
        log::info!("fused_reduction: creating pass2 bind group");
        let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runmat-reduction-p2-bg"),
            layout: &bgl2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: partials_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p2_buf.as_ref().as_entire_binding(),
                },
            ],
        });
        log::info!("fused_reduction: pass2 bind group created");

        // Dispatch two passes
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-reduction-2pass-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-reduction-pass1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&*pipeline_p1);
            pass.set_bind_group(0, &bg1, &[]);
            let g0 = (num_slices as u32).max(1);
            let g1 = chunks.max(1);
            log::info!("fused_reduction: pass1 dispatch groups=({}, {}, 1)", g0, g1);
            pass.dispatch_workgroups(g0, g1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-reduction-pass2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&*pipeline_p2);
            pass.set_bind_group(0, &bg2, &[]);
            let g0 = (num_slices as u32).max(1);
            log::info!("fused_reduction: pass2 dispatch groups=({}, 1, 1)", g0);
            pass.dispatch_workgroups(g0, 1, 1);
        }
        log::info!("fused_reduction: submitting (two-pass)");
        self.submit(encoder);
        log::info!("fused_reduction: two-pass submitted");

        Ok(self.register_existing_buffer(out_buffer, output_shape.to_vec(), num_slices.max(1)))
    }

    fn warmup(&self) {
        self.warmup_internal();
    }

    fn fused_cache_counters(&self) -> (u64, u64) {
        (
            self.fused_cache_hits.load(Ordering::Relaxed),
            self.fused_cache_misses.load(Ordering::Relaxed),
        )
    }

    fn default_reduction_workgroup_size(&self) -> u32 {
        self.reduction_workgroup_size_default
    }

    fn two_pass_threshold(&self) -> usize {
        self.reduction_two_pass_threshold
    }

    fn scatter_column(
        &self,
        matrix: &GpuTensorHandle,
        col_index: usize,
        values: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let m_entry = self.get_entry(matrix)?;
        if m_entry.shape.len() != 2 {
            return Err(anyhow!("scatter_column: only 2D tensors supported"));
        }
        let rows = m_entry.shape[0];
        let cols = m_entry.shape[1];
        if col_index >= cols {
            return Err(anyhow!("scatter_column: column index out of bounds"));
        }
        let v_entry = self.get_entry(values)?;
        let v_rows = if v_entry.shape.len() == 1 {
            v_entry.shape[0]
        } else if v_entry.shape.len() == 2 {
            v_entry.shape[0]
        } else {
            return Err(anyhow!("scatter_column: values must be vector or [rows,1]"));
        };
        if v_rows != rows {
            return Err(anyhow!("scatter_column: length mismatch"));
        }

        // Simple kernel: copy values into matrix column j
        let shader = r#"struct T { data: array<f32> };
@group(0) @binding(0) var<storage, read> V: T;
@group(0) @binding(1) var<storage, read> M: T;
@group(0) @binding(2) var<storage, read_write> Out: T;
struct P { rows:u32, cols:u32, j:u32 }
@group(0) @binding(3) var<uniform> Pm: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x; if (r >= Pm.rows) { return; }
  let dst = r + Pm.j * Pm.rows;
  Out.data[dst] = V.data[r];
}
"#;
        // Reuse fused_elementwise-like dispatch: build BGL with 3 storage + 1 uniform
        let out_buffer = self.create_storage_buffer(rows * cols, "runmat-scatter-col-out");
        // Copy input matrix into out first (blit kernel or copy pass)
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("runmat-scatter-col-copy") });
            enc.copy_buffer_to_buffer(
                m_entry.buffer.as_ref(), 0,
                out_buffer.as_ref(), 0,
                (rows * cols * self.element_size) as u64,
            );
            self.submit(enc);
        }
        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("runmat-scatter-col-bgl"),
            entries: &[
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
        });
        let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("runmat-scatter-col-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("runmat-scatter-col-module"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader)),
        });
        let key = self.compute_pipeline_hash_bytes(shader.as_bytes(), "runmat-scatter-col-bgl", Some(256));
        let pipeline = self.get_or_create_pipeline(key, &pl, &module, "runmat-scatter-col", Some(shader.as_bytes()), Some("runmat-scatter-col-bgl"), Some(256));
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Pm { rows: u32, cols: u32, j: u32 }
        let params = Pm { rows: rows as u32, cols: cols as u32, j: col_index as u32 };
        let pbuf = self.uniform_buffer(&params, "runmat-scatter-col-params");
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runmat-scatter-col-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: v_entry.buffer.as_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: m_entry.buffer.as_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buffer.as_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pbuf.as_entire_binding() },
            ],
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("runmat-scatter-col-enc") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("runmat-scatter-col-pass"), timestamp_writes: None });
            pass.set_pipeline(&*pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let groups = dispatch_size(rows as u32, 256);
            if groups > 0 { pass.dispatch_workgroups(groups, 1, 1); }
        }
        self.submit(enc);
        Ok(self.register_existing_buffer(out_buffer, vec![rows, cols], rows * cols))
    }

    fn scatter_row(
        &self,
        matrix: &GpuTensorHandle,
        row_index: usize,
        values: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let m_entry = self.get_entry(matrix)?;
        if m_entry.shape.len() != 2 { return Err(anyhow!("scatter_row: only 2D tensors supported")); }
        let rows = m_entry.shape[0];
        let cols = m_entry.shape[1];
        if row_index >= rows { return Err(anyhow!("scatter_row: row index out of bounds")); }
        let v_entry = self.get_entry(values)?;
        let v_cols = if v_entry.shape.len() == 1 { v_entry.shape[0] } else if v_entry.shape.len() == 2 { v_entry.shape[1] } else { return Err(anyhow!("scatter_row: values must be vector or [1,cols]")); };
        if v_cols != cols { return Err(anyhow!("scatter_row: length mismatch")); }

        let shader = r#"struct T { data: array<f32> };
@group(0) @binding(0) var<storage, read> V: T;
@group(0) @binding(1) var<storage, read> M: T;
@group(0) @binding(2) var<storage, read_write> Out: T;
struct P { rows:u32, cols:u32, i:u32 }
@group(0) @binding(3) var<uniform> Pm: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; if (c >= Pm.cols) { return; }
  let dst = Pm.i + c * Pm.rows;
  Out.data[dst] = V.data[c];
}
"#;
        let out_buffer = self.create_storage_buffer(rows * cols, "runmat-scatter-row-out");
        {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("runmat-scatter-row-copy") });
            enc.copy_buffer_to_buffer(m_entry.buffer.as_ref(), 0, out_buffer.as_ref(), 0, (rows * cols * self.element_size) as u64);
            self.submit(enc);
        }
        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("runmat-scatter-row-bgl"), entries: &[
            storage_read_entry(0), storage_read_entry(1), storage_read_write_entry(2), uniform_entry(3)
        ]});
        let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("runmat-scatter-row-pl"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("runmat-scatter-row-module"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader)) });
        let key = self.compute_pipeline_hash_bytes(shader.as_bytes(), "runmat-scatter-row-bgl", Some(256));
        let pipeline = self.get_or_create_pipeline(key, &pl, &module, "runmat-scatter-row", Some(shader.as_bytes()), Some("runmat-scatter-row-bgl"), Some(256));
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Pm { rows: u32, cols: u32, i: u32 }
        let params = Pm { rows: rows as u32, cols: cols as u32, i: row_index as u32 };
        let pbuf = self.uniform_buffer(&params, "runmat-scatter-row-params");
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("runmat-scatter-row-bg"), layout: &bgl, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: v_entry.buffer.as_ref().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: m_entry.buffer.as_ref().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buffer.as_ref().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pbuf.as_entire_binding() },
        ] });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("runmat-scatter-row-enc") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("runmat-scatter-row-pass"), timestamp_writes: None });
            pass.set_pipeline(&*pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let groups = dispatch_size(cols as u32, 256);
            if groups > 0 { pass.dispatch_workgroups(groups, 1, 1); }
        }
        self.submit(enc);
        Ok(self.register_existing_buffer(out_buffer, vec![rows, cols], rows * cols))
    }

    fn last_warmup_millis(&self) -> Option<u64> {
        Some(self.last_warmup_millis.load(Ordering::Relaxed))
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
    offset: u32,
    total: u32,
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
    let local = gid.x;
    if local >= params.len {
        return;
    }
    let idx = params.offset + local;
    if idx >= params.total {
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
    offset: u32,
    total: u32,
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
    let local = gid.x;
    if local >= params.len {
        return;
    }
    let idx = params.offset + local;
    if idx >= params.total {
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
    offset: u32,
    total: u32,
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
    let local = gid.x;
    if local >= params.len {
        return;
    }
    let idx = params.offset + local;
    if idx >= params.total {
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
    offset: u32,
    total: u32,
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
    let local = gid.x;
    if local >= params.len {
        return;
    }
    let idx = params.offset + local;
    if idx >= params.total {
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

fn isNan(x: f64) -> bool { return x != x; }

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
    offset: u32,
    total: u32,
    scalar: f32,
    scalar_pad: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isNan(x: f32) -> bool { return x != x; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len {
        return;
    }
    let idx = params.offset + local;
    if idx >= params.total {
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
    offset: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isNan(x: f64) -> bool { return x != x; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len {
        return;
    }
    let rows = params.rows;
    let cols = params.cols;
    let idx = params.offset + local;
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
    offset: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isNan(x: f32) -> bool { return x != x; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len {
        return;
    }
    let rows = params.rows;
    let cols = params.cols;
    let idx = params.offset + local;
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
    offset: u32,
    total: u32,
};

@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutBuf: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> tile: array<f64, 256>;

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
        case 1u: { return f64(1.0) / f64(0.0); }
        case 2u: { return -f64(1.0) / f64(0.0); }
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
    tile[lid.x] = acc;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if stride == 0u {
            break;
        }
        if lid.x < stride {
            tile[lid.x] = combine(tile[lid.x], tile[lid.x + stride], params.op);
        }
        stride = stride / 2u;
        workgroupBarrier();
    }
    if lid.x == 0u {
        OutBuf.data[wid.x] = tile[0u];
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
    offset: u32,
    total: u32,
};

@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutBuf: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> tile: array<f32, 256>;

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
        case 0u: { return 0.0f; }
        case 1u: { return 1.0f / 0.0f; }
        case 2u: { return -1.0f / 0.0f; }
        default: { return 0.0f; }
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
    tile[lid.x] = acc;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if stride == 0u {
            break;
        }
        if lid.x < stride {
            tile[lid.x] = combine(tile[lid.x], tile[lid.x + stride], params.op);
        }
        stride = stride / 2u;
        workgroupBarrier();
    }
    if lid.x == 0u {
        OutBuf.data[wid.x] = tile[0u];
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

fn isNan(x: f64) -> bool { return x != x; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if params.dim == 1u {
        if idx >= params.cols {
            return;
        }
        var acc: f64 = 0.0;
        var saw_nan: bool = false;
        for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
            let linear = r + idx * params.rows;
            let v = InBuf.data[linear];
            if isNan(v) {
                saw_nan = true;
            } else {
                acc = acc + v;
            }
        }
        if saw_nan {
            OutBuf.data[idx] = f64(0.0) / f64(0.0);
        } else {
            if params.op == 1u { acc = acc / f64(params.rows); }
            OutBuf.data[idx] = acc;
        }
    } else {
        if idx >= params.rows {
            return;
        }
        var acc: f64 = 0.0;
        var saw_nan: bool = false;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let linear = idx + c * params.rows;
            let v = InBuf.data[linear];
            if isNan(v) {
                saw_nan = true;
            } else {
                acc = acc + v;
            }
        }
        if saw_nan {
            OutBuf.data[idx] = f64(0.0) / f64(0.0);
        } else {
            if params.op == 1u { acc = acc / f64(params.cols); }
            OutBuf.data[idx] = acc;
        }
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

fn isNan(x: f32) -> bool { return x != x; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if params.dim == 1u {
        if idx >= params.cols {
            return;
        }
        var acc: f32 = 0.0;
        var saw_nan: bool = false;
        for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
            let linear = r + idx * params.rows;
            let v = InBuf.data[linear];
            if isNan(v) {
                saw_nan = true;
            } else {
                acc = acc + v;
            }
        }
        if saw_nan {
            OutBuf.data[idx] = f32(0.0) / f32(0.0);
        } else {
            if params.op == 1u { acc = acc / f32(params.rows); }
            OutBuf.data[idx] = acc;
        }
    } else {
        if idx >= params.rows {
            return;
        }
        var acc: f32 = 0.0;
        var saw_nan: bool = false;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let linear = idx + c * params.rows;
            let v = InBuf.data[linear];
            if isNan(v) {
                saw_nan = true;
            } else {
                acc = acc + v;
            }
        }
        if saw_nan {
            OutBuf.data[idx] = f32(0.0) / f32(0.0);
        } else {
            if params.op == 1u { acc = acc / f32(params.cols); }
            OutBuf.data[idx] = acc;
        }
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
            best = f64(1.0) / f64(0.0);
        } else {
            best = -f64(1.0) / f64(0.0);
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
            best = f64(1.0) / f64(0.0);
        } else {
            best = -f64(1.0) / f64(0.0);
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
        var best: f32;
        if params.op == 0u {
            best = 1.0f / 0.0f;
        } else {
            best = -1.0f / 0.0f;
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
        OutIdx.data[idx] = f32(best_idx);
    } else {
        if idx >= params.rows {
            return;
        }
        var best: f32;
        if params.op == 0u {
            best = 1.0f / 0.0f;
        } else {
            best = -1.0f / 0.0f;
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
        OutIdx.data[idx] = f32(best_idx);
    }
}
"#;
