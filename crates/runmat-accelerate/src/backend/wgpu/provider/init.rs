use anyhow::{anyhow, Result};
use log::{info, warn};
#[cfg(not(target_arch = "wasm32"))]
use pollster::block_on;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use super::{
    canonical_vendor_name, install_device_error_handlers, parse_two_pass_mode,
    ImageNormalizeTuning, NumericPrecision, ReductionTwoPassMode, WgpuProvider,
    WgpuProviderOptions, WorkgroupConfig,
};
use crate::backend::wgpu::autotune::AutotuneController;
use crate::backend::wgpu::cache::bind_group::BindGroupCache;
use crate::backend::wgpu::config::{
    self, DEFAULT_REDUCTION_WG, DEFAULT_TWO_PASS_THRESHOLD, MATMUL_TILE, WORKGROUP_SIZE,
};
use crate::backend::wgpu::pipelines::{ImageNormalizeBootstrap, WgpuPipelines};
use crate::backend::wgpu::residency::BufferResidency;
use crate::backend::wgpu::resources::KernelResourceRegistry;
use crate::telemetry::AccelTelemetry;

impl WgpuProvider {
    pub(super) fn buffer_residency_pool_limit() -> usize {
        const VAR: &str = "RUNMAT_WGPU_POOL_MAX_PER_KEY";
        match std::env::var(VAR) {
            Ok(raw) => match raw.parse::<usize>() {
                Ok(value) => {
                    log::info!(
                        "RunMat Accelerate: buffer residency pool capacity set to {} via {}",
                        value,
                        VAR
                    );
                    value
                }
                Err(err) => {
                    log::warn!(
                        "RunMat Accelerate: failed to parse {}='{}' ({}); using default {}",
                        VAR,
                        raw,
                        err,
                        Self::BUFFER_RESIDENCY_MAX_PER_KEY
                    );
                    Self::BUFFER_RESIDENCY_MAX_PER_KEY
                }
            },
            Err(_) => Self::BUFFER_RESIDENCY_MAX_PER_KEY,
        }
    }

    pub(super) fn parse_buffer_residency_max_poolable_bytes(
        raw_override: Option<&str>,
        adapter_max_buffer_size: u64,
    ) -> u64 {
        let default_limit = if adapter_max_buffer_size == 0 {
            256u64 << 20
        } else {
            (256u64 << 20).min(adapter_max_buffer_size)
        };
        match raw_override {
            Some(raw) => match raw.parse::<u64>() {
                Ok(value) => {
                    if adapter_max_buffer_size == 0 {
                        value
                    } else {
                        value.min(adapter_max_buffer_size)
                    }
                }
                Err(_) => default_limit,
            },
            None => default_limit,
        }
    }

    pub(super) fn buffer_residency_max_poolable_bytes(adapter_max_buffer_size: u64) -> u64 {
        const VAR: &str = "RUNMAT_WGPU_POOL_MAX_BUFFER_BYTES";
        match std::env::var(VAR) {
            Ok(raw) => {
                let parsed = Self::parse_buffer_residency_max_poolable_bytes(
                    Some(raw.as_str()),
                    adapter_max_buffer_size,
                );
                if raw.parse::<u64>().is_ok() {
                    log::info!(
                        "RunMat Accelerate: max pooled buffer size set to {} bytes via {}",
                        parsed,
                        VAR
                    );
                } else {
                    let default_limit = Self::parse_buffer_residency_max_poolable_bytes(
                        None,
                        adapter_max_buffer_size,
                    );
                    log::warn!(
                        "RunMat Accelerate: failed to parse {}='{}'; using default {} bytes",
                        VAR,
                        raw,
                        default_limit
                    );
                }
                parsed
            }
            Err(_) => {
                Self::parse_buffer_residency_max_poolable_bytes(None, adapter_max_buffer_size)
            }
        }
    }

    pub async fn new_async(opts: WgpuProviderOptions) -> Result<Self> {
        let mut instance_desc = wgpu::InstanceDescriptor::default();
        #[cfg(all(not(target_arch = "wasm32"), target_os = "windows"))]
        {
            instance_desc.dx12_shader_compiler = wgpu::util::dx12_shader_compiler_from_env()
                .unwrap_or(wgpu::Dx12Compiler::Dxc {
                    dxil_path: None,
                    dxc_path: None,
                });
        }
        #[cfg(all(not(target_arch = "wasm32"), not(target_os = "windows")))]
        {
            if let Some(compiler) = wgpu::util::dx12_shader_compiler_from_env() {
                instance_desc.dx12_shader_compiler = compiler;
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            instance_desc.backends = wgpu::Backends::BROWSER_WEBGPU;
        }

        let instance = Arc::new(wgpu::Instance::new(instance_desc));
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: opts.power_preference,
                force_fallback_adapter: opts.force_fallback_adapter,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| anyhow!("wgpu: no compatible adapter found"))?;

        let adapter_info = adapter.get_info();
        #[cfg(not(target_arch = "wasm32"))]
        let adapter_features = adapter.features();
        let forced_precision = std::env::var("RUNMAT_WGPU_FORCE_PRECISION")
            .ok()
            .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
                "f32" | "float32" | "32" => Some(NumericPrecision::F32),
                "f64" | "float64" | "64" => Some(NumericPrecision::F64),
                _ => None,
            });

        #[cfg(target_arch = "wasm32")]
        let precision = {
            if forced_precision == Some(NumericPrecision::F64) {
                warn!("RunMat Accelerate: f64 precision is unavailable on WebGPU/wasm builds; using f32");
            }
            NumericPrecision::F32
        };

        #[cfg(not(target_arch = "wasm32"))]
        let precision = {
            let mut p = forced_precision.unwrap_or(NumericPrecision::F32);
            if p == NumericPrecision::F64 && !adapter_features.contains(wgpu::Features::SHADER_F64)
            {
                warn!(
                    "RunMat Accelerate: requested f64 precision but adapter lacks SHADER_F64; falling back to f32"
                );
                p = NumericPrecision::F32;
            }
            p
        };

        if forced_precision.is_none() {
            info!(
                "RunMat Accelerate: defaulting to {} kernels for adapter '{}'",
                match precision {
                    NumericPrecision::F64 => "f64",
                    NumericPrecision::F32 => "f32",
                },
                adapter_info.name
            );
        }

        let two_pass_threshold = std::env::var("RUNMAT_TWO_PASS_THRESHOLD")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_TWO_PASS_THRESHOLD);
        let requested_scalar_wg = config::env_requested_workgroup_size().unwrap_or(WORKGROUP_SIZE);
        let requested_matmul_tile = config::env_requested_matmul_tile().unwrap_or(MATMUL_TILE);
        let requested_reduction_wg =
            config::env_requested_reduction_workgroup_size().unwrap_or(DEFAULT_REDUCTION_WG);
        let reduction_two_pass_mode = match std::env::var("RUNMAT_REDUCTION_TWO_PASS") {
            Ok(raw) if !raw.trim().is_empty() => match parse_two_pass_mode(&raw) {
                Some(mode) => mode,
                None => {
                    warn!(
                        "RUNMAT_REDUCTION_TWO_PASS='{}' not recognized (expected auto|force_on|force_off); defaulting to auto",
                        raw
                    );
                    ReductionTwoPassMode::Auto
                }
            },
            _ => ReductionTwoPassMode::Auto,
        };

        let required_features = match precision {
            NumericPrecision::F64 => wgpu::Features::SHADER_F64,
            NumericPrecision::F32 => wgpu::Features::empty(),
        };
        let limits = adapter.limits();

        #[cfg(not(target_arch = "wasm32"))]
        let (device_raw, queue_raw) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("RunMat WGPU Device"),
                    required_features,
                    required_limits: limits.clone(),
                },
                None,
            )
            .await?;
        #[cfg(target_arch = "wasm32")]
        let (device_raw, queue_raw) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("RunMat WGPU Device"),
                    required_features,
                    required_limits: limits.clone(),
                },
                None,
            )
            .await
            .map_err(|err| anyhow!(err.to_string()))?;
        let device = Arc::new(device_raw);
        install_device_error_handlers(&device);
        let queue = Arc::new(queue_raw);
        let adapter = Arc::new(adapter);
        let satisfied_limits = device.limits();

        let workgroup_config = WorkgroupConfig::new(
            &satisfied_limits,
            requested_scalar_wg,
            requested_reduction_wg,
            requested_matmul_tile,
        );
        crate::backend::wgpu::config::set_effective_workgroup_size(workgroup_config.scalar);
        crate::backend::wgpu::config::set_effective_matmul_tile(workgroup_config.matmul_tile);
        info!(
            "WGPU adapter '{}' ready: scalar_wg={} reduction_wg={} matmul_tile={} precision={} wg_limits=({}, {}, {}) max_invocations={}",
            adapter_info.name,
            workgroup_config.scalar,
            workgroup_config.reduction_default,
            workgroup_config.matmul_tile,
            match precision {
                NumericPrecision::F64 => "f64",
                NumericPrecision::F32 => "f32",
            },
            workgroup_config.max_x,
            workgroup_config.max_y,
            workgroup_config.max_z,
            workgroup_config.adapter_max_invocations
        );

        let reduction_wg_default = workgroup_config.reduction_default;
        let cache_device_id = adapter_info.device;
        let runtime_device_id = runmat_accelerate_api::next_device_id();
        let element_size = match precision {
            NumericPrecision::F64 => std::mem::size_of::<f64>(),
            NumericPrecision::F32 => std::mem::size_of::<f32>(),
        };

        match precision {
            NumericPrecision::F64 => info!(
                "WGPU adapter '{}' supports shader-f64; using f64 kernels",
                adapter_info.name
            ),
            NumericPrecision::F32 => {
                info!("WGPU adapter '{}' using f32 kernels", adapter_info.name)
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        let pipeline_cache_dir = {
            let dir = if let Ok(custom) = std::env::var("RUNMAT_PIPELINE_CACHE_DIR") {
                PathBuf::from(custom)
            } else if let Some(base) = dirs::cache_dir() {
                base.join("runmat")
                    .join("pipelines")
                    .join(format!("device-{}", cache_device_id))
            } else {
                PathBuf::from("target")
                    .join("tmp")
                    .join(format!("wgpu-pipeline-cache-{}", cache_device_id))
            };
            Some(dir)
        };
        #[cfg(target_arch = "wasm32")]
        let pipeline_cache_dir: Option<PathBuf> = None;

        #[cfg(not(target_arch = "wasm32"))]
        let autotune_base_dir = std::env::var("RUNMAT_AUTOTUNE_DIR")
            .ok()
            .map(PathBuf::from)
            .or_else(|| {
                dirs::data_local_dir().map(|mut dir| {
                    dir.push("runmat");
                    dir
                })
            })
            .or_else(|| pipeline_cache_dir.clone());
        #[cfg(target_arch = "wasm32")]
        let autotune_base_dir: Option<PathBuf> = None;

        let autotune_device_tag = format!(
            "{}-{:08x}",
            canonical_vendor_name(&adapter_info),
            cache_device_id
        );
        if let Some(dir) = &autotune_base_dir {
            let reduction_path = dir.join("autotune").join("fused_reduction");
            info!(
                "Reduction autotune cache dir {:?} (tag {})",
                reduction_path, autotune_device_tag
            );
        }
        let reduction_autotune = AutotuneController::new_from_env(
            "RUNMAT_REDUCTION_AUTOTUNE",
            "fused_reduction",
            autotune_base_dir.clone(),
            &autotune_device_tag,
        );
        if let Some(dir) = &autotune_base_dir {
            let image_path = dir.join("autotune").join("image_normalize");
            info!(
                "ImageNormalize autotune cache dir {:?} (tag {})",
                image_path, autotune_device_tag
            );
        }
        let image_norm_autotune = AutotuneController::new_from_env(
            "RUNMAT_IMAGE_NORMALIZE_AUTOTUNE",
            "image_normalize",
            autotune_base_dir.clone(),
            &autotune_device_tag,
        );

        info!(
            "Reduction two-pass mode={} threshold={} workgroup_size={}",
            reduction_two_pass_mode.as_str(),
            two_pass_threshold,
            reduction_wg_default
        );

        let bootstrap_tuning = ImageNormalizeTuning {
            batch_tile: 1,
            values_per_thread: 1,
            lane_count: 32,
            spatial_tile: 1,
        };
        let sanitized_bootstrap =
            workgroup_config.sanitize_image_normalize_tuning(bootstrap_tuning, 1);
        let image_norm_bootstrap = ImageNormalizeBootstrap {
            batch_tile: sanitized_bootstrap.batch_tile,
            values_per_thread: sanitized_bootstrap.values_per_thread,
            lane_count: sanitized_bootstrap.lane_count,
            spatial_tile: sanitized_bootstrap.spatial_tile,
        };
        let pipelines = WgpuPipelines::new(&device, precision, image_norm_bootstrap);

        let buffer_pool_limit = Self::buffer_residency_pool_limit();
        let max_poolable_bytes =
            Self::buffer_residency_max_poolable_bytes(satisfied_limits.max_buffer_size);

        Ok(Self {
            instance,
            device,
            queue,
            adapter,
            adapter_info,
            adapter_limits: satisfied_limits,
            workgroup_config,
            buffers: Mutex::new(HashMap::new()),
            buffer_residency: BufferResidency::new(buffer_pool_limit),
            buffer_residency_max_poolable_bytes: max_poolable_bytes,
            next_id: AtomicU64::new(1),
            pipelines,
            runtime_device_id,
            cache_device_id,
            precision,
            element_size,
            fused_pipeline_cache: Mutex::new(HashMap::new()),
            bind_group_layout_cache: Mutex::new(HashMap::new()),
            bind_group_layout_tags: Mutex::new(HashMap::new()),
            bind_group_cache: BindGroupCache::default(),
            kernel_resources: KernelResourceRegistry::default(),
            metrics: crate::backend::wgpu::metrics::WgpuMetrics::default(),
            telemetry: AccelTelemetry::default(),
            reduction_two_pass_mode,
            reduction_two_pass_threshold: two_pass_threshold,
            reduction_workgroup_size_default: reduction_wg_default,
            pipeline_cache_dir,
            reduction_autotune,
            image_norm_autotune,
            image_norm_pipeline_cache: Mutex::new(HashMap::new()),
            autotune_base_dir,
            autotune_device_tag,
            pow2_of: Mutex::new(HashMap::new()),
            moments_cache: Mutex::new(HashMap::new()),
            fft_twiddle_cache: Mutex::new(HashMap::new()),
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(opts: WgpuProviderOptions) -> Result<Self> {
        block_on(Self::new_async(opts))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new(opts: WgpuProviderOptions) -> Result<Self> {
        Err(anyhow!(
            "RunMat Accelerate: synchronous WGPU initialization is unavailable on wasm targets. Use new_async instead (opts: {:?}).",
            opts
        ))
    }
}
