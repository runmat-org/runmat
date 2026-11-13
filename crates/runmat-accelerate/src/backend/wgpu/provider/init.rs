use anyhow::{anyhow, Result};
use log::{info, warn};
#[cfg(not(target_arch = "wasm32"))]
use pollster::block_on;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::moments_cache::{CacheBudget, ReductionReuseCache};
use super::owner_shared::OwnerShared;
use super::{
    canonical_vendor_name, install_device_error_handlers, parse_two_pass_mode,
    ImageNormalizeTuning, NumericPrecision, ReductionTwoPassMode, WgpuDeviceHealth,
    WgpuPrecisionPolicy, WgpuProvider, WgpuProviderOptions, WorkgroupConfig,
};
use crate::backend::wgpu::autotune::AutotuneController;
use crate::backend::wgpu::cache::bind_group::BindGroupCache;
use crate::backend::wgpu::config::{
    self, DEFAULT_REDUCTION_WG, DEFAULT_TWO_PASS_THRESHOLD, MATMUL_TILE, WORKGROUP_SIZE,
};
use crate::backend::wgpu::pipelines::{FrozenLinearLaunch, ImageNormalizeBootstrap, WgpuPipelines};
use crate::backend::wgpu::resources::KernelResourceRegistry;
use crate::telemetry::AccelTelemetry;

#[path = "init/timestamp_features.rs"]
mod timestamp_features;
#[path = "init/adapter.rs"]
mod adapter;
#[cfg(all(not(target_arch = "wasm32"), target_os = "windows"))]
#[path = "init/windows_shader_compiler.rs"]
mod windows_shader_compiler;

#[cfg(not(target_arch = "wasm32"))]
fn native_required_features(precision: NumericPrecision) -> wgpu::Features {
    match precision {
        NumericPrecision::F32 => wgpu::Features::empty(),
        NumericPrecision::F64 => {
            crate::backend::wgpu::shader_compilation::capabilities::REQUIRED_FEATURES
        }
    }
}

impl WgpuProvider {
    /// Provider-local reuse ceiling, independent of the adapter residency
    /// pool and aggregate placement hardware capacity. Zero disables reuse.
    pub(super) fn reduction_reuse_budget() -> CacheBudget {
        fn parse(var: &str, default: u64) -> u64 {
            match std::env::var(var) {
                Ok(raw) => match raw.parse::<u64>() {
                    Ok(value) => value,
                    Err(_) => {
                        log::warn!(
                            "invalid {var}='{raw}'; using reduction-cache default {default}"
                        );
                        default
                    }
                },
                Err(_) => default,
            }
        }
        let entries = parse("RUNMAT_WGPU_MOMENTS_CACHE_MAX_ENTRIES", 128);
        let max_entries = usize::try_from(entries).unwrap_or_else(|_| {
            warn!("WGPU reduction-cache entry limit {entries} exceeds usize; using default 128");
            128
        });
        CacheBudget {
            max_entries,
            max_bytes: parse("RUNMAT_WGPU_MOMENTS_CACHE_MAX_BYTES", 64 << 20),
        }
    }

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
            // D3D12 is the native presentation backend used by RunMat Desktop.
            // Keeping compute and presentation on this one backend is required
            // for provider-owned buffers to remain WGPU-resident through draw.
            // Advanced users can still opt into another backend explicitly.
            instance_desc.backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::DX12);
            instance_desc.backend_options.dx12.shader_compiler =
                windows_shader_compiler::resolve()?;
        }
        #[cfg(all(not(target_arch = "wasm32"), not(target_os = "windows")))]
        {
            if let Some(compiler) = wgpu::Dx12Compiler::from_env() {
                instance_desc.backend_options.dx12.shader_compiler = compiler;
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            instance_desc.backends = wgpu::Backends::BROWSER_WEBGPU;
        }

        let (instance, adapter) = adapter::request(instance_desc, &opts).await?;

        let adapter_info = adapter.get_info();
        let adapter_features = adapter.features();
        let requested_precision = match opts.precision {
            WgpuPrecisionPolicy::Automatic => environment_precision()?,
            WgpuPrecisionPolicy::RequireF32 => Some(NumericPrecision::F32),
            WgpuPrecisionPolicy::RequireF64 => Some(NumericPrecision::F64),
        };

        #[cfg(target_arch = "wasm32")]
        let precision = select_precision(requested_precision, false)?;

        #[cfg(not(target_arch = "wasm32"))]
        let precision = {
            if requested_precision == Some(NumericPrecision::F64) {
                crate::backend::wgpu::shader_compilation::capabilities::validate_adapter(&adapter)?;
            }
            // Preserve the established default precision policy. Supporting
            // complete binary64 math does not select a different default or
            // authorize narrowing an explicitly requested binary64 route.
            select_precision(
                requested_precision,
                adapter_features.contains(wgpu::Features::SHADER_F64),
            )?
        };

        if requested_precision.is_none() {
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

        let optional_timestamp_features =
            timestamp_features::requested(adapter_info.backend, adapter_features);
        #[cfg(not(target_arch = "wasm32"))]
        let required_features = native_required_features(precision) | optional_timestamp_features;
        #[cfg(target_arch = "wasm32")]
        let required_features = optional_timestamp_features;
        let limits = adapter.limits();

        #[cfg(not(target_arch = "wasm32"))]
        let (device_raw, queue_raw) = adapter
            .request_device(
                &crate::backend::wgpu::compat::device_descriptor(
                    Some("RunMat WGPU Device"),
                    required_features,
                    limits.clone(),
                ),
                None,
            )
            .await?;
        #[cfg(target_arch = "wasm32")]
        let (device_raw, queue_raw) = adapter
            .request_device(
                &crate::backend::wgpu::compat::device_descriptor(
                    Some("RunMat WGPU Device"),
                    required_features,
                    limits.clone(),
                ),
                None,
            )
            .await
            .map_err(|err| anyhow!(err.to_string()))?;
        let device_health = Arc::new(WgpuDeviceHealth::default());
        let timestamp_period_ns = queue_raw.get_timestamp_period();
        let (access, emission_authority) =
            runmat_accelerate_api::WgpuDeviceAccess::new_profiled(device_raw, queue_raw);
        #[cfg(target_arch = "wasm32")]
        access.with_profiled_shared(&emission_authority, |emission| {
            install_device_error_handlers(emission.device(), device_health.clone())
        })?;
        let adapter = Arc::new(adapter);
        let satisfied_limits = access
            .with_profiled_shared(&emission_authority, |emission| emission.device().limits())?;
        let enabled_features = access
            .with_profiled_shared(&emission_authority, |emission| emission.device().features())?;
        let timestamp_capability = enabled_features
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| {
                crate::backend::wgpu::dispatch::observation::TimestampCapability::new(
                    timestamp_period_ns,
                    enabled_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
                )
            })
            .flatten();
        let observation_limit =
            std::num::NonZeroU32::new(64).expect("nonzero provider observation safety limit");
        // The same optional byte admission also covers browser direct-work
        // completion fences when timestamp queries are unavailable.
        let query_requested_byte_limit = u64::from(observation_limit.get())
            .checked_mul(crate::backend::wgpu::dispatch::observation::MAX_QUERY_REQUESTED_BYTES)
            .ok_or_else(|| anyhow!("WGPU observation byte bound overflow"))?;

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
        let linear_launch = FrozenLinearLaunch::new(
            workgroup_config.scalar,
            satisfied_limits.max_compute_workgroups_per_dimension,
        )
        .ok_or_else(|| anyhow!("enabled WGPU linear launch limits are invalid"))?;
        let global_reduce_launch =
            crate::backend::wgpu::pipelines::FrozenGlobalReductionLaunch::new(
                workgroup_config.reduction_default,
                precision,
                &satisfied_limits,
            )
            .ok_or_else(|| anyhow!("enabled WGPU global reduction launch limits are invalid"))?;
        let pipelines = WgpuPipelines::new(
            &access,
            precision,
            image_norm_bootstrap,
            linear_launch,
            global_reduce_launch,
            workgroup_config.matmul_tile,
        );
        let f32_real_floating = (precision == NumericPrecision::F64).then(|| {
            crate::backend::wgpu::pipelines::F32RealFloatingPipelines::new(
                &access,
                linear_launch,
                crate::backend::wgpu::pipelines::FrozenShaderTokens::new(
                    linear_launch,
                    workgroup_config.matmul_tile,
                ),
            )
        });
        let ordered_real_mean = (precision == NumericPrecision::F64)
            .then(|| crate::backend::wgpu::pipelines::OrderedRealMeanPipelines::new(&access));

        let sparse_pipelines =
            super::sparse::SparsePipelines::new(&access, precision, workgroup_config.scalar);

        let buffer_pool_limit = Self::buffer_residency_pool_limit();
        let max_poolable_bytes =
            Self::buffer_residency_max_poolable_bytes(satisfied_limits.max_buffer_size);

        let bind_group_cache = Arc::new(BindGroupCache::default());
        let reduction_reuse = OwnerShared::new(Mutex::new(ReductionReuseCache::try_new(
            Self::reduction_reuse_budget(),
        )?));
        let kernel_resources = OwnerShared::new(KernelResourceRegistry::default());
        let allocations = OwnerShared::new(super::allocation_state::AllocationState::new(
            runtime_device_id,
            buffer_pool_limit,
            max_poolable_bytes,
            bind_group_cache.clone(),
            OwnerShared::downgrade(&kernel_resources),
        ));
        allocations.set_reduction_reuse(OwnerShared::downgrade(&reduction_reuse))?;
        #[cfg(not(target_arch = "wasm32"))]
        let observation_stopped = access.with_profiled_shared(&emission_authority, |emission| {
            install_device_error_handlers(emission.device(), &allocations, device_health.clone())
        })?;
        #[cfg(not(target_arch = "wasm32"))]
        let observation_progress = super::observation_progress::NativeObservationProgress::start(
            access.clone(),
            observation_limit,
            observation_stopped,
            device_health.clone(),
        )
        .map_err(|error| log::warn!("provider observation progress unavailable: {error}"))
        .ok();

        let provider = Self {
            capability_snapshot: std::sync::OnceLock::new(),
            instance,
            access,
            emission_authority,
            device_health,
            adapter,
            adapter_info,
            adapter_limits: satisfied_limits,
            enabled_features,
            timestamp_capability,
            workgroup_config,
            linear_launch,
            global_reduce_launch,
            allocations,
            sparse: super::sparse::SparseState::default(),
            sparse_pipelines,
            pipelines,
            f32_real_floating,
            ordered_real_mean,
            runtime_device_id,
            #[cfg(not(target_arch = "wasm32"))]
            observation_progress,
            observation_budget: Arc::new(
                runmat_accelerate_api::observation::ProviderObservationBudget::with_wgpu_query_bytes(
                    observation_limit,
                    query_requested_byte_limit,
                ),
            ),
            cache_device_id,
            precision,
            element_size,
            fused_pipeline_cache: Mutex::new(super::pipeline_cache::PipelineCache::default()),
            validated_fusion_cache: Mutex::new(HashMap::new()),
            warmed_pipeline_cache: Mutex::new(HashMap::new()),
            bind_group_layout_cache: Mutex::new(HashMap::new()),
            bind_group_layout_tags: Mutex::new(HashMap::new()),
            bind_group_cache,
            kernel_resources,
            metrics: crate::backend::wgpu::metrics::WgpuMetrics::default(),
            telemetry: AccelTelemetry::default(),
            #[cfg(test)]
            numeric_upload_post_buffer_fault: Mutex::new(None),
            #[cfg(test)]
            numeric_upload_post_buffer_count: std::sync::atomic::AtomicUsize::new(0),
            reduction_two_pass_mode,
            reduction_two_pass_threshold: two_pass_threshold,
            reduction_workgroup_size_default: reduction_wg_default,
            pipeline_cache_dir,
            reduction_autotune,
            image_norm_autotune,
            image_norm_pipeline_cache: Mutex::new(HashMap::new()),
            autotune_base_dir,
            autotune_device_tag,
            reduction_reuse,
            fft_twiddle_cache: Mutex::new(HashMap::new()),
        };
        let mut snapshot = crate::placement::wgpu_capability_snapshot(
            &provider,
            provider.adapter_limits.max_buffer_size,
        );
        snapshot.sparse = Some(super::sparse::capabilities(&provider));
        // Construction owns the provider until its immutable contract is set.
        let _ = provider.capability_snapshot.set(snapshot);
        Ok(provider)
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

fn select_precision(
    requested: Option<NumericPrecision>,
    exact_f64_available: bool,
) -> Result<NumericPrecision> {
    match requested.unwrap_or(NumericPrecision::F32) {
        NumericPrecision::F64 if !exact_f64_available => Err(anyhow!(
            "exact f64 WGPU precision was requested, but the selected adapter does not expose SHADER_F64"
        )),
        precision => Ok(precision),
    }
}

fn environment_precision() -> Result<Option<NumericPrecision>> {
    match std::env::var("RUNMAT_WGPU_FORCE_PRECISION") {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" | "32" => Ok(Some(NumericPrecision::F32)),
            "f64" | "float64" | "64" => Ok(Some(NumericPrecision::F64)),
            _ => Err(anyhow!(
                "RUNMAT_WGPU_FORCE_PRECISION must be one of f32, float32, 32, f64, float64, or 64"
            )),
        },
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(anyhow!(
            "RUNMAT_WGPU_FORCE_PRECISION is not valid Unicode: {error}"
        )),
    }
}

#[cfg(test)]
mod precision_tests {
    use super::{select_precision, NumericPrecision};

    #[test]
    fn exact_f64_selection_fails_closed_without_hardware_support() {
        assert!(select_precision(Some(NumericPrecision::F64), false).is_err());
        assert_eq!(
            select_precision(Some(NumericPrecision::F64), true).expect("supported exact f64"),
            NumericPrecision::F64
        );
        assert_eq!(
            select_precision(None, false).expect("default native precision"),
            NumericPrecision::F32
        );
        assert_eq!(
            select_precision(None, true).expect("default precision with binary64 available"),
            NumericPrecision::F32
        );
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod ieee_feature_tests {
    use super::*;

    #[test]
    fn native_binary64_requests_the_complete_controlled_arithmetic_contract() {
        let capable = native_required_features(NumericPrecision::F64);
        assert_eq!(
            capable,
            crate::backend::wgpu::shader_compilation::capabilities::REQUIRED_FEATURES
        );
        assert_eq!(
            native_required_features(NumericPrecision::F32),
            wgpu::Features::empty()
        );
    }
}
