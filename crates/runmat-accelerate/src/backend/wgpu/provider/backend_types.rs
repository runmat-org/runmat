use super::*;

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

pub(super) type MomentsKey = (u64, Vec<usize>);
pub(super) type MomentsValue = (GpuTensorHandle, GpuTensorHandle);
pub(super) type MomentsCache = HashMap<MomentsKey, MomentsValue>;

// Core WGPU provider state (device, caches, pipelines)
pub struct WgpuProvider {
    pub(super) instance: Arc<wgpu::Instance>,
    pub(super) device: Arc<wgpu::Device>,
    pub(super) queue: Arc<wgpu::Queue>,
    pub(super) adapter: Arc<wgpu::Adapter>,
    pub(super) adapter_info: wgpu::AdapterInfo,
    pub(super) adapter_limits: wgpu::Limits,
    pub(super) workgroup_config: WorkgroupConfig,
    pub(super) buffers: Mutex<HashMap<u64, BufferEntry>>, // in-memory handle table
    pub(super) buffer_residency: BufferResidency,
    pub(super) buffer_residency_max_poolable_bytes: u64,
    pub(super) next_id: AtomicU64,
    pub(super) pipelines: WgpuPipelines,
    pub(super) runtime_device_id: u32,
    pub(super) cache_device_id: u32,
    pub(super) precision: NumericPrecision,
    pub(super) element_size: usize,
    pub(super) fused_pipeline_cache: Mutex<HashMap<u64, Arc<wgpu::ComputePipeline>>>,
    pub(super) bind_group_layout_cache: Mutex<HashMap<String, Arc<wgpu::BindGroupLayout>>>,
    pub(super) bind_group_layout_tags: Mutex<HashMap<usize, String>>,
    pub(super) bind_group_cache: BindGroupCache,
    pub(super) kernel_resources: KernelResourceRegistry,
    pub(super) metrics: crate::backend::wgpu::metrics::WgpuMetrics,
    pub(super) telemetry: AccelTelemetry,
    pub(super) reduction_two_pass_mode: ReductionTwoPassMode,
    pub(super) reduction_two_pass_threshold: usize,
    pub(super) reduction_workgroup_size_default: u32,
    pub(super) pipeline_cache_dir: Option<std::path::PathBuf>,
    pub(super) reduction_autotune: AutotuneController<ReductionAutotuneKey, ReductionTuning>,
    pub(super) image_norm_autotune: AutotuneController<ImageNormalizeKey, ImageNormalizeTuning>,
    pub(super) image_norm_pipeline_cache:
        Mutex<HashMap<ImageNormalizeTuning, Arc<wgpu::ComputePipeline>>>,
    #[allow(dead_code)]
    pub(super) autotune_base_dir: Option<PathBuf>,
    #[allow(dead_code)]
    pub(super) autotune_device_tag: String,
    // Optimization caches
    pub(super) pow2_of: Mutex<HashMap<u64, u64>>, // squared_buffer_id -> base_buffer_id
    pub(super) moments_cache: Mutex<MomentsCache>, // (base_buffer_id, dims) -> (mean, ex2)
    pub(super) fft_twiddle_cache: Mutex<HashMap<(usize, u8), Arc<wgpu::Buffer>>>, // (len, mode) -> twiddle buffer
}

#[cfg(target_arch = "wasm32")]
unsafe impl Send for WgpuProvider {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for WgpuProvider {}

#[derive(Clone)]
pub(super) struct BufferEntry {
    pub(super) buffer: Arc<wgpu::Buffer>,
    pub(super) len: usize,
    pub(super) shape: Vec<usize>,
    pub(super) storage: GpuTensorStorage,
    pub(super) precision: NumericPrecision,
    pub(super) usage: BufferUsageClass,
    pub(super) last_submission_id: Option<u32>,
}

#[derive(Clone, Copy)]
pub(super) struct MatrixOperandView {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) lda: u32,
    pub(super) transpose: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct WorkgroupConfig {
    pub(super) scalar: u32,
    pub(super) reduction_default: u32,
    pub(super) matmul_tile: u32,
    pub(super) max_x: u32,
    pub(super) max_y: u32,
    pub(super) max_z: u32,
    pub(super) adapter_max_invocations: u32,
}

impl WorkgroupConfig {
    pub(super) fn new(
        limits: &wgpu::Limits,
        requested_scalar: u32,
        requested_reduction: u32,
        requested_tile: u32,
    ) -> Self {
        let max_x = Self::normalize_dim(limits.max_compute_workgroup_size_x, 1024);
        let max_y = Self::normalize_dim(limits.max_compute_workgroup_size_y, 1024);
        let max_z = Self::normalize_dim(limits.max_compute_workgroup_size_z, 64);
        let adapter_max_invocations =
            Self::normalize_invocations(limits.max_compute_invocations_per_workgroup, 1536);

        let scalar =
            Self::clamp_linear_workgroup(requested_scalar, max_x, adapter_max_invocations, 32);
        let reduction =
            Self::clamp_linear_workgroup(requested_reduction, max_x, adapter_max_invocations, 32);
        let matmul_tile =
            Self::clamp_matmul_tile(requested_tile, max_x, max_y, adapter_max_invocations, 8);

        Self {
            scalar,
            reduction_default: reduction,
            matmul_tile,
            max_x,
            max_y,
            max_z,
            adapter_max_invocations,
        }
    }

    fn normalize_dim(value: u32, fallback: u32) -> u32 {
        if value == 0 {
            fallback
        } else {
            value
        }
    }

    fn normalize_invocations(value: u32, fallback: u32) -> u32 {
        if value == 0 {
            fallback
        } else {
            value
        }
    }

    fn clamp_linear_workgroup(requested: u32, max_dim: u32, max_inv: u32, align: u32) -> u32 {
        let mut value = requested.max(1);
        let allowed_max = max_dim.min(max_inv).max(1);
        let align = if allowed_max < align { 1 } else { align };
        value = value.min(allowed_max);
        value = Self::align_down(value, align).max(1);
        value = Self::floor_power_of_two(value);
        if value == 0 {
            value = allowed_max;
        }
        while value > allowed_max && value > 1 {
            value = Self::align_down((value / 2).max(1), align).max(1);
        }
        value.max(1).min(allowed_max)
    }

    fn clamp_matmul_tile(requested: u32, max_x: u32, max_y: u32, max_inv: u32, align: u32) -> u32 {
        let mut tile = requested.max(1);
        tile = tile.min(max_x).min(max_y);
        let inv_limit = (max_inv as f64).sqrt().floor() as u32;
        tile = tile.min(inv_limit.max(1));
        let align = if tile < align { 1 } else { align };
        tile = Self::align_down(tile, align).max(1);
        tile = Self::floor_power_of_two(tile);
        if tile == 0 {
            tile = 1;
        }
        while tile > 1
            && (tile > max_x || tile > max_y || (tile as u64 * tile as u64) > max_inv as u64)
        {
            tile = if tile > align {
                Self::align_down(tile - align, align).max(1)
            } else {
                (tile / 2).max(1)
            };
        }
        tile.max(1)
    }

    fn align_down(value: u32, align: u32) -> u32 {
        if align <= 1 {
            return value;
        }
        let remainder = value % align;
        if remainder == 0 {
            value
        } else {
            value.saturating_sub(remainder)
        }
    }

    fn floor_power_of_two(value: u32) -> u32 {
        if value == 0 {
            return 1;
        }
        1 << (31 - value.leading_zeros())
    }

    pub(super) fn sanitize_image_normalize_tuning(
        &self,
        mut tuning: ImageNormalizeTuning,
        batches: u32,
    ) -> ImageNormalizeTuning {
        let original = tuning;
        let max_lane_dim = self.max_x.max(32);
        tuning.lane_count = tuning.lane_count.clamp(32, max_lane_dim).max(32);
        tuning.lane_count =
            WgpuProvider::round_up_to_multiple(tuning.lane_count, 32).min(max_lane_dim);
        tuning.values_per_thread = tuning.values_per_thread.clamp(1, 8);
        let max_spatial = self.max_y.max(1);
        tuning.spatial_tile = tuning.spatial_tile.clamp(1, max_spatial);
        let max_invocations = self.adapter_max_invocations.max(64);
        while tuning.lane_count.saturating_mul(tuning.spatial_tile) > max_invocations {
            if tuning.lane_count > 32 {
                tuning.lane_count -= 32;
            } else if tuning.spatial_tile > 1 {
                tuning.spatial_tile -= 1;
            } else {
                break;
            }
        }
        tuning.batch_tile = tuning.batch_tile.clamp(1, batches.max(1));
        if tuning != original {
            debug!(
                "sanitize_image_normalize_tuning batches={} lane {} -> {} spatial {} -> {} values/thread {} -> {} batch_tile {} -> {} (max_invocations={}, limits=({}, {}, {}))",
                batches,
                original.lane_count,
                tuning.lane_count,
                original.spatial_tile,
                tuning.spatial_tile,
                original.values_per_thread,
                tuning.values_per_thread,
                original.batch_tile,
                tuning.batch_tile,
                self.adapter_max_invocations,
                self.max_x,
                self.max_y,
                self.max_z
            );
        }
        tuning
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) enum ReductionMode {
    SinglePass,
    TwoPass { chunk_rows: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct ReductionTuning {
    pub(super) mode: ReductionMode,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct ReductionAutotuneKey {
    pub(super) precision: u8,
    pub(super) slices_bucket: u32,
    pub(super) reduce_bucket: u32,
}

impl ReductionAutotuneKey {
    pub(super) fn new(precision: NumericPrecision, num_slices: usize, reduce_len: usize) -> Self {
        Self {
            precision: match precision {
                NumericPrecision::F64 => 64,
                NumericPrecision::F32 => 32,
            },
            slices_bucket: bucketize_dimension(num_slices),
            reduce_bucket: bucketize_dimension(reduce_len),
        }
    }
}

pub(super) fn bucketize_dimension(value: usize) -> u32 {
    if value == 0 {
        return 0;
    }
    let mut bucket = 1u64;
    let target = value as u64;
    while bucket < target && bucket < u32::MAX as u64 {
        bucket <<= 1;
    }
    bucket.min(u32::MAX as u64) as u32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct ImageNormalizeTuning {
    pub(super) batch_tile: u32,
    pub(super) values_per_thread: u32,
    pub(super) lane_count: u32,
    pub(super) spatial_tile: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct ImageNormalizeKey {
    pub(super) version: u8,
    pub(super) precision: u8,
    pub(super) plane_bucket: u32,
    pub(super) batch_bucket: u32,
}

impl ImageNormalizeKey {
    pub(super) fn new(precision: NumericPrecision, batches: u32, plane: u32) -> Self {
        Self {
            version: WgpuProvider::IMAGE_NORMALIZE_AUTOTUNE_VERSION,
            precision: match precision {
                NumericPrecision::F64 => 64,
                NumericPrecision::F32 => 32,
            },
            plane_bucket: bucketize_dimension(plane as usize),
            batch_bucket: bucketize_dimension(batches as usize),
        }
    }
}
