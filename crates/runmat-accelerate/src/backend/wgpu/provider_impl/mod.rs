// Internal note: this file has become a bit too large.
// Subsequent provider call implementations that would otherwise
// be added in this file should, going forwards, be added to
// ./provider_impl/*.rs instead. This module will be refactored into
// submodules in that manner in the future.

use anyhow::{anyhow, ensure, Result};
use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use futures::channel::oneshot;
use log::{debug, error, info, warn};
use once_cell::sync::OnceCell;
#[cfg(not(target_arch = "wasm32"))]
use pollster::block_on;
use rand::seq::SliceRandom;
use runmat_accelerate_api::{
    AccelContextHandle, AccelContextKind, AccelDownloadFuture, AccelProvider, AccelProviderFuture,
    ApiDeviceInfo, CorrcoefNormalization, CorrcoefOptions, CorrcoefRows, CovNormalization, CovRows,
    CovarianceOptions, FindDirection, FspecialRequest, GpuTensorHandle, GpuTensorStorage,
    HostTensorOwned, HostTensorView, ImfilterOptions, ImfilterPadding, IsMemberOptions,
    IsMemberResult, MeshgridAxisView, PagefunOp, PagefunRequest, ProviderBandwidth,
    ProviderCholResult, ProviderCondNorm, ProviderConv1dOptions, ProviderConvMode,
    ProviderConvOrientation, ProviderCummaxResult, ProviderCumminResult, ProviderEigResult,
    ProviderFindResult, ProviderHermitianKind, ProviderIirFilterOptions, ProviderIirFilterResult,
    ProviderInvOptions, ProviderLinsolveOptions, ProviderLinsolveResult, ProviderLuResult,
    ProviderMeshgridResult, ProviderNanMode, ProviderNormOrder, ProviderPinvOptions,
    ProviderPolyderQuotient, ProviderPolyfitResult, ProviderPolyvalOptions, ProviderPrecision,
    ProviderQrOptions, ProviderQrPivot, ProviderQrPowerIterResult, ProviderQrResult,
    ProviderScanDirection, ProviderStdNormalization, ProviderSymmetryKind, ReduceDimResult,
    ReductionFlavor, ReductionTwoPassMode, SetdiffOptions, SetdiffResult, SortComparison,
    SortOrder, SortResult, SortRowsColumnSpec, SpawnHandleConcurrency, UnionOptions, UnionResult,
    UniqueOptions, UniqueResult, WgpuBufferRef, WgpuContextHandle,
};
use runmat_builtins::{Tensor, Value};
use runmat_runtime::builtins::common::shape::normalize_scalar_shape;
use runmat_runtime::builtins::image::filters::fspecial::{
    spec_from_request as runtime_fspecial_spec_from_request, FspecialFilterSpec,
};
use runmat_runtime::builtins::image::filters::imfilter::{
    apply_imfilter_tensor as runtime_apply_imfilter_tensor, build_imfilter_plan,
};

use runmat_runtime::builtins::math::linalg::ops::{
    mldivide_host_real_for_provider, mrdivide_host_real_for_provider,
};
use runmat_runtime::builtins::math::linalg::solve::cond::cond_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::inv::inv_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::linsolve::linsolve_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::norm::norm_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::pinv::pinv_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::rank::rank_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::rcond::rcond_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::structure::bandwidth::ensure_matrix_shape as ensure_bandwidth_shape;
use runmat_runtime::builtins::math::linalg::structure::ishermitian::ishermitian_host_real_data;
use runmat_runtime::builtins::math::linalg::structure::issymmetric::ensure_matrix_shape as ensure_symmetry_shape;
use runmat_runtime::builtins::math::linalg::structure::symrcm::symrcm_host_real_data;
use runmat_runtime::builtins::math::poly::polyfit::polyfit_host_real_for_provider;
use runmat_runtime::builtins::math::reduction::{compute_median_inplace, matlab_gradient_shape};
use runmat_runtime::RuntimeError;
use runmat_time::Instant;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use tracing::info_span;
use wgpu::util::DeviceExt;

mod constructors;
mod elementwise;
mod fft;
mod image;
mod indexing;
mod polynomial;
mod rnd;
mod reduction;
mod signal;
mod solve;
mod tensor;
mod window;

use self::window::WindowKind;

use crate::backend::wgpu::autotune::AutotuneController;
use crate::backend::wgpu::cache::{
    bind_group::BindGroupCache, key as cache_key, persist as cache_persist,
};
use crate::backend::wgpu::config::{
    self, DEFAULT_REDUCTION_WG, DEFAULT_TWO_PASS_THRESHOLD, MATMUL_TILE, WORKGROUP_SIZE,
};
use crate::backend::wgpu::params::{
    BandwidthParams, Conv1dParams, CummaxParams, CumminParams, CumprodParams, CumsumParams,
    DiffParams, FilterParams, GradientParamsF32, GradientParamsF64, ImageNormalizeUniforms,
    LinearGatherParams, LinearScatterParams, QrPowerIterParams, SymmetryParamsF32,
    SymmetryParamsF64, SyrkParams, IMAGE_NORMALIZE_FLAG_BIAS, IMAGE_NORMALIZE_FLAG_GAIN,
    IMAGE_NORMALIZE_FLAG_GAMMA, SYRK_FLAG_ACCUMULATE, SYRK_FLAG_FILL_BOTH,
};
use crate::backend::wgpu::pipelines::{ImageNormalizeBootstrap, WgpuPipelines};
use crate::backend::wgpu::residency::{BufferResidency, BufferUsageClass};
use crate::backend::wgpu::resources::{KernelResourceRegistry, UniformBufferKey};
use crate::backend::wgpu::shaders::image_normalize::{
    IMAGE_NORMALIZE_SHADER_F32, IMAGE_NORMALIZE_SHADER_F64,
};
use crate::backend::wgpu::shaders::logical::{
    ELEM_EQ_SHADER_F32, ELEM_EQ_SHADER_F64, ELEM_GE_SHADER_F32, ELEM_GE_SHADER_F64,
    ELEM_GT_SHADER_F32, ELEM_GT_SHADER_F64, ELEM_LE_SHADER_F32, ELEM_LE_SHADER_F64,
    ELEM_LT_SHADER_F32, ELEM_LT_SHADER_F64, ELEM_NE_SHADER_F32, ELEM_NE_SHADER_F64,
    LOGICAL_AND_SHADER_F32, LOGICAL_AND_SHADER_F64, LOGICAL_ISFINITE_SHADER_F32,
    LOGICAL_ISFINITE_SHADER_F64, LOGICAL_ISINF_SHADER_F32, LOGICAL_ISINF_SHADER_F64,
    LOGICAL_ISNAN_SHADER_F32, LOGICAL_ISNAN_SHADER_F64, LOGICAL_NOT_SHADER_F32,
    LOGICAL_NOT_SHADER_F64, LOGICAL_OR_SHADER_F32, LOGICAL_OR_SHADER_F64,
    LOGICAL_XOR_SHADER_F32, LOGICAL_XOR_SHADER_F64,
};
use crate::backend::wgpu::types::NumericPrecision;
const QR_DEVICE_MAX_COLS: usize = 64;
const QR_DEVICE_MAX_ELEMS: usize = 1_000_000;
use crate::fusion::{active_fusion, active_group_plan_clone};
use crate::host_lu::{lu_factor_host, LuHostFactors};
use crate::sortrows_host::{sort_rows_host, SortRowsHostOutputs};
use crate::telemetry::AccelTelemetry;

fn runtime_flow_to_anyhow(_context: &str, err: RuntimeError) -> anyhow::Error {
    anyhow::Error::new(err)
}

fn validate_compute_binding_counts(
    operation: &str,
    storage_bindings: usize,
    total_bindings: usize,
    limits: &wgpu::Limits,
) -> Result<()> {
    let storage_limit = limits.max_storage_buffers_per_shader_stage as usize;
    ensure!(
        storage_bindings <= storage_limit,
        "{}: requires {} storage buffers, but this WebGPU adapter supports {} per shader stage",
        operation,
        storage_bindings,
        storage_limit
    );

    let binding_limit = limits.max_bindings_per_bind_group as usize;
    ensure!(
        total_bindings <= binding_limit,
        "{}: requires {} bind group entries, but this WebGPU adapter supports {} per bind group",
        operation,
        total_bindings,
        binding_limit
    );

    Ok(())
}

fn checked_binding_count(operation: &str, left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| anyhow!("{}: binding count overflow", operation))
}

fn gpu_per_buffer_limit_error(
    operation: &str,
    requested_bytes: u64,
    max_bytes: u64,
) -> anyhow::Error {
    let requested_mib = requested_bytes as f64 / (1024.0 * 1024.0);
    let max_mib = max_bytes as f64 / (1024.0 * 1024.0);
    anyhow!(
        "{operation}: requested {requested_bytes} bytes ({requested_mib:.2} MiB) exceeds this device per-buffer limit of {max_bytes} bytes ({max_mib:.2} MiB). This is a per-buffer backend limit (not total VRAM). Split the data into smaller arrays/chunks and process iteratively."
    )
}

fn gpu_dispatch_length_limit_error(operation: &str, len: usize) -> anyhow::Error {
    anyhow!(
        "{operation}: tensor length {len} exceeds the current GPU kernel indexing limit of {} elements. Split the operation into smaller chunks and process iteratively.",
        u32::MAX
    )
}

#[cfg(test)]
mod compute_binding_count_tests {
    use super::{checked_binding_count, validate_compute_binding_counts, WgpuProvider};

    #[test]
    fn rejects_storage_bindings_over_adapter_stage_limit() {
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 10,
            ..Default::default()
        };

        let err = validate_compute_binding_counts("fused_elementwise_multi", 11, 12, &limits)
            .expect_err(
                "storage binding overflow should be rejected before creating a WGPU layout",
            );

        assert!(err.to_string().contains("requires 11 storage buffers"));
    }

    #[test]
    fn accepts_bindings_at_adapter_limits() {
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 10,
            max_bindings_per_bind_group: 11,
            ..Default::default()
        };

        validate_compute_binding_counts("fused_elementwise", 10, 11, &limits)
            .expect("limits are inclusive");
    }

    #[test]
    fn rejects_bindings_over_bind_group_limit() {
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 10,
            max_bindings_per_bind_group: 11,
            ..Default::default()
        };

        let err = validate_compute_binding_counts("fused_elementwise", 10, 12, &limits)
            .expect_err("bind group entry overflow should be rejected");

        assert!(err.to_string().contains("requires 12 bind group entries"));
    }

    #[test]
    fn rejects_binding_count_overflow() {
        let err = checked_binding_count("fused_elementwise", usize::MAX, 1)
            .expect_err("binding count overflow should be rejected");

        assert!(err.to_string().contains("binding count overflow"));
    }

    #[test]
    fn poolable_bytes_uses_default_and_caps_to_adapter_limit() {
        assert_eq!(
            WgpuProvider::parse_buffer_residency_max_poolable_bytes(None, 0),
            256u64 << 20
        );
        assert_eq!(
            WgpuProvider::parse_buffer_residency_max_poolable_bytes(None, 128u64 << 20),
            128u64 << 20
        );
    }

    #[test]
    fn poolable_bytes_honors_env_override_and_adapter_cap() {
        assert_eq!(
            WgpuProvider::parse_buffer_residency_max_poolable_bytes(
                Some("1073741824"),
                512u64 << 20
            ),
            512u64 << 20
        );
    }

    #[test]
    fn poolable_bytes_accepts_zero_to_disable_pooling() {
        assert_eq!(
            WgpuProvider::parse_buffer_residency_max_poolable_bytes(Some("0"), 2u64 << 30),
            0
        );
    }

    #[test]
    fn poolable_bytes_invalid_override_falls_back_to_default() {
        assert_eq!(
            WgpuProvider::parse_buffer_residency_max_poolable_bytes(Some("bad"), 512u64 << 20),
            256u64 << 20
        );
    }
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

type MomentsKey = (u64, Vec<usize>);
type MomentsValue = (GpuTensorHandle, GpuTensorHandle);
type MomentsCache = HashMap<MomentsKey, MomentsValue>;

// Core WGPU provider state (device, caches, pipelines)
pub struct WgpuProvider {
    instance: Arc<wgpu::Instance>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter: Arc<wgpu::Adapter>,
    adapter_info: wgpu::AdapterInfo,
    adapter_limits: wgpu::Limits,
    workgroup_config: WorkgroupConfig,
    buffers: Mutex<HashMap<u64, BufferEntry>>, // in-memory handle table
    buffer_residency: BufferResidency,
    buffer_residency_max_poolable_bytes: u64,
    next_id: AtomicU64,
    pipelines: WgpuPipelines,
    runtime_device_id: u32,
    cache_device_id: u32,
    precision: NumericPrecision,
    element_size: usize,
    fused_pipeline_cache: Mutex<HashMap<u64, Arc<wgpu::ComputePipeline>>>,
    bind_group_layout_cache: Mutex<HashMap<String, Arc<wgpu::BindGroupLayout>>>,
    bind_group_layout_tags: Mutex<HashMap<usize, String>>,
    bind_group_cache: BindGroupCache,
    kernel_resources: KernelResourceRegistry,
    metrics: crate::backend::wgpu::metrics::WgpuMetrics,
    telemetry: AccelTelemetry,
    reduction_two_pass_mode: ReductionTwoPassMode,
    reduction_two_pass_threshold: usize,
    reduction_workgroup_size_default: u32,
    pipeline_cache_dir: Option<std::path::PathBuf>,
    reduction_autotune: AutotuneController<ReductionAutotuneKey, ReductionTuning>,
    image_norm_autotune: AutotuneController<ImageNormalizeKey, ImageNormalizeTuning>,
    image_norm_pipeline_cache: Mutex<HashMap<ImageNormalizeTuning, Arc<wgpu::ComputePipeline>>>,
    #[allow(dead_code)]
    autotune_base_dir: Option<PathBuf>,
    #[allow(dead_code)]
    autotune_device_tag: String,
    // Optimization caches
    pow2_of: Mutex<HashMap<u64, u64>>, // squared_buffer_id -> base_buffer_id
    moments_cache: Mutex<MomentsCache>, // (base_buffer_id, dims) -> (mean, ex2)
    fft_twiddle_cache: Mutex<HashMap<(usize, u8), Arc<wgpu::Buffer>>>, // (len, mode) -> twiddle buffer
}

#[cfg(target_arch = "wasm32")]
unsafe impl Send for WgpuProvider {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for WgpuProvider {}

#[derive(Clone)]
struct BufferEntry {
    buffer: Arc<wgpu::Buffer>,
    len: usize,
    shape: Vec<usize>,
    storage: GpuTensorStorage,
    precision: NumericPrecision,
    usage: BufferUsageClass,
    last_submission_id: Option<u32>,
}

#[derive(Clone, Copy)]
struct MatrixOperandView {
    rows: usize,
    cols: usize,
    lda: u32,
    transpose: bool,
}

#[derive(Clone, Copy, Debug)]
struct WorkgroupConfig {
    scalar: u32,
    reduction_default: u32,
    matmul_tile: u32,
    max_x: u32,
    max_y: u32,
    max_z: u32,
    adapter_max_invocations: u32,
}

impl WorkgroupConfig {
    fn new(
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

    fn sanitize_image_normalize_tuning(
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
enum ReductionMode {
    SinglePass,
    TwoPass { chunk_rows: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct ReductionTuning {
    mode: ReductionMode,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct ReductionAutotuneKey {
    precision: u8,
    slices_bucket: u32,
    reduce_bucket: u32,
}

impl ReductionAutotuneKey {
    fn new(precision: NumericPrecision, num_slices: usize, reduce_len: usize) -> Self {
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

fn bucketize_dimension(value: usize) -> u32 {
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
struct ImageNormalizeTuning {
    batch_tile: u32,
    values_per_thread: u32,
    lane_count: u32,
    spatial_tile: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct ImageNormalizeKey {
    version: u8,
    precision: u8,
    plane_bucket: u32,
    batch_bucket: u32,
}

impl ImageNormalizeKey {
    fn new(precision: NumericPrecision, batches: u32, plane: u32) -> Self {
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

fn parse_two_pass_mode(raw: &str) -> Option<ReductionTwoPassMode> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "auto" => Some(ReductionTwoPassMode::Auto),
        "force_on" | "on" | "true" | "1" => Some(ReductionTwoPassMode::ForceOn),
        "force_off" | "off" | "false" | "0" => Some(ReductionTwoPassMode::ForceOff),
        _ => None,
    }
}

fn build_matrix_operand_view(
    handle: &GpuTensorHandle,
    entry: &BufferEntry,
) -> Result<MatrixOperandView> {
    if entry.shape.len() < 2 {
        return Err(anyhow!(
            "matrix operand requires at least 2D tensor (buffer {} shape {:?})",
            handle.buffer_id,
            entry.shape
        ));
    }
    let rows = entry.shape[0];
    let cols = entry.shape[1];
    if let Some(info) = runmat_accelerate_api::handle_transpose_info(handle) {
        if rows != info.base_cols || cols != info.base_rows {
            return Err(anyhow!(
                "transpose metadata mismatch for buffer {}",
                handle.buffer_id
            ));
        }
        let lda = u32::try_from(info.base_rows)
            .map_err(|_| anyhow!("leading dimension exceeds GPU limits"))?;
        Ok(MatrixOperandView {
            rows,
            cols,
            lda,
            transpose: true,
        })
    } else {
        let lda =
            u32::try_from(rows).map_err(|_| anyhow!("leading dimension exceeds GPU limits"))?;
        Ok(MatrixOperandView {
            rows,
            cols,
            lda,
            transpose: false,
        })
    }
}

fn canonical_vendor_name(info: &wgpu::AdapterInfo) -> String {
    match info.vendor {
        0x10DE => "NVIDIA".to_string(),
        0x1002 | 0x1022 => "AMD".to_string(),
        0x8086 => "Intel".to_string(),
        0x106B => "Apple".to_string(),
        0x13B5 => "ARM".to_string(),
        0x5143 => "Qualcomm".to_string(),
        0x1414 => "Microsoft".to_string(),
        0x1AE0 => "Google".to_string(),
        0x1C5C => "Huawei".to_string(),
        0 => info
            .name
            .split_whitespace()
            .next()
            .unwrap_or("unknown")
            .to_string(),
        other => {
            let prefix = info.name.split_whitespace().next().unwrap_or("vendor");
            format!("{prefix} (0x{other:04x})")
        }
    }
}

const POLYDER_EPS: f64 = 1.0e-12;

#[derive(Clone, Copy)]
enum PolynomialOrientation {
    Scalar,
    Row,
    Column,
}

fn polynomial_orientation(shape: &[usize]) -> Result<PolynomialOrientation> {
    let mut non_unit = 0usize;
    let mut orientation = PolynomialOrientation::Scalar;
    for (idx, &dim) in shape.iter().enumerate() {
        if dim > 1 {
            non_unit += 1;
            orientation = if idx == 0 {
                PolynomialOrientation::Column
            } else {
                PolynomialOrientation::Row
            };
        }
    }
    if non_unit > 1 {
        Err(anyhow!(
            "polyder: coefficient tensors must be vectors on the GPU"
        ))
    } else {
        Ok(orientation)
    }
}

fn conv_orientation_for(orientation: PolynomialOrientation) -> ProviderConvOrientation {
    match orientation {
        PolynomialOrientation::Column => ProviderConvOrientation::Column,
        PolynomialOrientation::Scalar | PolynomialOrientation::Row => ProviderConvOrientation::Row,
    }
}

fn shape_for_orientation(orientation: PolynomialOrientation, len: usize) -> Vec<usize> {
    if len <= 1 {
        return vec![1, 1];
    }
    match orientation {
        PolynomialOrientation::Scalar | PolynomialOrientation::Row => vec![1, len],
        PolynomialOrientation::Column => vec![len, 1],
    }
}

fn trim_leading_zeros_real(coeffs: &[f64]) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![0.0];
    }
    if let Some(idx) = coeffs.iter().position(|c| c.abs() > POLYDER_EPS) {
        coeffs[idx..].to_vec()
    } else {
        vec![0.0]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PolyderParams {
    input_len: u32,
    output_len: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PolyintParamsF64 {
    input_len: u32,
    output_len: u32,
    constant: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PolyintParamsF32 {
    input_len: u32,
    output_len: u32,
    constant: f32,
    _pad0: f32,
}

fn normalize_eye_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => normalize_scalar_shape(shape),
        1 => {
            let n = shape[0];
            normalize_scalar_shape(&[n, n])
        }
        _ => normalize_scalar_shape(shape),
    }
}

fn normalize_concat_shape(mut shape: Vec<usize>, dim_zero: usize) -> Vec<usize> {
    if shape.is_empty() {
        return normalize_scalar_shape(&shape);
    }
    let min_len = ((dim_zero + 1).max(2)).min(shape.len());
    while shape.len() > min_len && shape.last() == Some(&1) {
        shape.pop();
    }
    normalize_scalar_shape(&shape)
}

fn normalize_gradient_shape(shape: &[usize], len: usize) -> Vec<usize> {
    matlab_gradient_shape(shape, len)
}

fn conv1d_output_shape(len: usize, orientation: ProviderConvOrientation) -> Vec<usize> {
    match (orientation, len) {
        (ProviderConvOrientation::Row, 0) => vec![1, 0],
        (ProviderConvOrientation::Row, _) => vec![1, len],
        (ProviderConvOrientation::Column, 0) => vec![0, 1],
        (ProviderConvOrientation::Column, _) => vec![len, 1],
    }
}

fn conv1d_window(
    signal_len: usize,
    kernel_len: usize,
    mode: ProviderConvMode,
) -> Result<(usize, usize, usize)> {
    if signal_len == 0 || kernel_len == 0 {
        return Ok((0, 0, 0));
    }
    let full_len = signal_len
        .checked_add(kernel_len)
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| anyhow!("conv1d: result length overflow"))?;
    let (output_len, start_offset) = match mode {
        ProviderConvMode::Full => (full_len, 0usize),
        ProviderConvMode::Same => {
            let start = if kernel_len == 0 {
                0
            } else {
                (kernel_len - 1) / 2
            };
            let len = signal_len.min(full_len.saturating_sub(start));
            (len, start)
        }
        ProviderConvMode::Valid => {
            if signal_len < kernel_len {
                (0usize, 0usize)
            } else {
                (signal_len - kernel_len + 1, kernel_len - 1)
            }
        }
    };
    if output_len == 0 {
        return Ok((0, start_offset, full_len));
    }
    ensure!(
        start_offset
            .checked_add(output_len)
            .map(|v| v <= full_len)
            .unwrap_or(false),
        "conv1d: window exceeds full convolution length"
    );
    Ok((output_len, start_offset, full_len))
}

fn product_checked(dims: &[usize]) -> Option<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn canonical_matrix_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![1, shape[0]],
        _ => {
            let mut out = shape.to_vec();
            if out.len() == 1 {
                out.push(1);
            }
            out
        }
    }
}

fn pad_dims(mut dims: Vec<usize>, rank: usize) -> Vec<usize> {
    if dims.len() < rank {
        dims.resize(rank, 1);
    } else if dims.len() > rank {
        dims.truncate(rank);
    }
    dims
}

fn compute_page_strides(dims: &[usize]) -> Vec<usize> {
    let mut stride = 1usize;
    let mut out = Vec::with_capacity(dims.len());
    for &dim in dims {
        out.push(stride);
        stride = stride.saturating_mul(dim.max(1));
    }
    out
}

fn decode_multi_index(mut index: usize, dims: &[usize], out: &mut [usize]) {
    for (dim, &extent) in dims.iter().enumerate() {
        if extent == 0 {
            out[dim] = 0;
        } else {
            out[dim] = index % extent;
            index /= extent;
        }
    }
}

fn broadcast_linear_index(dims: &[usize], strides: &[usize], multi_index: &[usize]) -> usize {
    let mut linear = 0usize;
    for ((&extent, &stride), &coord) in dims.iter().zip(strides.iter()).zip(multi_index.iter()) {
        if extent == 0 {
            return 0;
        }
        let actual = if extent == 1 { 0 } else { coord };
        linear += actual * stride;
    }
    linear
}

fn gaussian_normalizer(rows: usize, cols: usize, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return 0.0;
    }
    let row_center = (rows as f64 - 1.0) / 2.0;
    let col_center = (cols as f64 - 1.0) / 2.0;
    let denom = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for col in 0..cols {
        let dx = col as f64 - col_center;
        for row in 0..rows {
            let dy = row as f64 - row_center;
            sum += (-((dx * dx + dy * dy) / denom)).exp();
        }
    }
    if sum <= 0.0 || !sum.is_finite() {
        0.0
    } else {
        1.0 / sum
    }
}

fn shapes_compatible(expected: &[usize], actual: &[usize]) -> bool {
    let max_len = expected.len().max(actual.len());
    for idx in 0..max_len {
        let e = expected.get(idx).copied().unwrap_or(1);
        let a = actual.get(idx).copied().unwrap_or(1);
        if e != a {
            return false;
        }
    }
    true
}

fn filter_state_shape(mut base: Vec<usize>, dim_idx: usize, state_len: usize) -> Vec<usize> {
    if base.len() <= dim_idx {
        base.extend(std::iter::repeat_n(1, dim_idx + 1 - base.len()));
    }
    if !base.is_empty() {
        base[dim_idx] = state_len;
    }
    base
}

pub(crate) fn host_tensor_from_value(label: &str, value: Value) -> Result<Tensor> {
    match value {
        Value::Tensor(tensor) => Ok(tensor),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(|e| anyhow!("{label}: {e}")),
        Value::Int(i) => {
            Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| anyhow!("{label}: {e}"))
        }
        Value::Bool(b) => Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|e| anyhow!("{label}: {e}")),
        Value::ComplexTensor(_) => Err(anyhow!(
            "{label}: complex outputs are not supported by the wgpu provider"
        )),
        other => Err(anyhow!("{label}: unexpected value {other:?}")),
    }
}

fn median_from_slice(values: &[f64]) -> f64 {
    if values.is_empty() || values.iter().any(|v| v.is_nan()) {
        f64::NAN
    } else {
        let mut tmp = values.to_vec();
        compute_median_inplace(&mut tmp)
    }
}

fn diag_offset_abs(offset: isize) -> usize {
    if offset >= 0 {
        offset as usize
    } else {
        let magnitude = -(offset as i128);
        magnitude as usize
    }
}
fn diag_matrix_size_checked(len: usize, offset: isize) -> Result<(usize, usize)> {
    let shift = diag_offset_abs(offset);
    let size = len
        .checked_add(shift)
        .ok_or_else(|| anyhow!("diag: result dimension exceeds GPU limits"))?;
    let total = size
        .checked_mul(size)
        .ok_or_else(|| anyhow!("diag: result size exceeds GPU limits"))?;
    Ok((size, total))
}

fn diag_length(rows: usize, cols: usize, offset: isize) -> usize {
    if rows == 0 || cols == 0 {
        return 0;
    }
    if offset >= 0 {
        let shift = offset as usize;
        if shift >= cols {
            0
        } else {
            rows.min(cols - shift)
        }
    } else {
        let shift = diag_offset_abs(offset);
        if shift >= rows {
            0
        } else {
            (rows - shift).min(cols)
        }
    }
}

fn diag_rows_cols(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], 1),
        _ => (shape[0], shape[1]),
    }
}

fn diag_is_vector_like(rows: usize, cols: usize, dims: usize) -> bool {
    rows == 1 || cols == 1 || dims <= 1
}
fn diag_ensure_shape(shape: &[usize]) -> Result<()> {
    if shape.len() > 2 && shape.iter().skip(2).any(|&d| d != 1) {
        Err(anyhow!("diag: input must be 2-D"))
    } else {
        Ok(())
    }
}

fn apply_tril_mask_host(data: &mut [f64], shape: &[usize], offset: isize) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    let plane = rows.saturating_mul(cols);
    if plane == 0 {
        ensure!(
            data.is_empty(),
            "tril: shape/product mismatch ({} vs {})",
            0,
            data.len()
        );
        return Ok(());
    }
    let pages = if shape.len() <= 2 {
        1usize
    } else {
        shape[2..].iter().product::<usize>()
    };
    if pages == 0 {
        ensure!(
            data.is_empty(),
            "tril: shape/product mismatch ({} vs {})",
            0,
            data.len()
        );
        return Ok(());
    }
    let expected = plane
        .checked_mul(pages)
        .ok_or_else(|| anyhow!("tril: dimension product overflow"))?;
    ensure!(
        expected == data.len(),
        "tril: shape/product mismatch ({} vs {})",
        expected,
        data.len()
    );
    for page in 0..pages {
        let base = page * plane;
        for col in 0..cols {
            let col_base = base + col * rows;
            for row in 0..rows {
                if (row as isize) - (col as isize) < -offset {
                    data[col_base + row] = 0.0;
                }
            }
        }
    }
    Ok(())
}

fn apply_triu_mask_host(data: &mut [f64], shape: &[usize], offset: isize) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    let plane = rows.saturating_mul(cols);
    if plane == 0 {
        ensure!(
            data.is_empty(),
            "triu: shape/product mismatch ({} vs {})",
            0,
            data.len()
        );
        return Ok(());
    }
    let pages = if shape.len() <= 2 {
        1usize
    } else {
        shape[2..].iter().product::<usize>()
    };
    if pages == 0 {
        ensure!(
            data.is_empty(),
            "triu: shape/product mismatch ({} vs {})",
            0,
            data.len()
        );
        return Ok(());
    }
    let expected = plane
        .checked_mul(pages)
        .ok_or_else(|| anyhow!("triu: dimension product overflow"))?;
    ensure!(
        expected == data.len(),
        "triu: shape/product mismatch ({} vs {})",
        expected,
        data.len()
    );
    for page in 0..pages {
        let base = page * plane;
        for col in 0..cols {
            let col_base = base + col * rows;
            let col_isize = col as isize;
            for row in 0..rows {
                let diff = col_isize - (row as isize);
                if diff < offset {
                    data[col_base + row] = 0.0;
                }
            }
        }
    }
    Ok(())
}

fn stride_before_for(shape: &[usize], dim: usize) -> usize {
    if dim == 0 {
        return 1;
    }
    let upper = dim.min(shape.len());
    shape[..upper]
        .iter()
        .copied()
        .fold(1usize, |acc, extent| acc.saturating_mul(extent.max(1)))
}

fn stride_after_for(shape: &[usize], dim: usize) -> usize {
    if dim + 1 >= shape.len() {
        return 1;
    }
    shape[(dim + 1)..]
        .iter()
        .copied()
        .fold(1usize, |acc, extent| acc.saturating_mul(extent.max(1)))
}
fn dimension_length_zero_based(shape: &[usize], dim: usize) -> usize {
    shape.get(dim).copied().unwrap_or(1)
}
fn compare_values_for_sort(
    a: f64,
    b: f64,
    order: SortOrder,
    comparison: SortComparison,
) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => match order {
            SortOrder::Ascend => Ordering::Greater,
            SortOrder::Descend => Ordering::Less,
        },
        (false, true) => match order {
            SortOrder::Ascend => Ordering::Less,
            SortOrder::Descend => Ordering::Greater,
        },
        (false, false) => compare_finite_for_sort(a, b, order, comparison),
    }
}

fn compare_finite_for_sort(
    a: f64,
    b: f64,
    order: SortOrder,
    comparison: SortComparison,
) -> Ordering {
    let primary = if matches!(comparison, SortComparison::Abs) {
        let abs_cmp = a.abs().partial_cmp(&b.abs()).unwrap_or(Ordering::Equal);
        if abs_cmp == Ordering::Equal {
            Ordering::Equal
        } else {
            match order {
                SortOrder::Ascend => abs_cmp,
                SortOrder::Descend => abs_cmp.reverse(),
            }
        }
    } else {
        Ordering::Equal
    };
    if primary != Ordering::Equal {
        return primary;
    }
    match order {
        SortOrder::Ascend => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
        SortOrder::Descend => b.partial_cmp(&a).unwrap_or(Ordering::Equal),
    }
}

fn sort_host_tensor(
    data: &[f64],
    shape: &[usize],
    dim: usize,
    order: SortOrder,
    comparison: SortComparison,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let expected_len = if shape.is_empty() {
        1usize
    } else {
        product_checked(shape)
            .ok_or_else(|| anyhow!("sort_dim: tensor size exceeds supported limits"))?
    };
    ensure!(
        expected_len == data.len(),
        "sort_dim: tensor data length {} does not match shape {:?}",
        data.len(),
        shape
    );

    if data.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let dim_len = dimension_length_zero_based(shape, dim);
    let mut sorted = data.to_vec();
    let mut indices = if dim_len == 0 {
        Vec::new()
    } else {
        vec![1.0; sorted.len()]
    };

    if dim_len <= 1 {
        return Ok((sorted, indices));
    }

    let stride_before = stride_before_for(shape, dim);
    let stride_after = stride_after_for(shape, dim);
    let mut buffer: Vec<(usize, f64)> = Vec::with_capacity(dim_len);

    for after in 0..stride_after {
        for before in 0..stride_before {
            buffer.clear();
            for k in 0..dim_len {
                let idx = before + k * stride_before + after * stride_before * dim_len;
                buffer.push((k, data[idx]));
            }
            buffer.sort_by(|a, b| compare_values_for_sort(a.1, b.1, order, comparison));
            for (pos, (original_index, value)) in buffer.iter().enumerate() {
                let target = before + pos * stride_before + after * stride_before * dim_len;
                sorted[target] = *value;
                indices[target] = (*original_index + 1) as f64;
            }
        }
    }

    Ok((sorted, indices))
}
const RNG_DEFAULT_SEED: u64 = 0x9e3779b97f4a7c15;
const MAX_SAFE_INTEGER: u64 = 1 << 53;
const RNG_MULTIPLIER: u64 = 6364136223846793005;
const RNG_INCREMENT: u64 = 1;

fn advance_rng_state(state: u64, mut delta: u64) -> u64 {
    let mut acc_mult = 1u64;
    let mut acc_plus = 0u64;
    let mut cur_mult = RNG_MULTIPLIER;
    let mut cur_plus = RNG_INCREMENT;

    while delta > 0 {
        if (delta & 1) != 0 {
            acc_mult = acc_mult.wrapping_mul(cur_mult);
            acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
        }
        cur_plus = cur_plus.wrapping_mul(cur_mult.wrapping_add(1));
        cur_mult = cur_mult.wrapping_mul(cur_mult);
        delta >>= 1;
    }

    acc_mult.wrapping_mul(state).wrapping_add(acc_plus)
}
fn seed_from_state(state: u64) -> u32 {
    let high = (state >> 32) as u32;
    let low = state as u32;
    let mut seed = low ^ high.rotate_left(13);
    if seed == 0 {
        seed = 0x9E37_79B9;
    }
    seed | 1
}

fn philox_keys_from_state(state: u64) -> (u32, u32) {
    let lo = state as u32;
    let hi = (state >> 32) as u32;
    let mut key0 = lo ^ hi.rotate_left(7);
    if key0 == 0 {
        key0 = 0x9E37_79B9;
    }
    let mut key1 = hi ^ lo.rotate_right(3);
    if key1 == 0 {
        key1 = 0xBB67_AE85;
    }
    (key0, key1)
}

fn rng_state() -> &'static Mutex<u64> {
    static RNG: OnceCell<Mutex<u64>> = OnceCell::new();
    RNG.get_or_init(|| Mutex::new(RNG_DEFAULT_SEED))
}
static NEXT_SUBMISSION_ID: AtomicU32 = AtomicU32::new(1);

impl WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    async fn map_readback_bytes(
        &self,
        staging: wgpu::Buffer,
        size_bytes: u64,
        context: &str,
    ) -> Result<Vec<u8>> {
        let size_usize = usize::try_from(size_bytes)
            .map_err(|_| anyhow!("{context}: readback size overflow"))?;
        let slice = staging.slice(..);
        let (tx, rx) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.device.poll(wgpu::Maintain::Wait);
        }
        let map_result = rx
            .await
            .map_err(|_| anyhow!("{context}: map_async callback dropped"))?;
        map_result.map_err(|e: wgpu::BufferAsyncError| anyhow!(e))?;
        let data = slice.get_mapped_range();
        let mut out = vec![0u8; size_usize];
        out.copy_from_slice(&data);
        drop(data);
        staging.unmap();
        Ok(out)
    }

    fn map_readback_bytes_sync(
        &self,
        staging: wgpu::Buffer,
        size_bytes: u64,
        context: &str,
    ) -> Result<Vec<u8>> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (staging, size_bytes);
            Err(anyhow!("{context}: readback requires async path on wasm"))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            block_on(self.map_readback_bytes(staging, size_bytes, context))
        }
    }
    const BUFFER_RESIDENCY_MAX_PER_KEY: usize = 8;
    const IMAGE_NORMALIZE_AUTOTUNE_VERSION: u8 = 1;
    const IMAGE_NORMALIZE_STREAM_COLD_CAP: u32 = 8;
    const IMAGE_NORMALIZE_TARGET_SAMPLES_PER_LANE: f64 = 256.0;
    const IMAGE_NORMALIZE_TARGET_LOOP_ITERS_PER_LANE: f64 = 16.0;

    fn precision_tag(&self) -> &'static str {
        match self.precision {
            NumericPrecision::F64 => "f64",
            NumericPrecision::F32 => "f32",
        }
    }

    fn record_kernel_launch_basic(
        &self,
        kernel: &'static str,
        shape: &[(&'static str, u64)],
        tuning: &[(&'static str, u64)],
    ) {
        self.telemetry
            .record_kernel_launch(kernel, Some(self.precision_tag()), shape, tuning);
    }

    fn record_matmul_kernel_launch(
        &self,
        m: usize,
        n: usize,
        k: usize,
        use_vec4: bool,
        chunked: bool,
    ) {
        let shape = [("m", m as u64), ("n", n as u64), ("k", k as u64)];
        let tuning = [
            ("vec4", if use_vec4 { 1 } else { 0 }),
            ("chunked", if chunked { 1 } else { 0 }),
        ];
        self.record_kernel_launch_basic("matmul", &shape, &tuning);
    }

    fn create_storage_buffer_checked_with_usage(
        &self,
        len: usize,
        label: &str,
        usage: BufferUsageClass,
    ) -> Result<Arc<wgpu::Buffer>> {
        // Centralised guard + warning for oversized allocations
        let size_bytes = (len as u64) * self.element_size as u64;
        if size_bytes > self.adapter_limits.max_buffer_size {
            return Err(gpu_per_buffer_limit_error(
                label,
                size_bytes,
                self.adapter_limits.max_buffer_size,
            ));
        }
        let (buffer, reused) = self.create_storage_buffer_for_usage(usage, len, label);
        if reused && std::env::var("RUNMAT_DEBUG_RESIDENCY").is_ok() {
            log::debug!(
                "[residency_debug] reused buffer label={} usage={:?} len={} ptr={:p}",
                label,
                usage,
                len,
                buffer.as_ref()
            );
        }
        if !reused && size_bytes >= (256u64 << 20) {
            log::warn!(
                "{}: large GPU allocation ({} bytes) len={} elems",
                label,
                size_bytes,
                len
            );
        }
        Ok(buffer)
    }

    fn create_storage_buffer_checked(&self, len: usize, label: &str) -> Result<Arc<wgpu::Buffer>> {
        self.create_storage_buffer_checked_with_usage(len, label, BufferUsageClass::Generic)
    }

    fn image_normalize_vector_width(&self) -> u32 {
        match self.precision {
            NumericPrecision::F64 => 2,
            NumericPrecision::F32 => 4,
        }
    }

    fn round_up_to_multiple(value: u32, mult: u32) -> u32 {
        if mult <= 1 {
            return value;
        }
        let remainder = value % mult;
        if remainder == 0 {
            value
        } else {
            value.saturating_add(mult - remainder).max(mult)
        }
    }

    fn select_image_normalize_tuning(&self, batches: u32, plane: u32) -> ImageNormalizeTuning {
        let batches = batches.max(1);
        let plane = plane.max(1);
        let mut lane =
            ((plane as f64) / Self::IMAGE_NORMALIZE_TARGET_SAMPLES_PER_LANE).ceil() as u32;
        lane = lane.max(32);
        let max_lane_dim = self.workgroup_config.max_x.max(32);
        lane = lane.min(max_lane_dim);
        lane = Self::round_up_to_multiple(lane, 32).max(32);
        let plane_per_lane = (plane as f64 / lane as f64).max(1.0);
        let mut values_per_thread =
            ((plane_per_lane / Self::IMAGE_NORMALIZE_TARGET_LOOP_ITERS_PER_LANE).ceil() as u32)
                .clamp(1, 8);
        if plane <= 512 {
            values_per_thread = values_per_thread.min(4);
        }
        let spatial_tile = if plane <= 1024 {
            1
        } else if plane <= 4096 {
            2
        } else {
            4
        };
        let mut batch_tile = if plane >= 8192 {
            batches.min(16)
        } else {
            batches.min(32)
        };
        if batches <= 4 {
            batch_tile = batches;
        }
        let tuning = ImageNormalizeTuning {
            batch_tile: batch_tile.max(1),
            values_per_thread,
            lane_count: lane,
            spatial_tile,
        };
        let sanitized = self
            .workgroup_config
            .sanitize_image_normalize_tuning(tuning, batches);
        debug!(
            "select_image_normalize_tuning batches={} plane={} raw={:?} sanitized={:?}",
            batches, plane, tuning, sanitized
        );
        sanitized
    }

    fn resolve_image_normalize_tuning(
        &self,
        batches: u32,
        plane: u32,
    ) -> (ImageNormalizeTuning, bool) {
        let key = ImageNormalizeKey::new(self.precision, batches, plane);
        if self.image_norm_autotune.is_enabled() {
            if let Some(tuning) = self.image_norm_autotune.get(&key) {
                let sanitized = self
                    .workgroup_config
                    .sanitize_image_normalize_tuning(tuning, batches);
                if sanitized != tuning {
                    debug!(
                        "image_normalize autotune sanitized cached key {:?}: {:?} -> {:?}",
                        key, tuning, sanitized
                    );
                    debug!(
                        "resolve_image_normalize_tuning returning cached {:?} for key {:?}",
                        sanitized, key
                    );
                    self.image_norm_autotune.insert(key, sanitized);
                } else {
                    debug!(
                        "image_normalize autotune reusing cached key {:?}: {:?}",
                        key, tuning
                    );
                }
                return (sanitized, true);
            }
            let tuning = self.select_image_normalize_tuning(batches, plane);
            debug!(
                "image_normalize autotune inserted key {:?}: {:?}",
                key, tuning
            );
            self.image_norm_autotune.insert(key, tuning);
            (tuning, false)
        } else {
            let tuning = self.select_image_normalize_tuning(batches, plane);
            debug!(
                "resolve_image_normalize_tuning returning fresh {:?} for key {:?}",
                tuning, key
            );
            (tuning, false)
        }
    }

    fn image_normalize_hot_stream_cap(&self, plane: u32, batches: u32) -> u32 {
        if batches == 0 {
            return 0;
        }
        let plane = plane.max(1);
        let bytes_per_batch = plane as u64 * self.element_size as u64;
        if bytes_per_batch == 0 {
            return batches;
        }
        let target_bytes = self
            .image_normalize_stream_target_bytes()
            .max(bytes_per_batch);
        let max_batches = target_bytes / bytes_per_batch;
        max_batches
            .clamp(1, batches as u64)
            .try_into()
            .unwrap_or(batches)
    }

    fn image_normalize_stream_target_bytes(&self) -> u64 {
        if let Ok(raw) = std::env::var("RUNMAT_IMAGE_NORMALIZE_STREAM_TARGET_BYTES") {
            if let Ok(parsed) = raw.parse::<u64>() {
                return parsed.max(1);
            }
        }
        let limit = self.adapter_limits.max_buffer_size;
        let default = 6u64 * 1024 * 1024 * 1024;
        default.min(limit).max((self.element_size as u64) * 4)
    }

    fn image_normalize_pipeline(
        &self,
        tuning: &ImageNormalizeTuning,
    ) -> Result<Arc<wgpu::ComputePipeline>> {
        if let Ok(cache) = self.image_norm_pipeline_cache.lock() {
            if let Some(existing) = cache.get(tuning) {
                return Ok(existing.clone());
            }
        }
        info!(
            "Compiling image_normalize pipeline tuning: batch_tile={} values/thread={} lane={} spatial={}",
            tuning.batch_tile, tuning.values_per_thread, tuning.lane_count, tuning.spatial_tile
        );
        let template = match self.precision {
            NumericPrecision::F64 => IMAGE_NORMALIZE_SHADER_F64,
            NumericPrecision::F32 => IMAGE_NORMALIZE_SHADER_F32,
        };
        let shader_src = template
            .replace("@BT@", &tuning.batch_tile.to_string())
            .replace("@VP@", &tuning.values_per_thread.to_string())
            .replace("@WG@", &tuning.lane_count.to_string())
            .replace("@ST@", &tuning.spatial_tile.to_string())
            .replace("@BV@", &self.image_normalize_vector_width().to_string());
        let module = self
            .device_ref()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("runmat-image-normalize-shader-dyn"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_src)),
            });
        let pipeline_layout = crate::backend::wgpu::cache::factory::create_pipeline_layout_single(
            self.device_ref(),
            "runmat-image-normalize-pipeline-dyn",
            &self.pipelines.image_normalize.layout,
        );
        let pipeline = crate::backend::wgpu::cache::factory::create_compute_pipeline(
            self.device_ref(),
            "runmat-image-normalize-pipeline-dyn",
            &pipeline_layout,
            &module,
        );
        let arc = Arc::new(pipeline);
        if let Ok(mut cache) = self.image_norm_pipeline_cache.lock() {
            cache.insert(*tuning, arc.clone());
        }
        Ok(arc)
    }
    pub fn device_id(&self) -> u32 {
        self.cache_device_id
    }

    pub(crate) fn device_ref(&self) -> &wgpu::Device {
        self.device.as_ref()
    }
    pub(crate) fn queue_ref(&self) -> &wgpu::Queue {
        self.queue.as_ref()
    }

    fn warmup_from_disk(&self) {
        if std::env::var("RUNMAT_DISABLE_PIPELINE_WARMUP").is_ok() {
            return;
        }
        crate::backend::wgpu::warmup::warmup_from_disk(
            &self.device,
            self.pipeline_cache_dir.as_deref(),
            self.precision,
            |bytes, tag, wg| self.compute_pipeline_hash_bytes(bytes, tag, wg),
            |key, pl, module, label, src, tag, wg| {
                self.get_or_create_pipeline(key, pl, module, label, src, tag, wg)
            },
            |pipeline| {
                crate::backend::wgpu::warmup::noop_after_create(&self.device, &self.queue, pipeline)
            },
        );
    }

    fn cached_bind_group_layout<F>(&self, key: &str, build: F) -> Arc<wgpu::BindGroupLayout>
    where
        F: FnOnce(&wgpu::Device) -> wgpu::BindGroupLayout,
    {
        if let Ok(cache) = self.bind_group_layout_cache.lock() {
            if let Some(layout) = cache.get(key).cloned() {
                return layout;
            }
        }
        let layout = Arc::new(build(self.device_ref()));
        let ptr = layout.as_ref() as *const wgpu::BindGroupLayout as usize;
        if let Ok(mut tags) = self.bind_group_layout_tags.lock() {
            tags.entry(ptr).or_insert_with(|| key.to_string());
        }
        if let Ok(mut cache) = self.bind_group_layout_cache.lock() {
            cache.insert(key.to_string(), layout.clone());
        }
        layout
    }

    fn cached_bind_group_layout_for_tag(&self, tag: &str) -> Option<Arc<wgpu::BindGroupLayout>> {
        if let Ok(cache) = self.bind_group_layout_cache.lock() {
            if let Some(layout) = cache.get(tag).cloned() {
                return Some(layout);
            }
        }
        let layout =
            crate::backend::wgpu::bindings::build_bgl_for_layout_tag(self.device_ref(), tag)?;
        let layout = Arc::new(layout);
        let ptr = layout.as_ref() as *const wgpu::BindGroupLayout as usize;
        if let Ok(mut tags) = self.bind_group_layout_tags.lock() {
            tags.entry(ptr).or_insert_with(|| tag.to_string());
        }
        if let Ok(mut cache) = self.bind_group_layout_cache.lock() {
            cache.insert(tag.to_string(), layout.clone());
        }
        Some(layout)
    }

    fn cached_fusion_bind_group_layout(&self, inputs_len: usize) -> Arc<wgpu::BindGroupLayout> {
        let key = format!("runmat-fusion-layout-{}", inputs_len);
        self.cached_bind_group_layout(&key, |device| {
            crate::backend::wgpu::bindings::build_fusion_bgl(device, inputs_len)
        })
    }

    pub fn try_compile_kernel(&self, label: &str, wgsl_src: &str) -> Result<()> {
        crate::backend::wgpu::debug::try_compile_kernel(&self.device, label, wgsl_src);
        Ok(())
    }

    pub fn probe_kernel_with_buffers(&self, label: &str, wgsl_src: &str, wg: u32) -> Result<()> {
        crate::backend::wgpu::debug::probe_kernel_with_buffers(
            &self.device,
            &self.queue,
            label,
            wgsl_src,
            wg,
        );
        Ok(())
    }

    async fn image_normalize_cpu_fallback(
        &self,
        input: &GpuTensorHandle,
        desc: &runmat_accelerate_api::ImageNormalizeDescriptor,
    ) -> Result<GpuTensorHandle> {
        let mut host = <Self as AccelProvider>::download(self, input).await?;
        ensure!(
            host.shape.len() == 3,
            "image_normalize: expected 3-D tensor, got {:?}",
            host.shape
        );
        ensure!(
            host.shape[0] == desc.batch
                && host.shape[1] == desc.height
                && host.shape[2] == desc.width,
            "image_normalize: descriptor dims {:?} do not match tensor shape {:?}",
            (desc.batch, desc.height, desc.width),
            host.shape
        );

        let batch = desc.batch;
        let height = desc.height;
        let width = desc.width;
        let plane = height * width;

        if plane == 0 {
            let view = HostTensorView {
                data: &host.data,
                shape: &host.shape,
            };
            return self.upload(&view);
        }

        let stride_h = batch;
        let stride_w = batch * height;

        let gain = desc.gain.unwrap_or(1.0);
        let bias = desc.bias.unwrap_or(0.0);
        let gamma = desc.gamma;

        for b in 0..batch {
            let mut sum = 0.0;
            for w in 0..width {
                let base_w = w * stride_w;
                for h in 0..height {
                    let idx = b + h * stride_h + base_w;
                    sum += host.data[idx];
                }
            }
            let mean = sum / plane as f64;

            let mut sq_sum = 0.0;
            for w in 0..width {
                let base_w = w * stride_w;
                for h in 0..height {
                    let idx = b + h * stride_h + base_w;
                    let diff = host.data[idx] - mean;
                    sq_sum += diff * diff;
                }
            }
            let variance = sq_sum / plane as f64;
            let sigma = (variance + desc.epsilon).sqrt();
            let inv_sigma = if sigma > 0.0 { 1.0 / sigma } else { 0.0 };

            for w in 0..width {
                let base_w = w * stride_w;
                for h in 0..height {
                    let idx = b + h * stride_h + base_w;
                    let mut value = (host.data[idx] - mean) * inv_sigma;
                    if desc.gain.is_some() {
                        value *= gain;
                    }
                    if desc.bias.is_some() {
                        value += bias;
                    }
                    value = value.max(0.0);
                    if let Some(gamma) = gamma {
                        value = value.powf(gamma);
                    }
                    host.data[idx] = value;
                }
            }
        }

        let view = HostTensorView {
            data: &host.data,
            shape: &host.shape,
        };
        self.upload(&view)
    }

    /// Get or create a compute pipeline from cache using a caller-provided hash key.
    #[allow(clippy::too_many_arguments)]
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
            self.metrics.inc_hit();
            return p;
        }
        self.metrics.inc_miss();
        // Persist WGSL + meta for warmup on next run
        self.persist_pipeline_meta(
            hash_key,
            label,
            persist_layout_tag,
            persist_workgroup_size,
            persist_wgsl_src,
        );
        let p = Arc::new(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(pipeline_layout),
                    module,
                    entry_point: "main",
                }),
        );
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
        cache_key::compute_pipeline_hash_bytes(shader_bytes, layout_tag, workgroup_size)
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
            cache_persist::persist_pipeline_meta(
                dir,
                hash_key,
                label,
                layout_tag,
                workgroup_size,
                self.precision,
                wgsl_src,
            );
        }
    }

    fn buffer_residency_pool_limit() -> usize {
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

    fn parse_buffer_residency_max_poolable_bytes(
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

    fn buffer_residency_max_poolable_bytes(adapter_max_buffer_size: u64) -> u64 {
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

    fn register_existing_buffer(
        &self,
        buffer: Arc<wgpu::Buffer>,
        shape: Vec<usize>,
        len: usize,
    ) -> GpuTensorHandle {
        self.register_existing_buffer_with_usage(buffer, shape, len, BufferUsageClass::Generic)
    }

    fn register_existing_buffer_with_storage(
        &self,
        buffer: Arc<wgpu::Buffer>,
        shape: Vec<usize>,
        len: usize,
        storage: GpuTensorStorage,
    ) -> GpuTensorHandle {
        self.register_existing_buffer_with_usage_and_storage(
            buffer,
            shape,
            len,
            storage,
            BufferUsageClass::Generic,
        )
    }

    fn register_existing_buffer_with_usage(
        &self,
        buffer: Arc<wgpu::Buffer>,
        shape: Vec<usize>,
        len: usize,
        usage: BufferUsageClass,
    ) -> GpuTensorHandle {
        self.register_existing_buffer_with_usage_and_storage(
            buffer,
            shape,
            len,
            GpuTensorStorage::Real,
            usage,
        )
    }

    fn register_existing_buffer_with_usage_and_storage(
        &self,
        buffer: Arc<wgpu::Buffer>,
        shape: Vec<usize>,
        len: usize,
        storage: GpuTensorStorage,
        usage: BufferUsageClass,
    ) -> GpuTensorHandle {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let entry = BufferEntry {
            buffer,
            len,
            shape: shape.clone(),
            storage: storage.clone(),
            precision: self.precision,
            usage,
            last_submission_id: None,
        };
        self.buffers
            .lock()
            .expect("buffer mutex poisoned")
            .insert(id, entry);
        log::trace!("wgpu register id={} len={} shape={:?}", id, len, &shape);
        let handle = GpuTensorHandle {
            shape,
            device_id: self.runtime_device_id,
            buffer_id: id,
        };
        runmat_accelerate_api::set_handle_logical(&handle, false);
        runmat_accelerate_api::set_handle_storage(&handle, storage);
        runmat_accelerate_api::clear_handle_transpose(&handle);
        handle
    }

    fn remember_matmul_sources(
        &self,
        product: &GpuTensorHandle,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) {
        if lhs.device_id != self.runtime_device_id || rhs.device_id != self.runtime_device_id {
            return;
        }
        log::debug!(
            "remember_matmul_sources: product={} lhs={} rhs={} active_fusion={:?}",
            product.buffer_id,
            lhs.buffer_id,
            rhs.buffer_id,
            active_fusion()
        );
        self.kernel_resources
            .remember_matmul_sources(product, lhs, rhs);
    }

    fn mark_buffer_usage(&self, handle: &GpuTensorHandle, usage: BufferUsageClass) {
        if let Ok(mut guard) = self.buffers.lock() {
            if let Some(entry) = guard.get_mut(&handle.buffer_id) {
                entry.usage = usage;
            }
        }
    }

    fn record_buffer_submission(&self, buffer_id: u64, submission_id: u32) {
        if let Ok(mut guard) = self.buffers.lock() {
            if let Some(entry) = guard.get_mut(&buffer_id) {
                entry.last_submission_id = Some(submission_id);
            }
        }
    }

    fn qr_factor_device(
        &self,
        matrix: &GpuTensorHandle,
        rows: usize,
        cols: usize,
        reuse_q: Option<&GpuTensorHandle>,
        label: &str,
        retain_r_inv: bool,
    ) -> Result<(GpuTensorHandle, GpuTensorHandle, Option<GpuTensorHandle>)> {
        ensure!(rows >= cols, "qr: rows must be >= cols for device path");
        ensure!(
            cols > 0,
            "qr: zero-column input not supported for device path"
        );

        let gram_handle = self.syrk_exec(matrix)?;

        let gram_entry = self.get_entry(&gram_handle)?;
        let gram_len = cols * cols;
        ensure!(
            gram_entry.len == gram_len,
            "qr: gram len mismatch (expected {}, got {})",
            gram_len,
            gram_entry.len
        );
        let gram_bytes = (gram_len as u64) * (self.element_size as u64);
        let gram_scratch = self.kernel_resources.scratch_storage_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::ScratchBufferKind::QrGram,
            gram_bytes,
            "runmat-qr-gram-scratch",
        );
        if gram_bytes > 0 {
            let gram_copy_label = format!("{label}-gram-copy");
            let mut encoder =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(gram_copy_label.as_str()),
                    });
            encoder.copy_buffer_to_buffer(
                gram_entry.buffer.as_ref(),
                0,
                gram_scratch.as_ref(),
                0,
                gram_bytes,
            );
            self.submit(encoder);
        }

        let len_out = cols * cols;
        let r_bytes = (len_out as u64) * (self.element_size as u64);
        let r_buffer = self.kernel_resources.scratch_storage_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::ScratchBufferKind::QrR,
            r_bytes,
            "runmat-qr-r-scratch",
        );
        let r_inv_buffer = self.kernel_resources.scratch_storage_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::ScratchBufferKind::QrRInv,
            r_bytes,
            "runmat-qr-rinv-scratch",
        );

        let params = QrPowerIterParams {
            cols: cols as u32,
            stride: cols as u32,
            _pad0: [0, 0],
        };
        let params_buffer = self.kernel_resources.uniform_buffer(
            self.device_ref(),
            UniformBufferKey::QrPowerIterParams,
            std::mem::size_of::<QrPowerIterParams>() as u64,
            "runmat-qr-power-params",
        );
        self.queue
            .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));

        let layout = &self.pipelines.qr_power_iter.layout;
        let bind_entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gram_scratch.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: r_buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: r_inv_buffer.as_ref().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ];
        let bind_group = self
            .bind_group_cache
            .get_or_create(layout, &bind_entries, || {
                Arc::new(
                    self.device_ref()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-qr-power-bind"),
                            layout,
                            entries: &bind_entries,
                        }),
                )
            });
        crate::backend::wgpu::dispatch::qr_power_iter::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.qr_power_iter.pipeline,
            bind_group.as_ref(),
        );

        let _ = self.free(&gram_handle);

        let r_shape = vec![cols, cols];
        let r_handle = self.register_existing_buffer_with_usage(
            r_buffer.clone(),
            r_shape.clone(),
            len_out,
            BufferUsageClass::FusionOut,
        );
        self.mark_buffer_usage(&r_handle, BufferUsageClass::FusionOut);

        let r_inv_handle = self.register_existing_buffer_with_usage(
            r_inv_buffer.clone(),
            r_shape,
            len_out,
            BufferUsageClass::FusionOut,
        );
        self.mark_buffer_usage(&r_inv_handle, BufferUsageClass::FusionOut);

        let q_temp =
            self.matmul_exec_with_usage(matrix, &r_inv_handle, BufferUsageClass::FusionOut)?;

        let q_temp_entry = self.get_entry(&q_temp)?;
        let q_result = if let Some(target) = reuse_q {
            let target_entry = self.get_entry(target)?;
            if Arc::strong_count(&target_entry.buffer) <= 2 && target_entry.len == q_temp_entry.len
            {
                let bytes = (target_entry.len as u64) * self.element_size as u64;
                if bytes > 0 {
                    let copy_label = format!("{label}-reuse-copy");
                    let mut encoder =
                        self.device_ref()
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some(copy_label.as_str()),
                            });
                    encoder.copy_buffer_to_buffer(
                        q_temp_entry.buffer.as_ref(),
                        0,
                        target_entry.buffer.as_ref(),
                        0,
                        bytes,
                    );
                    self.submit(encoder);
                }
                let _ = self.free(&q_temp);
                self.mark_buffer_usage(target, BufferUsageClass::FusionOut);
                target.clone()
            } else {
                q_temp
            }
        } else {
            q_temp
        };

        let r_inv_result = if retain_r_inv {
            Some(r_inv_handle)
        } else {
            let _ = self.free(&r_inv_handle);
            None
        };

        Ok((q_result, r_handle, r_inv_result))
    }

    async fn qr_power_iter_host(
        &self,
        product: &GpuTensorHandle,
        options: &ProviderQrOptions,
    ) -> Result<Option<ProviderQrPowerIterResult>> {
        let host_product = <Self as AccelProvider>::download(self, product).await?;
        let tensor =
            Tensor::new(host_product.data.clone(), host_product.shape.clone()).map_err(|e| {
                anyhow!("qr_power_iter: failed to construct host tensor for fallback: {e}")
            })?;
        let host_result = self.qr_host_result(tensor, options).await?;
        let _ = self.free(product);
        Ok(Some(ProviderQrPowerIterResult {
            q: host_result.q,
            r: host_result.r,
            perm_matrix: host_result.perm_matrix,
            perm_vector: host_result.perm_vector,
        }))
    }

    fn try_qr_device(
        &self,
        matrix: &GpuTensorHandle,
        options: &ProviderQrOptions,
    ) -> Result<Option<ProviderQrResult>> {
        if !options.economy {
            return Ok(None);
        }
        if options.pivot != ProviderQrPivot::Matrix {
            return Ok(None);
        }
        if self.precision() != ProviderPrecision::F32 {
            return Ok(None);
        }
        let entry = self.get_entry(matrix)?;
        if entry.shape.len() != 2 {
            return Ok(None);
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        if rows < cols || cols == 0 {
            return Ok(None);
        }
        if cols > QR_DEVICE_MAX_COLS {
            return Ok(None);
        }
        if rows
            .checked_mul(cols)
            .map(|v| v > QR_DEVICE_MAX_ELEMS)
            .unwrap_or(true)
        {
            return Ok(None);
        }

        let (q_handle, r_handle, _) =
            self.qr_factor_device(matrix, rows, cols, None, "runmat-qr-direct", false)?;

        let mut perm_matrix = vec![0.0f64; cols * cols];
        for i in 0..cols {
            perm_matrix[i + i * cols] = 1.0;
        }
        let perm_vector: Vec<f64> = (1..=cols).map(|v| v as f64).collect();

        let perm_matrix_shape = [cols, cols];
        let perm_matrix_handle = self.upload(&HostTensorView {
            data: &perm_matrix,
            shape: &perm_matrix_shape,
        })?;
        let perm_vector_shape = vec![cols, 1];
        let perm_vector_handle = self.upload(&HostTensorView {
            data: &perm_vector,
            shape: &perm_vector_shape,
        })?;

        Ok(Some(ProviderQrResult {
            q: q_handle,
            r: r_handle,
            perm_matrix: perm_matrix_handle,
            perm_vector: perm_vector_handle,
        }))
    }

    async fn qr_host_result(
        &self,
        tensor: Tensor,
        options: &ProviderQrOptions,
    ) -> Result<ProviderQrResult> {
        let mut args = Vec::new();
        if options.economy {
            args.push(Value::Num(0.0));
        }
        if matches!(options.pivot, ProviderQrPivot::Vector) {
            args.push(Value::from("vector"));
        }
        let eval = runmat_runtime::builtins::math::linalg::factor::qr::evaluate(
            Value::Tensor(tensor),
            &args,
        )
        .await
        .map_err(|err| runtime_flow_to_anyhow("qr", err))?;

        let q_tensor = host_tensor_from_value("qr", eval.q())?;
        let r_tensor = host_tensor_from_value("qr", eval.r())?;
        let perm_matrix_tensor = host_tensor_from_value("qr", eval.permutation_matrix())?;
        let perm_vector_tensor = host_tensor_from_value("qr", eval.permutation_vector())?;

        let q = self.upload(&HostTensorView {
            data: &q_tensor.data,
            shape: &q_tensor.shape,
        })?;
        let r = self.upload(&HostTensorView {
            data: &r_tensor.data,
            shape: &r_tensor.shape,
        })?;
        let perm_matrix = self.upload(&HostTensorView {
            data: &perm_matrix_tensor.data,
            shape: &perm_matrix_tensor.shape,
        })?;
        let perm_vector = self.upload(&HostTensorView {
            data: &perm_vector_tensor.data,
            shape: &perm_vector_tensor.shape,
        })?;

        Ok(ProviderQrResult {
            q,
            r,
            perm_matrix,
            perm_vector,
        })
    }

    async fn trim_polynomial_handle(
        &self,
        handle: GpuTensorHandle,
        orientation: PolynomialOrientation,
    ) -> Result<GpuTensorHandle> {
        let host = <Self as AccelProvider>::download(self, &handle).await?;
        let trimmed = trim_leading_zeros_real(&host.data);
        if trimmed.len() == host.data.len() {
            return Ok(handle);
        }
        let shape_vec = shape_for_orientation(orientation, trimmed.len());
        let new_handle = if trimmed.is_empty() {
            self.register_existing_buffer(
                self.create_storage_buffer(0, "runmat-polyder-trim-empty"),
                shape_vec,
                0,
            )
        } else {
            match self.precision {
                NumericPrecision::F64 => {
                    let buffer = Arc::new(self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("runmat-polyder-trim-f64"),
                            contents: cast_slice(trimmed.as_slice()),
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC,
                        },
                    ));
                    self.register_existing_buffer(buffer, shape_vec, trimmed.len())
                }
                NumericPrecision::F32 => {
                    let data_f32: Vec<f32> = trimmed.iter().map(|v| *v as f32).collect();
                    let buffer = Arc::new(self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("runmat-polyder-trim-f32"),
                            contents: cast_slice(&data_f32),
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC,
                        },
                    ));
                    self.register_existing_buffer(buffer, shape_vec, trimmed.len())
                }
            }
        };
        self.free(&handle).ok();
        Ok(new_handle)
    }

    fn create_storage_buffer_for_usage(
        &self,
        usage: BufferUsageClass,
        len: usize,
        label: &str,
    ) -> (Arc<wgpu::Buffer>, bool) {
        self.buffer_residency
            .acquire(self.device_ref(), usage, len, self.element_size, label)
    }

    fn create_storage_buffer(&self, len: usize, label: &str) -> Arc<wgpu::Buffer> {
        self.create_storage_buffer_for_usage(BufferUsageClass::Generic, len, label)
            .0
    }

    fn uniform_buffer<T: Pod>(&self, data: &T, label: &str) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn prepare_matmul_pipeline(&self) {
        let mut enc = self
            .device_ref()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-matmul-warmup"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-matmul-warmup-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.matmul.pipeline);
        }
        self.submit(enc);

        let enc = self
            .device_ref()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-matmul-flush-gap"),
            });
        self.submit(enc);
    }

    fn prepare_matmul_vec4_pipeline(&self) {
        let mut enc = self
            .device_ref()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-matmul-vec4-warmup"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-matmul-vec4-warmup-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.matmul_vec4.pipeline);
        }
        self.submit(enc);

        let enc = self
            .device_ref()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-matmul-vec4-flush-gap"),
            });
        self.submit(enc);
    }

    fn submit(&self, encoder: wgpu::CommandEncoder) -> u32 {
        let submission_id = NEXT_SUBMISSION_ID.fetch_add(1, AtomicOrdering::Relaxed);
        let _span = info_span!(
            "gpu.dispatch",
            label = "runmat-wgpu-submit",
            submission_id = submission_id
        )
        .entered();
        log::trace!("wgpu submit {}: begin", submission_id);
        let work = encoder.finish();
        let submission_index = self.queue.submit(Some(work));
        log::trace!(
            "wgpu submit {}: submitted, polling (index={:?})",
            submission_id,
            submission_index
        );
        // On Web/WASM, blocking `Maintain::Wait` can starve the worker event loop, preventing
        // plot presentation and other cooperative tasks from running. Prefer non-blocking polls.
        #[cfg(target_arch = "wasm32")]
        let poll_start = Instant::now();
        #[cfg(target_arch = "wasm32")]
        self.device.poll(wgpu::Maintain::Poll);
        #[cfg(target_arch = "wasm32")]
        log::trace!("wgpu submit {}: poll complete", submission_id);
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.queue.on_submitted_work_done(move || {
                log::trace!(
                    "wgpu submit {}: on_submitted_work_done callback fired",
                    submission_id
                );
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            log::trace!(
                "wgpu submit {}: on_submitted_work_done unavailable on wasm",
                submission_id
            );
        }
        #[cfg(target_arch = "wasm32")]
        {
            let elapsed = poll_start.elapsed();
            if elapsed > Duration::from_millis(250) {
                log::warn!(
                    "wgpu submit {}: device.poll took {}ms on wasm target",
                    submission_id,
                    elapsed.as_millis()
                );
            }
        }
        submission_id
    }
    fn get_entry(&self, handle: &GpuTensorHandle) -> Result<BufferEntry> {
        if handle.device_id != self.runtime_device_id {
            return Err(anyhow!(
                "handle device mismatch: expected {}, got {}",
                self.runtime_device_id,
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
                storage: entry.storage.clone(),
                precision: entry.precision,
                usage: entry.usage,
                last_submission_id: entry.last_submission_id,
            })
            .ok_or_else(|| anyhow!("buffer not found: {}", handle.buffer_id))
    }
}
// Internal exec methods for WgpuProvider
impl WgpuProvider {
    pub(crate) fn bandwidth_exec(&self, matrix: &GpuTensorHandle) -> Result<ProviderBandwidth> {
        let entry = self.get_entry(matrix)?;
        let (rows, cols) =
            ensure_bandwidth_shape(&entry.shape).map_err(|e| anyhow!("bandwidth: {e}"))?;
        if rows == 0 || cols == 0 {
            return Ok(ProviderBandwidth { lower: 0, upper: 0 });
        }
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("bandwidth: matrix dimensions too large"))?;
        if total == 0 {
            return Ok(ProviderBandwidth { lower: 0, upper: 0 });
        }
        if total > entry.len {
            return Err(anyhow!(
                "bandwidth: shape/product mismatch ({} vs {})",
                total,
                entry.len
            ));
        }
        if total as u64 > u32::MAX as u64 {
            return Err(anyhow!("bandwidth: matrix exceeds GPU limits"));
        }

        let pipeline = &self.pipelines.bandwidth;
        let output_init = [0u32, 0u32];
        let output_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("runmat-bandwidth-output"),
                contents: cast_slice(&output_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let params = BandwidthParams {
            rows: rows as u32,
            cols: cols as u32,
            len: total as u32,
            _pad: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-bandwidth-params");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runmat-bandwidth-bind-group"),
            layout: &pipeline.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: entry.buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(total as u32, 256);
        crate::backend::wgpu::dispatch::elementwise::run(
            self.device_ref(),
            self.queue_ref(),
            &pipeline.pipeline,
            &bind_group,
            groups,
        );

        let staging_size = (std::mem::size_of::<u32>() * 2) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-bandwidth-staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-bandwidth-copy"),
            });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, staging_size);
        self.submit(encoder);

        let bytes = self.map_readback_bytes_sync(staging, staging_size, "bandwidth")?;
        let words: &[u32] = cast_slice(&bytes);
        let lower = words.first().copied().unwrap_or(0);
        let upper = words.get(1).copied().unwrap_or(0);

        Ok(ProviderBandwidth { lower, upper })
    }
    pub(crate) fn syrk_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if entry.shape.len() != 2 {
            return Err(anyhow!("syrk: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let out_shape = vec![cols, cols];
        let len = cols
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("syrk: output size overflow"))?;

        let out_bytes = (len as u64) * (self.element_size as u64);
        let out_buffer = self.kernel_resources.scratch_storage_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::ScratchBufferKind::SyrkOut,
            out_bytes,
            "runmat-syrk-out-scratch",
        );
        if len == 0 {
            return Ok(self.register_existing_buffer_with_usage(
                out_buffer,
                out_shape,
                0,
                BufferUsageClass::SyrkOut,
            ));
        }

        let rows_u32 =
            u32::try_from(rows).map_err(|_| anyhow!("syrk: row count exceeds GPU limits"))?;
        let cols_u32 =
            u32::try_from(cols).map_err(|_| anyhow!("syrk: column count exceeds GPU limits"))?;
        let lda_u32 = rows_u32;
        let ldc_u32 = cols_u32;

        let tile = crate::backend::wgpu::config::effective_matmul_tile();
        let groups_x = crate::backend::wgpu::dispatch::common::dispatch_size_dim(cols_u32, tile);
        let groups_y = groups_x;

        const SYRK_ROW_CHUNK: usize = 32768;
        let mut offset = 0usize;
        let mut first_chunk = true;
        while offset < rows {
            let remaining = rows - offset;
            let chunk_rows = remaining.min(SYRK_ROW_CHUNK.max(1));
            let chunk_rows_u32 = u32::try_from(chunk_rows)
                .map_err(|_| anyhow!("syrk: chunk rows exceed GPU limits"))?;
            let row_offset_u32 = u32::try_from(offset)
                .map_err(|_| anyhow!("syrk: row offset exceeds GPU limits"))?;

            let mut flags = SYRK_FLAG_FILL_BOTH;
            if !first_chunk {
                flags |= SYRK_FLAG_ACCUMULATE;
            }

            let params = SyrkParams {
                rows_total: rows_u32,
                cols: cols_u32,
                lda: lda_u32,
                ldc: ldc_u32,
                row_offset: row_offset_u32,
                chunk_rows: chunk_rows_u32,
                flags,
                offset_out: 0,
            };
            let params_buffer = self.kernel_resources.uniform_buffer(
                self.device_ref(),
                UniformBufferKey::SyrkParams,
                std::mem::size_of::<crate::backend::wgpu::params::SyrkParams>() as u64,
                "runmat-syrk-params",
            );
            self.queue
                .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
            let bind_entries = [
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
            ];
            let layout = &self.pipelines.syrk.layout;
            let bind_group = self
                .bind_group_cache
                .get_or_create(layout, &bind_entries, || {
                    Arc::new(
                        self.device_ref()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-syrk-bind"),
                                layout,
                                entries: &bind_entries,
                            }),
                    )
                });

            crate::backend::wgpu::dispatch::syrk::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.syrk.pipeline,
                bind_group.as_ref(),
                groups_x,
                groups_y,
            );

            offset += chunk_rows;
            first_chunk = false;
        }

        Ok(self.register_existing_buffer_with_usage(
            out_buffer,
            out_shape,
            len,
            BufferUsageClass::SyrkOut,
        ))
    }

    pub(crate) fn matmul_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.matmul_exec_with_usage(a, b, BufferUsageClass::MatmulOut)
    }
    fn matmul_exec_with_usage(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        out_usage: BufferUsageClass,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape.len() != 2 || entry_b.shape.len() != 2 {
            return Err(anyhow!("matmul: only 2D tensors supported"));
        }

        let view_a = build_matrix_operand_view(a, &entry_a).map_err(|e| anyhow!("matmul: {e}"))?;
        let view_b = build_matrix_operand_view(b, &entry_b).map_err(|e| anyhow!("matmul: {e}"))?;

        if view_a.cols != view_b.rows {
            return Err(anyhow!("matmul: inner dimensions must match"));
        }

        let m = view_a.rows;
        let n = view_b.cols;
        let k = view_a.cols;

        let debug_matmul = std::env::var("RUNMAT_DEBUG_MATMUL").is_ok();
        if debug_matmul {
            log::debug!(
                "[matmul_debug] ptr_a={:p} ptr_b={:p}",
                entry_a.buffer.as_ref(),
                entry_b.buffer.as_ref()
            );
            log::debug!(
                "[matmul_debug] m={} n={} k={} lda={} ldb={} transpose_a={} transpose_b={}",
                m,
                n,
                k,
                view_a.lda,
                view_b.lda,
                view_a.transpose,
                view_b.transpose
            );
        }

        let out_shape = vec![m, n];
        let len = m * n;
        if len == 0 {
            let (out_buffer, _) =
                self.create_storage_buffer_for_usage(out_usage, 0, "runmat-matmul-out");
            return Ok(
                self.register_existing_buffer_with_usage(out_buffer, out_shape, 0, out_usage)
            );
        }

        let m_u32 = u32::try_from(m).map_err(|_| anyhow!("matmul: m exceeds GPU limits"))?;
        let n_u32 = u32::try_from(n).map_err(|_| anyhow!("matmul: n exceeds GPU limits"))?;
        let k_u32 = u32::try_from(k).map_err(|_| anyhow!("matmul: k exceeds GPU limits"))?;

        const K_CHUNK: usize = 8192;
        const K_CHUNK_SWITCH: usize = 65536; // only chunk for very large k to avoid regressions

        let can_vec4 = self.precision == NumericPrecision::F32
            && !view_a.transpose
            && !view_b.transpose
            && m % 4 == 0
            && m >= 4
            && n > 0;
        let disable_vec4 = std::env::var("RUNMAT_DISABLE_VEC4")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "True"))
            .unwrap_or(false);
        let use_vec4 = can_vec4 && k < K_CHUNK_SWITCH && !disable_vec4;
        let enable_chunk = !view_a.transpose && !view_b.transpose && k >= K_CHUNK_SWITCH;
        if debug_matmul {
            log::debug!(
                "[matmul_debug] can_vec4={} use_vec4={} enable_chunk={} usage={:?}",
                can_vec4,
                use_vec4,
                enable_chunk,
                out_usage
            );
        }

        let start = Instant::now();

        if enable_chunk {
            self.prepare_matmul_pipeline();
            self.device_ref().poll(wgpu::Maintain::Poll);
            let lda_u32 = view_a.lda;
            let ldb_u32 = view_b.lda;
            // Accumulator handle across chunks
            let mut acc: Option<GpuTensorHandle> = None;
            let mut k_off: usize = 0;
            let partial_storage = self.create_storage_buffer_checked_with_usage(
                len,
                "runmat-matmul-partial",
                BufferUsageClass::MatmulPartial,
            )?;
            while k_off < k {
                let k_sub = std::cmp::min(K_CHUNK, k - k_off);
                // Create partial output buffer and bind group
                let partial_buffer = partial_storage.clone();
                let offset_a_elems = k_off
                    .checked_mul(view_a.rows)
                    .ok_or_else(|| anyhow!("matmul: offset overflow"))?;
                let offset_a_u32 = u32::try_from(offset_a_elems)
                    .map_err(|_| anyhow!("matmul: A offset exceeds GPU limits"))?;
                let offset_b_u32 = u32::try_from(k_off)
                    .map_err(|_| anyhow!("matmul: B offset exceeds GPU limits"))?;
                let params = crate::backend::wgpu::params::MatmulParams {
                    m: m_u32,
                    n: n_u32,
                    k: k_sub as u32,
                    lda: lda_u32,
                    ldb: ldb_u32,
                    ldc: m_u32,
                    offset_a: offset_a_u32,
                    offset_b: offset_b_u32,
                    offset_out: 0,
                    flags: 0,
                };
                let params_buffer = self.kernel_resources.uniform_buffer(
                    self.device_ref(),
                    UniformBufferKey::MatmulParams,
                    std::mem::size_of::<crate::backend::wgpu::params::MatmulParams>() as u64,
                    "runmat-matmul-params",
                );
                self.queue
                    .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
                let bind_entries = [
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
                        resource: partial_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ];
                let layout = &self.pipelines.matmul.layout;
                let bg =
                    self.bind_group_cache
                        .get_or_create(layout, &bind_entries, || {
                            Arc::new(self.device_ref().create_bind_group(
                                &wgpu::BindGroupDescriptor {
                                    label: Some("runmat-matmul-bind"),
                                    layout,
                                    entries: &bind_entries,
                                },
                            ))
                        });
                let tile = crate::backend::wgpu::config::effective_matmul_tile();
                let groups_x =
                    crate::backend::wgpu::dispatch::common::dispatch_size_dim(n_u32, tile);
                let groups_y =
                    crate::backend::wgpu::dispatch::common::dispatch_size_dim(m_u32, tile);
                crate::backend::wgpu::dispatch::matmul::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.matmul.pipeline,
                    bg.as_ref(),
                    groups_x,
                    groups_y,
                );
                // Wrap partial buffer into handle
                let partial = self.register_existing_buffer_with_usage(
                    partial_buffer,
                    out_shape.clone(),
                    len,
                    BufferUsageClass::MatmulPartial,
                );
                acc = match acc {
                    None => Some(partial),
                    Some(prev) => {
                        let sum = self.binary_op_exec(
                            crate::backend::wgpu::types::BinaryOpCode::Add,
                            &prev,
                            &partial,
                        )?;
                        self.free(&prev).ok();
                        self.free(&partial).ok();
                        Some(sum)
                    }
                };
                k_off += k_sub;
            }
            let handle = acc.expect("matmul chunking produced no output");
            self.remember_matmul_sources(&handle, a, b);
            self.mark_buffer_usage(&handle, out_usage);
            self.telemetry.record_matmul_duration(start.elapsed());
            self.record_matmul_kernel_launch(m, n, k, use_vec4, true);
            return Ok(handle);
        }

        // Default single-dispatch path
        let out_buffer =
            self.create_storage_buffer_checked_with_usage(len, "runmat-matmul-out", out_usage)?;
        if use_vec4 {
            self.prepare_matmul_vec4_pipeline();
        } else {
            self.prepare_matmul_pipeline();
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        let mut flags = 0u32;
        if view_a.transpose {
            flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_A;
        }
        if view_b.transpose {
            flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_B;
        }
        let params = crate::backend::wgpu::params::MatmulParams {
            m: m_u32,
            n: n_u32,
            k: k_u32,
            lda: view_a.lda,
            ldb: view_b.lda,
            ldc: m_u32,
            offset_a: 0,
            offset_b: 0,
            offset_out: 0,
            flags,
        };
        let params_buffer = self.kernel_resources.uniform_buffer(
            self.device_ref(),
            UniformBufferKey::MatmulParams,
            std::mem::size_of::<crate::backend::wgpu::params::MatmulParams>() as u64,
            "runmat-matmul-params",
        );
        self.queue
            .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
        let layout = if use_vec4 {
            &self.pipelines.matmul_vec4.layout
        } else {
            &self.pipelines.matmul.layout
        };
        let pipeline = if use_vec4 {
            &self.pipelines.matmul_vec4.pipeline
        } else {
            &self.pipelines.matmul.pipeline
        };
        let bind_entries = [
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
        ];
        let bg = if out_usage == BufferUsageClass::MatmulOut {
            Arc::new(
                self.device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-matmul-bind"),
                        layout,
                        entries: &bind_entries,
                    }),
            )
        } else {
            self.bind_group_cache
                .get_or_create(layout, &bind_entries, || {
                    Arc::new(
                        self.device_ref()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-matmul-bind"),
                                layout,
                                entries: &bind_entries,
                            }),
                    )
                })
        };
        let tile = crate::backend::wgpu::config::effective_matmul_tile();
        let groups_x = crate::backend::wgpu::dispatch::common::dispatch_size_dim(n_u32, tile);
        let groups_y = if use_vec4 {
            let rows_vec = (m as u32) / 4;
            crate::backend::wgpu::dispatch::common::dispatch_size_dim(rows_vec, tile)
        } else {
            crate::backend::wgpu::dispatch::common::dispatch_size_dim(m as u32, tile)
        };
        crate::backend::wgpu::dispatch::matmul::run(
            self.device_ref(),
            self.queue_ref(),
            pipeline,
            bg.as_ref(),
            groups_x,
            groups_y,
        );
        let out_ptr = out_buffer.as_ref() as *const wgpu::Buffer;
        let handle =
            self.register_existing_buffer_with_usage(out_buffer, out_shape, len, out_usage);
        if debug_matmul {
            log::debug!("[matmul_debug] out_ptr={:p} len={}", out_ptr, len);
        }
        self.remember_matmul_sources(&handle, a, b);
        self.telemetry.record_matmul_duration(start.elapsed());
        self.record_matmul_kernel_launch(m, n, k, use_vec4, false);
        Ok(handle)
    }
    pub(crate) fn pagefun_exec(&self, request: &PagefunRequest) -> Result<GpuTensorHandle> {
        match request.op {
            PagefunOp::Mtimes => self.pagefun_mtimes_exec(request),
        }
    }
    fn pagefun_mtimes_exec(&self, request: &PagefunRequest) -> Result<GpuTensorHandle> {
        ensure!(
            request.inputs.len() == 2,
            "pagefun: @mtimes expects exactly two inputs"
        );
        ensure!(
            request.input_page_dims.len() == request.inputs.len(),
            "pagefun: input metadata mismatch"
        );

        let lhs = &request.inputs[0];
        let rhs = &request.inputs[1];
        let entry_a = self.get_entry(lhs)?;
        let entry_b = self.get_entry(rhs)?;

        let view_a = build_matrix_operand_view(lhs, &entry_a)
            .map_err(|e| anyhow!("pagefun @mtimes: {e}"))?;
        let view_b = build_matrix_operand_view(rhs, &entry_b)
            .map_err(|e| anyhow!("pagefun @mtimes: {e}"))?;

        let canonical_a = canonical_matrix_shape(&entry_a.shape);
        let canonical_b = canonical_matrix_shape(&entry_b.shape);
        ensure!(
            canonical_a.len() >= 2 && canonical_b.len() >= 2,
            "pagefun: @mtimes operands must be at least 2-D"
        );

        let rows = view_a.rows;
        let k_a = view_a.cols;
        let k_b = view_b.rows;
        let cols = view_b.cols;
        ensure!(
            k_a == k_b,
            "pagefun: inner matrix dimensions must agree ({} vs {})",
            k_a,
            k_b
        );

        let rank = request.page_dims.len();
        let lhs_dims = pad_dims(request.input_page_dims[0].clone(), rank);
        let rhs_dims = pad_dims(request.input_page_dims[1].clone(), rank);
        let lhs_strides = compute_page_strides(&lhs_dims);
        let rhs_strides = compute_page_strides(&rhs_dims);

        let lhs_page_size = rows
            .checked_mul(k_a)
            .ok_or_else(|| anyhow!("pagefun: lhs page size overflow"))?;
        let rhs_page_size = k_b
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("pagefun: rhs page size overflow"))?;
        let out_page_size = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("pagefun: output page size overflow"))?;

        let page_volume = if rank == 0 {
            1
        } else {
            product_checked(&request.page_dims)
                .ok_or_else(|| anyhow!("pagefun: page dimensions overflow"))?
        };

        let total_len = out_page_size
            .checked_mul(page_volume)
            .ok_or_else(|| anyhow!("pagefun: output size overflow"))?;
        let out_buffer =
            self.create_storage_buffer_checked(total_len, "runmat-pagefun-mtimes-out")?;

        if total_len == 0 {
            return Ok(self.register_existing_buffer(
                out_buffer,
                request.output_shape.clone(),
                total_len,
            ));
        }

        let m_u32 = u32::try_from(rows)
            .map_err(|_| anyhow!("pagefun: matrix row count exceeds GPU limits"))?;
        let n_u32 = u32::try_from(cols)
            .map_err(|_| anyhow!("pagefun: matrix column count exceeds GPU limits"))?;
        let k_u32 = u32::try_from(k_a)
            .map_err(|_| anyhow!("pagefun: shared dimension exceeds GPU limits"))?;

        let lda = view_a.lda;
        let ldb = view_b.lda;
        let ldc = m_u32;

        let tile = crate::backend::wgpu::config::effective_matmul_tile();
        let groups_x = crate::backend::wgpu::dispatch::common::dispatch_size_dim(n_u32, tile);
        let groups_y = crate::backend::wgpu::dispatch::common::dispatch_size_dim(m_u32, tile);

        self.prepare_matmul_pipeline();
        self.device_ref().poll(wgpu::Maintain::Poll);

        let start = Instant::now();

        let mut multi_index = vec![0usize; rank];
        for page_idx in 0..page_volume {
            if rank > 0 {
                decode_multi_index(page_idx, &request.page_dims, &mut multi_index);
            }

            let lhs_linear = broadcast_linear_index(&lhs_dims, &lhs_strides, &multi_index);
            let rhs_linear = broadcast_linear_index(&rhs_dims, &rhs_strides, &multi_index);

            let lhs_offset_elements = lhs_linear
                .checked_mul(lhs_page_size)
                .ok_or_else(|| anyhow!("pagefun: lhs offset overflow"))?;
            let rhs_offset_elements = rhs_linear
                .checked_mul(rhs_page_size)
                .ok_or_else(|| anyhow!("pagefun: rhs offset overflow"))?;
            let out_offset_elements = page_idx
                .checked_mul(out_page_size)
                .ok_or_else(|| anyhow!("pagefun: output offset overflow"))?;

            let lhs_end = lhs_offset_elements
                .checked_add(lhs_page_size)
                .ok_or_else(|| anyhow!("pagefun: lhs offset overflow"))?;
            let rhs_end = rhs_offset_elements
                .checked_add(rhs_page_size)
                .ok_or_else(|| anyhow!("pagefun: rhs offset overflow"))?;
            let out_end = out_offset_elements
                .checked_add(out_page_size)
                .ok_or_else(|| anyhow!("pagefun: output offset overflow"))?;

            ensure!(
                lhs_end <= entry_a.len,
                "pagefun: lhs page out of bounds (page {})",
                page_idx
            );
            ensure!(
                rhs_end <= entry_b.len,
                "pagefun: rhs page out of bounds (page {})",
                page_idx
            );
            ensure!(
                out_end <= total_len,
                "pagefun: output page out of bounds (page {})",
                page_idx
            );

            let offset_a_u32 = u32::try_from(lhs_offset_elements)
                .map_err(|_| anyhow!("pagefun: lhs offset exceeds GPU limits"))?;
            let offset_b_u32 = u32::try_from(rhs_offset_elements)
                .map_err(|_| anyhow!("pagefun: rhs offset exceeds GPU limits"))?;
            let offset_out_u32 = u32::try_from(out_offset_elements)
                .map_err(|_| anyhow!("pagefun: output offset exceeds GPU limits"))?;

            let mut flags = 0u32;
            if view_a.transpose {
                flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_A;
            }
            if view_b.transpose {
                flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_B;
            }

            let params = crate::backend::wgpu::params::MatmulParams {
                m: m_u32,
                n: n_u32,
                k: k_u32,
                lda,
                ldb,
                ldc,
                offset_a: offset_a_u32,
                offset_b: offset_b_u32,
                offset_out: offset_out_u32,
                flags,
            };

            let params_buffer = self.uniform_buffer(&params, "runmat-pagefun-mtimes-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-pagefun-mtimes-bind"),
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

            crate::backend::wgpu::dispatch::matmul::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.matmul.pipeline,
                &bind_group,
                groups_x,
                groups_y,
            );
        }

        self.telemetry.record_matmul_duration(start.elapsed());

        let handle =
            self.register_existing_buffer(out_buffer, request.output_shape.clone(), total_len);

        Ok(handle)
    }
    async fn centered_gram_exec_kernel(
        &self,
        matrix: &GpuTensorHandle,
        matrix_entry: &BufferEntry,
        means: &GpuTensorHandle,
        rows: usize,
        cols: usize,
        denom: f64,
    ) -> Result<GpuTensorHandle> {
        let rows_f64 = rows as f64;
        let means_entry = self.get_entry(means)?;
        let mut means_used = means.clone();
        let mut casted_means = false;
        if means_entry.precision != matrix_entry.precision {
            means_used = self
                .cast_tensor_precision(means, matrix_entry.precision)
                .await?;
            casted_means = true;
        }

        // Compute X^T * X using the SYRK pipeline (no explicit transpose required).
        let xtx = self.syrk_exec(matrix)?;

        // Form n * μ μᵀ without materialising a centered copy of X.
        let means_scaled = self.scalar_mul(&means_used, rows_f64)?;
        let means_col = self
            .reshape(&means_scaled, &[cols, 1])
            .map_err(|e| anyhow!("centered_gram: reshape means col failed: {e}"))?;
        let means_row_scaled = self
            .reshape(&means_scaled, &[1, cols])
            .map_err(|e| anyhow!("centered_gram: reshape means row failed: {e}"))?;

        let outer_scaled = self.matmul_exec_with_usage(
            &means_col,
            &means_row_scaled,
            BufferUsageClass::FusionOut,
        )?;
        let outer = self.scalar_mul(&outer_scaled, 1.0 / rows_f64)?;

        let _ = self.free(&means_col);
        let _ = self.free(&means_row_scaled);
        let _ = self.free(&outer_scaled);

        let centered =
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &xtx, &outer)?;

        let _ = self.free(&xtx);
        let _ = self.free(&outer);
        let _ = self.free(&means_scaled);

        let handle = self.scalar_mul(&centered, 1.0 / denom)?;
        let _ = self.free(&centered);

        self.mark_buffer_usage(&handle, BufferUsageClass::FusionOut);

        if std::env::var("RUNMAT_DEBUG_CENTERED_GRAM").is_ok() {
            if let Err(err) = self
                .debug_centered_gram(
                    matrix,
                    matrix_entry.precision,
                    &means_used,
                    &handle,
                    rows,
                    cols,
                    denom,
                )
                .await
            {
                log::warn!("centered_gram debug instrumentation failed: {err}");
            }
        }

        if casted_means {
            let _ = self.free(&means_used);
        }

        Ok(handle)
    }
    #[allow(clippy::too_many_arguments)]
    async fn debug_centered_gram(
        &self,
        matrix: &GpuTensorHandle,
        precision: NumericPrecision,
        means: &GpuTensorHandle,
        output: &GpuTensorHandle,
        rows: usize,
        cols: usize,
        denom: f64,
    ) -> Result<()> {
        let matrix_host = <Self as AccelProvider>::download(self, matrix).await?;
        let means_gpu = <Self as AccelProvider>::download(self, means).await?;
        let output_gpu = <Self as AccelProvider>::download(self, output).await?;
        if matrix_host.data.len() != rows * cols {
            return Err(anyhow!(
                "centered_gram debug: matrix download length mismatch ({} vs {})",
                matrix_host.data.len(),
                rows * cols
            ));
        }

        let mut mean_ref = vec![0.0f64; cols];
        for (col, mean_slot) in mean_ref.iter_mut().enumerate().take(cols) {
            let mut sum = 0.0f64;
            let base = col * rows;
            for row in 0..rows {
                sum += matrix_host.data[base + row];
            }
            *mean_slot = sum / (rows as f64);
        }

        let mut max_mean_diff = 0.0f64;
        for (mean, gpu_mean) in mean_ref.iter().zip(means_gpu.data.iter()) {
            let diff = (*mean - *gpu_mean).abs();
            if diff > max_mean_diff {
                max_mean_diff = diff;
            }
        }

        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..cols).collect();
        indices.shuffle(&mut rng);
        indices.truncate(cols.min(32));
        indices.sort_unstable();

        let mut max_abs_err = 0.0f64;
        let mut max_abs_idx = (0usize, 0usize);
        let mut max_rel_err = 0.0f64;
        let mut max_rel_idx = (0usize, 0usize);
        let mut max_diag_neg = 0.0f64;
        let mut max_diag_idx = 0usize;

        for &j in &indices {
            for &i in &indices {
                let mut sum = 0.0f64;
                let base_i = i * rows;
                let base_j = j * rows;
                for row in 0..rows {
                    let centered_i = matrix_host.data[base_i + row] - mean_ref[i];
                    let centered_j = matrix_host.data[base_j + row] - mean_ref[j];
                    sum += centered_i * centered_j;
                }
                sum /= denom;

                let gpu_val = output_gpu.data[i + j * cols];
                let abs_err = (gpu_val - sum).abs();
                if i == j && std::env::var("RUNMAT_DEBUG_CENTERED_GRAM_TRACE").is_ok() {
                    log::info!(
                        "centered_gram diag sample col={} gpu={:.6e} ref={:.6e}",
                        i,
                        gpu_val,
                        sum
                    );
                }
                if abs_err > max_abs_err {
                    max_abs_err = abs_err;
                    max_abs_idx = (i, j);
                }
                if sum.abs() > 0.0 {
                    let rel_err = abs_err / sum.abs();
                    if rel_err > max_rel_err {
                        max_rel_err = rel_err;
                        max_rel_idx = (i, j);
                    }
                }
                if i == j && gpu_val < 0.0 {
                    let neg = gpu_val.abs();
                    if neg > max_diag_neg {
                        max_diag_neg = neg;
                        max_diag_idx = i;
                    }
                }
            }
        }

        let sample_preview: Vec<usize> = indices.iter().copied().take(16).collect();
        let rows_out = output_gpu.shape.first().copied().unwrap_or(cols);
        let diag_len = cols.min(rows_out);
        let mut trace = 0.0f64;
        for d in 0..diag_len {
            let idx = d + d * rows_out;
            if let Some(val) = output_gpu.data.get(idx) {
                trace += *val;
            }
        }
        log::info!(
            "centered_gram debug [{}]: rows={} cols={} sample_cols={} trace={:.6e} max_mean_diff={:.3e} max_abs_err={:.3e} at ({}, {}) max_rel_err={:.3e} at ({}, {}) max_diag_neg={:.3e} at ({}) samples={:?}",
            match precision {
                NumericPrecision::F32 => "f32",
                NumericPrecision::F64 => "f64",
            },
            rows,
            cols,
            indices.len(),
            trace,
            max_mean_diff,
            max_abs_err,
            max_abs_idx.0,
            max_abs_idx.1,
            max_rel_err,
            max_rel_idx.0,
            max_rel_idx.1,
            max_diag_neg,
            max_diag_idx,
            sample_preview
        );

        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    async fn debug_qr_power_iter(
        &self,
        product: &GpuTensorHandle,
        product_entry: &BufferEntry,
        pre_product_max: Option<f64>,
        pre_q_max: Option<f64>,
        q_result: &GpuTensorHandle,
        r_handle: &GpuTensorHandle,
        r_inv_handle: &GpuTensorHandle,
        gram_host: Option<&HostTensorOwned>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if rows == 0 || cols == 0 {
            return Ok(());
        }

        let product_host = <Self as AccelProvider>::download(self, product).await?;
        let q_gpu_host = <Self as AccelProvider>::download(self, q_result).await?;
        let r_gpu_host = <Self as AccelProvider>::download(self, r_handle).await?;
        let r_inv_gpu_host = <Self as AccelProvider>::download(self, r_inv_handle).await?;
        let max_r_inv_abs = r_inv_gpu_host
            .data
            .iter()
            .fold(0.0f64, |acc, v| acc.max(v.abs()));

        if product_host.data.len() != rows * cols
            || q_gpu_host.data.len() != rows * cols
            || r_gpu_host.data.len() != cols * cols
            || r_inv_gpu_host.data.len() != cols * cols
        {
            return Err(anyhow!(
                "qr_power_iter debug: length mismatch (rows={}, cols={})",
                rows,
                cols
            ));
        }

        let gram_cow: Cow<'_, HostTensorOwned> = if let Some(host) = gram_host {
            Cow::Borrowed(host)
        } else {
            let product_t_tmp = self.transpose_exec(product)?;
            let gram_tmp =
                self.matmul_exec_with_usage(&product_t_tmp, product, BufferUsageClass::FusionOut)?;
            let _ = self.free(&product_t_tmp);
            let owned = <Self as AccelProvider>::download(self, &gram_tmp).await?;
            let _ = self.free(&gram_tmp);
            Cow::Owned(owned)
        };
        let gram_view: &HostTensorOwned = gram_cow.as_ref();

        if gram_view.data.len() != cols * cols {
            return Err(anyhow!(
                "qr_power_iter debug: Gram data mismatch (cols={})",
                cols
            ));
        }

        let mut min_r_diag = f64::MAX;
        let mut max_r_diag = f64::MIN;
        for i in 0..cols {
            let diag = r_gpu_host.data[i + i * cols];
            min_r_diag = min_r_diag.min(diag);
            max_r_diag = max_r_diag.max(diag);
        }

        let mut min_gram_diag = f64::MAX;
        let mut max_gram_diag = f64::MIN;
        for i in 0..cols {
            let diag = gram_view.data[i + i * cols];
            min_gram_diag = min_gram_diag.min(diag);
            max_gram_diag = max_gram_diag.max(diag);
        }

        let mut q_ref = vec![0.0f64; rows * cols];
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = 0.0f64;
                for k in 0..cols {
                    sum += product_host.data[row + k * rows] * r_inv_gpu_host.data[k + col * cols];
                }
                q_ref[row + col * rows] = sum;
            }
        }

        let mut max_q_diff = 0.0f64;
        let mut max_q_diff_idx = 0usize;
        let mut max_q_abs = 0.0f64;
        let mut min_q_abs = f64::MAX;
        let mut non_zero_q = false;
        for (idx, (val, ref_val)) in q_gpu_host
            .data
            .iter()
            .zip(q_ref.iter())
            .enumerate()
            .take(rows * cols)
        {
            let diff = (*val - *ref_val).abs();
            if diff > max_q_diff {
                max_q_diff = diff;
                max_q_diff_idx = idx;
            }
            let abs_val = val.abs();
            if abs_val > max_q_abs {
                max_q_abs = abs_val;
            }
            if abs_val < min_q_abs {
                min_q_abs = abs_val;
            }
            if abs_val > 1.0e-12 {
                non_zero_q = true;
            }
        }
        if min_q_abs == f64::MAX {
            min_q_abs = 0.0;
        }

        let mut max_qtq_diag = 0.0f64;
        let mut max_qtq_diag_idx = 0usize;
        let mut max_qtq_off = 0.0f64;
        let mut max_qtq_off_idx = (0usize, 0usize);
        let mut min_diag_val = f64::MAX;
        let mut max_diag_val = f64::MIN;
        for j in 0..cols {
            for i in 0..cols {
                let mut sum = 0.0f64;
                for row in 0..rows {
                    sum += q_gpu_host.data[row + i * rows] * q_gpu_host.data[row + j * rows];
                }
                if i == j {
                    let err = (sum - 1.0).abs();
                    if err > max_qtq_diag {
                        max_qtq_diag = err;
                        max_qtq_diag_idx = i;
                    }
                    if sum < min_diag_val {
                        min_diag_val = sum;
                    }
                    if sum > max_diag_val {
                        max_diag_val = sum;
                    }
                } else {
                    let err = sum.abs();
                    if err > max_qtq_off {
                        max_qtq_off = err;
                        max_qtq_off_idx = (i, j);
                    }
                }
            }
        }

        let mut max_residual = 0.0f64;
        let mut max_residual_idx = (0usize, 0usize);
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = 0.0f64;
                for k in 0..cols {
                    sum += q_gpu_host.data[row + k * rows] * r_gpu_host.data[k + col * cols];
                }
                let diff = (sum - product_host.data[row + col * rows]).abs();
                if diff > max_residual {
                    max_residual = diff;
                    max_residual_idx = (row, col);
                }
            }
        }

        let mut gq_gpu = vec![0.0f64; rows * cols];
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = 0.0f64;
                for l in 0..cols {
                    sum += gram_view.data[l + col * cols] * q_gpu_host.data[row + l * rows];
                }
                gq_gpu[row + col * rows] = sum;
            }
        }
        let mut gq_ref = vec![0.0f64; rows * cols];
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = 0.0f64;
                for l in 0..cols {
                    sum += gram_view.data[l + col * cols] * q_ref[row + l * rows];
                }
                gq_ref[row + col * rows] = sum;
            }
        }

        let mut gpu_topk = 0.0f64;
        let mut ref_topk = 0.0f64;
        for col in 0..cols {
            let mut diag_gpu = 0.0f64;
            let mut diag_ref = 0.0f64;
            for row in 0..rows {
                diag_gpu += q_gpu_host.data[row + col * rows] * gq_gpu[row + col * rows];
                diag_ref += q_ref[row + col * rows] * gq_ref[row + col * rows];
            }
            gpu_topk += diag_gpu;
            ref_topk += diag_ref;
        }
        let topk_diff = gpu_topk - ref_topk;
        let max_product_abs = product_host
            .data
            .iter()
            .fold(0.0f64, |acc, v| acc.max(v.abs()));

        log::info!(
            "qr_power_iter debug: rows={} cols={} max_q_diff={:.3e} at idx={} max_q_abs={:.3e} min_q_abs={:.3e} non_zero_q={} max_qtq_diag_err={:.3e} at col={} max_qtq_off={:.3e} at ({}, {}) min_diag={:.3e} max_diag={:.3e} max_residual={:.3e} at ({}, {}) max_product_abs_pre={:?} max_product_abs={:.3e} max_q_abs_pre={:?} max_r_inv_abs={:.3e} min_r_diag={:.3e} max_r_diag={:.3e} min_gram_diag={:.3e} max_gram_diag={:.3e} gpu_topk={:.6e} ref_topk={:.6e} diff={:.3e}",
            rows,
            cols,
            max_q_diff,
            max_q_diff_idx,
            max_q_abs,
            min_q_abs,
            non_zero_q,
            max_qtq_diag,
            max_qtq_diag_idx,
            max_qtq_off,
            max_qtq_off_idx.0,
            max_qtq_off_idx.1,
            min_diag_val,
            max_diag_val,
            max_residual,
            max_residual_idx.0,
            max_residual_idx.1,
            pre_product_max,
            max_product_abs,
            pre_q_max,
            max_r_inv_abs,
            min_r_diag,
            max_r_diag,
            min_gram_diag,
            max_gram_diag,
            gpu_topk,
            ref_topk,
            topk_diff
        );

        if !non_zero_q || max_product_abs <= 1.0e-12 {
            let active = active_fusion();
            let plan = active_group_plan_clone();
            log::warn!(
                "qr_power_iter zero-data alert: product={} len={} non_zero_q={} max_product_abs_pre={:?} max_product_abs={:.3e} max_q_abs_pre={:?} active={:?} plan_inputs={:?} stack_pattern={:?}",
                product.buffer_id,
                product_entry.len,
                non_zero_q,
                pre_product_max,
                max_product_abs,
                pre_q_max,
                active,
                plan.as_ref().map(|p| p.inputs.clone()),
                plan.as_ref().map(|p| p.stack_pattern.clone())
            );
        }

        Ok(())
    }
    pub(crate) async fn covariance_exec(
        &self,
        matrix: &GpuTensorHandle,
        options: &CovarianceOptions,
    ) -> Result<GpuTensorHandle> {
        if options.rows != CovRows::All {
            return Err(anyhow!(
                "covariance: rows option {:?} not supported by WGPU provider",
                options.rows
            ));
        }
        if options.has_weight_vector {
            return Err(anyhow!(
                "covariance: weight vectors are not supported by WGPU provider"
            ));
        }

        let entry = self.get_entry(matrix)?;
        let shape = entry.shape.clone();
        let (rows, cols) = match shape.len() {
            0 => (1usize, 1usize),
            1 => (shape[0], 1usize),
            2 => (shape[0], shape[1]),
            _ => {
                return Err(anyhow!(
                    "covariance: inputs must be 2-D matrices or vectors (got shape {:?})",
                    shape
                ))
            }
        };

        if cols == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-cov-empty");
            return Ok(self.register_existing_buffer(out_buffer, vec![0, 0], 0));
        }

        if rows == 0 {
            return self.fill_exec(&[cols, cols], f64::NAN);
        }

        let denom = match options.normalization {
            CovNormalization::Unbiased => (rows as f64) - 1.0,
            CovNormalization::Biased => rows as f64,
        };

        if denom <= 0.0 {
            return self.fill_exec(&[cols, cols], f64::NAN);
        }

        let means = self.reduce_dim_sum_mean_exec(
            matrix,
            0,
            crate::backend::wgpu::types::DimReduceOp::Mean,
        )?;
        let result = self
            .centered_gram_exec_kernel(matrix, &entry, &means, rows, cols, denom)
            .await;
        let _ = self.free(&means);
        result
    }
    pub(crate) async fn corrcoef_exec(
        &self,
        matrix: &GpuTensorHandle,
        options: &CorrcoefOptions,
    ) -> Result<GpuTensorHandle> {
        if options.rows != CorrcoefRows::All {
            return Err(anyhow!(
                "corrcoef: rows option {:?} not supported by WGPU provider",
                options.rows
            ));
        }

        let entry = self.get_entry(matrix)?;
        let shape = entry.shape.clone();
        let (rows, cols) = match shape.len() {
            0 => (1usize, 1usize),
            1 => (shape[0], 1usize),
            2 => (shape[0], shape[1]),
            _ => {
                return Err(anyhow!(
                    "corrcoef: inputs must be 2-D matrices or vectors (got shape {:?})",
                    shape
                ))
            }
        };

        if cols == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-corrcoef-empty");
            return Ok(self.register_existing_buffer(out_buffer, vec![0, 0], 0));
        }

        if rows == 0 {
            return self.fill_exec(&[cols, cols], f64::NAN);
        }

        let denom = match options.normalization {
            CorrcoefNormalization::Unbiased => (rows as f64) - 1.0,
            CorrcoefNormalization::Biased => rows as f64,
        };

        if denom <= 0.0 {
            return self.fill_exec(&[cols, cols], f64::NAN);
        }

        let means = self.reduce_dim_sum_mean_exec(
            matrix,
            0,
            crate::backend::wgpu::types::DimReduceOp::Mean,
        )?;
        let ones = self.fill_exec(&[rows, 1], 1.0)?;
        let means_full = self.matmul_exec(&ones, &means)?;
        let centered = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Sub,
            matrix,
            &means_full,
        )?;
        let squared = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &centered,
            &centered,
        )?;
        let centered_t = self.transpose_exec(&centered)?;
        let covariance = self.matmul_exec(&centered_t, &centered)?;
        let inv_denom = 1.0 / denom;
        let inv_cov = self.fill_exec(&covariance.shape, inv_denom)?;
        let covariance_scaled = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &covariance,
            &inv_cov,
        )?;

        let variance_sum = self.reduce_dim_sum_mean_exec(
            &squared,
            0,
            crate::backend::wgpu::types::DimReduceOp::Sum,
        )?;
        let inv_var = self.fill_exec(&variance_sum.shape, inv_denom)?;
        let variance = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &variance_sum,
            &inv_var,
        )?;

        // Clamp tiny negative variances to zero to stabilise sqrt
        let mut host_variance = <Self as AccelProvider>::download(self, &variance).await?;
        for value in host_variance.data.iter_mut() {
            if *value < 0.0 && *value > -1.0e-12 {
                *value = 0.0;
            }
        }
        let view = HostTensorView {
            data: &host_variance.data,
            shape: &host_variance.shape,
        };
        let variance_adjusted = self.upload(&view)?;
        self.free(&variance)?;

        let std = self.unary_op_exec(
            crate::backend::wgpu::types::UnaryOpCode::Sqrt,
            &variance_adjusted,
        )?;
        let std_t = self.transpose_exec(&std)?;
        let std_outer = self.matmul_exec(&std_t, &std)?;
        let correlation = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Div,
            &covariance_scaled,
            &std_outer,
        )?;

        // Free temporaries
        let _ = self.free(&means);
        let _ = self.free(&ones);
        let _ = self.free(&means_full);
        let _ = self.free(&centered);
        let _ = self.free(&centered_t);
        let _ = self.free(&covariance);
        let _ = self.free(&inv_cov);
        let _ = self.free(&covariance_scaled);
        let _ = self.free(&squared);
        let _ = self.free(&variance_sum);
        let _ = self.free(&inv_var);
        let _ = self.free(&variance_adjusted);
        let _ = self.free(&std);
        let _ = self.free(&std_t);
        let _ = self.free(&std_outer);

        Ok(correlation)
    }
    async fn cast_tensor_precision(
        &self,
        tensor: &GpuTensorHandle,
        target: NumericPrecision,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(tensor)?;
        if entry.precision == target {
            return Ok(tensor.clone());
        }

        let mut host = <Self as AccelProvider>::download(self, tensor).await?;
        if matches!(target, NumericPrecision::F32) {
            for value in host.data.iter_mut() {
                *value = (*value as f32) as f64;
            }
        }

        let view = HostTensorView {
            data: host.data.as_slice(),
            shape: host.shape.as_slice(),
        };
        self.upload(&view)
    }

    pub(crate) fn dot_exec(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        let entry_lhs = self.get_entry(lhs)?;
        let entry_rhs = self.get_entry(rhs)?;
        ensure!(
            entry_lhs.shape == entry_rhs.shape,
            "dot: shape mismatch between inputs"
        );
        if entry_lhs.shape.is_empty() {
            return self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, lhs, rhs);
        }
        if entry_lhs.shape.len() != 2 {
            return Err(anyhow!(
                "dot: only 2D tensors are currently supported by the WGPU provider"
            ));
        }

        let shape = entry_lhs.shape.clone();
        let default_dim = shape
            .iter()
            .position(|&extent| extent != 1)
            .map(|idx| idx + 1)
            .unwrap_or(1);
        let target_dim = dim.unwrap_or(default_dim);
        let dim_index = target_dim.saturating_sub(1);

        if dim_index >= shape.len() {
            return self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, lhs, rhs);
        }
        if dim_index > 1 {
            return Err(anyhow!(
                "dot: unsupported dimension {} for WGPU provider",
                target_dim
            ));
        }

        let product =
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, lhs, rhs)?;

        let reduce = self.reduce_dim_sum_mean_exec(
            &product,
            dim_index,
            crate::backend::wgpu::types::DimReduceOp::Sum,
        );
        match reduce {
            Ok(handle) => {
                let _ = self.free(&product);
                Ok(handle)
            }
            Err(err) => {
                let _ = self.free(&product);
                Err(err)
            }
        }
    }

    pub(crate) fn cross_exec(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        let entry_lhs = self.get_entry(lhs)?;
        let entry_rhs = self.get_entry(rhs)?;
        ensure!(
            entry_lhs.shape == entry_rhs.shape,
            "cross: shape mismatch between inputs"
        );

        let shape = if entry_lhs.shape.is_empty() {
            vec![1, 1]
        } else {
            entry_lhs.shape.clone()
        };
        let rank = shape.len();
        let target_dim = match dim {
            Some(target_dim) => {
                ensure!(
                    target_dim >= 1 && target_dim <= rank,
                    "cross: dimension {} exceeds the number of array dimensions ({})",
                    target_dim,
                    rank
                );
                ensure!(
                    shape[target_dim - 1] == 3,
                    "cross: dimension {} must have length 3",
                    target_dim
                );
                target_dim
            }
            None => shape
                .iter()
                .position(|&extent| extent == 3)
                .map(|idx| idx + 1)
                .ok_or_else(|| anyhow!("cross: inputs must have a dimension of length 3"))?,
        };
        let dim_index = target_dim - 1;
        let total_len = entry_lhs.len;
        if total_len == 0 {
            return self.zeros_exec(&shape);
        }

        let stride_before = product_checked(&shape[..dim_index])
            .ok_or_else(|| anyhow!("cross: internal dimension overflow"))?;
        let stride_after = product_checked(&shape[dim_index + 1..])
            .ok_or_else(|| anyhow!("cross: internal dimension overflow"))?;
        let slice_stride = stride_before
            .checked_mul(3)
            .ok_or_else(|| anyhow!("cross: internal dimension overflow"))?;
        let slice_count = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cross: internal dimension overflow"))?;

        let mut comp1 = Vec::with_capacity(slice_count);
        let mut comp2 = Vec::with_capacity(slice_count);
        let mut comp3 = Vec::with_capacity(slice_count);
        for after in 0..stride_after {
            let slice_base = after
                .checked_mul(slice_stride)
                .ok_or_else(|| anyhow!("cross: internal index overflow"))?;
            for before in 0..stride_before {
                let idx1 = slice_base + before;
                let idx2 = idx1 + stride_before;
                let idx3 = idx2 + stride_before;
                comp1.push(
                    u32::try_from(idx1).map_err(|_| anyhow!("cross: GPU index exceeds limits"))?,
                );
                comp2.push(
                    u32::try_from(idx2).map_err(|_| anyhow!("cross: GPU index exceeds limits"))?,
                );
                comp3.push(
                    u32::try_from(idx3).map_err(|_| anyhow!("cross: GPU index exceeds limits"))?,
                );
            }
        }

        let mut reduced_shape = shape.clone();
        reduced_shape[dim_index] = 1;

        // Track every intermediate handle outside the computation closure so that
        // handles allocated before a failing `?` are still freed on error.
        let mut to_free: Vec<GpuTensorHandle> = Vec::with_capacity(15);

        let compute_result: Result<GpuTensorHandle> = (|| {
            let a1 = self.gather_linear_exec(lhs, &comp1, &reduced_shape)?;
            to_free.push(a1.clone());
            let a2 = self.gather_linear_exec(lhs, &comp2, &reduced_shape)?;
            to_free.push(a2.clone());
            let a3 = self.gather_linear_exec(lhs, &comp3, &reduced_shape)?;
            to_free.push(a3.clone());
            let b1 = self.gather_linear_exec(rhs, &comp1, &reduced_shape)?;
            to_free.push(b1.clone());
            let b2 = self.gather_linear_exec(rhs, &comp2, &reduced_shape)?;
            to_free.push(b2.clone());
            let b3 = self.gather_linear_exec(rhs, &comp3, &reduced_shape)?;
            to_free.push(b3.clone());

            let a2b3 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a2, &b3)?;
            to_free.push(a2b3.clone());
            let a3b2 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a3, &b2)?;
            to_free.push(a3b2.clone());
            let c1 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &a2b3, &a3b2)?;
            to_free.push(c1.clone());

            let a3b1 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a3, &b1)?;
            to_free.push(a3b1.clone());
            let a1b3 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a1, &b3)?;
            to_free.push(a1b3.clone());
            let c2 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &a3b1, &a1b3)?;
            to_free.push(c2.clone());

            let a1b2 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a1, &b2)?;
            to_free.push(a1b2.clone());
            let a2b1 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, &a2, &b1)?;
            to_free.push(a2b1.clone());
            let c3 =
                self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, &a1b2, &a2b1)?;
            to_free.push(c3.clone());

            let out = self.zeros_exec(&shape)?;
            let scatter_result = (|| -> Result<()> {
                self.scatter_linear_exec(&out, &comp1, &c1)?;
                self.scatter_linear_exec(&out, &comp2, &c2)?;
                self.scatter_linear_exec(&out, &comp3, &c3)?;
                Ok(())
            })();

            match scatter_result {
                Ok(()) => Ok(out),
                Err(err) => {
                    let _ = self.free(&out);
                    Err(err)
                }
            }
        })();

        for h in &to_free {
            let _ = self.free(h);
        }

        compute_result
    }
}

impl AccelProvider for WgpuProvider {
    fn export_context(&self, kind: AccelContextKind) -> Option<AccelContextHandle> {
        match kind {
            AccelContextKind::Plotting => Some(AccelContextHandle::Wgpu(WgpuContextHandle {
                instance: self.instance.clone(),
                device: self.device.clone(),
                queue: self.queue.clone(),
                adapter: self.adapter.clone(),
                adapter_info: self.adapter_info.clone(),
                limits: self.adapter_limits.clone(),
                features: self.device.features(),
            })),
        }
    }

    #[cfg(feature = "wgpu")]
    fn export_wgpu_buffer(&self, handle: &GpuTensorHandle) -> Option<WgpuBufferRef> {
        self.get_entry(handle).ok().map(|entry| WgpuBufferRef {
            buffer: entry.buffer,
            len: entry.len,
            shape: entry.shape,
            element_size: self.element_size,
            precision: match entry.precision {
                NumericPrecision::F32 => ProviderPrecision::F32,
                NumericPrecision::F64 => ProviderPrecision::F64,
            },
        })
    }

    fn device_id(&self) -> u32 {
        self.runtime_device_id
    }

    fn spawn_handle_concurrency(&self) -> SpawnHandleConcurrency {
        SpawnHandleConcurrency::SynchronizedMutation
    }

    fn gather_linear(
        &self,
        source: &GpuTensorHandle,
        indices: &[u32],
        output_shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        self.gather_linear_exec(source, indices, output_shape)
    }

    fn scatter_linear(
        &self,
        target: &GpuTensorHandle,
        indices: &[u32],
        values: &GpuTensorHandle,
    ) -> Result<()> {
        self.scatter_linear_exec(target, indices, values)
    }

    fn zeros(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.zeros_exec(shape)
    }

    fn zeros_like(&self, prototype: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.zeros_exec(&prototype.shape)
    }
    fn precision(&self) -> ProviderPrecision {
        match self.precision {
            NumericPrecision::F32 => ProviderPrecision::F32,
            NumericPrecision::F64 => ProviderPrecision::F64,
        }
    }

    fn fill(&self, shape: &[usize], value: f64) -> Result<GpuTensorHandle> {
        self.fill_exec(shape, value)
    }

    fn fill_like(&self, prototype: &GpuTensorHandle, value: f64) -> Result<GpuTensorHandle> {
        self.fill_exec(&prototype.shape, value)
    }

    fn eye(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.eye_exec(shape)
    }

    fn eye_like(&self, prototype: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.eye_exec(&prototype.shape)
    }

    fn meshgrid(&self, axes: &[MeshgridAxisView<'_>]) -> Result<ProviderMeshgridResult> {
        self.meshgrid_exec(axes)
    }

    fn linspace(&self, start: f64, stop: f64, count: usize) -> Result<GpuTensorHandle> {
        self.linspace_exec(start, stop, count)
    }

    fn random_uniform(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.random_uniform_exec(shape)
    }

    fn random_normal(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.random_normal_exec(shape)
    }

    fn random_exponential(&self, mu: f64, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.random_exponential_exec(mu, shape)
    }

    fn random_normrnd(&self, mu: f64, sigma: f64, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.random_normrnd_exec(mu, sigma, shape)
    }

    fn random_unifrnd(&self, a: f64, b: f64, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.random_unifrnd_exec(a, b, shape)
    }

    fn stochastic_evolution(
        &self,
        state: &GpuTensorHandle,
        drift: f64,
        scale: f64,
        steps: u32,
    ) -> Result<GpuTensorHandle> {
        self.stochastic_evolution_exec(state, drift, scale, steps)
    }

    fn fspecial(&self, request: &FspecialRequest) -> Result<GpuTensorHandle> {
        self.fspecial_exec(request)
    }

    fn peaks(&self, n: usize) -> Result<GpuTensorHandle> {
        self.peaks_exec(n)
    }

    fn peaks_xy(&self, x: &GpuTensorHandle, y: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.peaks_xy_exec(x, y)
    }

    fn hann_window(&self, len: usize, periodic: bool) -> Result<GpuTensorHandle> {
        self.window_exec(WindowKind::Hann, len, periodic)
    }

    fn hamming_window(&self, len: usize, periodic: bool) -> Result<GpuTensorHandle> {
        self.window_exec(WindowKind::Hamming, len, periodic)
    }

    fn blackman_window(&self, len: usize, periodic: bool) -> Result<GpuTensorHandle> {
        self.window_exec(WindowKind::Blackman, len, periodic)
    }

    fn imfilter<'a>(
        &'a self,
        image: &'a GpuTensorHandle,
        kernel: &'a GpuTensorHandle,
        options: &'a ImfilterOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.imfilter_exec(image, kernel, options).await })
    }

    fn random_integer_range(
        &self,
        lower: i64,
        upper: i64,
        shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        self.random_integer_range_exec(lower, upper, shape)
    }

    fn set_rng_state(&self, state: u64) -> Result<()> {
        let mut guard = rng_state()
            .lock()
            .map_err(|_| anyhow::anyhow!("wgpu provider: RNG mutex poisoned"))?;
        *guard = if state == 0 { RNG_DEFAULT_SEED } else { state };
        Ok(())
    }

    fn random_permutation(&self, n: usize, k: usize) -> Result<GpuTensorHandle> {
        self.randperm_exec(n, k)
    }

    fn random_permutation_like(
        &self,
        _prototype: &GpuTensorHandle,
        n: usize,
        k: usize,
    ) -> Result<GpuTensorHandle> {
        self.randperm_exec(n, k)
    }

    fn polyval(
        &self,
        coeffs: &GpuTensorHandle,
        points: &GpuTensorHandle,
        options: &ProviderPolyvalOptions,
    ) -> Result<GpuTensorHandle> {
        self.polyval_exec(coeffs, points, options)
    }

    fn polyfit<'a>(
        &'a self,
        x: &'a GpuTensorHandle,
        y: &'a GpuTensorHandle,
        degree: usize,
        weights: Option<&'a GpuTensorHandle>,
    ) -> AccelProviderFuture<'a, ProviderPolyfitResult> {
        Box::pin(async move {
            let x_host = <Self as AccelProvider>::download(self, x).await?;
            let y_host = <Self as AccelProvider>::download(self, y).await?;
            ensure!(
                x_host.data.len() == y_host.data.len(),
                "polyfit: X and Y vectors must match in length"
            );
            let weights_host = match weights {
                Some(handle) => Some(<Self as AccelProvider>::download(self, handle).await?),
                None => None,
            };
            let weights_slice = weights_host.as_ref().map(|w| w.data.as_slice());
            let host_result =
                polyfit_host_real_for_provider(&x_host.data, &y_host.data, degree, weights_slice)
                    .map_err(|err| anyhow!(err))?;
            Ok(ProviderPolyfitResult {
                coefficients: host_result.coefficients,
                r_matrix: host_result.r_matrix,
                normr: host_result.normr,
                df: host_result.df,
                mu: host_result.mu,
            })
        })
    }

    fn polyder_single<'a>(
        &'a self,
        polynomial: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.polyder_exec(polynomial).await })
    }

    fn polyder_product<'a>(
        &'a self,
        p: &'a GpuTensorHandle,
        q: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.polyder_product_exec(p, q).await })
    }

    fn polyder_quotient<'a>(
        &'a self,
        u: &'a GpuTensorHandle,
        v: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, ProviderPolyderQuotient> {
        Box::pin(async move { self.polyder_quotient_exec(u, v).await })
    }

    fn polyint(&self, polynomial: &GpuTensorHandle, constant: f64) -> Result<GpuTensorHandle> {
        self.polyint_exec(polynomial, constant)
    }

    fn diag_from_vector(&self, vector: &GpuTensorHandle, offset: isize) -> Result<GpuTensorHandle> {
        self.diag_from_vector_exec(vector, offset)
    }

    fn diag_extract(&self, matrix: &GpuTensorHandle, offset: isize) -> Result<GpuTensorHandle> {
        self.diag_extract_exec(matrix, offset)
    }

    fn tril<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        offset: isize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.tril_exec(matrix, offset).await })
    }

    fn triu<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        offset: isize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.triu_exec(matrix, offset).await })
    }

    fn reduce_mean_nd<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dims_zero_based: &'a [usize],
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_nd_mean_exec(a, dims_zero_based).await })
    }

    fn reduce_moments_nd<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dims_zero_based: &'a [usize],
    ) -> AccelProviderFuture<'a, runmat_accelerate_api::ProviderMoments2> {
        Box::pin(async move { self.reduce_moments_nd_exec(a, dims_zero_based) })
    }

    fn elem_add<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Add, a, b)
        })
    }

    fn elem_mul<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, a, b)
        })
    }

    fn elem_sub<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, a, b)
        })
    }

    fn elem_max<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Max, a, b)
        })
    }

    fn elem_min<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Min, a, b)
        })
    }

    fn elem_div<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Div, a, b)
        })
    }

    fn elem_pow<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Pow, a, b)
        })
    }

    fn elem_ge<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_ge_exec(a, b) })
    }

    fn elem_le<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_le_exec(a, b) })
    }

    fn elem_lt<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_lt_exec(a, b) })
    }

    fn elem_gt<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_gt_exec(a, b) })
    }

    fn elem_eq<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_eq_exec(a, b) })
    }

    fn elem_ne<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_ne_exec(a, b) })
    }

    fn logical_and(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_and_exec(a, b)
    }

    fn logical_or(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_or_exec(a, b)
    }

    fn logical_xor(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_xor_exec(a, b)
    }

    fn logical_not(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_not_exec(a)
    }

    fn logical_islogical(&self, a: &GpuTensorHandle) -> Result<bool> {
        let _ = self.get_entry(a)?;
        Ok(runmat_accelerate_api::handle_is_logical(a))
    }

    fn logical_isreal(&self, a: &GpuTensorHandle) -> Result<bool> {
        let _ = self.get_entry(a)?;
        Ok(true)
    }

    fn logical_isfinite(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_isfinite_exec(a)
    }

    fn logical_isnan(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_isnan_exec(a)
    }

    fn logical_isinf(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_isinf_exec(a)
    }

    fn elem_hypot<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Hypot, a, b)
        })
    }

    fn elem_atan2<'a>(
        &'a self,
        y: &'a GpuTensorHandle,
        x: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Atan2, y, x)
        })
    }

    fn unary_sin<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sin, a) },
        )
    }

    fn unary_sinc<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sinc, a) },
        )
    }

    fn unary_gamma<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Gamma, a) },
        )
    }

    fn unary_factorial<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Factorial, a)
        })
    }

    fn unary_asinh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Asinh, a) },
        )
    }

    fn unary_sinh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sinh, a) },
        )
    }

    fn unary_cosh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Cosh, a) },
        )
    }

    fn unary_asin<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Asin, a) },
        )
    }

    fn unary_acos<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Acos, a) },
        )
    }

    fn unary_acosh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Acosh, a) },
        )
    }

    fn unary_tan<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Tan, a) },
        )
    }

    fn unary_tanh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Tanh, a) },
        )
    }

    fn unary_atan<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Atan, a) },
        )
    }
    fn unary_atanh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Atanh, a) },
        )
    }

    fn unary_ceil<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Ceil, a) },
        )
    }

    fn unary_floor<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Floor, a) },
        )
    }

    fn unary_fix<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Fix, a) },
        )
    }

    fn unary_cos<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Cos, a) },
        )
    }

    fn unary_abs<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Abs, a) },
        )
    }

    fn unary_conj<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Conj, a) },
        )
    }

    fn unary_exp<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Exp, a) },
        )
    }

    fn unary_log<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Log, a) },
        )
    }

    fn unary_log1p<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Log1p, a) },
        )
    }

    fn unary_sqrt<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, a) },
        )
    }

    fn unary_double<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.precision != NumericPrecision::F64 {
                return Err(anyhow!(
                    "wgpu provider: shader-f64 unavailable; cannot materialise double precision"
                ));
            }
            let entry = self.get_entry(a)?;
            Ok(self.register_existing_buffer(entry.buffer, entry.shape, entry.len))
        })
    }

    fn unary_single<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Single, a) },
        )
    }

    fn unary_pow2<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let out = self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Pow2, a)?;
            // Record squared->base mapping for later reduction fusion (moments reuse)
            if let Ok(mut map) = self.pow2_of.lock() {
                map.insert(out.buffer_id, a.buffer_id);
            }
            Ok(out)
        })
    }

    fn unary_nextpow2<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::NextPow2, a)
        })
    }

    fn pow2_scale(
        &self,
        mantissa: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        if mantissa.shape != exponent.shape {
            return Err(anyhow!("pow2_scale requires matching shapes"));
        }
        let pow = self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Pow2, exponent)?;
        let result = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            mantissa,
            &pow,
        );
        let _ = self.free(&pow);
        result
    }

    fn scalar_rsub(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::RSub, a, scalar)
    }

    fn scalar_rdiv(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::RDiv, a, scalar)
    }

    fn scalar_add(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Add, a, scalar)
    }

    fn scalar_sub(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Sub, a, scalar)
    }

    fn scalar_mul(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Mul, a, scalar)
    }

    fn scalar_max(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Max, a, scalar)
    }

    fn scalar_min(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Min, a, scalar)
    }

    fn scalar_div(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Div, a, scalar)
    }

    fn sort_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        order: SortOrder,
        comparison: SortComparison,
    ) -> AccelProviderFuture<'a, SortResult> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, a).await?;
            let shape = host.shape.clone();
            let (values, indices) =
                sort_host_tensor(&host.data, &host.shape, dim, order, comparison)?;
            Ok(SortResult {
                values: HostTensorOwned {
                    data: values,
                    shape: shape.clone(),
                    storage: GpuTensorStorage::Real,
                },
                indices: HostTensorOwned {
                    data: indices,
                    shape,
                    storage: GpuTensorStorage::Real,
                },
            })
        })
    }
    fn sort_rows<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        columns: &'a [SortRowsColumnSpec],
        comparison: SortComparison,
    ) -> AccelProviderFuture<'a, SortResult> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, a).await?;
            let SortRowsHostOutputs {
                values,
                indices,
                indices_shape,
            } = sort_rows_host(&host.data, &host.shape, columns, comparison)?;
            Ok(SortResult {
                values: HostTensorOwned {
                    data: values,
                    shape: host.shape.clone(),
                    storage: GpuTensorStorage::Real,
                },
                indices: HostTensorOwned {
                    data: indices,
                    shape: indices_shape,
                    storage: GpuTensorStorage::Real,
                },
            })
        })
    }

    fn transpose(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.transpose_exec(a)
    }

    fn permute(&self, handle: &GpuTensorHandle, order: &[usize]) -> Result<GpuTensorHandle> {
        self.permute_exec(handle, order)
    }

    fn flip(&self, handle: &GpuTensorHandle, axes: &[usize]) -> Result<GpuTensorHandle> {
        self.flip_exec(handle, axes)
    }

    fn conv1d(
        &self,
        signal: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: ProviderConv1dOptions,
    ) -> Result<GpuTensorHandle> {
        self.conv1d_exec(signal, kernel, options)
    }
    fn iir_filter<'a>(
        &'a self,
        b: &'a GpuTensorHandle,
        a: &'a GpuTensorHandle,
        x: &'a GpuTensorHandle,
        options: ProviderIirFilterOptions,
    ) -> AccelProviderFuture<'a, ProviderIirFilterResult> {
        Box::pin(async move { self.iir_filter_exec(b, a, x, options).await })
    }
    fn conv2d(
        &self,
        _signal: &GpuTensorHandle,
        _kernel: &GpuTensorHandle,
        _mode: ProviderConvMode,
    ) -> Result<GpuTensorHandle> {
        Err(anyhow!("conv2d not implemented for the WGPU provider yet"))
    }

    fn diff_dim(
        &self,
        handle: &GpuTensorHandle,
        order: usize,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        self.diff_exec(handle, dim, order)
    }

    fn gradient_dim(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        spacing: f64,
    ) -> Result<GpuTensorHandle> {
        self.gradient_exec(handle, dim, spacing)
    }

    fn cumsum_scan(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        self.cumsum_exec(input, dim, direction, nan_mode)
    }

    fn cumprod_scan(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        self.cumprod_exec(input, dim, direction, nan_mode)
    }

    fn cummin_scan(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<runmat_accelerate_api::ProviderCumminResult> {
        self.cummin_exec(input, dim, direction, nan_mode)
    }

    fn cummax_scan(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<runmat_accelerate_api::ProviderCummaxResult> {
        self.cummax_exec(input, dim, direction, nan_mode)
    }

    fn circshift(&self, handle: &GpuTensorHandle, shifts: &[isize]) -> Result<GpuTensorHandle> {
        self.circshift_exec(handle, shifts)
    }

    fn fft_dim<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fft_dim_exec(handle, len, dim).await })
    }

    fn ifft_dim<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.ifft_dim_exec(handle, len, dim).await })
    }

    fn fft_extract_real<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fft_extract_real_exec(handle) })
    }

    fn unique<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        options: &'a UniqueOptions,
    ) -> AccelProviderFuture<'a, UniqueResult> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, handle).await?;
            let HostTensorOwned { data, shape, .. } = host;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("unique: {e}"))?;
            let eval = match runmat_runtime::builtins::array::sorting_sets::unique::unique_numeric_from_tensor(
                tensor, options,
            ) {
                Ok(eval) => eval,
                Err(err) => {
                    return Err(anyhow!("unique: {err}"));
                }
            };
            match eval.into_numeric_unique_result() {
                Ok(result) => Ok(result),
                Err(err) => Err(anyhow!("unique: {err}")),
            }
        })
    }
    fn ismember<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        options: &'a IsMemberOptions,
    ) -> AccelProviderFuture<'a, IsMemberResult> {
        Box::pin(async move {
            let host_a = <Self as AccelProvider>::download(self, a).await?;
            let host_b = <Self as AccelProvider>::download(self, b).await?;
            let tensor_a =
                Tensor::new(host_a.data, host_a.shape).map_err(|e| anyhow!("ismember: {e}"))?;
            let tensor_b =
                Tensor::new(host_b.data, host_b.shape).map_err(|e| anyhow!("ismember: {e}"))?;
            let eval = match runmat_runtime::builtins::array::sorting_sets::ismember::ismember_numeric_from_tensors(
                tensor_a,
                tensor_b,
                options.rows,
            ) {
                Ok(eval) => eval,
                Err(err) => {
                    return Err(anyhow!("ismember: {err}"));
                }
            };
            match eval.into_numeric_ismember_result() {
                Ok(result) => Ok(result),
                Err(err) => Err(anyhow!("ismember: {err}")),
            }
        })
    }

    fn union<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        options: &'a UnionOptions,
    ) -> AccelProviderFuture<'a, UnionResult> {
        Box::pin(async move {
            let host_a = <Self as AccelProvider>::download(self, a).await?;
            let host_b = <Self as AccelProvider>::download(self, b).await?;
            let tensor_a =
                Tensor::new(host_a.data, host_a.shape).map_err(|e| anyhow!("union: {e}"))?;
            let tensor_b =
                Tensor::new(host_b.data, host_b.shape).map_err(|e| anyhow!("union: {e}"))?;
            let eval = match runmat_runtime::builtins::array::sorting_sets::union::union_numeric_from_tensors(
                tensor_a, tensor_b, options,
            ) {
                Ok(eval) => eval,
                Err(err) => {
                    return Err(anyhow!("union: {err}"));
                }
            };
            match eval.into_numeric_union_result() {
                Ok(result) => Ok(result),
                Err(err) => Err(anyhow!("union: {err}")),
            }
        })
    }
    fn setdiff<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        options: &'a SetdiffOptions,
    ) -> AccelProviderFuture<'a, SetdiffResult> {
        Box::pin(async move {
            let host_a = <Self as AccelProvider>::download(self, a).await?;
            let host_b = <Self as AccelProvider>::download(self, b).await?;
            let tensor_a =
                Tensor::new(host_a.data, host_a.shape).map_err(|e| anyhow!("setdiff: {e}"))?;
            let tensor_b =
                Tensor::new(host_b.data, host_b.shape).map_err(|e| anyhow!("setdiff: {e}"))?;
            let eval = match runmat_runtime::builtins::array::sorting_sets::setdiff::setdiff_numeric_from_tensors(
                tensor_a, tensor_b, options,
            ) {
                Ok(eval) => eval,
                Err(err) => {
                    return Err(anyhow!("setdiff: {err}"));
                }
            };
            match eval.into_numeric_setdiff_result() {
                Ok(result) => Ok(result),
                Err(err) => Err(anyhow!("setdiff: {err}")),
            }
        })
    }

    fn cat(&self, dim: usize, inputs: &[GpuTensorHandle]) -> Result<GpuTensorHandle> {
        self.cat_exec(dim, inputs)
    }

    fn repmat(&self, handle: &GpuTensorHandle, reps: &[usize]) -> Result<GpuTensorHandle> {
        self.repmat_exec(handle, reps)
    }

    fn kron(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.kron_exec(a, b)
    }
    fn cross(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        self.cross_exec(lhs, rhs, dim)
    }
    fn reshape(&self, handle: &GpuTensorHandle, new_shape: &[usize]) -> Result<GpuTensorHandle> {
        let new_len = if new_shape.is_empty() {
            1
        } else {
            product_checked(new_shape)
                .ok_or_else(|| anyhow!("reshape: dimension product exceeds GPU limits"))?
        };
        let mut buffers = self.buffers.lock().expect("buffer mutex poisoned");
        let entry = buffers
            .get_mut(&handle.buffer_id)
            .ok_or_else(|| anyhow!("reshape: unknown buffer {}", handle.buffer_id))?;
        ensure!(
            entry.len == new_len,
            "reshape: product of dimensions ({}) must equal original tensor length ({})",
            new_len,
            entry.len
        );
        entry.shape = new_shape.to_vec();
        let mut updated = handle.clone();
        updated.shape = new_shape.to_vec();
        Ok(updated)
    }

    fn lu<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, ProviderLuResult> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, a).await?;
            let LuHostFactors {
                combined,
                lower,
                upper,
                perm_matrix,
                pivot_vector,
                combined_shape,
                lower_shape,
                upper_shape,
                perm_shape,
                pivot_shape,
            } = lu_factor_host(&host.data, &host.shape)?;
            let combined = self.upload(&HostTensorView {
                data: &combined,
                shape: &combined_shape,
            })?;
            let lower = self.upload(&HostTensorView {
                data: &lower,
                shape: &lower_shape,
            })?;
            let upper = self.upload(&HostTensorView {
                data: &upper,
                shape: &upper_shape,
            })?;
            let perm_matrix = self.upload(&HostTensorView {
                data: &perm_matrix,
                shape: &perm_shape,
            })?;
            let perm_vector = self.upload(&HostTensorView {
                data: &pivot_vector,
                shape: &pivot_shape,
            })?;
            Ok(ProviderLuResult {
                combined,
                lower,
                upper,
                perm_matrix,
                perm_vector,
            })
        })
    }

    fn chol<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        lower: bool,
    ) -> AccelProviderFuture<'a, ProviderCholResult> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, a).await?;
            let tensor = Tensor::new(host.data.clone(), host.shape.clone())
                .map_err(|e| anyhow!("chol: {e}"))?;
            let mut args = Vec::new();
            if lower {
                args.push(Value::from("lower"));
            }
            let eval = runmat_runtime::builtins::math::linalg::factor::chol::evaluate(
                Value::Tensor(tensor),
                &args,
            )
            .await
            .map_err(|err| runtime_flow_to_anyhow("chol", err))?;
            let factor_tensor = host_tensor_from_value("chol", eval.factor())?;
            let factor = self.upload(&HostTensorView {
                data: &factor_tensor.data,
                shape: &factor_tensor.shape,
            })?;
            Ok(ProviderCholResult {
                factor,
                info: eval.flag_index() as u32,
            })
        })
    }
    fn qr<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        options: ProviderQrOptions,
    ) -> AccelProviderFuture<'a, ProviderQrResult> {
        Box::pin(async move {
            if let Some(result) = self.try_qr_device(handle, &options)? {
                return Ok(result);
            }
            let host = <Self as AccelProvider>::download(self, handle).await?;
            let tensor = Tensor::new(host.data.clone(), host.shape.clone())
                .map_err(|e| anyhow!("qr: {e}"))?;
            self.qr_host_result(tensor, &options).await
        })
    }

    fn take_matmul_sources(
        &self,
        product: &GpuTensorHandle,
    ) -> Option<(GpuTensorHandle, GpuTensorHandle)> {
        let res = self.kernel_resources.take_matmul_sources(product);
        log::debug!(
            "take_matmul_sources: product={} found={} active_fusion={:?}",
            product.buffer_id,
            res.is_some(),
            active_fusion()
        );
        res
    }

    fn qr_power_iter<'a>(
        &'a self,
        product: &'a GpuTensorHandle,
        product_lhs: Option<&'a GpuTensorHandle>,
        q_handle: &'a GpuTensorHandle,
        options: &'a ProviderQrOptions,
    ) -> AccelProviderFuture<'a, Option<ProviderQrPowerIterResult>> {
        Box::pin(async move {
            let debug_qr = std::env::var("RUNMAT_DEBUG_QR").is_ok();
            if !options.economy {
                return Ok(None);
            }

            let product_entry = self.get_entry(product)?;
            if product_entry.shape.len() != 2 {
                return Ok(None);
            }
            let rows = product_entry.shape[0];
            let cols = product_entry.shape[1];
            if rows == 0 || cols == 0 {
                return Ok(None);
            }
            if cols > QR_DEVICE_MAX_COLS {
                if debug_qr {
                    log::debug!(
                        "qr_power_iter: column count {} exceeds device kernel limit {}; falling back",
                        cols,
                        QR_DEVICE_MAX_COLS
                    );
                }
                return Ok(None);
            }
            if self.precision() != ProviderPrecision::F32 {
                if debug_qr {
                    log::debug!(
                        "qr_power_iter: precision {:?} unsupported for device QR kernel; falling back",
                        self.precision()
                    );
                }
                return Ok(None);
            }
            let q_entry = self.get_entry(q_handle)?;
            if q_entry.shape != product_entry.shape {
                return Ok(None);
            }
            let k = cols;

            let mut pre_product_max = match <Self as AccelProvider>::download(self, product).await {
                Ok(host) => Some(
                    host.data
                        .iter()
                        .fold(0.0f64, |acc, value| acc.max(value.abs())),
                ),
                Err(err) => {
                    log::warn!("qr_power_iter pre-download failed: {err}");
                    None
                }
            };

            let pre_q_max = match <Self as AccelProvider>::download(self, q_handle).await {
                Ok(host) => Some(
                    host.data
                        .iter()
                        .fold(0.0f64, |acc, value| acc.max(value.abs())),
                ),
                Err(err) => {
                    log::warn!("qr_power_iter q-handle pre-download failed: {err}");
                    None
                }
            };

            const PRODUCT_EPS: f64 = 1.0e-12;
            const Q_EPS: f64 = 1.0e-6;
            if pre_product_max.unwrap_or(0.0) <= PRODUCT_EPS && pre_q_max.unwrap_or(0.0) > Q_EPS {
                let debug_zero_host = std::env::var("RUNMAT_DEBUG_QR_ZEROHOST").is_ok();
                if debug_zero_host {
                    if let Some(lhs_handle) = product_lhs {
                        let lhs_download =
                            <Self as AccelProvider>::download(self, lhs_handle).await;
                        let q_download = <Self as AccelProvider>::download(self, q_handle).await;
                        match (lhs_download, q_download) {
                            (Ok(lhs_host), Ok(q_host)) => {
                                let lhs_rows = lhs_host.shape.first().copied().unwrap_or(0);
                                let lhs_cols = lhs_host.shape.get(1).copied().unwrap_or(0);
                                let q_rows = q_host.shape.first().copied().unwrap_or(0);
                                let q_cols = q_host.shape.get(1).copied().unwrap_or(0);
                                if lhs_rows == q_rows
                                    && lhs_cols == q_rows
                                    && q_rows == rows
                                    && q_cols == cols
                                {
                                    let mut max_host_product = 0.0f64;
                                    for col in 0..cols {
                                        for row in 0..rows {
                                            let mut sum = 0.0f64;
                                            for k_idx in 0..lhs_cols {
                                                let lhs_idx = row + k_idx * lhs_rows;
                                                let q_idx = k_idx + col * q_rows;
                                                sum += lhs_host.data[lhs_idx] * q_host.data[q_idx];
                                            }
                                            max_host_product = max_host_product.max(sum.abs());
                                        }
                                    }
                                    log::info!(
                                    "qr_power_iter host check: rows={} cols={} host_max_product={:.6e}",
                                    rows,
                                    cols,
                                    max_host_product
                                );
                                } else {
                                    log::info!(
                                    "qr_power_iter host check skipped: lhs_shape={:?} q_shape={:?} rows={} cols={}",
                                    lhs_host.shape,
                                    q_host.shape,
                                    rows,
                                    cols
                                );
                                }
                            }
                            (lhs_res, q_res) => {
                                log::info!(
                                    "qr_power_iter host check download failed: lhs={:?} q={:?} product_id={}",
                                    lhs_res.err(),
                                    q_res.err(),
                                    product.buffer_id
                                );
                            }
                        }
                    } else {
                        log::info!(
                            "qr_power_iter host check skipped: product_lhs unavailable (product_id={})",
                            product.buffer_id
                        );
                    }
                }
                if let Some(lhs_handle) = product_lhs {
                    log::warn!(
                        "qr_power_iter: detected zero matmul product (product_id={} max_product_abs_pre={:?} max_q_abs_pre={:?}); recomputing",
                        product.buffer_id,
                        pre_product_max,
                        pre_q_max
                    );
                    if let Ok(lhs_entry) = self.get_entry(lhs_handle) {
                        if let Ok(rhs_entry) = self.get_entry(q_handle) {
                            let lhs_view = build_matrix_operand_view(lhs_handle, &lhs_entry)
                                .unwrap_or(MatrixOperandView {
                                    rows: 0,
                                    cols: 0,
                                    lda: 0,
                                    transpose: false,
                                });
                            let rhs_view = build_matrix_operand_view(q_handle, &rhs_entry)
                                .unwrap_or(MatrixOperandView {
                                    rows: 0,
                                    cols: 0,
                                    lda: 0,
                                    transpose: false,
                                });
                            log::info!(
                                "qr_power_iter recompute operands: product_id={} lhs_shape={:?} rhs_shape={:?} lhs_view={{rows:{} cols:{} lda:{} transpose:{}}} rhs_view={{rows:{} cols:{} lda:{} transpose:{}}}",
                                product.buffer_id,
                                lhs_entry.shape,
                                rhs_entry.shape,
                                lhs_view.rows,
                                lhs_view.cols,
                                lhs_view.lda,
                                lhs_view.transpose,
                                rhs_view.rows,
                                rhs_view.cols,
                                rhs_view.lda,
                                rhs_view.transpose
                            );
                            log::info!(
                                "qr_power_iter recompute buffers: product_id={} lhs_ptr={:p} rhs_ptr={:p}",
                                product.buffer_id,
                                lhs_entry.buffer.as_ref(),
                                rhs_entry.buffer.as_ref()
                            );
                        }
                    }
                    let recomputed = self.matmul_exec_with_usage(
                        lhs_handle,
                        q_handle,
                        BufferUsageClass::FusionOut,
                    )?;
                    let mut recomputed_max: Option<f64> = None;
                    if debug_zero_host {
                        match <Self as AccelProvider>::download(self, &recomputed).await {
                            Ok(host) => {
                                let max_recomputed = host
                                    .data
                                    .iter()
                                    .fold(0.0f64, |acc, value| acc.max(value.abs()));
                                log::info!(
                                    "qr_power_iter recompute check: product_id={} max_recomputed_abs={:.6e}",
                                    product.buffer_id,
                                    max_recomputed
                                );
                                recomputed_max = Some(max_recomputed);
                            }
                            Err(err) => {
                                log::info!(
                                    "qr_power_iter recompute check failed: product_id={} err={}",
                                    product.buffer_id,
                                    err
                                );
                            }
                        }
                    }
                    let recomputed_entry = self.get_entry(&recomputed)?;
                    log::info!(
                        "qr_power_iter recompute start: product_id={} original_len={} recomputed_len={}",
                        product.buffer_id,
                        product_entry.len,
                        recomputed_entry.len
                    );
                    let bytes = (recomputed_entry.len as u64) * self.element_size as u64;
                    log::info!(
                        "qr_power_iter recompute copy detail: product_id={} product_ptr={:p} recomputed_ptr={:p}",
                        product.buffer_id,
                        product_entry.buffer.as_ref(),
                        recomputed_entry.buffer.as_ref()
                    );
                    if bytes > 0 {
                        let mut encoder =
                            self.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("runmat-qr-product-recompute"),
                                });
                        encoder.copy_buffer_to_buffer(
                            recomputed_entry.buffer.as_ref(),
                            0,
                            product_entry.buffer.as_ref(),
                            0,
                            bytes,
                        );
                        self.submit(encoder);
                    }

                    let max_val = if let Some(val) = recomputed_max {
                        val
                    } else {
                        match <Self as AccelProvider>::download(self, product).await {
                            Ok(host) => host
                                .data
                                .iter()
                                .fold(0.0f64, |acc, value| acc.max(value.abs())),
                            Err(err) => {
                                log::warn!("qr_power_iter recompute verification failed: {err}");
                                0.0
                            }
                        }
                    };
                    log::info!(
                        "qr_power_iter recompute copy: product_id={} bytes={} post_max={:.6e}",
                        product.buffer_id,
                        bytes,
                        max_val
                    );
                    if max_val == 0.0 {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            let q_download =
                                <Self as AccelProvider>::download(self, q_handle).await;
                            if let Ok(lhs_dump) =
                                <Self as AccelProvider>::download(self, lhs_handle).await
                            {
                                if let Ok(ref q_dump) = q_download {
                                    let dump_dir = Path::new("target/matmul_zero");
                                    let _ = fs::create_dir_all(dump_dir);
                                    let lhs_path = dump_dir.join(format!(
                                        "lhs_{}_{}.bin",
                                        product.buffer_id,
                                        lhs_dump.data.len()
                                    ));
                                    let rhs_path = dump_dir.join(format!(
                                        "rhs_{}_{}.bin",
                                        product.buffer_id,
                                        q_dump.data.len()
                                    ));
                                    let _ =
                                        fs::write(&lhs_path, cast_slice(lhs_dump.data.as_slice()));
                                    let _ =
                                        fs::write(&rhs_path, cast_slice(q_dump.data.as_slice()));
                                    log::warn!(
                                        "qr_power_iter dump written: product_id={} lhs_path={} rhs_path={}",
                                        product.buffer_id,
                                        lhs_path.display(),
                                        rhs_path.display()
                                    );
                                }
                            }
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            log::warn!("qr_power_iter: skipping matmul dump because filesystem APIs are unavailable on wasm");
                        }
                        log::warn!(
                            "qr_power_iter: recomputed product is still zero; falling back to host QR"
                        );
                        let _ = self.free(&recomputed);
                        if let Some(handle) = self.qr_power_iter_host(product, options).await? {
                            return Ok(Some(handle));
                        }
                        return Ok(None);
                    }
                    pre_product_max = Some(max_val);

                    let _ = self.free(&recomputed);
                } else {
                    log::warn!(
                        "qr_power_iter: zero product detected for buffer {} without lhs handle; proceeding with existing data",
                        product.buffer_id
                    );
                }
            }

            let (q_result, r_handle, mut r_inv_opt) = self.qr_factor_device(
                product,
                rows,
                cols,
                Some(q_handle),
                "runmat-qr-power",
                true,
            )?;

            let mut fallback_needed = false;
            if let Ok(host_r) = <Self as AccelProvider>::download(self, &r_handle).await {
                for col in 0..cols {
                    let diag = host_r.data[col + col * cols];
                    if !diag.is_finite() || diag.abs() <= 1.0e-12 {
                        fallback_needed = true;
                        break;
                    }
                }
            }

            if fallback_needed {
                if let Some(handle) = r_inv_opt.take() {
                    let _ = self.free(&handle);
                }
                let _ = self.free(&q_result);
                let _ = self.free(&r_handle);
                return self.qr_power_iter_host(product, options).await;
            }

            if pre_product_max.unwrap_or(0.0) <= 1.0e-8 {
                if let Some(handle) = r_inv_opt.take() {
                    let _ = self.free(&handle);
                }
                let _ = self.free(&q_result);
                let _ = self.free(&r_handle);
                return self.qr_power_iter_host(product, options).await;
            }

            if debug_qr {
                if let Err(err) = self
                    .debug_qr_power_iter(
                        product,
                        &product_entry,
                        pre_product_max,
                        pre_q_max,
                        &q_result,
                        &r_handle,
                        r_inv_opt
                            .as_ref()
                            .expect("retain_r_inv=true must provide inverse handle"),
                        None::<&runmat_accelerate_api::HostTensorOwned>,
                        rows,
                        cols,
                    )
                    .await
                {
                    log::warn!("qr_power_iter debug failed: {err}");
                }
            }

            if let Some(handle) = r_inv_opt.take() {
                let _ = self.free(&handle);
            }

            let mut perm_matrix = vec![0.0f64; k * k];
            for i in 0..k {
                perm_matrix[i + i * k] = 1.0;
            }
            let perm_vector: Vec<f64> = (1..=k).map(|v| v as f64).collect();

            let perm_matrix_shape = [k, k];
            let perm_matrix_handle = self.upload(&HostTensorView {
                data: &perm_matrix,
                shape: &perm_matrix_shape,
            })?;
            let perm_vector_shape = vec![k, 1];
            let perm_vector_handle = self.upload(&HostTensorView {
                data: &perm_vector,
                shape: &perm_vector_shape,
            })?;

            let _ = self.free(product);

            Ok(Some(ProviderQrPowerIterResult {
                q: q_result,
                r: r_handle,
                perm_matrix: perm_matrix_handle,
                perm_vector: perm_vector_handle,
            }))
        })
    }
    fn matmul<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.matmul_exec(a, b) })
    }

    fn syrk(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.syrk_exec(a)
    }
    fn matmul_epilogue<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        ep: &'a runmat_accelerate_api::MatmulEpilogue,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            use runmat_accelerate_api::ProviderPrecision;
            let entry_a = self.get_entry(a)?;
            let entry_b = self.get_entry(b)?;
            if entry_a.shape.len() != 2 || entry_b.shape.len() != 2 {
                return Err(anyhow!("matmul_epilogue: only 2D tensors supported"));
            }
            let view_a = build_matrix_operand_view(a, &entry_a)
                .map_err(|e| anyhow!("matmul_epilogue: {e}"))?;
            let view_b = build_matrix_operand_view(b, &entry_b)
                .map_err(|e| anyhow!("matmul_epilogue: {e}"))?;

            if view_a.cols != view_b.rows {
                return Err(anyhow!("matmul_epilogue: inner dimensions must match"));
            }
            let m = view_a.rows;
            let n = view_b.cols;
            let k = view_a.cols;

            let out_shape = vec![m, n];
            let len = m * n;
            let out_buffer =
                self.create_storage_buffer_checked(len, "runmat-matmul-epilogue-out")?;
            if len == 0 {
                return Ok(self.register_existing_buffer(out_buffer, out_shape, len));
            }

            let start = Instant::now();

            let m_u32 =
                u32::try_from(m).map_err(|_| anyhow!("matmul_epilogue: m exceeds GPU limits"))?;
            let n_u32 =
                u32::try_from(n).map_err(|_| anyhow!("matmul_epilogue: n exceeds GPU limits"))?;
            let k_u32 =
                u32::try_from(k).map_err(|_| anyhow!("matmul_epilogue: k exceeds GPU limits"))?;
            let mut flags = 0u32;
            if view_a.transpose {
                flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_A;
            }
            if view_b.transpose {
                flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_B;
            }

            let params = crate::backend::wgpu::params::MatmulParams {
                m: m_u32,
                n: n_u32,
                k: k_u32,
                lda: view_a.lda,
                ldb: view_b.lda,
                ldc: m_u32,
                offset_a: 0,
                offset_b: 0,
                offset_out: 0,
                flags,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-matmul-epilogue-params");

            // Resolve optional scales and epilogue params by precision
            use crate::backend::wgpu::params::{
                MATMUL_EPILOGUE_FLAG_CLAMP_MAX, MATMUL_EPILOGUE_FLAG_CLAMP_MIN,
                MATMUL_EPILOGUE_FLAG_COL_DIV, MATMUL_EPILOGUE_FLAG_COL_SCALE,
                MATMUL_EPILOGUE_FLAG_DIAG_WRITE, MATMUL_EPILOGUE_FLAG_POW,
                MATMUL_EPILOGUE_FLAG_ROW_DIV, MATMUL_EPILOGUE_FLAG_ROW_SCALE,
            };
            let has_row = ep.row_scale.is_some();
            let has_col = ep.col_scale.is_some();
            let dummy_rowcol = self.create_storage_buffer(1, "runmat-matmul-epilogue-dummy-scale");
            let row_buf = match &ep.row_scale {
                Some(h) => self.get_entry(h)?.buffer.clone(),
                None => dummy_rowcol.clone(),
            };
            let col_buf = match &ep.col_scale {
                Some(h) => self.get_entry(h)?.buffer.clone(),
                None => dummy_rowcol.clone(),
            };

            let (diag_rows, diag_stride, diag_offset, has_diag) = match &ep.diag_output {
                Some(_) => {
                    return Err(anyhow!(
                        "matmul_epilogue: diag_output is not supported by the WGPU provider yet"
                    ));
                }
                None => (0u32, 1u32, 0u32, false),
            };

            let mut flags: u32 = 0;
            if has_row {
                flags |= MATMUL_EPILOGUE_FLAG_ROW_SCALE;
                if matches!(ep.row_op, runmat_accelerate_api::ScaleOp::Divide) {
                    flags |= MATMUL_EPILOGUE_FLAG_ROW_DIV;
                }
            }
            if has_col {
                flags |= MATMUL_EPILOGUE_FLAG_COL_SCALE;
                if matches!(ep.col_op, runmat_accelerate_api::ScaleOp::Divide) {
                    flags |= MATMUL_EPILOGUE_FLAG_COL_DIV;
                }
            }

            let mut clamp_min = 0.0f64;
            if let Some(v) = ep.clamp_min {
                clamp_min = v;
                flags |= MATMUL_EPILOGUE_FLAG_CLAMP_MIN;
            }
            let mut clamp_max = 0.0f64;
            if let Some(v) = ep.clamp_max {
                clamp_max = v;
                flags |= MATMUL_EPILOGUE_FLAG_CLAMP_MAX;
            }
            let mut pow_exponent = 1.0f64;
            if let Some(v) = ep.pow_exponent {
                pow_exponent = v;
                flags |= MATMUL_EPILOGUE_FLAG_POW;
            }
            if has_diag {
                flags |= MATMUL_EPILOGUE_FLAG_DIAG_WRITE;
            }

            let tile = crate::backend::wgpu::config::effective_matmul_tile();
            let groups_x =
                crate::backend::wgpu::dispatch::common::dispatch_size_dim(n as u32, tile);
            let groups_y =
                crate::backend::wgpu::dispatch::common::dispatch_size_dim(m as u32, tile);

            // Build a layout tag incorporating the epilogue mask for cache keying
            let layout_tag = format!("runmat-matmul-epilogue-layout-flags-{flags:08x}");

            // Create module from the static WGSL (token substitution handled inside)
            let (shader_src, ep_buf, pipeline_layout) = match self.precision() {
                ProviderPrecision::F64 => {
                    let ep_params = crate::backend::wgpu::params::MatmulEpilogueParamsF64 {
                        alpha: ep.alpha,
                        beta: ep.beta,
                        clamp_min,
                        clamp_max,
                        pow_exponent,
                        flags,
                        diag_offset,
                        diag_stride,
                        diag_rows,
                        _pad: 0,
                        _pad2: 0,
                    };
                    let ep_buf = self.uniform_buffer(&ep_params, "runmat-matmul-epilogue-uniform");
                    let pl = crate::backend::wgpu::cache::factory::create_pipeline_layout_single(
                        self.device_ref(),
                        "runmat-matmul-epilogue-pl",
                        &self.pipelines.matmul_epilogue.layout,
                    );
                    (
                        crate::backend::wgpu::shaders::matmul::MATMUL_EPILOGUE_SHADER_F64,
                        ep_buf,
                        pl,
                    )
                }
                ProviderPrecision::F32 => {
                    let ep_params = crate::backend::wgpu::params::MatmulEpilogueParamsF32 {
                        alpha: ep.alpha as f32,
                        beta: ep.beta as f32,
                        clamp_min: clamp_min as f32,
                        clamp_max: clamp_max as f32,
                        pow_exponent: pow_exponent as f32,
                        flags,
                        diag_offset,
                        diag_stride,
                        diag_rows,
                        _pad: 0,
                    };
                    let ep_buf = self.uniform_buffer(&ep_params, "runmat-matmul-epilogue-uniform");
                    let pl = crate::backend::wgpu::cache::factory::create_pipeline_layout_single(
                        self.device_ref(),
                        "runmat-matmul-epilogue-pl",
                        &self.pipelines.matmul_epilogue.layout,
                    );
                    (
                        crate::backend::wgpu::shaders::matmul::MATMUL_EPILOGUE_SHADER_F32,
                        ep_buf,
                        pl,
                    )
                }
            };

            let module = crate::backend::wgpu::pipelines::create_shader_module(
                self.device_ref(),
                "runmat-matmul-epilogue-module",
                shader_src,
            );
            let key =
                self.compute_pipeline_hash_bytes(shader_src.as_bytes(), &layout_tag, Some(tile));
            let pipeline = self.get_or_create_pipeline(
                key,
                &pipeline_layout,
                &module,
                "runmat-matmul-epilogue",
                Some(shader_src.as_bytes()),
                Some(&layout_tag),
                Some(tile),
            );

            let bg = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-matmul-epilogue-bind"),
                    layout: &self.pipelines.matmul_epilogue.layout,
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
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: row_buf.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: col_buf.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: ep_buf.as_entire_binding(),
                        },
                    ],
                });
            crate::backend::wgpu::dispatch::matmul::run(
                self.device_ref(),
                self.queue_ref(),
                &pipeline,
                &bg,
                groups_x,
                groups_y,
            );
            let handle = self.register_existing_buffer_with_usage(
                out_buffer,
                out_shape,
                len,
                BufferUsageClass::FusionOut,
            );

            self.telemetry.record_matmul_duration(start.elapsed());

            Ok(handle)
        })
    }
    fn pagefun(&self, request: &PagefunRequest) -> Result<GpuTensorHandle> {
        self.pagefun_exec(request)
    }
    fn image_normalize<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        desc: &'a runmat_accelerate_api::ImageNormalizeDescriptor,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let entry = self.get_entry(input)?;
            ensure!(
                entry.shape.len() == 3,
                "image_normalize: expected 3-D tensor, got {:?}",
                entry.shape
            );
            ensure!(
                entry.shape[0] == desc.batch
                    && entry.shape[1] == desc.height
                    && entry.shape[2] == desc.width,
                "image_normalize: descriptor dims {:?} do not match tensor shape {:?}",
                (desc.batch, desc.height, desc.width),
                entry.shape
            );

            if entry.len == 0 {
                return self.image_normalize_cpu_fallback(input, desc).await;
            }

            match self.precision {
                NumericPrecision::F64 => self.image_normalize_cpu_fallback(input, desc).await,
                NumericPrecision::F32 => {
                    ensure!(
                        desc.epsilon.is_finite(),
                        "image_normalize: epsilon must be finite"
                    );
                    ensure!(
                        desc.epsilon >= 0.0,
                        "image_normalize: epsilon must be non-negative"
                    );

                    let batches = entry.shape[0];
                    let height = entry.shape[1];
                    let width = entry.shape[2];
                    let plane = height
                        .checked_mul(width)
                        .ok_or_else(|| anyhow!("image_normalize: height*width overflow"))?;
                    ensure!(
                        entry.len == plane * batches,
                        "image_normalize: inconsistent tensor length {} vs dims {:?}",
                        entry.len,
                        entry.shape
                    );

                    let stride_h = batches;
                    let stride_w = batches
                        .checked_mul(height)
                        .ok_or_else(|| anyhow!("image_normalize: stride overflow"))?;

                    let batches_u32 = u32::try_from(batches)
                        .map_err(|_| anyhow!("image_normalize: batch size too large"))?;
                    let height_u32 = u32::try_from(height)
                        .map_err(|_| anyhow!("image_normalize: height too large"))?;
                    let width_u32 = u32::try_from(width)
                        .map_err(|_| anyhow!("image_normalize: width too large"))?;
                    let plane_u32 = u32::try_from(plane)
                        .map_err(|_| anyhow!("image_normalize: plane size too large"))?;
                    let stride_h_u32 = u32::try_from(stride_h)
                        .map_err(|_| anyhow!("image_normalize: stride_h too large"))?;
                    let stride_w_u32 = u32::try_from(stride_w)
                        .map_err(|_| anyhow!("image_normalize: stride_w too large"))?;
                    let (tuning, cache_hit) =
                        self.resolve_image_normalize_tuning(batches_u32, plane_u32);
                    log::debug!(
                    "image_normalize tuning batches={} plane={} lane={} spatial={} values/thread={} cache_hit={}",
                    batches_u32,
                    plane_u32,
                    tuning.lane_count,
                    tuning.spatial_tile,
                    tuning.values_per_thread,
                    cache_hit
                );
                    let pipeline = self.image_normalize_pipeline(&tuning)?;

                    let mut flags = 0u32;
                    if desc.gain.is_some() {
                        flags |= IMAGE_NORMALIZE_FLAG_GAIN;
                    }
                    if desc.bias.is_some() {
                        flags |= IMAGE_NORMALIZE_FLAG_BIAS;
                    }
                    if desc.gamma.is_some() {
                        flags |= IMAGE_NORMALIZE_FLAG_GAMMA;
                    }

                    let mut uniforms = ImageNormalizeUniforms {
                        batch_count: 0,
                        height: height_u32,
                        width: width_u32,
                        plane: plane_u32,
                        stride_h: stride_h_u32,
                        stride_w: stride_w_u32,
                        flags,
                        batch_stride: batches_u32,
                        batch_offset: 0,
                        _pad0: 0,
                        epsilon: desc.epsilon as f32,
                        gain: desc.gain.unwrap_or(1.0) as f32,
                        bias: desc.bias.unwrap_or(0.0) as f32,
                        gamma: desc.gamma.unwrap_or(1.0) as f32,
                        _pad1: 0,
                    };

                    let out_buffer = self.create_storage_buffer_checked_with_usage(
                        entry.len,
                        "runmat-image-normalize-out",
                        BufferUsageClass::FusionOut,
                    )?;
                    let uniform_buf = self.kernel_resources.uniform_buffer(
                        self.device_ref(),
                        UniformBufferKey::ImageNormalizeUniforms,
                        std::mem::size_of::<ImageNormalizeUniforms>() as u64,
                        "runmat-image-normalize-uniform",
                    );
                    let stream_hot_cap = self
                        .image_normalize_hot_stream_cap(plane_u32, batches_u32)
                        .max(1);
                    let cold_cap =
                        stream_hot_cap.min((Self::IMAGE_NORMALIZE_STREAM_COLD_CAP).max(1));
                    let chunk_limit = if cache_hit {
                        stream_hot_cap
                    } else {
                        cold_cap.max(1)
                    };

                    let bind_entries = [
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
                            resource: uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                    ];
                    let layout = &self.pipelines.image_normalize.layout;
                    let bind_group =
                        self.bind_group_cache
                            .get_or_create(layout, &bind_entries, || {
                                Arc::new(self.device_ref().create_bind_group(
                                    &wgpu::BindGroupDescriptor {
                                        label: Some("runmat-image-normalize-bind"),
                                        layout,
                                        entries: &bind_entries,
                                    },
                                ))
                            });

                    let mut offset = 0u32;
                    while offset < batches_u32 {
                        let remaining = batches_u32 - offset;
                        let chunk = remaining.min(chunk_limit).max(1);
                        uniforms.batch_count = chunk;
                        uniforms.batch_offset = offset;
                        self.queue
                            .write_buffer(uniform_buf.as_ref(), 0, bytes_of(&uniforms));
                        crate::backend::wgpu::dispatch::image_normalize::run(
                            self.device_ref(),
                            self.queue_ref(),
                            pipeline.as_ref(),
                            bind_group.as_ref(),
                            chunk,
                            tuning.batch_tile,
                        );
                        offset += chunk;
                    }

                    Ok(self.register_existing_buffer_with_usage(
                        out_buffer,
                        entry.shape.clone(),
                        entry.len,
                        BufferUsageClass::FusionOut,
                    ))
                }
            }
        })
    }
    fn matmul_power_step<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        epilogue: &'a runmat_accelerate_api::PowerStepEpilogue,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let rhs_entry = self.get_entry(rhs)?;
            let product = self.matmul_exec(lhs, rhs)?;
            let squared = self.binary_op_exec(
                crate::backend::wgpu::types::BinaryOpCode::Mul,
                &product,
                &product,
            )?;
            let mut sum_sq = self.reduce_dim_sum_mean_exec(
                &squared,
                0,
                crate::backend::wgpu::types::DimReduceOp::Sum,
            )?;
            let _ = self.free(&squared);
            if epilogue.epsilon != 0.0 {
                let eps = self.fill_exec(&sum_sq.shape, epilogue.epsilon)?;
                let adjusted = self.binary_op_exec(
                    crate::backend::wgpu::types::BinaryOpCode::Add,
                    &sum_sq,
                    &eps,
                )?;
                let _ = self.free(&sum_sq);
                let _ = self.free(&eps);
                sum_sq = adjusted;
            }
            let norms =
                self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, &sum_sq)?;
            let _ = self.free(&sum_sq);
            let normalized = self.binary_op_exec(
                crate::backend::wgpu::types::BinaryOpCode::Div,
                &product,
                &norms,
            )?;
            let _ = self.free(&product);
            let _ = self.free(&norms);

            let mut reused = false;
            let rhs_shape_match = rhs_entry.shape == normalized.shape;
            let rhs_transposed = runmat_accelerate_api::handle_transpose_info(rhs).is_some();
            let rhs_ref_count = Arc::strong_count(&rhs_entry.buffer);
            if rhs_shape_match && !rhs_transposed && rhs_entry.len > 0 && rhs_ref_count <= 2 {
                if let Ok(normalized_entry) = self.get_entry(&normalized) {
                    let bytes = (rhs_entry.len as u64) * self.element_size as u64;
                    if bytes > 0 {
                        let mut encoder = self.device_ref().create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("runmat-power-step-copy"),
                            },
                        );
                        encoder.copy_buffer_to_buffer(
                            normalized_entry.buffer.as_ref(),
                            0,
                            rhs_entry.buffer.as_ref(),
                            0,
                            bytes,
                        );
                        self.submit(encoder);
                    }
                    let _ = self.free(&normalized);
                    self.mark_buffer_usage(rhs, BufferUsageClass::FusionOut);
                    log::debug!(
                        "matmul_power_step: reused rhs buffer {} for normalized output (len={})",
                        rhs.buffer_id,
                        rhs_entry.len
                    );
                    reused = true;
                }
            }

            if reused {
                Ok(rhs.clone())
            } else {
                log::debug!(
                "matmul_power_step: fallback reuse (shape_match={} transpose={} len={} ref_count={})",
                rhs_shape_match,
                rhs_transposed,
                rhs_entry.len,
                rhs_ref_count
            );
                Ok(normalized)
            }
        })
    }
    fn covariance<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        second: Option<&'a GpuTensorHandle>,
        weights: Option<&'a GpuTensorHandle>,
        options: &'a CovarianceOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if options.rows != CovRows::All {
                return Err(anyhow!(
                    "covariance: rows option {:?} not supported by WGPU provider",
                    options.rows
                ));
            }
            if options.has_weight_vector || weights.is_some() {
                return Err(anyhow!(
                    "covariance: weight vectors are not supported by WGPU provider"
                ));
            }

            let combined = if let Some(rhs) = second {
                let left_entry = self.get_entry(matrix)?;
                let right_entry = self.get_entry(rhs)?;

                let rows_left = match left_entry.shape.len() {
                    0 => 1usize,
                    1 => left_entry.shape[0],
                    2 => left_entry.shape[0],
                    _ => {
                        return Err(anyhow!(
                            "covariance: inputs must be 2-D matrices or vectors (got shape {:?})",
                            left_entry.shape
                        ))
                    }
                };
                let rows_right = match right_entry.shape.len() {
                    0 => 1usize,
                    1 => right_entry.shape[0],
                    2 => right_entry.shape[0],
                    _ => {
                        return Err(anyhow!(
                            "covariance: inputs must be 2-D matrices or vectors (got shape {:?})",
                            right_entry.shape
                        ))
                    }
                };

                ensure!(
                    rows_left == rows_right,
                    "covariance: inputs must have the same number of rows (got {} and {})",
                    rows_left,
                    rows_right
                );

                let cat_inputs = vec![matrix.clone(), rhs.clone()];
                Some(self.cat_exec(2, &cat_inputs)?)
            } else {
                None
            };

            let result = {
                let source = combined.as_ref().unwrap_or(matrix);
                self.covariance_exec(source, options).await
            };

            if let Some(handle) = combined {
                let _ = self.free(&handle);
            }

            result
        })
    }
    fn corrcoef<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        options: &'a CorrcoefOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.corrcoef_exec(matrix, options).await })
    }
    fn linsolve<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        options: &'a ProviderLinsolveOptions,
    ) -> AccelProviderFuture<'a, ProviderLinsolveResult> {
        Box::pin(async move {
            if let Some(result) = self.try_linsolve_device(lhs, rhs, options).await? {
                return Ok(result);
            }
            let start = Instant::now();
            let HostTensorOwned {
                data: lhs_data,
                shape: lhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, lhs).await?;
            let HostTensorOwned {
                data: rhs_data,
                shape: rhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, rhs).await?;

            let lhs_tensor =
                Tensor::new(lhs_data, lhs_shape).map_err(|e| anyhow!("linsolve: {e}"))?;
            let rhs_tensor =
                Tensor::new(rhs_data, rhs_shape).map_err(|e| anyhow!("linsolve: {e}"))?;

            let (solution, rcond) =
                linsolve_host_real_for_provider(&lhs_tensor, &rhs_tensor, options)
                    .map_err(|e| anyhow!("{e}"))?;
            self.telemetry.record_linsolve_duration(start.elapsed());
            self.telemetry
                .record_solve_fallback("linsolve:host_reupload");

            let handle = self.upload(&HostTensorView {
                data: &solution.data,
                shape: &solution.shape,
            })?;

            Ok(ProviderLinsolveResult {
                solution: handle,
                reciprocal_condition: rcond,
            })
        })
    }
    fn inv<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        _options: ProviderInvOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, matrix).await?;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("inv: {e}"))?;
            let result = inv_host_real_for_provider(&tensor).map_err(|e| anyhow!("{e}"))?;
            self.upload(&HostTensorView {
                data: &result.data,
                shape: &result.shape,
            })
        })
    }

    fn pinv<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        options: ProviderPinvOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, matrix).await?;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("pinv: {e}"))?;
            let result = pinv_host_real_for_provider(&tensor, options.tolerance)
                .map_err(|e| anyhow!("{e}"))?;
            self.upload(&HostTensorView {
                data: &result.data,
                shape: &result.shape,
            })
        })
    }

    fn cond<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        norm: ProviderCondNorm,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, matrix).await?;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("cond: {e}"))?;
            let cond_value =
                cond_host_real_for_provider(&tensor, norm).map_err(|e| anyhow!("{e}"))?;
            let scalar = [cond_value];
            let shape = [1usize, 1usize];
            self.upload(&HostTensorView {
                data: &scalar,
                shape: &shape,
            })
        })
    }

    fn norm<'a>(
        &'a self,
        tensor: &'a GpuTensorHandle,
        order: ProviderNormOrder,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, tensor).await?;
            let host_tensor = Tensor::new(data, shape).map_err(|e| anyhow!("norm: {e}"))?;
            let value =
                norm_host_real_for_provider(&host_tensor, order).map_err(|e| anyhow!("{e}"))?;
            let scalar = [value];
            let shape = [1usize, 1usize];
            self.upload(&HostTensorView {
                data: &scalar,
                shape: &shape,
            })
        })
    }

    fn rank<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        tolerance: Option<f64>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, matrix).await?;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("rank: {e}"))?;
            let rank =
                rank_host_real_for_provider(&tensor, tolerance).map_err(|e| anyhow!("{e}"))? as f64;
            let scalar = [rank];
            let shape = [1usize, 1usize];
            self.upload(&HostTensorView {
                data: &scalar,
                shape: &shape,
            })
        })
    }

    fn rcond<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let HostTensorOwned { data, shape, .. } =
                <Self as AccelProvider>::download(self, matrix).await?;
            let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("rcond: {e}"))?;
            let estimate = rcond_host_real_for_provider(&tensor).map_err(|e| anyhow!("{e}"))?;
            let scalar = [estimate];
            let shape = [1usize, 1usize];
            self.upload(&HostTensorView {
                data: &scalar,
                shape: &shape,
            })
        })
    }

    fn mldivide<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let start = Instant::now();
            if let Some(result) = self
                .try_linsolve_device(lhs, rhs, &ProviderLinsolveOptions::default())
                .await?
            {
                self.telemetry.record_mldivide_duration(start.elapsed());
                return Ok(result.solution);
            }
            let HostTensorOwned {
                data: lhs_data,
                shape: lhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, lhs).await?;
            let HostTensorOwned {
                data: rhs_data,
                shape: rhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, rhs).await?;

            let lhs_tensor =
                Tensor::new(lhs_data, lhs_shape).map_err(|e| anyhow!("mldivide: {e}"))?;
            let rhs_tensor =
                Tensor::new(rhs_data, rhs_shape).map_err(|e| anyhow!("mldivide: {e}"))?;

            let result = mldivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
                .map_err(|e| anyhow!("{e}"))?;
            self.telemetry.record_mldivide_duration(start.elapsed());
            self.telemetry
                .record_solve_fallback("mldivide:host_reupload");

            let handle = self.upload(&HostTensorView {
                data: &result.data,
                shape: &result.shape,
            })?;
            Ok(handle)
        })
    }

    fn mrdivide<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let start = Instant::now();
            if let Some(result) = self.try_mrdivide_device(lhs, rhs).await? {
                self.telemetry.record_mrdivide_duration(start.elapsed());
                return Ok(result);
            }
            let HostTensorOwned {
                data: lhs_data,
                shape: lhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, lhs).await?;
            let HostTensorOwned {
                data: rhs_data,
                shape: rhs_shape,
                ..
            } = <Self as AccelProvider>::download(self, rhs).await?;

            let lhs_tensor =
                Tensor::new(lhs_data, lhs_shape).map_err(|e| anyhow!("mrdivide: {e}"))?;
            let rhs_tensor =
                Tensor::new(rhs_data, rhs_shape).map_err(|e| anyhow!("mrdivide: {e}"))?;

            let result = mrdivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
                .map_err(|e| anyhow!("{e}"))?;
            self.telemetry.record_mrdivide_duration(start.elapsed());
            self.telemetry
                .record_solve_fallback("mrdivide:host_reupload");

            let handle = self.upload(&HostTensorView {
                data: &result.data,
                shape: &result.shape,
            })?;
            Ok(handle)
        })
    }

    fn dot<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        dim: Option<usize>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.dot_exec(lhs, rhs, dim) })
    }
    fn eig<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        compute_left: bool,
    ) -> AccelProviderFuture<'a, ProviderEigResult> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, handle).await?;
            let tensor = Tensor::new(host.data.clone(), host.shape.clone())
                .map_err(|e| anyhow!("eig: {e}"))?;
            let eval = runmat_runtime::builtins::math::linalg::factor::eig::evaluate(
                Value::Tensor(tensor),
                &[],
                compute_left,
            )
            .await
            .map_err(|err| runtime_flow_to_anyhow("eig", err))?;

            let eigenvalues_tensor = host_tensor_from_value("eig", eval.eigenvalues())?;
            let diagonal_tensor = host_tensor_from_value("eig", eval.diagonal_matrix())?;
            let right_tensor = host_tensor_from_value("eig", eval.right())?;

            let left_value = if compute_left {
                Some(
                    eval.left()
                        .map_err(|err| runtime_flow_to_anyhow("eig", err))?,
                )
            } else {
                None
            };

            let left_tensor = match left_value {
                Some(value) => Some(host_tensor_from_value("eig", value)?),
                None => None,
            };

            let eigenvalues = self.upload(&HostTensorView {
                data: &eigenvalues_tensor.data,
                shape: &eigenvalues_tensor.shape,
            })?;
            let diagonal = self.upload(&HostTensorView {
                data: &diagonal_tensor.data,
                shape: &diagonal_tensor.shape,
            })?;
            let right = self.upload(&HostTensorView {
                data: &right_tensor.data,
                shape: &right_tensor.shape,
            })?;
            let left = match left_tensor {
                Some(tensor) => Some(self.upload(&HostTensorView {
                    data: &tensor.data,
                    shape: &tensor.shape,
                })?),
                None => None,
            };

            if compute_left && left.is_none() {
                return Err(anyhow!(
                    "eig: left eigenvectors are not available for the requested matrix"
                ));
            }

            Ok(ProviderEigResult {
                eigenvalues,
                diagonal,
                right,
                left,
            })
        })
    }

    fn reduce_sum_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Sum)
        })
    }
    fn reduce_nnz_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_dim_sum_mean_exec(
                a,
                dim,
                crate::backend::wgpu::types::DimReduceOp::CountNonZero,
            )
        })
    }
    fn reduce_prod_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Prod)
        })
    }
    fn reduce_mean_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Mean)
        })
    }
    fn reduce_any_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let op = if omit_nan {
                crate::backend::wgpu::types::DimReduceOp::AnyOmit
            } else {
                crate::backend::wgpu::types::DimReduceOp::AnyInclude
            };
            self.reduce_dim_sum_mean_exec(a, dim, op)
        })
    }
    fn reduce_any<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let op = if omit_nan {
                crate::backend::wgpu::types::DimReduceOp::AnyOmit
            } else {
                crate::backend::wgpu::types::DimReduceOp::AnyInclude
            };
            let first = self.reduce_dim_sum_mean_exec(a, 0, op)?;
            match self.reduce_dim_sum_mean_exec(&first, 1, op) {
                Ok(handle) => {
                    let _ = self.free(&first);
                    Ok(handle)
                }
                Err(err) => {
                    let _ = self.free(&first);
                    Err(err)
                }
            }
        })
    }

    fn reduce_all_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let op = if omit_nan {
                crate::backend::wgpu::types::DimReduceOp::AllOmit
            } else {
                crate::backend::wgpu::types::DimReduceOp::AllInclude
            };
            self.reduce_dim_sum_mean_exec(a, dim, op)
        })
    }

    fn reduce_all<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let op = if omit_nan {
                crate::backend::wgpu::types::DimReduceOp::AllOmit
            } else {
                crate::backend::wgpu::types::DimReduceOp::AllInclude
            };
            let total_elems = if a.shape.is_empty() {
                1
            } else {
                product_checked(&a.shape)
                    .ok_or_else(|| anyhow!("reduce_all: tensor size exceeds GPU limits"))?
            };
            if total_elems == 0 {
                return self.fill(&[1usize, 1usize], f64::NAN);
            }
            if a.shape.len() <= 2 {
                let first = self.reduce_dim_sum_mean_exec(a, 0, op)?;
                match self.reduce_dim_sum_mean_exec(&first, 1, op) {
                    Ok(handle) => {
                        let _ = self.free(&first);
                        Ok(handle)
                    }
                    Err(err) => {
                        let _ = self.free(&first);
                        Err(err)
                    }
                }
            } else {
                let original_shape = a.shape.clone();
                let flattened_shape = vec![total_elems, 1usize];
                let flattened = self.reshape(a, &flattened_shape)?;
                let result = self.reduce_dim_sum_mean_exec(&flattened, 0, op);
                let _ = self.reshape(a, &original_shape);
                result
            }
        })
    }

    fn reduce_median<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, a).await?;
            let median = median_from_slice(&host.data);
            let data = [median];
            let shape = [1usize, 1usize];
            self.upload(&HostTensorView {
                data: &data,
                shape: &shape,
            })
        })
    }

    fn reduce_median_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, a).await?;
            if host.shape.len() != 2 {
                return Err(anyhow!("reduce_median_dim: only 2D supported"));
            }
            let rows = host.shape[0];
            let cols = host.shape[1];
            let mut scratch = Vec::<f64>::with_capacity(rows.max(cols));
            let (out, shape) = if dim <= 1 {
                let mut values = vec![f64::NAN; cols];
                for (c, value) in values.iter_mut().enumerate().take(cols) {
                    scratch.clear();
                    let mut saw_nan = false;
                    for r in 0..rows {
                        let v = host.data[r + c * rows];
                        if v.is_nan() {
                            saw_nan = true;
                            scratch.clear();
                            break;
                        }
                        scratch.push(v);
                    }
                    *value = if saw_nan || scratch.is_empty() {
                        f64::NAN
                    } else {
                        compute_median_inplace(&mut scratch)
                    };
                }
                (values, vec![1usize, cols])
            } else {
                let mut values = vec![f64::NAN; rows];
                for (r, value) in values.iter_mut().enumerate().take(rows) {
                    scratch.clear();
                    let mut saw_nan = false;
                    for c in 0..cols {
                        let v = host.data[r + c * rows];
                        if v.is_nan() {
                            saw_nan = true;
                            scratch.clear();
                            break;
                        }
                        scratch.push(v);
                    }
                    *value = if saw_nan || scratch.is_empty() {
                        f64::NAN
                    } else {
                        compute_median_inplace(&mut scratch)
                    };
                }
                (values, vec![rows, 1usize])
            };
            self.upload(&HostTensorView {
                data: &out,
                shape: &shape,
            })
        })
    }

    fn reduce_sum<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Sum)
        })
    }

    fn reduce_nnz<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::CountNonZero)
        })
    }

    fn reduce_prod<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Prod)
        })
    }

    fn reduce_mean<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            // Mean over all elements: compute via single-pass sum then divide by len
            let sum_handle =
                self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Sum)?;
            let total_elems: usize = self.get_entry(a)?.len.max(1);
            let scalar = 1.0 / (total_elems as f64);
            let out = self.scalar_op_exec(
                crate::backend::wgpu::types::ScalarOpCode::Mul,
                &sum_handle,
                scalar,
            )?;
            // Free temporary sum buffer
            let _ = self.free(&sum_handle);
            Ok(out)
        })
    }
    fn reduce_std<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_std_exec(a, normalization, nan_mode) })
    }

    fn reduce_std_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_std_dim_exec(a, dim, normalization, nan_mode) })
    }

    fn reduce_min<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Min)
        })
    }

    fn reduce_max<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Max)
        })
    }

    fn reduce_min_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, ReduceDimResult> {
        Box::pin(async move {
            self.reduce_dim_minmax_exec(a, dim, crate::backend::wgpu::types::DimReduceExtrema::Min)
        })
    }

    fn reduce_max_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, ReduceDimResult> {
        Box::pin(async move {
            self.reduce_dim_minmax_exec(a, dim, crate::backend::wgpu::types::DimReduceExtrema::Max)
        })
    }

    fn find(
        &self,
        a: &GpuTensorHandle,
        limit: Option<usize>,
        direction: FindDirection,
    ) -> Result<ProviderFindResult> {
        self.find_exec(a, limit, direction)
    }
    fn issymmetric(
        &self,
        matrix: &GpuTensorHandle,
        kind: ProviderSymmetryKind,
        tolerance: f64,
    ) -> Result<bool> {
        let entry = self.get_entry(matrix)?;
        let (rows, cols) =
            ensure_symmetry_shape(&entry.shape).map_err(|e| anyhow!("issymmetric: {e}"))?;
        if rows != cols {
            return Ok(false);
        }
        if rows == 0 || cols == 0 {
            return Ok(true);
        }
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("issymmetric: matrix dimensions too large"))?;
        if total > entry.len {
            return Err(anyhow!(
                "issymmetric: shape/product mismatch ({} vs {})",
                total,
                entry.len
            ));
        }
        if total as u64 > u32::MAX as u64 {
            return Err(anyhow!("issymmetric: matrix exceeds GPU limits"));
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(anyhow!(
                "issymmetric: tolerance must be finite and non-negative"
            ));
        }

        let mode = match kind {
            ProviderSymmetryKind::Symmetric => 0u32,
            ProviderSymmetryKind::Skew => 1u32,
        };

        let output_init = [1u32];
        let output_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("runmat-issymmetric-output"),
                contents: cast_slice(&output_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let pipeline = &self.pipelines.symmetry;
        match entry.precision {
            NumericPrecision::F64 => {
                let params = SymmetryParamsF64 {
                    rows: rows as u32,
                    cols: cols as u32,
                    len: total as u32,
                    mode,
                    tolerance,
                    _pad: 0.0,
                };
                let params_buffer = self.uniform_buffer(&params, "runmat-issymmetric-params-f64");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-issymmetric-bind-group-f64"),
                    layout: &pipeline.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
                let groups =
                    crate::backend::wgpu::dispatch::common::dispatch_size(total as u32, 256);
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &pipeline.pipeline,
                    &bind_group,
                    groups,
                );
            }
            NumericPrecision::F32 => {
                let tol32 = tolerance.min(f32::MAX as f64).max(0.0) as f32;
                let params = SymmetryParamsF32 {
                    rows: rows as u32,
                    cols: cols as u32,
                    len: total as u32,
                    mode,
                    tolerance: tol32,
                    _pad: [0.0; 3],
                };
                let params_buffer = self.uniform_buffer(&params, "runmat-issymmetric-params-f32");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-issymmetric-bind-group-f32"),
                    layout: &pipeline.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });
                let groups =
                    crate::backend::wgpu::dispatch::common::dispatch_size(total as u32, 256);
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &pipeline.pipeline,
                    &bind_group,
                    groups,
                );
            }
        }

        let staging_size = std::mem::size_of::<u32>() as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-issymmetric-staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-issymmetric-copy"),
            });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, staging_size);
        self.submit(encoder);

        let bytes = self.map_readback_bytes_sync(staging, staging_size, "issymmetric")?;
        let words: &[u32] = cast_slice(&bytes);
        let flag = words.first().copied().unwrap_or(0);

        Ok(flag != 0)
    }

    fn ishermitian<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        kind: ProviderHermitianKind,
        tolerance: f64,
    ) -> AccelProviderFuture<'a, bool> {
        Box::pin(async move {
            if !tolerance.is_finite() || tolerance < 0.0 {
                return Err(anyhow!(
                    "ishermitian: tolerance must be finite and non-negative"
                ));
            }
            let host = <Self as AccelProvider>::download(self, matrix).await?;
            let skew = matches!(kind, ProviderHermitianKind::Skew);
            ishermitian_host_real_data(&host.shape, &host.data, skew, tolerance)
                .map_err(|e| anyhow!(e))
        })
    }

    fn bandwidth(&self, matrix: &GpuTensorHandle) -> Result<ProviderBandwidth> {
        self.bandwidth_exec(matrix)
    }

    fn sym_rcm<'a>(&'a self, matrix: &'a GpuTensorHandle) -> AccelProviderFuture<'a, Vec<usize>> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, matrix).await?;
            symrcm_host_real_data(&host.shape, &host.data).map_err(|e| anyhow!(e))
        })
    }
    fn read_scalar(&self, h: &GpuTensorHandle, linear_index: usize) -> Result<f64> {
        let entry = self.get_entry(h)?;
        let elem_size = match entry.precision {
            NumericPrecision::F64 => std::mem::size_of::<f64>() as u64,
            NumericPrecision::F32 => std::mem::size_of::<f32>() as u64,
        };
        let total_bytes = (linear_index as u64)
            .checked_mul(elem_size)
            .ok_or_else(|| anyhow!("read_scalar: index overflow"))?;
        if (linear_index + 1) > entry.len {
            return Err(anyhow!(
                "read_scalar: index {} out of bounds (len {})",
                linear_index + 1,
                entry.len
            ));
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-read-scalar-staging"),
            size: elem_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-read-scalar-enc"),
            });
        encoder.copy_buffer_to_buffer(entry.buffer.as_ref(), total_bytes, &staging, 0, elem_size);
        self.submit(encoder);
        let bytes = self.map_readback_bytes_sync(staging, elem_size, "read_scalar")?;
        let value = match entry.precision {
            NumericPrecision::F64 => {
                let words: &[f64] = cast_slice(&bytes);
                words.first().copied().unwrap_or(0.0)
            }
            NumericPrecision::F32 => {
                let words: &[f32] = cast_slice(&bytes);
                words.first().copied().unwrap_or(0.0) as f64
            }
        };
        Ok(value)
    }

    fn fused_elementwise(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
    ) -> Result<GpuTensorHandle> {
        let start = Instant::now();
        let result = self.fused_elementwise_exec(shader, inputs, output_shape, len);
        if result.is_ok() {
            let elapsed = start.elapsed();
            self.telemetry.record_fused_elementwise_duration(elapsed);
            let shape = [
                ("len", len as u64),
                ("inputs", inputs.len() as u64),
                ("rank", output_shape.len() as u64),
            ];
            let wg = crate::backend::wgpu::config::effective_workgroup_size() as u64;
            let tuning = [("wg", wg)];
            self.record_kernel_launch_basic("fused_elementwise", &shape, &tuning);
        }
        result
    }

    fn fused_elementwise_multi(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
        num_outputs: usize,
    ) -> Result<Vec<GpuTensorHandle>> {
        let start = Instant::now();
        let result =
            self.fused_elementwise_multi_exec(shader, inputs, output_shape, len, num_outputs);
        if result.is_ok() {
            let elapsed = start.elapsed();
            self.telemetry.record_fused_elementwise_duration(elapsed);
            let shape = [
                ("len", len as u64),
                ("inputs", inputs.len() as u64),
                ("rank", output_shape.len() as u64),
                ("num_outputs", num_outputs as u64),
            ];
            let wg = crate::backend::wgpu::config::effective_workgroup_size() as u64;
            let tuning = [("wg", wg)];
            self.record_kernel_launch_basic("fused_elementwise_multi", &shape, &tuning);
        }
        result
    }

    fn map_nan_to_zero(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-nan-to-zero-empty");
            return Ok(self.register_existing_buffer(out, entry.shape, 0));
        }
        let shader = match self.precision {
            NumericPrecision::F64 => crate::backend::wgpu::shaders::nan::NAN_TO_ZERO_SHADER_F64,
            NumericPrecision::F32 => crate::backend::wgpu::shaders::nan::NAN_TO_ZERO_SHADER_F32,
        };
        self.fused_elementwise(shader, std::slice::from_ref(a), &entry.shape, len)
    }
    fn not_nan_mask(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-not-nan-mask-empty");
            return Ok(self.register_existing_buffer(out, entry.shape, 0));
        }
        let shader = match self.precision {
            NumericPrecision::F64 => crate::backend::wgpu::shaders::nan::NOT_NAN_MASK_SHADER_F64,
            NumericPrecision::F32 => crate::backend::wgpu::shaders::nan::NOT_NAN_MASK_SHADER_F32,
        };
        self.fused_elementwise(shader, std::slice::from_ref(a), &entry.shape, len)
    }

    fn fused_reduction(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
        flavor: ReductionFlavor,
    ) -> Result<GpuTensorHandle> {
        let start = Instant::now();
        let result = self.fused_reduction_exec(
            shader,
            inputs,
            output_shape,
            reduce_len,
            num_slices,
            workgroup_size,
            flavor,
        );
        if result.is_ok() {
            let elapsed = start.elapsed();
            self.telemetry.record_fused_reduction_duration(elapsed);
            let actual_wg = if workgroup_size == 0 {
                self.default_reduction_workgroup_size()
            } else {
                workgroup_size
            } as u64;
            let flavor_tag = match flavor {
                ReductionFlavor::Sum => 0,
                ReductionFlavor::Mean => 1,
                ReductionFlavor::CustomScale(_) => 2,
            };
            let shape = [
                ("reduce_len", reduce_len as u64),
                ("slices", num_slices as u64),
                ("rank", output_shape.len() as u64),
            ];
            let tuning = [("wg", actual_wg), ("flavor", flavor_tag)];
            self.record_kernel_launch_basic("fused_reduction", &shape, &tuning);
        }
        result
    }
    fn warmup(&self) {
        if std::env::var("RUNMAT_WGPU_SKIP_WARMUP")
            .ok()
            .and_then(|v| {
                let trimmed = v.trim();
                if trimmed.is_empty() {
                    None
                } else if trimmed.eq_ignore_ascii_case("1")
                    || trimmed.eq_ignore_ascii_case("true")
                    || trimmed.eq_ignore_ascii_case("yes")
                {
                    Some(true)
                } else if trimmed.eq_ignore_ascii_case("0")
                    || trimmed.eq_ignore_ascii_case("false")
                    || trimmed.eq_ignore_ascii_case("no")
                {
                    Some(false)
                } else {
                    None
                }
            })
            .unwrap_or(false)
        {
            log::info!("RunMat Accelerate: skipping wgpu warmup (RUNMAT_WGPU_SKIP_WARMUP=1)");
            return;
        }

        let start = Instant::now();
        self.warmup_from_disk();
        // Proactively warm common pipelines used by normalization and reduction chains
        let pl = &self.pipelines;
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.binary.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.binary_broadcast.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.unary.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.scalar.pipeline,
        );
        crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
            self.device_ref(),
            self.queue_ref(),
            &pl.reduce_dim_sum_mean.pipeline,
        );
        crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
            self.device_ref(),
            self.queue_ref(),
            &pl.reduce_nd_mean.pipeline,
        );
        crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
            self.device_ref(),
            self.queue_ref(),
            &pl.reduce_global.pipeline,
        );
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pl.fill.pipeline,
        );

        let ms = start.elapsed().as_millis() as u64;
        self.metrics.set_last_warmup_millis(ms);
    }
    fn fused_cache_counters(&self) -> (u64, u64) {
        self.metrics.counters()
    }

    fn last_warmup_millis(&self) -> Option<u64> {
        Some(self.metrics.last_warmup_millis())
    }

    fn telemetry_snapshot(&self) -> runmat_accelerate_api::ProviderTelemetry {
        let (fusion_hits, fusion_misses) = self.metrics.counters();
        let (bind_hits, bind_misses) = self.bind_group_cache.counters();
        // Build per-layout telemetry by resolving layout pointers to tags
        let mut by_layout: Vec<runmat_accelerate_api::BindGroupLayoutTelemetry> = Vec::new();
        let per = self.bind_group_cache.per_layout_counters();
        if let Ok(tags) = self.bind_group_layout_tags.lock() {
            for (ptr, (h, m)) in per {
                let tag = tags
                    .get(&ptr)
                    .cloned()
                    .unwrap_or_else(|| format!("layout_ptr_{:#x}", ptr));
                by_layout.push(runmat_accelerate_api::BindGroupLayoutTelemetry {
                    tag,
                    hits: h,
                    misses: m,
                });
            }
        }
        self.telemetry.snapshot(
            fusion_hits,
            fusion_misses,
            bind_hits,
            bind_misses,
            Some(by_layout),
        )
    }

    fn reset_telemetry(&self) {
        self.telemetry.reset();
        self.metrics.reset();
        self.bind_group_cache.reset_counters();
    }

    fn default_reduction_workgroup_size(&self) -> u32 {
        self.reduction_workgroup_size_default
    }

    fn two_pass_threshold(&self) -> usize {
        self.reduction_two_pass_threshold
    }

    fn reduction_two_pass_mode(&self) -> ReductionTwoPassMode {
        self.reduction_two_pass_mode
    }

    fn scatter_column(
        &self,
        matrix: &GpuTensorHandle,
        col_index: usize,
        values: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.scatter_column_exec(matrix, col_index, values)
    }
    fn scatter_row(
        &self,
        matrix: &GpuTensorHandle,
        row_index: usize,
        values: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.scatter_row_exec(matrix, row_index, values)
    }

    fn sub2ind(
        &self,
        dims: &[usize],
        strides: &[usize],
        inputs: &[&GpuTensorHandle],
        scalar_mask: &[bool],
        len: usize,
        output_shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        self.sub2ind_exec(dims, strides, inputs, scalar_mask, len, output_shape)
    }

    fn supports_ind2sub(&self) -> bool {
        true
    }

    fn ind2sub(
        &self,
        dims: &[usize],
        strides: &[usize],
        indices: &GpuTensorHandle,
        total: usize,
        len: usize,
        output_shape: &[usize],
    ) -> Result<Vec<GpuTensorHandle>> {
        self.ind2sub_exec(dims, strides, indices, total, len, output_shape)
    }

    fn upload(&self, host: &HostTensorView) -> Result<GpuTensorHandle> {
        let _span = info_span!(
            "gpu.transfer.upload",
            shape = ?host.shape,
            len = host.data.len()
        )
        .entered();
        let len = host.data.len();
        let shape = host.shape.to_vec();
        let bytes = (len as u64).saturating_mul(self.element_size as u64);
        if bytes > self.adapter_limits.max_buffer_size {
            return Err(gpu_per_buffer_limit_error(
                "upload",
                bytes,
                self.adapter_limits.max_buffer_size,
            ));
        }
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
        self.telemetry.record_upload_bytes(bytes);
        Ok(self.register_existing_buffer(buffer, shape, len))
    }
    fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let span = info_span!(
                "gpu.transfer.download",
                shape = ?h.shape,
                buffer_id = h.buffer_id
            );
            let entry = {
                let _guard = span.enter();
                log::trace!("wgpu download id={} shape={:?}", h.buffer_id, &h.shape);
                self.get_entry(h)?
            };
            if let Some(last) = entry.last_submission_id {
                log::trace!(
                    "wgpu download id={} last_submission_id={}",
                    h.buffer_id,
                    last
                );
            } else {
                log::trace!("wgpu download id={} last_submission_id=<none>", h.buffer_id);
            }
            if entry.len == 0 {
                return Ok(HostTensorOwned {
                    data: Vec::new(),
                    shape: h.shape.clone(),
                    storage: runmat_accelerate_api::handle_storage(h),
                });
            }

            let size_bytes = (entry.len * self.element_size) as u64;

            // Shared post-map readback logic: decode mapped bytes, unmap, record telemetry,
            // apply transpose metadata, and return host tensor.
            let finish_readback =
                |staging: wgpu::Buffer, size_bytes: u64| -> Result<HostTensorOwned> {
                    let slice = staging.slice(..);
                    let data = slice.get_mapped_range();
                    log::trace!(
                        "wgpu download copying data id={} len={} bytes={}",
                        h.buffer_id,
                        entry.len,
                        size_bytes
                    );

                    let mut out = vec![0.0f64; entry.len];
                    match entry.precision {
                        NumericPrecision::F64 => out.copy_from_slice(cast_slice(&data)),
                        NumericPrecision::F32 => {
                            let f32_slice: &[f32] = cast_slice(&data);
                            for (dst, src) in out.iter_mut().zip(f32_slice.iter()) {
                                *dst = *src as f64;
                            }
                        }
                    }
                    drop(data);
                    staging.unmap();
                    log::trace!("wgpu download finished copy id={}", h.buffer_id);
                    self.telemetry.record_download_bytes(size_bytes);

                    let mut shape = h.shape.clone();
                    if let Some(info) = runmat_accelerate_api::handle_transpose_info(h) {
                        let base_rows = info.base_rows;
                        let base_cols = info.base_cols;
                        if base_rows * base_cols != out.len() {
                            return Err(anyhow!(
                                "download: transpose metadata mismatch for buffer {}",
                                h.buffer_id
                            ));
                        }
                        if shape.len() == 2 {
                            let rows_t = base_cols;
                            let cols_t = base_rows;
                            let mut transposed = vec![0.0f64; out.len()];
                            for col in 0..base_cols {
                                for row in 0..base_rows {
                                    let src_idx = row + col * base_rows;
                                    let dst_idx = col + row * base_cols;
                                    transposed[dst_idx] = out[src_idx];
                                }
                            }
                            out = transposed;
                            shape[0] = rows_t;
                            shape[1] = cols_t;
                        }
                    }

                    log::trace!(
                        "wgpu download complete id={} final_shape={:?}",
                        h.buffer_id,
                        shape
                    );

                    Ok(HostTensorOwned {
                        data: out,
                        shape,
                        storage: runmat_accelerate_api::handle_storage(h),
                    })
                };

            log::trace!(
                "wgpu download creating staging buffer id={} bytes={}",
                h.buffer_id,
                size_bytes
            );
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
            let (tx, rx) = oneshot::channel();

            let map_buffer_id = h.buffer_id;
            slice.map_async(wgpu::MapMode::Read, move |res| {
                log::trace!(
                    "wgpu download map_async callback id={} status={:?}",
                    map_buffer_id,
                    res
                );
                let _ = tx.send(res);
            });
            log::trace!(
                "wgpu download awaiting map_async completion id={} bytes={}",
                h.buffer_id,
                size_bytes
            );
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.device.poll(wgpu::Maintain::Wait);
            }
            let map_result = rx
                .await
                .map_err(|_| anyhow!("map_async callback dropped for buffer {}", h.buffer_id))?;

            log::trace!("wgpu download map_async success id={}", h.buffer_id);
            map_result.map_err(|e: wgpu::BufferAsyncError| anyhow!(e))?;
            finish_readback(staging, size_bytes)
        })
    }
    fn free(&self, h: &GpuTensorHandle) -> Result<()> {
        // Remove from handle table and return buffer to pool for reuse
        log::trace!("wgpu free id={}", h.buffer_id);
        let entry = self
            .buffers
            .lock()
            .expect("buffer mutex poisoned")
            .remove(&h.buffer_id);
        if let Some(entry) = entry {
            if entry.len > 0 {
                let size_bytes = (entry.len as u64).saturating_mul(self.element_size as u64);
                let poolable_by_size = self.buffer_residency_max_poolable_bytes > 0
                    && size_bytes <= self.buffer_residency_max_poolable_bytes;
                let buffer_ptr = entry.buffer.as_ref() as *const wgpu::Buffer as usize;
                // Always invalidate bind-group cache first so cache-held references
                // do not pin dropped buffers across loop iterations.
                self.bind_group_cache.invalidate_buffer(buffer_ptr);
                let strong_count = Arc::strong_count(&entry.buffer);
                if poolable_by_size && strong_count == 1 {
                    self.buffer_residency
                        .release(entry.usage, entry.len, entry.buffer.clone());
                } else {
                    log::trace!(
                        "buffer_residency: not pooling buffer id={} len={} bytes={} strong_count={} poolable_by_size={}",
                        h.buffer_id,
                        entry.len,
                        size_bytes,
                        strong_count,
                        poolable_by_size
                    );
                }
            }
        }
        self.kernel_resources.clear_matmul_source(h.buffer_id);
        runmat_accelerate_api::clear_handle_logical(h);
        runmat_accelerate_api::clear_handle_storage(h);
        runmat_accelerate_api::clear_handle_transpose(h);
        Ok(())
    }

    fn device_info(&self) -> String {
        format!(
            "{} ({:?})",
            self.adapter_info.name, self.adapter_info.backend
        )
    }

    fn device_info_struct(&self) -> ApiDeviceInfo {
        let backend = format!("{:?}", self.adapter_info.backend).to_ascii_lowercase();
        let memory_bytes = if self.adapter_limits.max_buffer_size > 0 {
            Some(self.adapter_limits.max_buffer_size)
        } else {
            None
        };
        ApiDeviceInfo {
            device_id: self.runtime_device_id,
            name: self.adapter_info.name.clone(),
            vendor: canonical_vendor_name(&self.adapter_info),
            memory_bytes,
            backend: Some(backend),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn install_device_error_handlers(device: &wgpu::Device) {
    device.on_uncaptured_error(Box::new(|error| {
        error!("WGPU uncaptured error: {:?}", error);
    }));
    device.set_device_lost_callback(|reason, message| {
        error!("WGPU device lost: reason={:?}, message={}", reason, message);
    });
}

#[cfg(target_arch = "wasm32")]
fn install_device_error_handlers(device: &wgpu::Device) {
    device.on_uncaptured_error(Box::new(|error| {
        error!("WGPU uncaptured error (wasm): {:?}", error);
    }));
    debug!("wgpu set_device_lost_callback not supported on wasm targets");
}
