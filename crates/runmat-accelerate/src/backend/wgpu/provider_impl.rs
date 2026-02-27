use anyhow::{anyhow, ensure, Result};
use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use futures::channel::oneshot;
use log::{debug, error, info, warn};
use num_complex::Complex;
use once_cell::sync::OnceCell;
#[cfg(not(target_arch = "wasm32"))]
use pollster::block_on;
use rand::seq::SliceRandom;
use runmat_accelerate_api::{
    AccelContextHandle, AccelContextKind, AccelDownloadFuture, AccelProvider, AccelProviderFuture,
    ApiDeviceInfo, CorrcoefNormalization, CorrcoefOptions, CorrcoefRows, CovNormalization, CovRows,
    CovarianceOptions, FindDirection, FspecialRequest, GpuTensorHandle, HostTensorOwned,
    HostTensorView, ImfilterOptions, ImfilterPadding, IsMemberOptions, IsMemberResult,
    MeshgridAxisView, PagefunOp, PagefunRequest, ProviderBandwidth, ProviderCholResult,
    ProviderCondNorm, ProviderConv1dOptions, ProviderConvMode, ProviderConvOrientation,
    ProviderCummaxResult, ProviderCumminResult, ProviderEigResult, ProviderFindResult,
    ProviderHermitianKind, ProviderIirFilterOptions, ProviderIirFilterResult, ProviderInvOptions,
    ProviderLinsolveOptions, ProviderLinsolveResult, ProviderLuResult, ProviderMeshgridResult,
    ProviderNanMode, ProviderNormOrder, ProviderPinvOptions, ProviderPolyderQuotient,
    ProviderPolyfitResult, ProviderPolyvalOptions, ProviderPrecision, ProviderQrOptions,
    ProviderQrPivot, ProviderQrPowerIterResult, ProviderQrResult, ProviderScanDirection,
    ProviderStdNormalization, ProviderSymmetryKind, ReduceDimResult, ReductionFlavor,
    ReductionTwoPassMode, SetdiffOptions, SetdiffResult, SortComparison, SortOrder, SortResult,
    SortRowsColumnSpec, UnionOptions, UnionResult, UniqueOptions, UniqueResult, WgpuBufferRef,
    WgpuContextHandle,
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
use runmat_runtime::builtins::math::reduction::compute_median_inplace;
use runmat_runtime::RuntimeError;
use runmat_time::Instant;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::info_span;
use wgpu::util::DeviceExt;

use crate::backend::wgpu::autotune::AutotuneController;
use crate::backend::wgpu::cache::{
    bind_group::BindGroupCache, key as cache_key, persist as cache_persist,
};
use crate::backend::wgpu::config::{
    self, DEFAULT_REDUCTION_WG, DEFAULT_TWO_PASS_THRESHOLD, MATMUL_TILE, WORKGROUP_SIZE,
};
use crate::backend::wgpu::params::{
    BandwidthParams, Conv1dParams, CummaxParams, CumminParams, CumprodParams, CumsumParams,
    DiffParams, FilterParams, ImageNormalizeUniforms, LinearGatherParams, LinearScatterParams,
    QrPowerIterParams, SymmetryParamsF32, SymmetryParamsF64, SyrkParams, IMAGE_NORMALIZE_FLAG_BIAS,
    IMAGE_NORMALIZE_FLAG_GAIN, IMAGE_NORMALIZE_FLAG_GAMMA, SYRK_FLAG_ACCUMULATE,
    SYRK_FLAG_FILL_BOTH,
};
use crate::backend::wgpu::pipelines::{ImageNormalizeBootstrap, WgpuPipelines};
use crate::backend::wgpu::residency::{BufferResidency, BufferUsageClass};
use crate::backend::wgpu::resources::{KernelResourceRegistry, UniformBufferKey};
use crate::backend::wgpu::shaders::image_normalize::{
    IMAGE_NORMALIZE_SHADER_F32, IMAGE_NORMALIZE_SHADER_F64,
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

const LOGICAL_AND_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let lhs_true = !(lhs == 0.0);
    let rhs_true = !(rhs == 0.0);
    let cond = lhs_true && rhs_true;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

const LOGICAL_AND_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let lhs_true = !(lhs == f64(0.0));
    let rhs_true = !(rhs == f64(0.0));
    let cond = lhs_true && rhs_true;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

const ELEM_EQ_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs == rhs;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

const ELEM_EQ_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs == rhs;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

const ELEM_NE_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = !(lhs == rhs);
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

const ELEM_NE_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = !(lhs == rhs);
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

const ELEM_LT_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs < rhs;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

const ELEM_LT_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs < rhs;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

const ELEM_LE_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs <= rhs;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

const ELEM_LE_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs <= rhs;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

const ELEM_GT_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs > rhs;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

const ELEM_GT_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs > rhs;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

const ELEM_GE_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs >= rhs;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

const ELEM_GE_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs >= rhs;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;
const LOGICAL_OR_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let lhs_true = !(lhs == 0.0);
    let rhs_true = !(rhs == 0.0);
    let cond = lhs_true || rhs_true;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

const LOGICAL_OR_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let lhs_true = !(lhs == f64(0.0));
    let rhs_true = !(rhs == f64(0.0));
    let cond = lhs_true || rhs_true;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

const LOGICAL_XOR_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let lhs_true = !(lhs == 0.0);
    let rhs_true = !(rhs == 0.0);
    let cond = lhs_true != rhs_true;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

const LOGICAL_XOR_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let lhs_true = !(lhs == f64(0.0));
    let rhs_true = !(rhs == f64(0.0));
    let cond = lhs_true != rhs_true;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

const LOGICAL_NOT_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = (value != 0.0);
    output.data[idx] = select(1.0, 0.0, cond);
}
"#;

const LOGICAL_NOT_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = (value != f64(0.0));
    output.data[idx] = select(f64(1.0), f64(0.0), cond);
}
"#;

const LOGICAL_ISNAN_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
fn isNan(x: f32) -> bool { return x != x; }

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = isNan(value);
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

const LOGICAL_ISNAN_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isNan(x: f64) -> bool { return x != x; }

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = isNan(value);
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;
const LOGICAL_ISINF_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
fn isInf(x: f32) -> bool { return (x == x) && !(abs(x) < 3.4028234663852886e38); }
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = isInf(value);
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

const LOGICAL_ISINF_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isInf(x: f64) -> bool { return (x == x) && !(abs(x) < f64(1.7976931348623157e308)); }

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = isInf(value);
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

const LOGICAL_ISFINITE_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isFinite(x: f32) -> bool { return (x == x) && (abs(x) < 3.4028234663852886e38); }

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = isFinite(value);
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

const LOGICAL_ISFINITE_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isFinite(x: f64) -> bool { return (x == x) && (abs(x) < f64(1.7976931348623157e308)); }

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = isFinite(value);
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;
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

fn fft_trim_trailing_ones(shape: &mut Vec<usize>, minimum_rank: usize) {
    while shape.len() > minimum_rank && shape.last() == Some(&1) {
        shape.pop();
    }
    *shape = normalize_scalar_shape(shape);
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
            return Err(anyhow!(
                "{}: requested {} bytes exceeds device max {}",
                label,
                size_bytes,
                self.adapter_limits.max_buffer_size
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

    fn register_existing_buffer_with_usage(
        &self,
        buffer: Arc<wgpu::Buffer>,
        shape: Vec<usize>,
        len: usize,
        usage: BufferUsageClass,
    ) -> GpuTensorHandle {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let entry = BufferEntry {
            buffer,
            len,
            shape: shape.clone(),
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

    pub(crate) fn fused_elementwise_exec(
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
            .map(|h| self.get_entry(h))
            .collect::<Result<Vec<_>>>()?;
        let (output_buffer, _) = self.create_storage_buffer_for_usage(
            BufferUsageClass::FusionOut,
            len,
            "runmat-fusion-output",
        );
        let bind_group_layout = self.cached_fusion_bind_group_layout(inputs.len());
        let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-fusion-pipeline-layout",
            bind_group_layout.as_ref(),
        );
        let layout_tag = {
            let mut tag = String::from("runmat-fusion-layout-");
            tag.push_str(&inputs.len().to_string());
            tag
        };
        let shader_hash = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            &layout_tag,
            Some(crate::backend::wgpu::config::effective_workgroup_size()),
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-fusion-shader",
            shader,
        );
        let pipeline = self.get_or_create_pipeline(
            shader_hash,
            &pipeline_layout,
            &module,
            "runmat-fusion-pipeline",
            Some(shader.as_bytes()),
            Some(&layout_tag),
            Some(crate::backend::wgpu::config::effective_workgroup_size()),
        );
        // Warmup once
        crate::backend::wgpu::dispatch::elementwise::warmup_noop(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
        );
        self.device_ref().poll(wgpu::Maintain::Poll);

        let broadcast_mode = shader.contains("out_shape") || shader.contains("a_shape");

        struct BroadcastUniformState {
            buffer: Arc<wgpu::Buffer>,
            template: Vec<u8>,
        }

        impl BroadcastUniformState {
            fn update(&mut self, queue: &wgpu::Queue, len: u32, offset: u32) {
                self.template[..4].copy_from_slice(&len.to_ne_bytes());
                self.template[4..8].copy_from_slice(&offset.to_ne_bytes());
                queue.write_buffer(self.buffer.as_ref(), 0, &self.template);
            }
        }

        enum FusionUniformState {
            Broadcast(BroadcastUniformState),
            Simple { buffer: Arc<wgpu::Buffer> },
        }

        let uniform_state = if broadcast_mode {
            let rank = output_shape.len();
            let max_rank = crate::backend::wgpu::params::BCAST_MAX_RANK;
            let mut bytes: Vec<u8> = Vec::with_capacity(4 * 4 + (max_rank * 4 * 4));
            let write_u32 = |buf: &mut Vec<u8>, v: u32| buf.extend_from_slice(&v.to_ne_bytes());
            write_u32(&mut bytes, 0); // len placeholder
            write_u32(&mut bytes, 0); // offset placeholder
            write_u32(&mut bytes, rank as u32);
            write_u32(&mut bytes, 0);
            let write_packed_array = |buf: &mut Vec<u8>, vals: &[u32]| {
                for &val in vals.iter() {
                    write_u32(buf, val);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                }
                for _ in vals.len()..max_rank {
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                    write_u32(buf, 0);
                }
            };
            let out_shape_u32: Vec<u32> = output_shape.iter().map(|&d| d as u32).collect();
            write_packed_array(&mut bytes, &out_shape_u32);
            for entry in &entries {
                let mut shape = entry.shape.clone();
                if shape.len() < rank {
                    let pad = rank - shape.len();
                    let mut v = vec![1usize; pad];
                    v.extend_from_slice(&shape);
                    shape = v;
                }
                let shape_u32: Vec<u32> = shape.iter().map(|&d| d as u32).collect();
                write_packed_array(&mut bytes, &shape_u32);
                let mut strides: Vec<u32> = vec![0; rank];
                let mut s: u64 = 1;
                for i in 0..rank {
                    strides[i] = if shape[i] == 1 { 0 } else { s as u32 };
                    s = s.saturating_mul(shape[i] as u64);
                }
                write_packed_array(&mut bytes, &strides);
            }
            let buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-fusion-params"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.queue.write_buffer(buffer.as_ref(), 0, &bytes);
            let uniform_buffer = buffer.clone();
            let state = BroadcastUniformState {
                buffer,
                template: bytes,
            };
            (FusionUniformState::Broadcast(state), uniform_buffer)
        } else {
            let buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-fusion-params"),
                size: std::mem::size_of::<crate::backend::wgpu::params::FusionParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            (
                FusionUniformState::Simple {
                    buffer: buffer.clone(),
                },
                buffer,
            )
        };

        let (mut uniform_state, uniform_buffer) = uniform_state;

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
        bind_entries.push(wgpu::BindGroupEntry {
            binding: (inputs.len() + 1) as u32,
            resource: uniform_buffer.as_ref().as_entire_binding(),
        });

        let bind_group =
            self.bind_group_cache
                .get_or_create(bind_group_layout.as_ref(), &bind_entries, || {
                    Arc::new(
                        self.device_ref()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-fusion-bind-group"),
                                layout: bind_group_layout.as_ref(),
                                entries: &bind_entries,
                            }),
                    )
                });

        let output_ptr = output_buffer.as_ref() as *const wgpu::Buffer as usize;
        let input_ids: Vec<u64> = inputs.iter().map(|h| h.buffer_id).collect();
        log::trace!(
            "fusion elementwise begin len={} out_ptr=0x{:x} inputs={:?}",
            len,
            output_ptr,
            input_ids
        );
        // Dispatch in chunks to satisfy 65535 limit
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset_elems = 0usize;
        let mut chunk_index = 0usize;
        let mut last_submission_id = None;
        while offset_elems < len {
            let remaining = len - offset_elems;
            let chunk_len = remaining.min(chunk_capacity);

            log::trace!(
                "fusion elementwise chunk start out_ptr=0x{:x} chunk_len={} offset={} chunk_index={}",
                output_ptr,
                chunk_len,
                offset_elems,
                chunk_index
            );
            match &mut uniform_state {
                FusionUniformState::Broadcast(state) => {
                    state.update(self.queue_ref(), chunk_len as u32, offset_elems as u32)
                }
                FusionUniformState::Simple { buffer } => {
                    let params = crate::backend::wgpu::params::FusionParams {
                        len: chunk_len as u32,
                        offset: offset_elems as u32,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    self.queue
                        .write_buffer(buffer.as_ref(), 0, bytes_of(&params));
                }
            }

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::effective_workgroup_size(),
            );
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-fusion-elementwise-encoder"),
                    });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-fusion-elementwise-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, bind_group.as_ref(), &[]);
                if workgroups > 0 {
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
            }
            let submission_id = self.submit(enc);
            last_submission_id = Some(submission_id);
            log::trace!(
                "fusion elementwise chunk complete out_ptr=0x{:x} chunk_len={} offset={} submission_id={}",
                output_ptr,
                chunk_len,
                offset_elems,
                submission_id
            );
            offset_elems += chunk_len;
            chunk_index += 1;
        }
        let handle = self.register_existing_buffer_with_usage(
            output_buffer,
            output_shape.to_vec(),
            len,
            BufferUsageClass::FusionOut,
        );
        log::trace!(
            "fusion elementwise complete buffer_id={} out_ptr=0x{:x} len={} chunks={} last_submission_id={:?}",
            handle.buffer_id,
            output_ptr,
            len,
            chunk_index,
            last_submission_id
        );
        if let Some(submission_id) = last_submission_id {
            self.record_buffer_submission(handle.buffer_id, submission_id);
        }
        Ok(handle)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fused_reduction_exec(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
        flavor: ReductionFlavor,
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

        let workgroup_size = if workgroup_size == 0 {
            self.default_reduction_workgroup_size()
        } else {
            workgroup_size
        };
        let tuning_key = ReductionAutotuneKey::new(self.precision, num_slices, reduce_len);
        if self.reduction_autotune.is_enabled() {
            if let Some(tuning) = self.reduction_autotune.get(&tuning_key) {
                return self.execute_reduction_with_strategy(
                    &tuning,
                    inputs,
                    output_shape,
                    shader,
                    reduce_len,
                    num_slices,
                    workgroup_size,
                    flavor,
                );
            }
            if let Some(handle) = self.maybe_autotune_reduction(
                &tuning_key,
                inputs,
                output_shape,
                shader,
                reduce_len,
                num_slices,
                workgroup_size,
                flavor,
            )? {
                return Ok(handle);
            }
            if let Some(tuning) = self.reduction_autotune.get(&tuning_key) {
                return self.execute_reduction_with_strategy(
                    &tuning,
                    inputs,
                    output_shape,
                    shader,
                    reduce_len,
                    num_slices,
                    workgroup_size,
                    flavor,
                );
            }
        }

        let fallback_tuning =
            self.heuristic_reduction_tuning(reduce_len, num_slices, workgroup_size);
        self.execute_reduction_with_strategy(
            &fallback_tuning,
            inputs,
            output_shape,
            shader,
            reduce_len,
            num_slices,
            workgroup_size,
            flavor,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_reduction_with_strategy(
        &self,
        tuning: &ReductionTuning,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        shader: &str,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
        flavor: ReductionFlavor,
    ) -> Result<GpuTensorHandle> {
        let mut prepared =
            self.prepare_reduction_tuning(tuning, reduce_len, num_slices, workgroup_size);
        if prepared.is_none()
            && !matches!(tuning.mode, ReductionMode::SinglePass)
            && self.can_use_single_pass(num_slices)
        {
            prepared = Some(ReductionTuning {
                mode: ReductionMode::SinglePass,
            });
        }
        let prepared = prepared.ok_or_else(|| {
            anyhow!(
                "fused_reduction: unable to schedule tuning {:?} for slices={} reduce_len={}",
                tuning.mode,
                num_slices,
                reduce_len
            )
        })?;

        match prepared.mode {
            ReductionMode::SinglePass => self.run_reduction_single_pass(
                inputs,
                output_shape,
                shader,
                reduce_len,
                num_slices,
                workgroup_size,
            ),
            ReductionMode::TwoPass { chunk_rows } => self.run_reduction_two_pass(
                inputs,
                output_shape,
                shader,
                reduce_len,
                num_slices,
                workgroup_size,
                chunk_rows,
                flavor,
            ),
        }
    }

    fn run_reduction_single_pass(
        &self,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        shader: &str,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
    ) -> Result<GpuTensorHandle> {
        let layout_tag = &format!("runmat-reduction-layout-{}", inputs.len());
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-fused-reduction-module",
            shader,
        );
        let bgl = self
            .cached_bind_group_layout_for_tag(layout_tag)
            .expect("reduction bgl");
        let pl = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-reduction-pl",
            bgl.as_ref(),
        );
        if std::env::var("RUNMAT_DEBUG_PIPELINE_ONLY").is_ok() {
            let out_len = num_slices.max(1);
            let out_buffer = self.create_storage_buffer_checked(out_len, "runmat-reduction-out")?;
            return Ok(self.register_existing_buffer(out_buffer, output_shape.to_vec(), out_len));
        }
        let key =
            self.compute_pipeline_hash_bytes(shader.as_bytes(), layout_tag, Some(workgroup_size));
        let pipeline = self.get_or_create_pipeline(
            key,
            &pl,
            &module,
            "runmat-reduction-pipeline",
            Some(shader.as_bytes()),
            Some(layout_tag),
            Some(workgroup_size),
        );
        crate::backend::wgpu::dispatch::reduction::warmup_noop_single(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
        );
        self.device_ref().poll(wgpu::Maintain::Poll);
        self.device_ref().poll(wgpu::Maintain::Poll);
        let flush_enc = self
            .device_ref()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-flush-single-pass-gap"),
            });
        self.submit(flush_enc);
        let out_len = num_slices.max(1);
        let mut out_buffer = self.create_storage_buffer_checked_with_usage(
            out_len,
            "runmat-reduction-out",
            BufferUsageClass::FusionOut,
        )?;
        {
            let out_ptr = out_buffer.as_ref() as *const wgpu::Buffer as usize;
            let mut alias = false;
            for h in inputs.iter() {
                let ptr = self.get_entry(h)?.buffer.as_ref() as *const wgpu::Buffer as usize;
                if ptr == out_ptr {
                    alias = true;
                    break;
                }
            }
            if alias {
                out_buffer = self.create_storage_buffer_checked_with_usage(
                    out_len,
                    "runmat-reduction-out-unique",
                    BufferUsageClass::FusionOut,
                )?;
            }
        }
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
        let params_buffer = self.kernel_resources.uniform_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::UniformBufferKey::ReductionParams,
            std::mem::size_of::<Params>() as u64,
            "runmat-reduction-params",
        );
        self.queue
            .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
        let mut entries_vec: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(inputs.len() + 2);
        let mut input_bufs: Vec<(Arc<wgpu::Buffer>, u64)> = Vec::with_capacity(inputs.len());
        for h in inputs.iter() {
            let e = self.get_entry(h)?;
            let bytes = (e.len * self.element_size) as u64;
            input_bufs.push((e.buffer.clone(), bytes));
        }
        let snapshot_inputs = std::env::var("RUNMAT_FUSED_SNAPSHOT_INPUTS").is_ok();
        let mut bind_input_bufs: Vec<Arc<wgpu::Buffer>> = Vec::with_capacity(inputs.len());
        if snapshot_inputs {
            for (buf, bytes) in input_bufs.iter() {
                let snap = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-fused-input-snapshot"),
                    size: *bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                let mut enc =
                    self.device_ref()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("runmat-fused-input-snapshot-copy"),
                        });
                enc.copy_buffer_to_buffer(buf.as_ref(), 0, snap.as_ref(), 0, *bytes);
                self.submit(enc);
                bind_input_bufs.push(snap);
            }
        } else {
            for (buf, _bytes) in input_bufs.iter() {
                bind_input_bufs.push(buf.clone());
            }
        }
        for (i, buf_arc) in bind_input_bufs.iter().enumerate() {
            entries_vec.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf_arc.as_ref().as_entire_binding(),
            });
        }
        entries_vec.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: out_buffer.as_ref().as_entire_binding(),
        });
        entries_vec.push(wgpu::BindGroupEntry {
            binding: (inputs.len() + 1) as u32,
            resource: params_buffer.as_ref().as_entire_binding(),
        });
        {
            let out_ptr = out_buffer.as_ref() as *const wgpu::Buffer as usize;
            let mut alias_found = false;
            for b in bind_input_bufs.iter() {
                let in_ptr = b.as_ref() as *const wgpu::Buffer as usize;
                if in_ptr == out_ptr {
                    alias_found = true;
                    break;
                }
            }
            if alias_found {
                return Err(anyhow!("fused_reduction(single-pass): input/output alias"));
            }
        }
        let groups = (num_slices as u32).max(1);
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            for (i, buf) in input_bufs.iter().enumerate() {
                log::debug!(
                    "[fused-reduction] binding={} role=read ptr={:p}",
                    i,
                    buf.0.as_ref()
                );
            }
            log::debug!(
                "[fused-reduction] binding={} role=read_write ptr={:p}",
                inputs.len(),
                out_buffer.as_ref()
            );
            log::debug!(
                "[fused-reduction] binding={} role=uniform ptr={:p}",
                inputs.len() + 1,
                params_buffer.as_ref()
            );
            log::debug!(
                "[fused-reduction] reduce_len={} slices={} wg={} groups={}",
                reduce_len,
                num_slices,
                workgroup_size,
                groups
            );
        }
        let disable_bg_cache = std::env::var("RUNMAT_DISABLE_FUSED_BG_CACHE").is_ok();
        let bg = if disable_bg_cache {
            Arc::new(
                self.device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-reduction-bg-direct"),
                        layout: bgl.as_ref(),
                        entries: &entries_vec,
                    }),
            )
        } else {
            self.bind_group_cache
                .get_or_create(bgl.as_ref(), &entries_vec, || {
                    Arc::new(
                        self.device_ref()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-reduction-bg"),
                                layout: bgl.as_ref(),
                                entries: &entries_vec,
                            }),
                    )
                })
        };
        crate::backend::wgpu::dispatch::reduction::run_single_pass(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
            bg.as_ref(),
            groups,
        );
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            log::debug!("[fused-reduction] single-pass dispatch complete");
        }
        Ok(self.register_existing_buffer(out_buffer, output_shape.to_vec(), out_len))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_reduction_two_pass(
        &self,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        shader: &str,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
        chunk_rows: u32,
        flavor: ReductionFlavor,
    ) -> Result<GpuTensorHandle> {
        let scalar_ty = match self.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => "f64",
            _ => "f32",
        };
        let flags = if shader.contains("const OMITNAN: bool = true") {
            1u32
        } else {
            0u32
        };
        let chunk_rows = chunk_rows.max(workgroup_size.max(1));
        let chunk_rows_u32 = chunk_rows;
        let total_chunks = (reduce_len as u64).div_ceil(chunk_rows as u64);
        let total_chunks_u32 =
            u32::try_from(total_chunks).map_err(|_| anyhow!("reduction: too many chunks"))?;
        let partials_len = num_slices.max(1) * (total_chunks as usize);
        let (pass1, pass2) = crate::backend::wgpu::shaders::reduction::build_two_pass_shaders(
            scalar_ty,
            workgroup_size,
        );
        let m1 = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-reduction-pass1",
            &pass1,
        );
        let m2 = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-reduction-pass2",
            &pass2,
        );
        let bgl1 = self
            .cached_bind_group_layout_for_tag("runmat-reduction-p1-bgl")
            .expect("p1 bgl");
        let bgl2 = self
            .cached_bind_group_layout_for_tag("runmat-reduction-p2-bgl")
            .expect("p2 bgl");
        let pl1 = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-reduction-p1-pl",
            bgl1.as_ref(),
        );
        let pl2 = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-reduction-p2-pl",
            bgl2.as_ref(),
        );
        if std::env::var("RUNMAT_DEBUG_PIPELINE_ONLY").is_ok() {
            let out_len = num_slices.max(1);
            let out_buffer = self.create_storage_buffer_checked(out_len, "runmat-reduction-out")?;
            return Ok(self.register_existing_buffer(out_buffer, output_shape.to_vec(), out_len));
        }
        let p2_key = self.compute_pipeline_hash_bytes(
            pass2.as_bytes(),
            "runmat-reduction-p2-bgl",
            Some(workgroup_size),
        );
        let pipeline_p2 = self.get_or_create_pipeline(
            p2_key,
            &pl2,
            &m2,
            "runmat-reduction-pass2",
            Some(pass2.as_bytes()),
            Some("runmat-reduction-p2-bgl"),
            Some(workgroup_size),
        );
        self.device_ref().poll(wgpu::Maintain::Poll);
        let flush_enc = self
            .device_ref()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-flush-before-pass1"),
            });
        self.submit(flush_enc);
        crate::backend::wgpu::dispatch::reduction::warmup_noop_after_pass2(
            self.device_ref(),
            self.queue_ref(),
            &pipeline_p2,
        );
        self.device_ref().poll(wgpu::Maintain::Poll);
        let p1_key = self.compute_pipeline_hash_bytes(
            pass1.as_bytes(),
            "runmat-reduction-p1-bgl",
            Some(workgroup_size),
        );
        let pipeline_p1 = self.get_or_create_pipeline(
            p1_key,
            &pl1,
            &m1,
            "runmat-reduction-pass1",
            Some(pass1.as_bytes()),
            Some("runmat-reduction-p1-bgl"),
            Some(workgroup_size),
        );
        self.device_ref().poll(wgpu::Maintain::Poll);
        let input_buf = self.get_entry(&inputs[0])?.buffer.clone();
        let input_buf = if std::env::var("RUNMAT_FUSED_SNAPSHOT_INPUTS").is_ok() {
            let bytes = (reduce_len * num_slices.max(1) * self.element_size) as u64;
            let snap = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-fused-p1-input-snapshot"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-fused-p1-input-snapshot-copy"),
                    });
            enc.copy_buffer_to_buffer(input_buf.as_ref(), 0, snap.as_ref(), 0, bytes);
            self.submit(enc);
            snap
        } else {
            input_buf
        };
        let partials_bytes = (partials_len * self.element_size) as u64;
        let mut partials_buffer = self.kernel_resources.scratch_storage_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::ScratchBufferKind::ReductionPartials,
            partials_bytes,
            "runmat-reduction-partials-scratch",
        );
        let out_len = num_slices.max(1);
        let mut out_buffer = self
            .create_storage_buffer_for_usage(
                BufferUsageClass::FusionOut,
                out_len,
                "runmat-reduction-out",
            )
            .0;
        {
            let in_ptr = input_buf.as_ref() as *const wgpu::Buffer as usize;
            let mut part_ptr = partials_buffer.as_ref() as *const wgpu::Buffer as usize;
            let out_ptr = out_buffer.as_ref() as *const wgpu::Buffer as usize;
            if part_ptr == in_ptr {
                let unique =
                    self.create_storage_buffer(partials_len, "runmat-reduction-partials-unique");
                partials_buffer = unique;
                part_ptr = partials_buffer.as_ref() as *const wgpu::Buffer as usize;
            }
            if out_ptr == in_ptr || out_ptr == part_ptr {
                out_buffer = self
                    .create_storage_buffer_for_usage(
                        BufferUsageClass::FusionOut,
                        out_len,
                        "runmat-reduction-out-unique",
                    )
                    .0;
            }
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct P1 {
            nrows: u32,
            ncols: u32,
            ld: u32,
            flags: u32,
            chunks: u32,
            chunk_stride: u32,
            chunk_rows: u32,
            _pad: u32,
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct P2F32 {
            ncols: u32,
            chunks: u32,
            flags: u32,
            scale: f32,
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct P2F64 {
            ncols: u32,
            chunks: u32,
            flags: u32,
            _pad: u32,
            scale: f64,
        }
        let max_dispatch = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS;
        let mut chunk_stride = total_chunks_u32.min(max_dispatch).max(1);
        let mut chunk_tiles = (total_chunks_u32 as u64).div_ceil(chunk_stride as u64);
        if chunk_tiles > max_dispatch as u64 {
            let required_stride = (total_chunks_u32 as u64).div_ceil(max_dispatch as u64);
            chunk_stride = required_stride.max(1).min(max_dispatch as u64) as u32;
            chunk_tiles = (total_chunks_u32 as u64).div_ceil(chunk_stride as u64);
            if chunk_tiles > max_dispatch as u64 {
                return Err(anyhow!(
                    "fused_reduction: chunk grid {} exceeds dispatch limits (stride {}, tiles {})",
                    total_chunks_u32,
                    chunk_stride,
                    chunk_tiles
                ));
            }
        }
        let p1u = P1 {
            nrows: reduce_len as u32,
            ncols: num_slices as u32,
            ld: reduce_len as u32,
            flags,
            chunks: total_chunks_u32,
            chunk_stride,
            chunk_rows: chunk_rows_u32,
            _pad: 0,
        };
        let p1_buf = self.kernel_resources.uniform_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::UniformBufferKey::ReductionPass1Params,
            std::mem::size_of::<P1>() as u64,
            "runmat-reduction-p1-params",
        );
        let p2_size = match self.precision {
            NumericPrecision::F64 => std::mem::size_of::<P2F64>() as u64,
            NumericPrecision::F32 => std::mem::size_of::<P2F32>() as u64,
        };
        let p2_buf = self.kernel_resources.uniform_buffer(
            self.device_ref(),
            crate::backend::wgpu::resources::UniformBufferKey::ReductionPass2Params,
            p2_size,
            "runmat-reduction-p2-params",
        );

        self.queue.write_buffer(p1_buf.as_ref(), 0, bytes_of(&p1u));
        let scale_value = flavor.scale(reduce_len);
        match self.precision {
            NumericPrecision::F64 => {
                let p2u = P2F64 {
                    ncols: num_slices as u32,
                    chunks: total_chunks_u32,
                    flags,
                    _pad: 0,
                    scale: scale_value,
                };
                self.queue.write_buffer(p2_buf.as_ref(), 0, bytes_of(&p2u));
            }
            NumericPrecision::F32 => {
                let p2u = P2F32 {
                    ncols: num_slices as u32,
                    chunks: total_chunks_u32,
                    flags,
                    scale: scale_value as f32,
                };
                self.queue.write_buffer(p2_buf.as_ref(), 0, bytes_of(&p2u));
            }
        }
        let entries_bg1 = vec![
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
        ];
        let bg1 = self
            .bind_group_cache
            .get_or_create(bgl1.as_ref(), &entries_bg1, || {
                Arc::new(
                    self.device_ref()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-reduction-p1-bg"),
                            layout: bgl1.as_ref(),
                            entries: &entries_bg1,
                        }),
                )
            });
        let entries_bg2 = vec![
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
        ];
        let bg2 = self
            .bind_group_cache
            .get_or_create(bgl2.as_ref(), &entries_bg2, || {
                Arc::new(
                    self.device_ref()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("runmat-reduction-p2-bg"),
                            layout: bgl2.as_ref(),
                            entries: &entries_bg2,
                        }),
                )
            });
        let g0 = (num_slices as u32).max(1);
        let g1 = chunk_stride.max(1);
        let g2 = (chunk_tiles as u32).max(1);
        crate::backend::wgpu::dispatch::reduction::run_two_pass(
            self.device_ref(),
            self.queue_ref(),
            &pipeline_p1,
            &pipeline_p2,
            bg1.as_ref(),
            bg2.as_ref(),
            g0,
            g1,
            g2,
        );
        Ok(self.register_existing_buffer(out_buffer, output_shape.to_vec(), num_slices.max(1)))
    }

    fn reduction_partials_budget_bytes(&self) -> u64 {
        if let Ok(raw) = std::env::var("RUNMAT_REDUCTION_PARTIALS_BUDGET_BYTES") {
            if let Ok(parsed) = raw.parse::<u64>() {
                if parsed > 0 {
                    return parsed;
                }
            }
        }
        let fallback = 4u64 << 30;
        let adapter_limit = if self.adapter_limits.max_buffer_size > 0 {
            self.adapter_limits.max_buffer_size
        } else {
            fallback
        };
        let mut budget = adapter_limit.saturating_mul(40) / 100;
        if budget == 0 {
            budget = adapter_limit;
        }
        let floor = 256u64 << 20;
        if adapter_limit >= floor {
            budget = budget.max(floor);
        }
        budget.min(adapter_limit)
    }

    fn sanitize_chunk_rows_for_limits(
        &self,
        chunk_rows: u32,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
    ) -> Option<u32> {
        if reduce_len == 0 {
            return None;
        }
        let wg = workgroup_size.max(1);
        let reduce_cap = reduce_len.min(u32::MAX as usize) as u32;
        let mut rows = chunk_rows.clamp(wg, reduce_cap.max(wg));
        if !rows.is_multiple_of(wg) {
            rows = rows.div_ceil(wg) * wg;
        }
        let slices = num_slices.max(1) as u64;
        let elem_bytes = self.element_size as u64;
        if elem_bytes == 0 {
            return None;
        }
        let per_chunk_bytes = slices.checked_mul(elem_bytes)?;
        let budget = self.reduction_partials_budget_bytes();
        if budget < per_chunk_bytes {
            return None;
        }
        let max_chunks = (budget / per_chunk_bytes).max(1);
        let mut required_chunk_rows = (reduce_len as u64).div_ceil(max_chunks);
        required_chunk_rows = required_chunk_rows.max(wg as u64);
        if required_chunk_rows > u32::MAX as u64 {
            return None;
        }
        let mut rows_u64 = rows as u64;
        if rows_u64 < required_chunk_rows {
            rows_u64 = required_chunk_rows;
        }
        rows_u64 = rows_u64.min(reduce_len.min(u32::MAX as usize) as u64);
        let mut rows_u32 = rows_u64 as u32;
        if !rows_u32.is_multiple_of(wg) {
            rows_u32 = rows_u32.div_ceil(wg) * wg;
        }
        rows_u32 = rows_u32.min(reduce_cap.max(wg));
        Some(rows_u32.max(wg))
    }

    fn prepare_reduction_tuning(
        &self,
        tuning: &ReductionTuning,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
    ) -> Option<ReductionTuning> {
        match tuning.mode {
            ReductionMode::SinglePass => Some(*tuning),
            ReductionMode::TwoPass { chunk_rows } => {
                let rows = self.sanitize_chunk_rows_for_limits(
                    chunk_rows,
                    reduce_len,
                    num_slices,
                    workgroup_size,
                )?;
                Some(ReductionTuning {
                    mode: ReductionMode::TwoPass { chunk_rows: rows },
                })
            }
        }
    }

    fn heuristic_reduction_tuning(
        &self,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
    ) -> ReductionTuning {
        let default_two_pass = ReductionTuning {
            mode: ReductionMode::TwoPass {
                chunk_rows: self.default_chunk_rows(reduce_len, workgroup_size),
            },
        };
        let single = ReductionTuning {
            mode: ReductionMode::SinglePass,
        };
        match self.reduction_two_pass_mode {
            ReductionTwoPassMode::ForceOn => default_two_pass,
            ReductionTwoPassMode::ForceOff => single,
            ReductionTwoPassMode::Auto => {
                if self.can_use_single_pass(num_slices) && reduce_len <= self.two_pass_threshold() {
                    single
                } else {
                    default_two_pass
                }
            }
        }
    }

    fn default_chunk_rows(&self, reduce_len: usize, workgroup_size: u32) -> u32 {
        let wg = workgroup_size.max(1) as usize;
        if reduce_len <= wg {
            return reduce_len.min(u32::MAX as usize) as u32;
        }
        let max_dispatch = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize;
        let target_chunks = max_dispatch.max(1);
        let mut rows = reduce_len.div_ceil(target_chunks).max(wg);
        rows = rows.div_ceil(wg) * wg;
        rows = rows.clamp(wg, reduce_len.max(wg));
        rows.min(u32::MAX as usize) as u32
    }

    fn can_use_single_pass(&self, num_slices: usize) -> bool {
        num_slices as u64 <= crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as u64
    }

    fn reduction_chunk_row_candidates(&self, reduce_len: usize, workgroup_size: u32) -> Vec<u32> {
        let mut values = Vec::new();
        let mut current = workgroup_size.max(1) as usize;
        while current < reduce_len {
            values.push(current.min(u32::MAX as usize) as u32);
            current = current.saturating_mul(2);
        }
        values.push(reduce_len.min(u32::MAX as usize) as u32);
        values.sort_unstable();
        values.dedup();
        values
    }

    fn reduction_strategy_candidates(
        &self,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
    ) -> Vec<ReductionTuning> {
        let mut strategies = Vec::new();
        let chunk_candidates = self.reduction_chunk_row_candidates(reduce_len, workgroup_size);
        let can_single = self.can_use_single_pass(num_slices);
        match self.reduction_two_pass_mode {
            ReductionTwoPassMode::ForceOff => {
                if can_single {
                    strategies.push(ReductionTuning {
                        mode: ReductionMode::SinglePass,
                    });
                }
            }
            ReductionTwoPassMode::ForceOn => {
                for chunk in &chunk_candidates {
                    strategies.push(ReductionTuning {
                        mode: ReductionMode::TwoPass { chunk_rows: *chunk },
                    });
                }
            }
            ReductionTwoPassMode::Auto => {
                if can_single {
                    strategies.push(ReductionTuning {
                        mode: ReductionMode::SinglePass,
                    });
                }
                for chunk in &chunk_candidates {
                    strategies.push(ReductionTuning {
                        mode: ReductionMode::TwoPass { chunk_rows: *chunk },
                    });
                }
            }
        }
        if strategies.is_empty() {
            strategies.push(ReductionTuning {
                mode: ReductionMode::TwoPass {
                    chunk_rows: self.default_chunk_rows(reduce_len, workgroup_size),
                },
            });
        }
        strategies
    }

    #[allow(clippy::too_many_arguments)]
    fn maybe_autotune_reduction(
        &self,
        key: &ReductionAutotuneKey,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        shader: &str,
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
        flavor: ReductionFlavor,
    ) -> Result<Option<GpuTensorHandle>> {
        if !self.reduction_autotune.is_enabled() {
            return Ok(None);
        }
        let candidates = self.reduction_strategy_candidates(reduce_len, num_slices, workgroup_size);
        if candidates.len() <= 1 {
            if let Some(tuning) = candidates.first() {
                self.reduction_autotune.insert(key.clone(), *tuning);
            }
            return Ok(None);
        }
        let mut best_tuning: Option<ReductionTuning> = None;
        let mut best_time: Option<Duration> = None;
        let mut best_handle: Option<GpuTensorHandle> = None;
        let mut tested = HashSet::new();
        let mut last_err: Option<anyhow::Error> = None;
        for tuning in candidates {
            let sanitized = self
                .prepare_reduction_tuning(&tuning, reduce_len, num_slices, workgroup_size)
                .or_else(|| {
                    if !matches!(tuning.mode, ReductionMode::SinglePass)
                        && self.can_use_single_pass(num_slices)
                    {
                        Some(ReductionTuning {
                            mode: ReductionMode::SinglePass,
                        })
                    } else {
                        None
                    }
                });
            let Some(sanitized) = sanitized else {
                continue;
            };
            if !tested.insert(sanitized) {
                continue;
            }
            let start = Instant::now();
            match self.execute_reduction_with_strategy(
                &sanitized,
                inputs,
                output_shape,
                shader,
                reduce_len,
                num_slices,
                workgroup_size,
                flavor,
            ) {
                Ok(handle) => {
                    let elapsed = start.elapsed();
                    if best_time.is_none_or(|t| elapsed < t) {
                        if let Some(existing) = best_handle.replace(handle) {
                            let _ = self.free(&existing);
                        }
                        best_time = Some(elapsed);
                        best_tuning = Some(sanitized);
                    } else {
                        let _ = self.free(&handle);
                    }
                }
                Err(err) => {
                    last_err = Some(err);
                }
            }
        }
        if let (Some(tuning), Some(handle)) = (best_tuning, best_handle) {
            self.reduction_autotune.insert(key.clone(), tuning);
            return Ok(Some(handle));
        }
        if let Some(err) = last_err {
            return Err(err);
        }
        Ok(None)
    }

    pub(crate) fn scatter_column_exec(
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
        let v_rows = match v_entry.shape.len() {
            1 | 2 => v_entry.shape[0],
            _ => {
                return Err(anyhow!("scatter_column: values must be vector or [rows,1]"));
            }
        };
        if v_rows != rows {
            return Err(anyhow!("scatter_column: length mismatch"));
        }
        let shader = crate::backend::wgpu::shaders::scatter::SCATTER_COL_SHADER_F32;
        let out_buffer = self.create_storage_buffer(rows * cols, "runmat-scatter-col-out");
        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-scatter-col-copy"),
                    });
            enc.copy_buffer_to_buffer(
                m_entry.buffer.as_ref(),
                0,
                out_buffer.as_ref(),
                0,
                (rows * cols * self.element_size) as u64,
            );
            self.submit(enc);
        }
        let bgl = crate::backend::wgpu::bindings::build_scatter_col_bgl(self.device_ref());
        let pl = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-scatter-col-pl",
            &bgl,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-scatter-col-module",
            shader,
        );
        let key = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            "runmat-scatter-col-bgl",
            Some(256),
        );
        let pipeline = self.get_or_create_pipeline(
            key,
            &pl,
            &module,
            "runmat-scatter-col",
            Some(shader.as_bytes()),
            Some("runmat-scatter-col-bgl"),
            Some(256),
        );
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Pm {
            rows: u32,
            cols: u32,
            j: u32,
        }
        let params = Pm {
            rows: rows as u32,
            cols: cols as u32,
            j: col_index as u32,
        };
        let pbuf = self.uniform_buffer(&params, "runmat-scatter-col-params");
        let bg = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-scatter-col-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: v_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: m_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pbuf.as_entire_binding(),
                    },
                ],
            });
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(rows as u32, 256);
        crate::backend::wgpu::dispatch::scatter::run(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
            &bg,
            groups,
        );
        Ok(self.register_existing_buffer(out_buffer, vec![rows, cols], rows * cols))
    }
    pub(crate) fn sub2ind_exec(
        &self,
        dims: &[usize],
        strides: &[usize],
        inputs: &[&GpuTensorHandle],
        scalar_mask: &[bool],
        len: usize,
        output_shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        if inputs.len() != dims.len() || inputs.len() != scalar_mask.len() {
            return Err(anyhow!(
                "sub2ind: expected {} subscripts for {} dimensions",
                dims.len(),
                dims.len()
            ));
        }
        let expected_len: usize = output_shape.iter().copied().product();
        if expected_len != len {
            return Err(anyhow!(
                "sub2ind: output shape does not match subscript sizes"
            ));
        }
        if len == 0 {
            let buffer = self.create_storage_buffer(0, "runmat-sub2ind-empty");
            return Ok(self.register_existing_buffer(buffer, output_shape.to_vec(), 0));
        }
        if dims.iter().any(|&d| d > u32::MAX as usize)
            || strides.iter().any(|&s| s > u32::MAX as usize)
            || len > u32::MAX as usize
        {
            return Err(anyhow!("sub2ind: dimensions exceed GPU kernel limits"));
        }

        let dims_u32: Vec<u32> = dims.iter().map(|&d| d as u32).collect();
        let strides_u32: Vec<u32> = strides.iter().map(|&s| s as u32).collect();
        let mask_u32: Vec<u32> = scalar_mask
            .iter()
            .map(|&m| if m { 1u32 } else { 0u32 })
            .collect();

        let mut input_buffers = Vec::with_capacity(inputs.len());
        for handle in inputs {
            input_buffers.push(self.get_entry(handle)?.buffer.clone());
        }

        let output_buffer = self.create_storage_buffer_checked(len, "runmat-sub2ind-out")?;
        let error_bytes = vec![0u8; std::mem::size_of::<u32>() * 4];
        let error_buffer = Arc::new(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("runmat-sub2ind-error"),
                contents: &error_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        ));

        let (scalar_ty, epsilon) = match self.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => ("f64", "1.0e-12"),
            runmat_accelerate_api::ProviderPrecision::F32 => ("f32", "1.0e-5"),
        };
        let workgroup_size = crate::backend::wgpu::config::WORKGROUP_SIZE;
        let shader = crate::backend::wgpu::shaders::sub2ind::build_sub2ind_shader(
            scalar_ty,
            &dims_u32,
            &strides_u32,
            &mask_u32,
            workgroup_size,
            epsilon,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-sub2ind-module",
            &shader,
        );
        let bgl =
            crate::backend::wgpu::bindings::build_sub2ind_bgl(self.device_ref(), inputs.len());
        let pl = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-sub2ind-pl",
            &bgl,
        );
        let layout_tag = format!("runmat-sub2ind-layout-{}", inputs.len());
        let key =
            self.compute_pipeline_hash_bytes(shader.as_bytes(), &layout_tag, Some(workgroup_size));
        let pipeline = self.get_or_create_pipeline(
            key,
            &pl,
            &module,
            "runmat-sub2ind",
            Some(shader.as_bytes()),
            Some(layout_tag.as_str()),
            Some(workgroup_size),
        );

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            len: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }
        let params = Params {
            len: len as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-sub2ind-params");

        let mut bind_entries = Vec::with_capacity(inputs.len() + 3);
        for (idx, buffer) in input_buffers.iter().enumerate() {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: idx as u32,
                resource: buffer.as_ref().as_entire_binding(),
            });
        }
        bind_entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: output_buffer.as_ref().as_entire_binding(),
        });
        bind_entries.push(wgpu::BindGroupEntry {
            binding: (inputs.len() + 1) as u32,
            resource: error_buffer.as_ref().as_entire_binding(),
        });
        bind_entries.push(wgpu::BindGroupEntry {
            binding: (inputs.len() + 2) as u32,
            resource: params_buffer.as_entire_binding(),
        });
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-sub2ind-bg"),
                layout: &bgl,
                entries: &bind_entries,
            });

        let groups =
            crate::backend::wgpu::dispatch::common::dispatch_size(len as u32, workgroup_size);
        crate::backend::wgpu::dispatch::sub2ind::run(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
            &bind_group,
            groups,
        );

        let error_size = (std::mem::size_of::<u32>() * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-sub2ind-error-staging"),
            size: error_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-sub2ind-error-copy"),
            });
        encoder.copy_buffer_to_buffer(error_buffer.as_ref(), 0, &staging, 0, error_size);
        self.submit(encoder);
        let bytes = self.map_readback_bytes_sync(staging, error_size, "sub2ind")?;
        let words: &[u32] = cast_slice(&bytes);
        let code = words.first().copied().unwrap_or(0);
        let dim_word = words.get(1).copied().unwrap_or(0);
        let extra = words.get(2).copied().unwrap_or(0);

        if code != 0 {
            let dim_index = dim_word.max(1) as usize;
            let dim_size = dims.get(dim_index.saturating_sub(1)).copied().unwrap_or(0);
            let err = match code {
                1 => anyhow!(
                    "sub2ind: subscript in dimension {} must be finite",
                    dim_index
                ),
                2 => anyhow!(
                    "sub2ind: subscript in dimension {} must be an integer",
                    dim_index
                ),
                3 => anyhow!(
                    "sub2ind: subscript {} exceeds dimension {} (size {})",
                    extra as isize,
                    dim_index,
                    dim_size
                ),
                _ => anyhow!("sub2ind: kernel reported error code {}", code),
            };
            return Err(err);
        }

        Ok(self.register_existing_buffer_with_usage(
            output_buffer,
            output_shape.to_vec(),
            len,
            BufferUsageClass::FusionOut,
        ))
    }

    pub(crate) fn ind2sub_exec(
        &self,
        dims: &[usize],
        strides: &[usize],
        indices: &GpuTensorHandle,
        total: usize,
        len: usize,
        output_shape: &[usize],
    ) -> Result<Vec<GpuTensorHandle>> {
        if dims.len() != strides.len() {
            return Err(anyhow!("ind2sub: size vector mismatch"));
        }
        let expected_len: usize = output_shape.iter().copied().product();
        if expected_len != len {
            return Err(anyhow!("ind2sub: output shape does not match index tensor"));
        }
        if len == 0 {
            let mut handles = Vec::with_capacity(dims.len());
            for _ in 0..dims.len() {
                let buffer = self.create_storage_buffer(0, "runmat-ind2sub-empty");
                handles.push(self.register_existing_buffer(buffer, output_shape.to_vec(), 0));
            }
            return Ok(handles);
        }
        if dims.iter().any(|&d| d > u32::MAX as usize)
            || strides.iter().any(|&s| s > u32::MAX as usize)
            || total > u32::MAX as usize
            || len > u32::MAX as usize
        {
            return Err(anyhow!("ind2sub: dimensions exceed GPU kernel limits"));
        }

        let entry = self.get_entry(indices)?;
        if entry.len != len {
            return Err(anyhow!(
                "ind2sub: index tensor length does not match provided shape"
            ));
        }

        let dims_u32: Vec<u32> = dims.iter().map(|&d| d as u32).collect();
        let strides_u32: Vec<u32> = strides.iter().map(|&s| s as u32).collect();

        let mut output_buffers = Vec::with_capacity(dims.len());
        for _ in 0..dims.len() {
            output_buffers.push(self.create_storage_buffer_checked(len, "runmat-ind2sub-out")?);
        }

        let error_bytes = vec![0u8; std::mem::size_of::<u32>() * 4];
        let error_buffer = Arc::new(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("runmat-ind2sub-error"),
                contents: &error_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        ));

        let (scalar_ty, epsilon) = match self.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => ("f64", "1.0e-12"),
            runmat_accelerate_api::ProviderPrecision::F32 => ("f32", "1.0e-5"),
        };
        let workgroup_size = crate::backend::wgpu::config::WORKGROUP_SIZE;
        let shader = crate::backend::wgpu::shaders::ind2sub::build_ind2sub_shader(
            scalar_ty,
            &dims_u32,
            &strides_u32,
            total as u32,
            workgroup_size,
            epsilon,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-ind2sub-module",
            &shader,
        );
        let bgl = crate::backend::wgpu::bindings::build_ind2sub_bgl(self.device_ref(), dims.len());
        let pl = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-ind2sub-pl",
            &bgl,
        );
        let layout_tag = format!("runmat-ind2sub-layout-{}", dims.len());
        let key =
            self.compute_pipeline_hash_bytes(shader.as_bytes(), &layout_tag, Some(workgroup_size));
        let pipeline = self.get_or_create_pipeline(
            key,
            &pl,
            &module,
            "runmat-ind2sub",
            Some(shader.as_bytes()),
            Some(layout_tag.as_str()),
            Some(workgroup_size),
        );

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Params {
            len: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }
        let params = Params {
            len: len as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-ind2sub-params");

        let mut bind_entries = Vec::with_capacity(dims.len() + 3);
        bind_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: entry.buffer.as_ref().as_entire_binding(),
        });
        for (idx, buffer) in output_buffers.iter().enumerate() {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: (idx + 1) as u32,
                resource: buffer.as_ref().as_entire_binding(),
            });
        }
        bind_entries.push(wgpu::BindGroupEntry {
            binding: (dims.len() + 1) as u32,
            resource: error_buffer.as_ref().as_entire_binding(),
        });
        bind_entries.push(wgpu::BindGroupEntry {
            binding: (dims.len() + 2) as u32,
            resource: params_buffer.as_entire_binding(),
        });
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-ind2sub-bg"),
                layout: &bgl,
                entries: &bind_entries,
            });

        let groups =
            crate::backend::wgpu::dispatch::common::dispatch_size(len as u32, workgroup_size);
        crate::backend::wgpu::dispatch::ind2sub::run(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
            &bind_group,
            groups,
        );

        let error_size = (std::mem::size_of::<u32>() * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-ind2sub-error-staging"),
            size: error_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-ind2sub-error-copy"),
            });
        encoder.copy_buffer_to_buffer(error_buffer.as_ref(), 0, &staging, 0, error_size);
        self.submit(encoder);
        let bytes = self.map_readback_bytes_sync(staging, error_size, "ind2sub")?;
        let words: &[u32] = cast_slice(&bytes);
        let code = words.first().copied().unwrap_or(0);

        if code != 0 {
            let err = match code {
                1..=3 => anyhow!("Linear indices must be positive integers."),
                4 => anyhow!(
                    "Index exceeds number of array elements. Index must not exceed {}.",
                    total
                ),
                _ => anyhow!("ind2sub: kernel reported error code {}", code),
            };
            return Err(err);
        }

        let mut handles = Vec::with_capacity(output_buffers.len());
        for buffer in output_buffers {
            handles.push(self.register_existing_buffer(buffer, output_shape.to_vec(), len));
        }
        Ok(handles)
    }

    pub(crate) fn scatter_row_exec(
        &self,
        matrix: &GpuTensorHandle,
        row_index: usize,
        values: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let m_entry = self.get_entry(matrix)?;
        if m_entry.shape.len() != 2 {
            return Err(anyhow!("scatter_row: only 2D tensors supported"));
        }
        let rows = m_entry.shape[0];
        let cols = m_entry.shape[1];
        if row_index >= rows {
            return Err(anyhow!("scatter_row: row index out of bounds"));
        }
        let v_entry = self.get_entry(values)?;
        let v_cols = if v_entry.shape.len() == 1 {
            v_entry.shape[0]
        } else if v_entry.shape.len() == 2 {
            v_entry.shape[1]
        } else {
            return Err(anyhow!("scatter_row: values must be vector or [1,cols]"));
        };
        if v_cols != cols {
            return Err(anyhow!("scatter_row: length mismatch"));
        }
        let shader = crate::backend::wgpu::shaders::scatter::SCATTER_ROW_SHADER_F32;
        let out_buffer = self.create_storage_buffer(rows * cols, "runmat-scatter-row-out");
        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-scatter-row-copy"),
                    });
            enc.copy_buffer_to_buffer(
                m_entry.buffer.as_ref(),
                0,
                out_buffer.as_ref(),
                0,
                (rows * cols * self.element_size) as u64,
            );
            self.submit(enc);
        }
        let bgl = crate::backend::wgpu::bindings::build_scatter_row_bgl(self.device_ref());
        let pl = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-scatter-row-pl",
            &bgl,
        );
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            self.device_ref(),
            "runmat-scatter-row-module",
            shader,
        );
        let key = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            "runmat-scatter-row-bgl",
            Some(256),
        );
        let pipeline = self.get_or_create_pipeline(
            key,
            &pl,
            &module,
            "runmat-scatter-row",
            Some(shader.as_bytes()),
            Some("runmat-scatter-row-bgl"),
            Some(256),
        );
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Pm {
            rows: u32,
            cols: u32,
            i: u32,
        }
        let params = Pm {
            rows: rows as u32,
            cols: cols as u32,
            i: row_index as u32,
        };
        let pbuf = self.uniform_buffer(&params, "runmat-scatter-row-params");
        let bg = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-scatter-row-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: v_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: m_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pbuf.as_entire_binding(),
                    },
                ],
            });
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            cols as u32,
            crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::scatter::run(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
            &bg,
            groups,
        );
        Ok(self.register_existing_buffer(out_buffer, vec![rows, cols], rows * cols))
    }
    pub(crate) fn permute_exec(
        &self,
        handle: &GpuTensorHandle,
        order: &[usize],
    ) -> Result<GpuTensorHandle> {
        ensure!(!order.is_empty(), "permute: order must not be empty");
        let rank = order.len();
        if rank > crate::backend::wgpu::params::PERMUTE_MAX_RANK {
            return Err(anyhow!(
                "permute: rank {} exceeds GPU support (max {})",
                rank,
                crate::backend::wgpu::params::PERMUTE_MAX_RANK
            ));
        }

        let entry = self.get_entry(handle)?;
        ensure!(
            entry.shape.len() <= rank,
            "permute: order length ({}) must be at least ndims(A) ({})",
            rank,
            entry.shape.len()
        );

        let mut src_shape = entry.shape.clone();
        if src_shape.len() < rank {
            src_shape.extend(std::iter::repeat_n(1usize, rank - src_shape.len()));
        }

        let total: usize = src_shape.iter().copied().product();
        ensure!(
            total == entry.len,
            "permute: shape/product mismatch ({} vs {})",
            total,
            entry.len
        );
        if entry.len > u32::MAX as usize {
            return Err(anyhow!("permute: tensor too large for GPU kernel"));
        }

        let mut dst_shape = vec![0usize; rank];
        let mut seen = vec![false; rank];
        for (dst_dim, &src_dim) in order.iter().enumerate() {
            ensure!(
                src_dim < rank,
                "permute: invalid dimension index {}",
                src_dim + 1
            );
            ensure!(
                !seen[src_dim],
                "permute: duplicate dimension index {}",
                src_dim + 1
            );
            seen[src_dim] = true;
            dst_shape[dst_dim] = src_shape[src_dim];
        }
        ensure!(
            seen.iter().all(|&flag| flag),
            "permute: order must include every dimension exactly once"
        );

        if src_shape.iter().any(|&d| d > u32::MAX as usize)
            || dst_shape.iter().any(|&d| d > u32::MAX as usize)
        {
            return Err(anyhow!("permute: dimensions exceed GPU kernel limits"));
        }

        let mut src_strides = vec![0usize; rank];
        let mut stride = 1usize;
        for (idx, &dim) in src_shape.iter().enumerate() {
            src_strides[idx] = stride;
            stride = stride
                .checked_mul(dim)
                .ok_or_else(|| anyhow!("permute: dimension product exceeds GPU limits"))?;
        }

        ensure!(
            dst_shape.iter().copied().product::<usize>() == entry.len,
            "permute: output shape/product mismatch"
        );

        let mut src_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::PERMUTE_MAX_RANK];
        let mut dst_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::PERMUTE_MAX_RANK];
        let mut order_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::PERMUTE_MAX_RANK];
        let mut strides_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::PERMUTE_MAX_RANK];
        for i in 0..rank {
            src_shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(src_shape[i] as u32);
            dst_shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(dst_shape[i] as u32);
            order_arr[i] = crate::backend::wgpu::params::AlignedU32::new(order[i] as u32);
            strides_arr[i] = crate::backend::wgpu::params::AlignedU32::new(src_strides[i] as u32);
        }

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-permute-out");
        let out_shape = dst_shape;
        if entry.len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, 0));
        }

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-permute-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-permute-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.permute.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-permute-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < entry.len {
            let remaining = entry.len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::PermuteParams {
                len: chunk_len as u32,
                offset: offset as u32,
                rank: rank as u32,
                _pad: 0,
                src_shape: src_shape_arr,
                dst_shape: dst_shape_arr,
                order: order_arr,
                src_strides: strides_arr,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-permute-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-permute-bind"),
                    layout: &self.pipelines.permute.layout,
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
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::permute::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.permute.pipeline,
                &bind_group,
                workgroups,
            );
            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, out_shape, entry.len))
    }
    pub(crate) fn circshift_exec(
        &self,
        handle: &GpuTensorHandle,
        shifts: &[isize],
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let mut ext_shape = if entry.shape.is_empty() {
            vec![1usize]
        } else {
            entry.shape.clone()
        };

        if shifts.len() > ext_shape.len() {
            ext_shape.extend(std::iter::repeat_n(1usize, shifts.len() - ext_shape.len()));
        }

        let rank = ext_shape.len();
        if rank == 0 {
            return Ok(handle.clone());
        }
        if rank > crate::backend::wgpu::params::CIRCSHIFT_MAX_RANK {
            return Err(anyhow!(
                "circshift: rank {} exceeds GPU support (max {})",
                rank,
                crate::backend::wgpu::params::CIRCSHIFT_MAX_RANK
            ));
        }

        let total = product_checked(&ext_shape)
            .ok_or_else(|| anyhow!("circshift: dimension product exceeds GPU limits"))?;
        ensure!(
            total == entry.len || (total == 0 && entry.len == 0),
            "circshift: shape/product mismatch ({} vs {})",
            total,
            entry.len
        );

        ensure!(
            entry.len <= u32::MAX as usize,
            "circshift: tensor too large for GPU kernel"
        );
        ensure!(
            ext_shape.iter().all(|&d| d <= u32::MAX as usize),
            "circshift: dimensions exceed GPU kernel limits"
        );

        let mut normalized = vec![0usize; rank];
        let mut has_effect = false;
        for axis in 0..rank {
            let size = ext_shape[axis];
            let shift = if axis < shifts.len() { shifts[axis] } else { 0 };
            if size <= 1 {
                normalized[axis] = 0;
                continue;
            }
            let size_isize = size as isize;
            let mut norm = shift % size_isize;
            if norm < 0 {
                norm += size_isize;
            }
            let norm_usize = norm as usize;
            normalized[axis] = norm_usize;
            if norm_usize != 0 {
                has_effect = true;
            }
        }

        if !has_effect {
            return Ok(handle.clone());
        }

        let mut strides = vec![0usize; rank];
        let mut stride = 1usize;
        for axis in 0..rank {
            strides[axis] = stride;
            stride = stride
                .checked_mul(ext_shape[axis].max(1))
                .ok_or_else(|| anyhow!("circshift: stride computation exceeds GPU limits"))?;
        }

        ensure!(
            strides.iter().all(|&s| s <= u32::MAX as usize),
            "circshift: strides exceed GPU kernel limits"
        );

        let mut shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::CIRCSHIFT_MAX_RANK];
        let mut strides_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::CIRCSHIFT_MAX_RANK];
        let mut shifts_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::CIRCSHIFT_MAX_RANK];
        for axis in 0..rank {
            shape_arr[axis] = crate::backend::wgpu::params::AlignedU32::new(ext_shape[axis] as u32);
            strides_arr[axis] = crate::backend::wgpu::params::AlignedU32::new(strides[axis] as u32);
            let denom = ext_shape[axis].max(1);
            shifts_arr[axis] =
                crate::backend::wgpu::params::AlignedU32::new((normalized[axis] % denom) as u32);
        }

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-circshift-out");
        let out_shape = entry.shape.clone();

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-circshift-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-circshift-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.circshift.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-circshift-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < entry.len {
            let remaining = entry.len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::CircshiftParams {
                len: chunk_len as u32,
                offset: offset as u32,
                rank: rank as u32,
                _pad: 0,
                shape: shape_arr,
                strides: strides_arr,
                shifts: shifts_arr,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-circshift-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-circshift-bind"),
                    layout: &self.pipelines.circshift.layout,
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
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::circshift::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.circshift.pipeline,
                &bind_group,
                workgroups,
            );
            offset += chunk_len;
        }

        let handle = self.register_existing_buffer(out_buffer, out_shape, entry.len);

        Ok(handle)
    }

    pub(crate) async fn tril_exec(
        &self,
        handle: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let rows = entry.shape.first().copied().unwrap_or(1);
        let cols = entry.shape.get(1).copied().unwrap_or(1);
        let plane = rows.saturating_mul(cols);
        if plane == 0 {
            return Ok(handle.clone());
        }
        if plane > entry.len {
            return self.tril_exec_fallback(handle, offset).await;
        }
        if entry.len % plane != 0 {
            return self.tril_exec_fallback(handle, offset).await;
        }
        let pages = entry.len / plane;
        let max_u32 = u32::MAX as usize;
        if rows > max_u32
            || cols > max_u32
            || plane > max_u32
            || entry.len > max_u32
            || pages > max_u32
            || rows > i32::MAX as usize
            || cols > i32::MAX as usize
        {
            return self.tril_exec_fallback(handle, offset).await;
        }

        let diag_offset = if offset > i32::MAX as isize {
            i32::MAX
        } else if offset < -(i32::MAX as isize) {
            -i32::MAX
        } else {
            offset as i32
        };

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-tril-out");
        let out_shape = entry.shape.clone();

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-tril-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-tril-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.tril.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-tril-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let plane_u32 = plane as u32;
        let mut offset_idx = 0usize;
        while offset_idx < entry.len {
            let remaining = entry.len - offset_idx;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::TrilParams {
                len: chunk_len as u32,
                start: offset_idx as u32,
                rows: rows_u32,
                cols: cols_u32,
                plane: plane_u32,
                diag_offset,
                _pad0: 0,
                _pad1: 0,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-tril-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-tril-bind"),
                    layout: &self.pipelines.tril.layout,
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
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::tril::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.tril.pipeline,
                &bind_group,
                workgroups,
            );
            offset_idx += chunk_len;
        }

        let handle = self.register_existing_buffer(out_buffer, out_shape, entry.len);

        Ok(handle)
    }

    async fn tril_exec_fallback(
        &self,
        handle: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let HostTensorOwned { mut data, shape } =
            <Self as AccelProvider>::download(self, handle).await?;
        apply_tril_mask_host(&mut data, &shape, offset)?;
        let view = HostTensorView {
            data: &data,
            shape: &shape,
        };
        <Self as AccelProvider>::upload(self, &view)
    }

    pub(crate) async fn triu_exec(
        &self,
        handle: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let rows = entry.shape.first().copied().unwrap_or(1);
        let cols = entry.shape.get(1).copied().unwrap_or(1);
        let plane = rows.saturating_mul(cols);
        if plane == 0 {
            return Ok(handle.clone());
        }
        if plane > entry.len || entry.len % plane != 0 {
            return self.triu_exec_fallback(handle, offset).await;
        }
        let pages = entry.len / plane;
        let max_u32 = u32::MAX as usize;
        if rows > max_u32
            || cols > max_u32
            || plane > max_u32
            || entry.len > max_u32
            || pages > max_u32
            || rows > i32::MAX as usize
            || cols > i32::MAX as usize
        {
            return self.triu_exec_fallback(handle, offset).await;
        }

        let diag_offset = if offset > i32::MAX as isize {
            i32::MAX
        } else if offset < -(i32::MAX as isize) {
            -i32::MAX
        } else {
            offset as i32
        };

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-triu-out");
        let out_shape = entry.shape.clone();

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-triu-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-triu-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.triu.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-triu-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let plane_u32 = plane as u32;
        let mut offset_idx = 0usize;
        while offset_idx < entry.len {
            let remaining = entry.len - offset_idx;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::TriuParams {
                len: chunk_len as u32,
                start: offset_idx as u32,
                rows: rows_u32,
                cols: cols_u32,
                plane: plane_u32,
                diag_offset,
                _pad0: 0,
                _pad1: 0,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-triu-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-triu-bind"),
                    layout: &self.pipelines.triu.layout,
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
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::triu::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.triu.pipeline,
                &bind_group,
                workgroups,
            );
            offset_idx += chunk_len;
        }

        let handle = self.register_existing_buffer(out_buffer, out_shape, entry.len);

        Ok(handle)
    }

    async fn triu_exec_fallback(
        &self,
        handle: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let HostTensorOwned { mut data, shape } =
            <Self as AccelProvider>::download(self, handle).await?;
        apply_triu_mask_host(&mut data, &shape, offset)?;
        let view = HostTensorView {
            data: &data,
            shape: &shape,
        };
        <Self as AccelProvider>::upload(self, &view)
    }
    pub(crate) fn flip_exec(
        &self,
        handle: &GpuTensorHandle,
        axes: &[usize],
    ) -> Result<GpuTensorHandle> {
        if axes.is_empty() {
            return Ok(handle.clone());
        }

        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let mut ext_shape = if entry.shape.is_empty() {
            vec![1usize]
        } else {
            entry.shape.clone()
        };

        if let Some(&max_axis) = axes.iter().max() {
            let needed = max_axis + 1;
            if needed > ext_shape.len() {
                ext_shape.extend(std::iter::repeat_n(1usize, needed - ext_shape.len()));
            }
        }

        let rank = ext_shape.len();
        if rank == 0 {
            return Ok(handle.clone());
        }
        if rank > crate::backend::wgpu::params::FLIP_MAX_RANK {
            return Err(anyhow!(
                "flip: rank {} exceeds GPU support (max {})",
                rank,
                crate::backend::wgpu::params::FLIP_MAX_RANK
            ));
        }

        let total = product_checked(&ext_shape)
            .ok_or_else(|| anyhow!("flip: dimension product exceeds GPU limits"))?;
        ensure!(
            total == entry.len || (total == 0 && entry.len == 0),
            "flip: shape/product mismatch ({} vs {})",
            total,
            entry.len
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "flip: tensor too large for GPU kernel"
        );
        ensure!(
            ext_shape.iter().all(|&d| d <= u32::MAX as usize),
            "flip: dimensions exceed GPU kernel limits"
        );

        let mut flags = vec![false; rank];
        for &axis in axes {
            if axis < rank {
                flags[axis] = !flags[axis];
            }
        }
        let has_effect = flags
            .iter()
            .enumerate()
            .any(|(idx, flag)| *flag && ext_shape[idx] > 1);
        if !has_effect {
            return Ok(handle.clone());
        }

        let mut strides = vec![0usize; rank];
        let mut stride = 1usize;
        for (idx, &dim) in ext_shape.iter().enumerate() {
            strides[idx] = stride;
            stride = stride
                .checked_mul(dim.max(1))
                .ok_or_else(|| anyhow!("flip: stride computation exceeds GPU limits"))?;
        }
        ensure!(
            strides.iter().all(|&s| s <= u32::MAX as usize),
            "flip: strides exceed GPU kernel limits"
        );

        let mut shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::FLIP_MAX_RANK];
        let mut strides_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::FLIP_MAX_RANK];
        let mut flags_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::FLIP_MAX_RANK];
        for i in 0..rank {
            shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(ext_shape[i] as u32);
            strides_arr[i] = crate::backend::wgpu::params::AlignedU32::new(strides[i] as u32);
            flags_arr[i] =
                crate::backend::wgpu::params::AlignedU32::new(if flags[i] { 1 } else { 0 });
        }

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-flip-out");
        let out_shape = entry.shape.clone();
        if entry.len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, 0));
        }

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-flip-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-flip-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.flip.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-flip-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < entry.len {
            let remaining = entry.len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::FlipParams {
                len: chunk_len as u32,
                offset: offset as u32,
                rank: rank as u32,
                _pad: 0,
                shape: shape_arr,
                strides: strides_arr,
                flags: flags_arr,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-flip-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-flip-bind"),
                    layout: &self.pipelines.flip.layout,
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
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::flip::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.flip.pipeline,
                &bind_group,
                workgroups,
            );
            offset += chunk_len;
        }

        let handle = self.register_existing_buffer(out_buffer, out_shape, entry.len);

        Ok(handle)
    }
    pub(crate) fn conv1d_exec(
        &self,
        signal: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: ProviderConv1dOptions,
    ) -> Result<GpuTensorHandle> {
        let entry_signal = self.get_entry(signal)?;
        let entry_kernel = self.get_entry(kernel)?;

        ensure!(
            entry_signal.precision == self.precision && entry_kernel.precision == self.precision,
            "conv1d: mixed precision tensors are not supported"
        );

        let signal_len = entry_signal.len;
        let kernel_len = entry_kernel.len;

        let (output_len, start_offset, _) = conv1d_window(signal_len, kernel_len, options.mode)?;

        if output_len == 0 {
            let out_shape = conv1d_output_shape(0, options.orientation);
            let out_buffer = self.create_storage_buffer(0, "runmat-conv1d-empty");
            return Ok(self.register_existing_buffer(out_buffer, out_shape, 0));
        }

        ensure!(
            signal_len <= u32::MAX as usize
                && kernel_len <= u32::MAX as usize
                && output_len <= u32::MAX as usize
                && start_offset <= u32::MAX as usize,
            "conv1d: tensor exceeds GPU kernel limits"
        );

        let out_shape = conv1d_output_shape(output_len, options.orientation);
        let out_buffer = self.create_storage_buffer_checked(output_len, "runmat-conv1d-out")?;

        let params = Conv1dParams {
            signal_len: signal_len as u32,
            kernel_len: kernel_len as u32,
            output_len: output_len as u32,
            start_offset: start_offset as u32,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-conv1d-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-conv1d-bind"),
                layout: &self.pipelines.conv1d.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry_signal.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: entry_kernel.buffer.as_ref().as_entire_binding(),
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

        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            output_len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::conv::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.conv1d.pipeline,
            &bind_group,
            workgroups,
        );

        let handle = self.register_existing_buffer(out_buffer, out_shape, output_len);

        Ok(handle)
    }
    pub(crate) async fn iir_filter_exec(
        &self,
        b: &GpuTensorHandle,
        a: &GpuTensorHandle,
        x: &GpuTensorHandle,
        options: ProviderIirFilterOptions,
    ) -> Result<ProviderIirFilterResult> {
        let ProviderIirFilterOptions { dim, zi } = options;

        let entry_b = self.get_entry(b)?;
        let entry_a = self.get_entry(a)?;
        let entry_x = self.get_entry(x)?;

        ensure!(
            entry_b.precision == self.precision
                && entry_a.precision == self.precision
                && entry_x.precision == self.precision,
            "iir_filter: mixed precision tensors are not supported"
        );

        let nb = entry_b.len;
        let na = entry_a.len;
        ensure!(
            nb > 0,
            "iir_filter: numerator coefficients must not be empty"
        );
        ensure!(
            na > 0,
            "iir_filter: denominator coefficients must not be empty"
        );

        let b_host = <Self as AccelProvider>::download(self, b).await?;
        let a_host = <Self as AccelProvider>::download(self, a).await?;
        let a0 = *a_host
            .data
            .first()
            .ok_or_else(|| anyhow!("iir_filter: denominator coefficients cannot be empty"))?;
        ensure!(
            a0 != 0.0,
            "iir_filter: denominator coefficient a(1) must be non-zero"
        );

        let order = nb.max(na);
        ensure!(
            order <= u32::MAX as usize,
            "iir_filter: filter order exceeds GPU limits"
        );

        let mut b_norm = vec![0.0f64; order];
        let mut a_norm = vec![0.0f64; order];
        for i in 0..order {
            let b_coeff = if i < nb { b_host.data[i] } else { 0.0 };
            b_norm[i] = b_coeff / a0;
            if i == 0 {
                a_norm[0] = 1.0;
            } else {
                let a_coeff = if i < na { a_host.data[i] } else { 0.0 };
                a_norm[i] = a_coeff / a0;
            }
        }

        let state_len = order.saturating_sub(1);

        let mut shape_ext = entry_x.shape.clone();
        if dim >= shape_ext.len() {
            shape_ext.extend(std::iter::repeat_n(1, dim + 1 - shape_ext.len()));
        }
        ensure!(
            dim < shape_ext.len(),
            "iir_filter: dimension argument exceeds tensor rank"
        );
        let dim_idx = dim;
        let dim_len = shape_ext[dim_idx];

        let leading = if dim_idx == 0 {
            1usize
        } else {
            product_checked(&shape_ext[..dim_idx])
                .ok_or_else(|| anyhow!("iir_filter: tensor exceeds GPU limits"))?
        };
        let trailing = if dim_idx + 1 >= shape_ext.len() {
            1usize
        } else {
            product_checked(&shape_ext[dim_idx + 1..])
                .ok_or_else(|| anyhow!("iir_filter: tensor exceeds GPU limits"))?
        };
        let channel_count = leading
            .checked_mul(trailing)
            .ok_or_else(|| anyhow!("iir_filter: tensor exceeds GPU limits"))?;

        ensure!(
            shape_ext.len() <= crate::backend::wgpu::params::FILTER_MAX_RANK,
            "iir_filter: tensors exceed supported rank for GPU kernel"
        );

        let state_shape = filter_state_shape(shape_ext.clone(), dim_idx, state_len);
        ensure!(
            state_shape.len() <= crate::backend::wgpu::params::FILTER_MAX_RANK,
            "iir_filter: filter state rank exceeds GPU limits"
        );

        let state_total = if state_len == 0 {
            0usize
        } else {
            product_checked(&state_shape)
                .ok_or_else(|| anyhow!("iir_filter: filter state exceeds GPU limits"))?
        };

        if let Some(ref zi_handle) = zi {
            let zi_entry = self.get_entry(zi_handle)?;
            ensure!(
                zi_entry.precision == self.precision,
                "iir_filter: initial conditions use incompatible precision"
            );
            ensure!(
                shapes_compatible(&state_shape, &zi_entry.shape),
                "iir_filter: initial conditions are not compatible with the requested dimension"
            );
            let zi_dim = if dim_idx < zi_entry.shape.len() {
                zi_entry.shape[dim_idx]
            } else {
                1
            };
            ensure!(
                zi_dim == state_len,
                "iir_filter: initial conditions must have {} states along dimension {}",
                state_len,
                dim + 1
            );
            if state_total == 0 {
                ensure!(
                    zi_entry.len == 0,
                    "iir_filter: initial conditions have {} elements but zero were expected",
                    zi_entry.len
                );
            } else {
                ensure!(
                    zi_entry.len == state_total,
                    "iir_filter: initial state vector length mismatch (expected {}, found {})",
                    state_total,
                    zi_entry.len
                );
            }
        }

        ensure!(
            entry_x.len <= u32::MAX as usize,
            "iir_filter: signal length exceeds GPU limits"
        );
        ensure!(
            leading <= u32::MAX as usize
                && trailing <= u32::MAX as usize
                && channel_count <= u32::MAX as usize,
            "iir_filter: tensor exceeds GPU kernel limits"
        );
        ensure!(
            dim_len <= u32::MAX as usize,
            "iir_filter: dimension length exceeds GPU limits"
        );
        ensure!(
            state_len <= u32::MAX as usize,
            "iir_filter: filter order exceeds GPU limits"
        );
        ensure!(
            state_total <= u32::MAX as usize,
            "iir_filter: filter state size exceeds GPU limits"
        );

        let state_buffer_len = if state_len == 0 {
            0usize
        } else {
            state_len
                .checked_mul(channel_count)
                .ok_or_else(|| anyhow!("iir_filter: state buffer length overflow"))?
        };
        ensure!(
            state_buffer_len <= u32::MAX as usize,
            "iir_filter: state buffer length exceeds GPU limits"
        );

        let mut cleanup_handles: Vec<GpuTensorHandle> = Vec::new();
        let result = (|| -> Result<ProviderIirFilterResult> {
            let b_shape = [order, 1usize];
            let b_view = HostTensorView {
                data: &b_norm,
                shape: &b_shape,
            };
            let b_norm_handle = self.upload(&b_view)?;
            cleanup_handles.push(b_norm_handle.clone());

            let a_shape = [order, 1usize];
            let a_view = HostTensorView {
                data: &a_norm,
                shape: &a_shape,
            };
            let a_norm_handle = self.upload(&a_view)?;
            cleanup_handles.push(a_norm_handle.clone());

            let b_norm_entry = self.get_entry(&b_norm_handle)?;
            let a_norm_entry = self.get_entry(&a_norm_handle)?;

            let out_buffer = self.create_storage_buffer(entry_x.len, "runmat-iir-filter-out");
            let states_buffer =
                self.create_storage_buffer(state_buffer_len, "runmat-iir-filter-state");
            let final_state_buffer =
                self.create_storage_buffer(state_total, "runmat-iir-filter-final");

            let (zi_buffer, zi_present_flag) = if let Some(ref zi_handle) = zi {
                let zi_entry = self.get_entry(zi_handle)?;
                (zi_entry.buffer, 1u32)
            } else {
                (
                    self.create_storage_buffer(state_total, "runmat-iir-filter-zi"),
                    0u32,
                )
            };

            let mut signal_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
                crate::backend::wgpu::params::FILTER_MAX_RANK];
            for (idx, dim_len) in shape_ext.iter().enumerate() {
                signal_shape_arr[idx] =
                    crate::backend::wgpu::params::AlignedU32::new(*dim_len as u32);
            }
            let mut state_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
                crate::backend::wgpu::params::FILTER_MAX_RANK];
            for (idx, dim_len) in state_shape.iter().enumerate() {
                state_shape_arr[idx] =
                    crate::backend::wgpu::params::AlignedU32::new(*dim_len as u32);
            }

            let params = FilterParams {
                dim_len: dim_len as u32,
                leading: leading as u32,
                trailing: trailing as u32,
                order: order as u32,
                state_len: state_len as u32,
                signal_len: entry_x.len as u32,
                channel_count: channel_count as u32,
                zi_present: zi_present_flag,
                dim_idx: dim_idx as u32,
                rank: shape_ext.len() as u32,
                state_rank: state_shape.len() as u32,
                _pad: 0,
                signal_shape: signal_shape_arr,
                state_shape: state_shape_arr,
            };

            let params_buffer = self.uniform_buffer(&params, "runmat-iir-filter-params");

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-iir-filter-bind"),
                    layout: &self.pipelines.filter.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry_x.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_norm_entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: a_norm_entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: zi_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: states_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: final_state_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                channel_count as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::filter::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.filter.pipeline,
                &bind_group,
                workgroups,
            );

            let output_handle =
                self.register_existing_buffer(out_buffer, entry_x.shape.clone(), entry_x.len);
            let final_state_handle =
                self.register_existing_buffer(final_state_buffer, state_shape.clone(), state_total);

            Ok(ProviderIirFilterResult {
                output: output_handle,
                final_state: Some(final_state_handle),
            })
        })();

        for handle in cleanup_handles {
            let _ = self.free(&handle);
        }

        result
    }
    pub(crate) fn diff_once_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;

        let mut ext_shape = if entry.shape.is_empty() {
            vec![if entry.len == 0 { 1 } else { entry.len }]
        } else {
            entry.shape.clone()
        };
        while ext_shape.len() <= dim {
            ext_shape.push(1);
        }

        let len_dim = ext_shape[dim];

        let mut out_shape = entry.shape.clone();
        while out_shape.len() <= dim {
            out_shape.push(1);
        }

        if len_dim <= 1 || entry.len == 0 {
            out_shape[dim] = out_shape[dim].saturating_sub(1);
            let out_len = product_checked(&out_shape).unwrap_or(0);
            let out_buffer = self.create_storage_buffer(out_len, "runmat-diff-empty");
            return Ok(self.register_existing_buffer(out_buffer, out_shape, out_len));
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            product_checked(&ext_shape[..dim])
                .ok_or_else(|| anyhow!("diff: stride computation overflow"))?
                .max(1)
        };
        let stride_after = if dim + 1 >= ext_shape.len() {
            1usize
        } else {
            product_checked(&ext_shape[dim + 1..])
                .ok_or_else(|| anyhow!("diff: stride computation overflow"))?
                .max(1)
        };

        let expected_len = stride_before
            .checked_mul(len_dim)
            .and_then(|v| v.checked_mul(stride_after))
            .ok_or_else(|| anyhow!("diff: tensor size exceeds GPU limits"))?;
        ensure!(
            expected_len == entry.len,
            "diff: tensor shape mismatch (expected {} elements, got {})",
            expected_len,
            entry.len
        );

        let segment_out = len_dim - 1;
        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("diff: segment count exceeds GPU limits"))?;
        let out_len = segments
            .checked_mul(segment_out)
            .ok_or_else(|| anyhow!("diff: output size exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(len_dim)
            .ok_or_else(|| anyhow!("diff: block size exceeds GPU limits"))?;

        ensure!(
            len_dim <= u32::MAX as usize
                && stride_before <= u32::MAX as usize
                && stride_after <= u32::MAX as usize
                && segments <= u32::MAX as usize
                && block <= u32::MAX as usize
                && out_len <= u32::MAX as usize
                && entry.len <= u32::MAX as usize,
            "diff: tensor exceeds GPU kernel limits"
        );

        let out_buffer = self.create_storage_buffer(out_len, "runmat-diff-out");
        out_shape[dim] = len_dim - 1;

        let params = DiffParams {
            stride_before: stride_before as u32,
            segments: segments as u32,
            segment_len: len_dim as u32,
            segment_out: segment_out as u32,
            block: block as u32,
            total_out: out_len as u32,
            total_in: entry.len as u32,
            _pad: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-diff-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-diff-bind"),
                layout: &self.pipelines.diff.layout,
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

        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            out_len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::diff::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.diff.pipeline,
            &bind_group,
            workgroups,
        );

        Ok(self.register_existing_buffer(out_buffer, out_shape, out_len))
    }

    pub(crate) fn diff_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        order: usize,
    ) -> Result<GpuTensorHandle> {
        if order == 0 {
            return Ok(handle.clone());
        }

        let mut current = handle.clone();
        let mut owns_current = false;
        for _ in 0..order {
            let next = self.diff_once_exec(&current, dim)?;
            if owns_current {
                let _ = self.free(&current);
            }
            current = next;
            owns_current = true;

            let entry = self.get_entry(&current)?;
            if entry.len == 0 {
                break;
            }
        }
        Ok(current)
    }

    pub(crate) fn repmat_exec(
        &self,
        handle: &GpuTensorHandle,
        reps: &[usize],
    ) -> Result<GpuTensorHandle> {
        ensure!(
            !reps.is_empty(),
            "repmat: replication factors must be specified"
        );
        let entry = self.get_entry(handle)?;
        let orig_rank = if entry.shape.is_empty() {
            1
        } else {
            entry.shape.len()
        };
        let rank = if reps.len() == 1 {
            orig_rank.max(2)
        } else {
            orig_rank.max(reps.len())
        };
        if rank > crate::backend::wgpu::params::REPMAT_MAX_RANK {
            return Err(anyhow!(
                "repmat: rank {} exceeds GPU support (max {})",
                rank,
                crate::backend::wgpu::params::REPMAT_MAX_RANK
            ));
        }

        let mut base_shape = vec![1usize; rank];
        for (idx, &dim) in entry.shape.iter().enumerate() {
            if idx < rank {
                base_shape[idx] = dim;
            }
        }

        let mut factors = vec![1usize; rank];
        if reps.len() == 1 {
            factors.fill(reps[0]);
        } else {
            for (idx, &factor) in reps.iter().enumerate() {
                if idx < rank {
                    factors[idx] = factor;
                }
            }
        }

        let mut new_shape = Vec::with_capacity(rank);
        for i in 0..rank {
            let new_dim = base_shape[i]
                .checked_mul(factors[i])
                .ok_or_else(|| anyhow!("repmat: requested output exceeds GPU limits"))?;
            new_shape.push(new_dim);
        }

        let orig_total = base_shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| anyhow!("repmat: dimension product exceeds GPU limits"))
        })?;

        ensure!(
            orig_total == entry.len || (orig_total == 0 && entry.len == 0),
            "repmat: internal shape mismatch"
        );

        let new_total = new_shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| anyhow!("repmat: requested output exceeds GPU limits"))
        })?;

        if new_total > u32::MAX as usize {
            return Err(anyhow!("repmat: tensor too large for GPU kernel"));
        }

        if base_shape.iter().any(|&d| d > u32::MAX as usize)
            || new_shape.iter().any(|&d| d > u32::MAX as usize)
        {
            return Err(anyhow!(
                "repmat: dimensions exceed GPU kernel coordinate precision"
            ));
        }

        let mut base_strides = vec![0usize; rank];
        let mut stride = 1usize;
        for i in 0..rank {
            base_strides[i] = stride;
            stride = stride
                .checked_mul(base_shape[i].max(1))
                .ok_or_else(|| anyhow!("repmat: stride computation exceeds GPU limits"))?;
        }

        if base_strides.iter().any(|&s| s > u32::MAX as usize) {
            return Err(anyhow!(
                "repmat: source strides exceed GPU kernel coordinate precision"
            ));
        }

        let mut base_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::REPMAT_MAX_RANK];
        let mut new_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::REPMAT_MAX_RANK];
        let mut strides_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::REPMAT_MAX_RANK];
        for i in 0..rank {
            base_shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(base_shape[i] as u32);
            new_shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(new_shape[i] as u32);
            strides_arr[i] = crate::backend::wgpu::params::AlignedU32::new(base_strides[i] as u32);
        }

        // Use checked allocation so we fail with a clear error instead of
        // creating an invalid WebGPU buffer (which later triggers a validation error).
        let out_buffer = self.create_storage_buffer_checked(new_total, "runmat-repmat-out")?;
        let out_shape = new_shape.clone();
        if new_total == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, 0));
        }

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-repmat-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-repmat-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.repmat.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-repmat-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < new_total {
            let remaining = new_total - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::RepmatParams {
                len: chunk_len as u32,
                offset: offset as u32,
                rank: rank as u32,
                _pad: 0,
                base_shape: base_shape_arr,
                new_shape: new_shape_arr,
                base_strides: strides_arr,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-repmat-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-repmat-bind"),
                    layout: &self.pipelines.repmat.layout,
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
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::repmat::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.repmat.pipeline,
                &bind_group,
                workgroups,
            );
            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, out_shape, new_total))
    }
    pub(crate) fn cat_exec(
        &self,
        dim: usize,
        inputs: &[GpuTensorHandle],
    ) -> Result<GpuTensorHandle> {
        ensure!(
            inputs.len() >= 2,
            "cat: at least two input arrays are required"
        );
        ensure!(dim >= 1, "cat: dimension must be >= 1");
        let dim_zero = dim - 1;

        let mut entries = Vec::with_capacity(inputs.len());
        for handle in inputs {
            entries.push(self.get_entry(handle)?);
        }

        let precision = entries[0].precision;
        for entry in &entries {
            ensure!(
                entry.precision == precision,
                "cat: input precision mismatch"
            );
        }

        let mut shapes: Vec<Vec<usize>> = entries.iter().map(|e| e.shape.clone()).collect();
        let mut rank = shapes
            .iter()
            .map(|s| if s.is_empty() { 0 } else { s.len() })
            .max()
            .unwrap_or(1);
        rank = rank.max(dim_zero + 1);
        if rank == 0 {
            rank = 1;
        }

        for shape in &mut shapes {
            if shape.is_empty() {
                shape.push(1);
            }
            while shape.len() < rank {
                shape.push(1);
            }
        }

        for (idx, shape) in shapes.iter().enumerate() {
            let expected = product_checked(shape)
                .ok_or_else(|| anyhow!("cat: input {} exceeds GPU limits", idx + 1))?;
            ensure!(
                expected == entries[idx].len,
                "cat: input {} has {} elements but the shape multiplies to {}",
                idx + 1,
                entries[idx].len,
                expected
            );
        }

        for axis in 0..rank {
            if axis == dim_zero {
                continue;
            }
            let reference = shapes[0][axis];
            for (idx, shape) in shapes.iter().enumerate().skip(1) {
                ensure!(
                    shape[axis] == reference,
                    "cat: dimension {} mismatch between input 1 (size {}) and input {} (size {})",
                    axis + 1,
                    reference,
                    idx + 1,
                    shape[axis]
                );
            }
        }

        let mut output_shape = shapes[0].clone();
        let mut concat_dim = 0usize;
        for shape in &shapes {
            concat_dim = concat_dim
                .checked_add(shape[dim_zero])
                .ok_or_else(|| anyhow!("cat: concatenated dimension exceeds GPU limits"))?;
        }
        output_shape[dim_zero] = concat_dim;

        let total_len = product_checked(&output_shape)
            .ok_or_else(|| anyhow!("cat: resulting array exceeds GPU limits"))?;

        let normalized_shape = normalize_concat_shape(output_shape.clone(), dim_zero);

        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-cat-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, normalized_shape, 0));
        }

        let inner = product_checked(&output_shape[..dim_zero])
            .ok_or_else(|| anyhow!("cat: internal dimension overflow"))?;
        let outer = product_checked(&output_shape[dim_zero + 1..])
            .ok_or_else(|| anyhow!("cat: internal dimension overflow"))?;

        let mut encoder =
            self.device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-cat-encoder"),
                });

        let mut dst_offset_elems = 0usize;
        for outer_idx in 0..outer {
            for (entry, shape) in entries.iter().zip(shapes.iter()) {
                let mid = shape[dim_zero];
                let chunk = mid
                    .checked_mul(inner)
                    .ok_or_else(|| anyhow!("cat: chunk size overflow"))?;
                if chunk == 0 {
                    continue;
                }
                let src_offset = outer_idx
                    .checked_mul(chunk)
                    .ok_or_else(|| anyhow!("cat: source offset overflow"))?;
                let bytes = chunk
                    .checked_mul(self.element_size)
                    .ok_or_else(|| anyhow!("cat: copy size overflow"))?;
                let src_bytes = src_offset
                    .checked_mul(self.element_size)
                    .ok_or_else(|| anyhow!("cat: source offset overflow"))?;
                let dst_bytes = dst_offset_elems
                    .checked_mul(self.element_size)
                    .ok_or_else(|| anyhow!("cat: destination offset overflow"))?;
                encoder.copy_buffer_to_buffer(
                    entry.buffer.as_ref(),
                    src_bytes as u64,
                    out_buffer.as_ref(),
                    dst_bytes as u64,
                    bytes as u64,
                );
                dst_offset_elems = dst_offset_elems
                    .checked_add(chunk)
                    .ok_or_else(|| anyhow!("cat: destination offset overflow"))?;
            }
        }

        debug_assert_eq!(dst_offset_elems, total_len);

        self.submit(encoder);

        Ok(self.register_existing_buffer(out_buffer, normalized_shape, total_len))
    }
    pub(crate) fn kron_exec(
        &self,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(left)?;
        let entry_b = self.get_entry(right)?;

        let rank = entry_a.shape.len().max(entry_b.shape.len()).max(1);
        ensure!(
            rank <= crate::backend::wgpu::params::KRON_MAX_RANK,
            "kron: rank {} exceeds GPU support (max {})",
            rank,
            crate::backend::wgpu::params::KRON_MAX_RANK
        );

        let mut shape_a = vec![1usize; rank];
        for (idx, &dim) in entry_a.shape.iter().enumerate() {
            if idx < rank {
                shape_a[idx] = dim;
            }
        }
        let mut shape_b = vec![1usize; rank];
        for (idx, &dim) in entry_b.shape.iter().enumerate() {
            if idx < rank {
                shape_b[idx] = dim;
            }
        }

        let mut shape_out = Vec::with_capacity(rank);
        for i in 0..rank {
            let dim = shape_a[i]
                .checked_mul(shape_b[i])
                .ok_or_else(|| anyhow!("kron: requested output exceeds GPU limits"))?;
            shape_out.push(dim);
        }

        let len_a = product_checked(&shape_a)
            .ok_or_else(|| anyhow!("kron: left operand size exceeds GPU limits"))?;
        let len_b = product_checked(&shape_b)
            .ok_or_else(|| anyhow!("kron: right operand size exceeds GPU limits"))?;
        let len_out = product_checked(&shape_out)
            .ok_or_else(|| anyhow!("kron: output size exceeds GPU limits"))?;

        ensure!(
            len_a == entry_a.len || (len_a == 0 && entry_a.len == 0),
            "kron: left operand shape mismatch"
        );
        ensure!(
            len_b == entry_b.len || (len_b == 0 && entry_b.len == 0),
            "kron: right operand shape mismatch"
        );

        if len_out == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-kron-out");
            return Ok(self.register_existing_buffer(out_buffer, shape_out, 0));
        }

        if len_out > u32::MAX as usize {
            return Err(anyhow!("kron: tensor too large for GPU kernel"));
        }

        for &dim in &shape_out {
            if dim > u32::MAX as usize {
                return Err(anyhow!(
                    "kron: dimensions exceed GPU kernel coordinate precision"
                ));
            }
        }

        let mut strides_a = vec![0usize; rank];
        let mut stride = 1usize;
        for i in 0..rank {
            strides_a[i] = stride;
            stride = stride
                .checked_mul(shape_a[i].max(1))
                .ok_or_else(|| anyhow!("kron: left stride overflow"))?;
        }

        let mut strides_b = vec![0usize; rank];
        stride = 1usize;
        for i in 0..rank {
            strides_b[i] = stride;
            stride = stride
                .checked_mul(shape_b[i].max(1))
                .ok_or_else(|| anyhow!("kron: right stride overflow"))?;
        }

        for &value in &strides_a {
            if value > u32::MAX as usize {
                return Err(anyhow!(
                    "kron: left strides exceed GPU kernel coordinate precision"
                ));
            }
        }
        for &value in &strides_b {
            if value > u32::MAX as usize {
                return Err(anyhow!(
                    "kron: right strides exceed GPU kernel coordinate precision"
                ));
            }
        }

        let mut shape_a_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::KRON_MAX_RANK];
        let mut shape_b_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::KRON_MAX_RANK];
        let mut shape_out_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::KRON_MAX_RANK];
        let mut stride_a_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::KRON_MAX_RANK];
        let mut stride_b_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::KRON_MAX_RANK];
        for i in 0..rank {
            shape_a_arr[i] = crate::backend::wgpu::params::AlignedU32::new(shape_a[i] as u32);
            shape_b_arr[i] = crate::backend::wgpu::params::AlignedU32::new(shape_b[i] as u32);
            shape_out_arr[i] = crate::backend::wgpu::params::AlignedU32::new(shape_out[i] as u32);
            stride_a_arr[i] = crate::backend::wgpu::params::AlignedU32::new(strides_a[i] as u32);
            stride_b_arr[i] = crate::backend::wgpu::params::AlignedU32::new(strides_b[i] as u32);
        }

        let out_buffer = self.create_storage_buffer_checked(len_out, "runmat-kron-out")?;
        let out_shape = shape_out.clone();

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-kron-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-kron-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.kron.pipeline);
            drop(pass);
            self.submit(enc);
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-kron-flush-gap"),
                });
            self.submit(enc);
        }

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < len_out {
            let remaining = len_out - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::KronParams {
                len: chunk_len as u32,
                offset: offset as u32,
                rank: rank as u32,
                _pad: 0,
                shape_a: shape_a_arr,
                shape_b: shape_b_arr,
                shape_out: shape_out_arr,
                stride_a: stride_a_arr,
                stride_b: stride_b_arr,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-kron-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-kron-bind"),
                    layout: &self.pipelines.kron.layout,
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
            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::kron::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.kron.pipeline,
                &bind_group,
                workgroups,
            );
            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, out_shape, len_out))
    }

    pub(crate) fn transpose_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if entry.shape.len() != 2 {
            return Err(anyhow!("transpose: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let len = entry.len;

        if let Some(info) = runmat_accelerate_api::handle_transpose_info(a) {
            let base_rows = info.base_rows;
            let base_cols = info.base_cols;
            let shape = vec![base_rows, base_cols];
            let handle = self.register_existing_buffer(entry.buffer.clone(), shape, len);
            runmat_accelerate_api::clear_handle_transpose(&handle);
            return Ok(handle);
        }

        let shape = vec![cols, rows];
        let handle = self.register_existing_buffer(entry.buffer.clone(), shape, len);
        runmat_accelerate_api::record_handle_transpose(&handle, rows, cols);
        Ok(handle)
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

    pub(crate) fn eye_exec(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let normalized = normalize_eye_shape(shape);
        if normalized.len() < 2 {
            return Err(anyhow!("eye: expected at least 2 dimensions"));
        }
        let total_len = product_checked(&normalized)
            .ok_or_else(|| anyhow!("eye: tensor size exceeds GPU limits"))?;
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-eye-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, normalized, total_len));
        }

        let rows = normalized[0];
        let cols = normalized[1];
        let diag_len = rows.min(cols);
        if diag_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, normalized, total_len));
        }
        let slice_stride = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("eye: matrix slice exceeds GPU limits"))?;
        let slices = if normalized.len() <= 2 {
            1usize
        } else {
            product_checked(&normalized[2..])
                .ok_or_else(|| anyhow!("eye: slice count exceeds GPU limits"))?
        };
        let diag_total = diag_len
            .checked_mul(slices)
            .ok_or_else(|| anyhow!("eye: diagonal count exceeds GPU limits"))?;
        if diag_total == 0 {
            return Ok(self.register_existing_buffer(out_buffer, normalized, total_len));
        }
        if rows > (u32::MAX as usize)
            || cols > (u32::MAX as usize)
            || slice_stride > (u32::MAX as usize)
            || diag_total > (u32::MAX as usize)
        {
            return Err(anyhow!("eye: dimensions exceed GPU dispatch limits"));
        }

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-eye-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-eye-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.eye.pipeline);
            drop(pass);
            self.submit(enc);
        }

        let params = crate::backend::wgpu::params::EyeParams {
            rows: rows as u32,
            cols: cols as u32,
            diag_len: diag_len as u32,
            slices: slices as u32,
            stride_slice: slice_stride as u32,
            diag_total: diag_total as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-eye-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-eye-bind"),
                layout: &self.pipelines.eye.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            diag_total as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::creation::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.eye.pipeline,
            &bind_group,
            workgroups,
            "runmat-eye-encoder",
            "runmat-eye-pass",
        );

        Ok(self.register_existing_buffer(out_buffer, normalized, total_len))
    }
    pub(crate) fn cumsum_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        // For reverse scans, compute as flip → forward-scan → flip to preserve exact semantics
        if matches!(direction, ProviderScanDirection::Reverse) {
            let flipped_in = self.flip_exec(handle, &[dim])?;
            let forward =
                self.cumsum_exec(&flipped_in, dim, ProviderScanDirection::Forward, nan_mode)?;
            let _ = self.free(&flipped_in);
            let flipped_out = self.flip_exec(&forward, &[dim])?;
            let _ = self.free(&forward);
            return Ok(flipped_out);
        }
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cumsum: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-cumsum-empty");
            return Ok(self.register_existing_buffer(out_buffer, entry.shape, 0));
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cumsum: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cumsum: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cumsum: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cumsum: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cumsum: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cumsum: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cumsum: tensor too large for GPU kernel"
        );

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-cumsum-out");
        if entry.len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry.shape, 0));
        }

        let mut flags = 0u32;
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CumsumParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-cumsum-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-cumsum-bind"),
                layout: &self.pipelines.cumsum.layout,
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
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            segments as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::scan::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.cumsum.pipeline,
            &bind_group,
            groups,
            "runmat-cumsum-encoder",
            "runmat-cumsum-pass",
        );
        Ok(self.register_existing_buffer(out_buffer, entry.shape, entry.len))
    }
    pub(crate) fn cumprod_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        // For reverse scans, compute as flip → forward-scan → flip to preserve exact semantics
        if matches!(direction, ProviderScanDirection::Reverse) {
            let flipped_in = self.flip_exec(handle, &[dim])?;
            let forward =
                self.cumprod_exec(&flipped_in, dim, ProviderScanDirection::Forward, nan_mode)?;
            let _ = self.free(&flipped_in);
            let flipped_out = self.flip_exec(&forward, &[dim])?;
            let _ = self.free(&forward);
            return Ok(flipped_out);
        }
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cumprod: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-cumprod-empty");
            return Ok(self.register_existing_buffer(out_buffer, entry.shape, 0));
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cumprod: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cumprod: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cumprod: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cumprod: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cumprod: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cumprod: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cumprod: tensor too large for GPU kernel"
        );

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-cumprod-out");
        if entry.len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry.shape, 0));
        }

        let mut flags = 0u32;
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CumprodParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-cumprod-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-cumprod-bind"),
                layout: &self.pipelines.cumprod.layout,
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
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            segments as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::scan::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.cumprod.pipeline,
            &bind_group,
            groups,
            "runmat-cumprod-encoder",
            "runmat-cumprod-pass",
        );
        Ok(self.register_existing_buffer(out_buffer, entry.shape, entry.len))
    }
    pub(crate) fn cummin_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<ProviderCumminResult> {
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            let values_buffer = self.create_storage_buffer(0, "runmat-cummin-values-empty");
            let indices_buffer = self.create_storage_buffer(0, "runmat-cummin-indices-empty");
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
            return Ok(ProviderCumminResult { values, indices });
        }

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cummin: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let values_buffer = self.create_storage_buffer(0, "runmat-cummin-values-empty");
            let indices_buffer = self.create_storage_buffer(0, "runmat-cummin-indices-empty");
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
            return Ok(ProviderCumminResult { values, indices });
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cummin: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cummin: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cummin: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cummin: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cummin: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cummin: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cummin: tensor too large for GPU kernel"
        );

        let values_buffer = self.create_storage_buffer(entry.len, "runmat-cummin-values");
        let indices_buffer = self.create_storage_buffer(entry.len, "runmat-cummin-indices");

        let mut flags = 0u32;
        if matches!(direction, ProviderScanDirection::Reverse) {
            flags |= 1;
        }
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CumminParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-cummin-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-cummin-bind"),
                layout: &self.pipelines.cummin.layout,
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

        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            segments as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::scan::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.cummin.pipeline,
            &bind_group,
            groups,
            "runmat-cummin-encoder",
            "runmat-cummin-pass",
        );

        let values = self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
        let indices = self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
        Ok(ProviderCumminResult { values, indices })
    }
    pub(crate) fn cummax_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<ProviderCummaxResult> {
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            let values_buffer = self.create_storage_buffer(0, "runmat-cummax-values-empty");
            let indices_buffer = self.create_storage_buffer(0, "runmat-cummax-indices-empty");
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
            return Ok(ProviderCummaxResult { values, indices });
        }

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cummax: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let values_buffer = self.create_storage_buffer(0, "runmat-cummax-values-empty");
            let indices_buffer = self.create_storage_buffer(0, "runmat-cummax-indices-empty");
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
            return Ok(ProviderCummaxResult { values, indices });
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cummax: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cummax: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cummax: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cummax: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cummax: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cummax: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cummax: tensor too large for GPU kernel"
        );

        let values_buffer = self.create_storage_buffer(entry.len, "runmat-cummax-values");
        let indices_buffer = self.create_storage_buffer(entry.len, "runmat-cummax-indices");

        let mut flags = 0u32;
        if matches!(direction, ProviderScanDirection::Reverse) {
            flags |= 1;
        }
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CummaxParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-cummax-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-cummax-bind"),
                layout: &self.pipelines.cummax.layout,
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

        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            segments as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::scan::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.cummax.pipeline,
            &bind_group,
            groups,
            "runmat-cummax-encoder",
            "runmat-cummax-pass",
        );

        let values = self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
        let indices = self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
        Ok(ProviderCummaxResult { values, indices })
    }
    pub(crate) fn fill_exec(&self, shape: &[usize], value: f64) -> Result<GpuTensorHandle> {
        let total_len = product_checked(shape)
            .ok_or_else(|| anyhow!("fill: tensor size exceeds GPU limits"))?;
        let shape_vec = shape.to_vec();
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-fill-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape_vec, 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "fill: tensor length exceeds GPU dispatch limits"
        );

        // chunked dispatch below will build per-chunk params buffers

        // Dispatch in chunks to satisfy per-dimension group limits (<= 65535)
        let wg_size = crate::backend::wgpu::config::WORKGROUP_SIZE;
        let max_groups: u32 = 65535;
        let max_elems_per_dispatch = (max_groups as usize) * (wg_size as usize);
        let mut processed: usize = 0;
        while processed < total_len {
            let remain = total_len - processed;
            let chunk_len = remain.min(max_elems_per_dispatch);

            // Per-chunk params (length)
            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = crate::backend::wgpu::params::FillParamsF64 {
                        value,
                        len: chunk_len as u32,
                        _pad: [0, 0, 0],
                    };
                    self.uniform_buffer(&params, "runmat-fill-params-f64")
                }
                NumericPrecision::F32 => {
                    let params = crate::backend::wgpu::params::FillParamsF32 {
                        value: value as f32,
                        len: chunk_len as u32,
                        _pad: [0, 0],
                    };
                    self.uniform_buffer(&params, "runmat-fill-params-f32")
                }
            };

            // Bind only the range for this chunk
            let byte_offset = (processed * self.element_size) as u64;
            let byte_size = std::num::NonZeroU64::new((chunk_len * self.element_size) as u64);
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-fill-bind"),
                    layout: &self.pipelines.fill.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: out_buffer.as_ref(),
                                offset: byte_offset,
                                size: byte_size,
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups =
                crate::backend::wgpu::dispatch::common::dispatch_size(chunk_len as u32, wg_size);
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.fill.pipeline,
                &bind_group,
                workgroups,
                "runmat-fill-encoder",
                "runmat-fill-pass",
            );

            processed += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, shape_vec, total_len))
    }
    pub(crate) async fn imfilter_exec(
        &self,
        image: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: &ImfilterOptions,
    ) -> Result<GpuTensorHandle> {
        if std::env::var("RUNMAT_WGPU_DISABLE_IMFILTER")
            .ok()
            .and_then(|v| match v.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" => Some(true),
                "0" | "false" | "no" => Some(false),
                _ => None,
            })
            .unwrap_or(false)
        {
            return self.imfilter_exec_fallback(image, kernel, options).await;
        }
        let image_entry = self.get_entry(image)?;
        let kernel_host = <Self as AccelProvider>::download(self, kernel).await?;
        let kernel_tensor = Tensor::new(kernel_host.data.clone(), kernel_host.shape.clone())
            .map_err(|e| anyhow!("imfilter: {e}"))?;

        let image_shape = if image_entry.shape.is_empty() {
            vec![1usize]
        } else {
            image_entry.shape.clone()
        };

        let plan = match build_imfilter_plan(&image_shape, &kernel_tensor, options, "imfilter") {
            Ok(plan) => plan,
            Err(err) => return Err(anyhow!(err)),
        };

        if plan.rank > crate::backend::wgpu::params::IMFILTER_MAX_RANK {
            return self.imfilter_exec_fallback(image, kernel, options).await;
        }

        let image_ext_product = plan
            .image_shape_ext
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("imfilter: image dimensions exceed GPU limits"))?;
        if image_ext_product != image_entry.len {
            return self.imfilter_exec_fallback(image, kernel, options).await;
        }

        let output_len = plan
            .output_shape_ext
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("imfilter: output dimensions exceed GPU limits"))?;
        if output_len > u32::MAX as usize || image_entry.len > u32::MAX as usize {
            return self.imfilter_exec_fallback(image, kernel, options).await;
        }

        let kernel_points_len = plan.kernel_points.len();
        if kernel_points_len > u32::MAX as usize {
            return self.imfilter_exec_fallback(image, kernel, options).await;
        }

        let mut kernel_offsets = Vec::with_capacity(kernel_points_len * plan.rank);
        let mut kernel_values_f64 = Vec::with_capacity(kernel_points_len);
        for point in &plan.kernel_points {
            if point.offsets.len() != plan.rank {
                return self.imfilter_exec_fallback(image, kernel, options).await;
            }
            for &offset in &point.offsets {
                if offset < i32::MIN as isize || offset > i32::MAX as isize {
                    return self.imfilter_exec_fallback(image, kernel, options).await;
                }
                kernel_offsets.push(offset as i32);
            }
            kernel_values_f64.push(point.value);
        }

        if kernel_offsets.len() > u32::MAX as usize {
            return self.imfilter_exec_fallback(image, kernel, options).await;
        }

        let kernel_offsets_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("runmat-imfilter-kernel-offsets"),
                    contents: bytemuck::cast_slice(&kernel_offsets),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let kernel_values_buffer = match self.precision {
            NumericPrecision::F64 => {
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-imfilter-kernel-values-f64"),
                        contents: bytemuck::cast_slice(&kernel_values_f64),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            }
            NumericPrecision::F32 => {
                let kernel_values_f32: Vec<f32> =
                    kernel_values_f64.iter().map(|&v| v as f32).collect();
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-imfilter-kernel-values-f32"),
                        contents: bytemuck::cast_slice(&kernel_values_f32),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            }
        };

        let out_buffer = self.create_storage_buffer_checked(output_len, "runmat-imfilter-out")?;
        if output_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, plan.final_shape.clone(), 0));
        }

        let mut image_shape_arr = [crate::backend::wgpu::params::AlignedU32::default();
            crate::backend::wgpu::params::IMFILTER_MAX_RANK];
        let mut image_strides_arr = [crate::backend::wgpu::params::AlignedU32::default();
            crate::backend::wgpu::params::IMFILTER_MAX_RANK];
        let mut output_shape_arr = [crate::backend::wgpu::params::AlignedU32::default();
            crate::backend::wgpu::params::IMFILTER_MAX_RANK];
        let mut base_offset_arr = [crate::backend::wgpu::params::PackedI32::default();
            crate::backend::wgpu::params::IMFILTER_MAX_RANK];

        for i in 0..plan.rank {
            let dim = plan.image_shape_ext[i];
            ensure!(
                dim <= u32::MAX as usize,
                "imfilter: image dimension exceeds GPU limits"
            );
            image_shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(dim as u32);

            let stride = plan.image_strides[i];
            ensure!(
                stride <= u32::MAX as usize,
                "imfilter: image stride exceeds GPU limits"
            );
            image_strides_arr[i] = crate::backend::wgpu::params::AlignedU32::new(stride as u32);

            let out_dim = plan.output_shape_ext[i];
            ensure!(
                out_dim <= u32::MAX as usize,
                "imfilter: output dimension exceeds GPU limits"
            );
            output_shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(out_dim as u32);

            let offset = plan.base_offset[i];
            ensure!(
                offset >= i32::MIN as isize && offset <= i32::MAX as isize,
                "imfilter: base offset exceeds GPU limits"
            );
            base_offset_arr[i] =
                crate::backend::wgpu::params::PackedI32::from_scalar(offset as i32);
        }

        let padding_mode = match options.padding {
            ImfilterPadding::Constant => 0u32,
            ImfilterPadding::Replicate => 1u32,
            ImfilterPadding::Symmetric => 2u32,
            ImfilterPadding::Circular => 3u32,
        };

        let kernel_points_u32 = kernel_points_len as u32;
        let image_len_u32 = image_entry.len as u32;

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-imfilter-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-imfilter-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.imfilter.pipeline);
            drop(pass);
            self.submit(enc);
        }

        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-imfilter-flush-gap"),
                });
            self.submit(enc);
        }

        let mut offset = 0usize;
        while offset < output_len {
            let remaining = output_len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let offset_u32 = offset as u32;
            let chunk_u32 = chunk_len as u32;

            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = crate::backend::wgpu::params::ImfilterParamsF64 {
                        len: chunk_u32,
                        offset: offset_u32,
                        rank: plan.rank as u32,
                        padding: padding_mode,
                        kernel_points: kernel_points_u32,
                        image_len: image_len_u32,
                        _pad0: 0,
                        _pad1: 0,
                        constant_value: options.constant_value,
                        _pad_const: 0.0,
                        image_shape: image_shape_arr,
                        image_strides: image_strides_arr,
                        output_shape: output_shape_arr,
                        base_offset: base_offset_arr,
                        _pad_tail: crate::backend::wgpu::params::AlignedU32::default(),
                    };

                    self.uniform_buffer(&params, "runmat-imfilter-params-f64")
                }
                NumericPrecision::F32 => {
                    let params = crate::backend::wgpu::params::ImfilterParamsF32 {
                        len: chunk_u32,
                        offset: offset_u32,
                        rank: plan.rank as u32,
                        padding: padding_mode,
                        kernel_points: kernel_points_u32,
                        image_len: image_len_u32,
                        _pad0: 0,
                        _pad1: 0,
                        constant_value: options.constant_value as f32,
                        _pad_const: [0.0; 3],
                        image_shape: image_shape_arr,
                        image_strides: image_strides_arr,
                        output_shape: output_shape_arr,
                        base_offset: base_offset_arr,
                        _pad_tail: crate::backend::wgpu::params::AlignedU32::default(),
                    };

                    self.uniform_buffer(&params, "runmat-imfilter-params-f32")
                }
            };

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-imfilter-bind"),
                    layout: &self.pipelines.imfilter.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: image_entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: kernel_offsets_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: kernel_values_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );

            crate::backend::wgpu::dispatch::imfilter::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.imfilter.pipeline,
                &bind_group,
                workgroups,
            );

            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, plan.final_shape.clone(), output_len))
    }

    pub(crate) fn zeros_exec(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let len: usize = shape.iter().copied().product();
        let buffer = self.create_storage_buffer_checked(len, "zeros")?;
        // Explicitly zero-initialize the storage buffer; pooled buffers may contain old data
        let size_bytes = (len.max(1) as u64) * (self.element_size as u64);
        if size_bytes > 0 {
            // Write zeros across the entire buffer
            let zero_bytes = vec![0u8; size_bytes as usize];
            self.queue.write_buffer(buffer.as_ref(), 0, &zero_bytes);
        }
        Ok(self.register_existing_buffer(buffer, shape.to_vec(), len))
    }

    pub(crate) fn meshgrid_exec(
        &self,
        axes: &[MeshgridAxisView<'_>],
    ) -> Result<ProviderMeshgridResult> {
        ensure!(
            axes.len() == 2 || axes.len() == 3,
            "meshgrid: provider expects two or three axes"
        );

        let x_axis = axes
            .first()
            .ok_or_else(|| anyhow!("meshgrid: missing X axis"))?
            .data;
        let y_axis = axes
            .get(1)
            .ok_or_else(|| anyhow!("meshgrid: missing Y axis"))?
            .data;
        let z_axis = axes.get(2).map(|axis| axis.data);

        let nx = x_axis.len();
        let ny = y_axis.len();
        let nz = z_axis.map(|axis| axis.len()).unwrap_or(1);

        let shape = if nz == 1 {
            vec![ny, nx]
        } else {
            vec![ny, nx, nz]
        };

        let total = product_checked(&shape)
            .ok_or_else(|| anyhow!("meshgrid: tensor size exceeds GPU limits"))?;

        let mut x_data = Vec::with_capacity(total);
        let mut y_data = Vec::with_capacity(total);
        let mut z_data = z_axis.map(|_| Vec::with_capacity(total));

        if let Some(axis) = z_axis {
            for &z_value in axis.iter().take(nz) {
                for &x_value in x_axis.iter().take(nx) {
                    for &y_value in y_axis.iter().take(ny) {
                        x_data.push(x_value);
                        y_data.push(y_value);
                        if let Some(ref mut z_vec) = z_data {
                            z_vec.push(z_value);
                        }
                    }
                }
            }
        } else {
            for &x_value in x_axis.iter().take(nx) {
                for &y_value in y_axis.iter().take(ny) {
                    x_data.push(x_value);
                    y_data.push(y_value);
                }
            }
        }

        let shape_slice = &shape;
        let x_view = HostTensorView {
            data: &x_data,
            shape: shape_slice,
        };
        let y_view = HostTensorView {
            data: &y_data,
            shape: shape_slice,
        };
        let x_handle = <Self as AccelProvider>::upload(self, &x_view)?;
        let y_handle = <Self as AccelProvider>::upload(self, &y_view)?;

        let mut outputs = vec![x_handle, y_handle];

        if let Some(z_vec) = z_data {
            let z_view = HostTensorView {
                data: &z_vec,
                shape: shape_slice,
            };
            let z_handle = <Self as AccelProvider>::upload(self, &z_view)?;
            outputs.push(z_handle);
        }

        Ok(ProviderMeshgridResult { outputs })
    }

    async fn imfilter_exec_fallback(
        &self,
        image: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: &ImfilterOptions,
    ) -> Result<GpuTensorHandle> {
        let image_host = <Self as AccelProvider>::download(self, image).await?;
        let kernel_host = <Self as AccelProvider>::download(self, kernel).await?;

        let image_tensor = Tensor::new(image_host.data.clone(), image_host.shape.clone())
            .map_err(|e| anyhow!("imfilter: {e}"))?;
        let kernel_tensor = Tensor::new(kernel_host.data.clone(), kernel_host.shape.clone())
            .map_err(|e| anyhow!("imfilter: {e}"))?;

        let result =
            runtime_apply_imfilter_tensor(&image_tensor, &kernel_tensor, options, "imfilter")
                .map_err(|err| anyhow!(err))?;
        let data_owned = result.data;
        let shape_owned = result.shape;
        let view = HostTensorView {
            data: &data_owned,
            shape: &shape_owned,
        };
        let handle = <Self as AccelProvider>::upload(self, &view)?;
        Ok(handle)
    }
    pub(crate) fn random_uniform_exec(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let total_len = product_checked(shape)
            .ok_or_else(|| anyhow!("rand: tensor size exceeds GPU limits"))?;
        if std::env::var("RUNMAT_DISABLE_RNG").is_ok()
            || std::env::var("RUNMAT_DISABLE_RAND").is_ok()
            || std::env::var("RUNMAT_DISABLE_RANDUNIFORM").is_ok()
        {
            // Debug-only CPU fallback: fill with deterministic values in [0,1)
            let mut out = vec![0.0f64; total_len];
            for (i, value) in out.iter_mut().enumerate().take(total_len) {
                *value = ((i as u64).wrapping_mul(1664525).wrapping_add(1013904223) % (1u64 << 32))
                    as f64
                    / 4294967296.0f64;
            }
            return self.upload(&HostTensorView { data: &out, shape });
        }
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-rng-uniform-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "rand: tensor length too large"
        );

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut rng_guard = rng_state()
            .lock()
            .map_err(|_| anyhow!("rand: provider RNG mutex poisoned"))?;
        let mut chunk_state = *rng_guard;

        let mut offset = 0usize;
        while offset < total_len {
            let remaining = total_len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let chunk_u32 = u32::try_from(chunk_len)
                .map_err(|_| anyhow!("rand: chunk length exceeds GPU dispatch limits"))?;
            let offset_u32 = u32::try_from(offset)
                .map_err(|_| anyhow!("rand: tensor offset exceeds GPU limits"))?;
            let (key0, key1) = philox_keys_from_state(chunk_state);

            let params = crate::backend::wgpu::params::RandomScalarParams {
                offset: offset_u32,
                chunk: chunk_u32,
                key0,
                key1,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-rng-uniform-params");

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-rng-uniform-bind"),
                    layout: &self.pipelines.random_uniform.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.random_uniform.pipeline,
                &bind_group,
                workgroups,
                "runmat-rng-uniform-encoder",
                "runmat-rng-uniform-pass",
            );

            chunk_state = advance_rng_state(chunk_state, u64::from(chunk_u32));
            offset += chunk_len;
        }

        *rng_guard = chunk_state;
        drop(rng_guard);

        Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), total_len))
    }

    pub(crate) fn random_normal_exec(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let total_len = product_checked(shape)
            .ok_or_else(|| anyhow!("randn: tensor size exceeds GPU limits"))?;
        if std::env::var("RUNMAT_DISABLE_RNG").is_ok()
            || std::env::var("RUNMAT_DISABLE_RAND").is_ok()
            || std::env::var("RUNMAT_DISABLE_RANDN").is_ok()
        {
            // Debug-only CPU fallback: simple deterministic normal-ish via Box-Muller on uint LCG
            let mut out = vec![0.0f64; total_len];
            let mut state: u64 = 0x9e3779b97f4a7c15;
            let next_u32 = |s: &mut u64| -> u32 {
                *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *s as u32
            };
            let mut i = 0usize;
            while i < total_len {
                let u1 = (next_u32(&mut state) as f64 + 1.0) / 4294967297.0;
                let u2 = (next_u32(&mut state) as f64 + 1.0) / 4294967297.0;
                let r = (-2.0f64 * u1.ln()).sqrt();
                let theta = 2.0f64 * std::f64::consts::PI * u2;
                let z0 = r * theta.cos();
                let z1 = r * theta.sin();
                out[i] = z0;
                if i + 1 < total_len {
                    out[i + 1] = z1;
                }
                i += 2;
            }
            return self.upload(&HostTensorView { data: &out, shape });
        }
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-rng-normal-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "randn: tensor length too large"
        );

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut rng_guard = rng_state()
            .lock()
            .map_err(|_| anyhow!("randn: provider RNG mutex poisoned"))?;
        let mut chunk_state = *rng_guard;

        let mut offset = 0usize;
        while offset < total_len {
            let remaining = total_len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let chunk_u32 = u32::try_from(chunk_len)
                .map_err(|_| anyhow!("randn: chunk length exceeds GPU dispatch limits"))?;
            let offset_u32 = u32::try_from(offset)
                .map_err(|_| anyhow!("randn: tensor offset exceeds GPU limits"))?;
            let (key0, key1) = philox_keys_from_state(chunk_state);

            let params = crate::backend::wgpu::params::RandomScalarParams {
                offset: offset_u32,
                chunk: chunk_u32,
                key0,
                key1,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-rng-normal-params");

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-rng-normal-bind"),
                    layout: &self.pipelines.random_normal.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.random_normal.pipeline,
                &bind_group,
                workgroups,
                "runmat-rng-normal-encoder",
                "runmat-rng-normal-pass",
            );

            let delta = u64::from(chunk_u32) * 2;
            chunk_state = advance_rng_state(chunk_state, delta);
            offset += chunk_len;
        }

        *rng_guard = chunk_state;
        drop(rng_guard);

        Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), total_len))
    }

    pub(crate) fn stochastic_evolution_exec(
        &self,
        state: &GpuTensorHandle,
        drift: f64,
        scale: f64,
        steps: u32,
    ) -> Result<GpuTensorHandle> {
        let total_len = product_checked(&state.shape)
            .ok_or_else(|| anyhow!("stochastic_evolution: tensor size exceeds GPU limits"))?;
        let start = Instant::now();
        let state_entry = self.get_entry(state)?;
        let out_buffer =
            self.create_storage_buffer_checked(total_len, "runmat-stochastic-evolution-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, state_entry.shape.clone(), 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "stochastic_evolution: tensor length too large"
        );

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut rng_guard = rng_state()
            .lock()
            .map_err(|_| anyhow!("stochastic_evolution: provider RNG mutex poisoned"))?;
        let mut chunk_state = *rng_guard;

        let pipeline = &self.pipelines.stochastic_evolution;
        let mut offset = 0usize;
        while offset < total_len {
            let remaining = total_len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let chunk_u32 = u32::try_from(chunk_len).map_err(|_| {
                anyhow!("stochastic_evolution: chunk length exceeds GPU dispatch limits")
            })?;
            let offset_u32 = u32::try_from(offset)
                .map_err(|_| anyhow!("stochastic_evolution: offset exceeds GPU limits"))?;
            let len_u32 = u32::try_from(total_len)
                .map_err(|_| anyhow!("stochastic_evolution: len overflow"))?;
            let (key0, key1) = philox_keys_from_state(chunk_state);

            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = crate::backend::wgpu::params::StochasticEvolutionParamsF64 {
                        offset: offset_u32,
                        chunk: chunk_u32,
                        len: len_u32,
                        steps,
                        key0,
                        key1,
                        _pad0: 0,
                        _pad1: 0,
                        drift,
                        scale,
                    };
                    self.uniform_buffer(&params, "runmat-stochastic-evolution-f64-params")
                }
                NumericPrecision::F32 => {
                    let params = crate::backend::wgpu::params::StochasticEvolutionParamsF32 {
                        offset: offset_u32,
                        chunk: chunk_u32,
                        len: len_u32,
                        steps,
                        key0,
                        key1,
                        _pad0: 0,
                        _pad1: 0,
                        drift: drift as f32,
                        scale: scale as f32,
                    };
                    self.uniform_buffer(&params, "runmat-stochastic-evolution-f32-params")
                }
            };

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-stochastic-evolution-bind"),
                    layout: &pipeline.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: state_entry.buffer.as_ref().as_entire_binding(),
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

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &pipeline.pipeline,
                &bind_group,
                workgroups,
                "runmat-stochastic-evolution-encoder",
                "runmat-stochastic-evolution-pass",
            );

            if steps > 0 {
                let advance = u64::from(chunk_u32) * u64::from(steps);
                chunk_state = advance_rng_state(chunk_state, advance);
            }
            offset += chunk_len;
        }

        *rng_guard = chunk_state;
        drop(rng_guard);

        self.telemetry
            .record_fused_elementwise_duration(start.elapsed());
        Ok(self.register_existing_buffer(out_buffer, state_entry.shape.clone(), total_len))
    }
    pub(crate) fn fspecial_exec(&self, request: &FspecialRequest) -> Result<GpuTensorHandle> {
        let spec =
            runtime_fspecial_spec_from_request(&request.filter).map_err(|err| anyhow!(err))?;

        let (rows, cols, kind, sigma, alpha, norm, center_x, center_y) = match &spec {
            FspecialFilterSpec::Average { rows, cols } => (
                *rows,
                *cols,
                0u32,
                0.0,
                0.0,
                1.0 / ((*rows as f64) * (*cols as f64)),
                0.0,
                0.0,
            ),
            FspecialFilterSpec::Gaussian { rows, cols, sigma } => {
                let norm = gaussian_normalizer(*rows, *cols, *sigma);
                ensure!(
                    norm.is_finite() && norm > 0.0,
                    "fspecial: gaussian normaliser invalid"
                );
                (
                    *rows,
                    *cols,
                    1u32,
                    *sigma,
                    0.0,
                    norm,
                    ((*cols as f64) - 1.0) / 2.0,
                    ((*rows as f64) - 1.0) / 2.0,
                )
            }
            FspecialFilterSpec::Laplacian { alpha } => {
                let norm = 4.0 / (alpha + 1.0);
                (3, 3, 2u32, 0.0, *alpha, norm, 0.0, 0.0)
            }
            FspecialFilterSpec::Prewitt => (3, 3, 3u32, 0.0, 0.0, 1.0, 0.0, 0.0),
            FspecialFilterSpec::Sobel => (3, 3, 4u32, 0.0, 0.0, 1.0, 0.0, 0.0),
            FspecialFilterSpec::Unsharp { alpha } => {
                let norm = 1.0 / (alpha + 1.0);
                (3, 3, 5u32, 0.0, *alpha, norm, 0.0, 0.0)
            }
            _ => {
                return Err(anyhow!(
                    "fspecial: filter not yet accelerated on the WGPU backend"
                ))
            }
        };

        ensure!(
            rows <= u32::MAX as usize && cols <= u32::MAX as usize,
            "fspecial: kernel dimensions exceed GPU limits"
        );
        let shape_vec = vec![rows, cols];
        let total_len = product_checked(&shape_vec)
            .ok_or_else(|| anyhow!("fspecial: tensor size exceeds GPU limits"))?;
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-fspecial-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape_vec, 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "fspecial: tensor length exceeds GPU dispatch limits"
        );

        let params_buffer = match self.precision {
            NumericPrecision::F64 => {
                let params = crate::backend::wgpu::params::FspecialParamsF64 {
                    rows: rows as u32,
                    cols: cols as u32,
                    kind,
                    len: total_len as u32,
                    sigma,
                    alpha,
                    norm,
                    center_x,
                    center_y,
                    extra0: 0.0,
                };
                self.uniform_buffer(&params, "runmat-fspecial-params-f64")
            }
            NumericPrecision::F32 => {
                let params = crate::backend::wgpu::params::FspecialParamsF32 {
                    rows: rows as u32,
                    cols: cols as u32,
                    kind,
                    len: total_len as u32,
                    sigma: sigma as f32,
                    alpha: alpha as f32,
                    norm: norm as f32,
                    _pad0: 0.0,
                    center_x: center_x as f32,
                    center_y: center_y as f32,
                    _pad1: [0.0, 0.0],
                };
                self.uniform_buffer(&params, "runmat-fspecial-params-f32")
            }
        };

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-fspecial-bind"),
                layout: &self.pipelines.fspecial.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            total_len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::creation::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.fspecial.pipeline,
            &bind_group,
            workgroups,
            "runmat-fspecial-encoder",
            "runmat-fspecial-pass",
        );

        Ok(self.register_existing_buffer(out_buffer, shape_vec, total_len))
    }
    pub(crate) fn random_integer_range_exec(
        &self,
        lower: i64,
        upper: i64,
        shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        ensure!(lower <= upper, "randi: lower bound must be <= upper bound");
        let total_len = product_checked(shape)
            .ok_or_else(|| anyhow!("randi: tensor size exceeds GPU limits"))?;
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-rng-int-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "randi: tensor length too large"
        );
        let span_i128 = (upper as i128)
            .checked_sub(lower as i128)
            .and_then(|d| d.checked_add(1))
            .ok_or_else(|| anyhow!("randi: integer range overflow"))?;
        ensure!(span_i128 > 0, "randi: integer range must be non-empty");
        ensure!(
            span_i128 <= (1i128 << 53),
            "randi: range cannot exceed 2^53"
        );
        let span_minus_one = span_i128 - 1;
        let span_u64 = span_i128 as u64;

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut rng_guard = rng_state()
            .lock()
            .map_err(|_| anyhow!("randi: provider RNG mutex poisoned"))?;
        let mut chunk_state = *rng_guard;

        let mut offset = 0usize;
        while offset < total_len {
            let remaining = total_len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let chunk_u32 = u32::try_from(chunk_len)
                .map_err(|_| anyhow!("randi: chunk length exceeds GPU dispatch limits"))?;
            let offset_u32 = u32::try_from(offset)
                .map_err(|_| anyhow!("randi: tensor offset exceeds GPU limits"))?;
            let seed = seed_from_state(chunk_state);

            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = crate::backend::wgpu::params::RandomIntParamsF64 {
                        lower: lower as f64,
                        upper: upper as f64,
                        span: span_u64 as f64,
                        span_minus_one: span_minus_one as f64,
                        offset: offset_u32,
                        chunk: chunk_u32,
                        seed,
                        _pad: 0,
                    };
                    self.uniform_buffer(&params, "runmat-rng-int-params-f64")
                }
                NumericPrecision::F32 => {
                    let params = crate::backend::wgpu::params::RandomIntParamsF32 {
                        lower: lower as f32,
                        upper: upper as f32,
                        span: span_u64 as f32,
                        span_minus_one: span_minus_one as f32,
                        offset: offset_u32,
                        chunk: chunk_u32,
                        seed,
                        _pad: 0,
                    };
                    self.uniform_buffer(&params, "runmat-rng-int-params-f32")
                }
            };

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-rng-int-bind"),
                    layout: &self.pipelines.random_int.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.random_int.pipeline,
                &bind_group,
                workgroups,
                "runmat-rng-int-encoder",
                "runmat-rng-int-pass",
            );

            chunk_state = advance_rng_state(chunk_state, u64::from(chunk_u32));
            offset += chunk_len;
        }

        *rng_guard = chunk_state;
        drop(rng_guard);

        Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), total_len))
    }
    pub(crate) fn randperm_exec(&self, n: usize, k: usize) -> Result<GpuTensorHandle> {
        ensure!(k <= n, "randperm: K must satisfy 0 <= K <= N");
        ensure!((n as u64) <= MAX_SAFE_INTEGER, "randperm: N exceeds 2^53");
        ensure!(
            n <= u32::MAX as usize,
            "randperm: N exceeds GPU dispatch limits"
        );
        ensure!(
            k <= u32::MAX as usize,
            "randperm: K exceeds GPU dispatch limits"
        );

        let effective_k = k.min(n);
        let shape_vec = vec![1, effective_k];
        let out_buffer = self.create_storage_buffer_checked(effective_k, "runmat-randperm-out")?;
        if effective_k == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape_vec, 0));
        }
        let mut rng_guard = rng_state()
            .lock()
            .map_err(|_| anyhow!("randperm: provider RNG mutex poisoned"))?;
        let seed = seed_from_state(*rng_guard);

        let params = crate::backend::wgpu::params::RandPermParams {
            n: n as u32,
            k: effective_k as u32,
            seed,
            _pad: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-randperm-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-randperm-bind"),
                layout: &self.pipelines.randperm.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        crate::backend::wgpu::dispatch::creation::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.randperm.pipeline,
            &bind_group,
            1,
            "runmat-randperm-encoder",
            "runmat-randperm-pass",
        );

        *rng_guard = advance_rng_state(*rng_guard, effective_k as u64);
        drop(rng_guard);

        Ok(self.register_existing_buffer(out_buffer, shape_vec, effective_k))
    }
    pub(crate) fn polyval_exec(
        &self,
        coeffs: &GpuTensorHandle,
        points: &GpuTensorHandle,
        options: &ProviderPolyvalOptions,
    ) -> Result<GpuTensorHandle> {
        let coeff_entry = self.get_entry(coeffs)?;
        let points_entry = self.get_entry(points)?;

        ensure!(
            coeff_entry.precision == self.precision && points_entry.precision == self.precision,
            "polyval: precision mismatch between tensors and provider"
        );
        ensure!(
            coeff_entry.len > 0,
            "polyval: coefficient vector must contain at least one element"
        );

        let len = points_entry.len;
        let shape = points_entry.shape.clone();
        if len == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-polyval-out");
            return Ok(self.register_existing_buffer(out_buffer, shape, 0));
        }

        ensure!(
            len <= u32::MAX as usize,
            "polyval: evaluation tensor exceeds GPU limits"
        );
        ensure!(
            coeff_entry.len <= u32::MAX as usize,
            "polyval: coefficient vector exceeds GPU limits"
        );

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-polyval-warmup"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-polyval-warmup-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.polyval.pipeline);
            drop(pass);
            self.submit(enc);
        }

        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-polyval-gap"),
                });
            self.submit(enc);
        }

        let out_buffer = self.create_storage_buffer_checked(len, "runmat-polyval-out")?;
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let (mu_mean, mu_scale) = options.mu.map(|m| (m.mean, m.scale)).unwrap_or((0.0, 1.0));
        let has_mu = options.mu.is_some() as u32;

        let mut offset = 0usize;
        while offset < len {
            let remaining = len - offset;
            let chunk_len = remaining.min(chunk_capacity).max(1);
            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = crate::backend::wgpu::params::PolyvalParamsF64 {
                        len: chunk_len as u32,
                        coeff_len: coeff_entry.len as u32,
                        offset: offset as u32,
                        has_mu,
                        mu_mean,
                        mu_scale,
                    };
                    self.uniform_buffer(&params, "runmat-polyval-params-f64")
                }
                NumericPrecision::F32 => {
                    let params = crate::backend::wgpu::params::PolyvalParamsF32 {
                        len: chunk_len as u32,
                        coeff_len: coeff_entry.len as u32,
                        offset: offset as u32,
                        has_mu,
                        mu_mean: mu_mean as f32,
                        mu_scale: mu_scale as f32,
                        _pad0: 0,
                        _pad1: 0,
                    };
                    self.uniform_buffer(&params, "runmat-polyval-params-f32")
                }
            };

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-polyval-bind"),
                    layout: &self.pipelines.polyval.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: coeff_entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: points_entry.buffer.as_ref().as_entire_binding(),
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

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );

            crate::backend::wgpu::dispatch::elementwise::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.polyval.pipeline,
                &bind_group,
                workgroups,
            );

            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, shape, len))
    }

    pub(crate) fn polyint_exec(
        &self,
        polynomial: &GpuTensorHandle,
        constant: f64,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(polynomial)?;
        ensure!(
            entry.precision == self.precision,
            "polyint: precision mismatch between tensor and provider"
        );
        let orientation = polynomial_orientation(&entry.shape)?;

        if entry.len == 0 {
            let shape = shape_for_orientation(orientation, 1);
            let buffer = match self.precision {
                NumericPrecision::F64 => Arc::new(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-polyint-const-f64"),
                        contents: cast_slice(&[constant]),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    },
                )),
                NumericPrecision::F32 => {
                    let value = constant as f32;
                    Arc::new(
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-polyint-const-f32"),
                                contents: cast_slice(&[value]),
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            }),
                    )
                }
            };
            return Ok(self.register_existing_buffer(buffer, shape, 1));
        }

        ensure!(
            entry.len <= u32::MAX as usize,
            "polyint: polynomial length exceeds GPU limits"
        );

        let output_len = entry.len + 1;
        let out_buffer = self.create_storage_buffer_checked(output_len, "runmat-polyint-out")?;
        let params_buffer = match self.precision {
            NumericPrecision::F64 => {
                let params = PolyintParamsF64 {
                    input_len: entry.len as u32,
                    output_len: output_len as u32,
                    constant,
                };
                self.uniform_buffer(&params, "runmat-polyint-params-f64")
            }
            NumericPrecision::F32 => {
                let params = PolyintParamsF32 {
                    input_len: entry.len as u32,
                    output_len: output_len as u32,
                    constant: constant as f32,
                    _pad0: 0.0,
                };
                self.uniform_buffer(&params, "runmat-polyint-params-f32")
            }
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runmat-polyint-bind"),
            layout: &self.pipelines.polyint.layout,
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

        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            output_len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-polyint-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-polyint-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.polyint.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.submit(encoder);

        let out_shape = shape_for_orientation(orientation, output_len);
        Ok(self.register_existing_buffer(out_buffer, out_shape, output_len))
    }
    pub(crate) async fn polyder_exec(
        &self,
        polynomial: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(polynomial)?;
        ensure!(
            entry.precision == self.precision,
            "polyder: precision mismatch between tensor and provider"
        );
        let orientation = polynomial_orientation(&entry.shape)?;
        if entry.len <= 1 {
            let shape = shape_for_orientation(orientation, 1);
            let buffer = match self.precision {
                NumericPrecision::F64 => Arc::new(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-polyder-const-f64"),
                        contents: cast_slice(&[0.0f64]),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    },
                )),
                NumericPrecision::F32 => {
                    let zeros: [f32; 1] = [0.0];
                    Arc::new(
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("runmat-polyder-const-f32"),
                                contents: cast_slice(&zeros),
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            }),
                    )
                }
            };
            return Ok(self.register_existing_buffer(buffer, shape, 1));
        }

        ensure!(
            entry.len <= u32::MAX as usize,
            "polyder: polynomial length exceeds GPU limits"
        );
        let output_len = entry.len - 1;
        let out_buffer = self.create_storage_buffer_checked(output_len, "runmat-polyder-out")?;
        let params = PolyderParams {
            input_len: entry.len as u32,
            output_len: output_len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-polyder-params");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runmat-polyder-bind"),
            layout: &self.pipelines.polyder.layout,
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
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            params.output_len,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runmat-polyder-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-polyder-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.polyder.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.submit(encoder);

        let out_shape = shape_for_orientation(orientation, output_len);
        let handle = self.register_existing_buffer(out_buffer, out_shape, output_len);
        self.trim_polynomial_handle(handle, orientation).await
    }

    pub(crate) async fn polyder_product_exec(
        &self,
        p: &GpuTensorHandle,
        q: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let p_entry = self.get_entry(p)?;
        let q_entry = self.get_entry(q)?;
        ensure!(
            p_entry.precision == self.precision && q_entry.precision == self.precision,
            "polyder: precision mismatch between tensors and provider"
        );
        let orientation = polynomial_orientation(&p_entry.shape)?;
        let conv_orientation = conv_orientation_for(orientation);

        let dp = self.polyder_exec(p).await?;
        let dq = self.polyder_exec(q).await?;
        let options = ProviderConv1dOptions {
            mode: ProviderConvMode::Full,
            orientation: conv_orientation,
        };
        let term1 = self.conv1d_exec(&dp, q, options)?;
        let term2 = self.conv1d_exec(p, &dq, options)?;
        let result = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Add,
            &term1,
            &term2,
        )?;
        self.free(&dp).ok();
        self.free(&dq).ok();
        self.free(&term1).ok();
        self.free(&term2).ok();
        self.trim_polynomial_handle(result, orientation).await
    }

    pub(crate) async fn polyder_quotient_exec(
        &self,
        u: &GpuTensorHandle,
        v: &GpuTensorHandle,
    ) -> Result<ProviderPolyderQuotient> {
        let u_entry = self.get_entry(u)?;
        let v_entry = self.get_entry(v)?;
        ensure!(
            u_entry.precision == self.precision && v_entry.precision == self.precision,
            "polyder: precision mismatch between tensors and provider"
        );
        let orientation_u = polynomial_orientation(&u_entry.shape)?;
        let orientation_v = polynomial_orientation(&v_entry.shape)?;
        let options_num = ProviderConv1dOptions {
            mode: ProviderConvMode::Full,
            orientation: conv_orientation_for(orientation_u),
        };
        let options_den = ProviderConv1dOptions {
            mode: ProviderConvMode::Full,
            orientation: conv_orientation_for(orientation_v),
        };

        let du = self.polyder_exec(u).await?;
        let dv = self.polyder_exec(v).await?;
        let term1 = self.conv1d_exec(&du, v, options_num)?;
        let term2 = self.conv1d_exec(u, &dv, options_num)?;
        let numerator_handle = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Sub,
            &term1,
            &term2,
        )?;
        let denominator_handle = self.conv1d_exec(v, v, options_den)?;
        self.free(&du).ok();
        self.free(&dv).ok();
        self.free(&term1).ok();
        self.free(&term2).ok();

        let numerator = self
            .trim_polynomial_handle(numerator_handle, orientation_u)
            .await?;
        let denominator = self
            .trim_polynomial_handle(denominator_handle, orientation_v)
            .await?;
        Ok(ProviderPolyderQuotient {
            numerator,
            denominator,
        })
    }
    pub(crate) fn linspace_exec(
        &self,
        start: f64,
        stop: f64,
        count: usize,
    ) -> Result<GpuTensorHandle> {
        if count > u32::MAX as usize {
            return Err(anyhow!("linspace: sequence length exceeds GPU limits"));
        }

        let shape = vec![1, count];
        let out_buffer = self.create_storage_buffer(count, "runmat-linspace-out");
        if count == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape, 0));
        }

        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-linspace-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-linspace-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.linspace.pipeline);
            drop(pass);
            self.submit(enc);
        }

        let step = if count <= 1 {
            0.0
        } else {
            (stop - start) / ((count - 1) as f64)
        };
        let total_u32 = count as u32;
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;

        while offset < count {
            let chunk_len = (count - offset).min(chunk_capacity).max(1);
            let offset_u32 = offset as u32;
            let chunk_u32 = chunk_len as u32;

            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = crate::backend::wgpu::params::LinspaceParamsF64 {
                        start,
                        step,
                        stop,
                        total: total_u32,
                        chunk: chunk_u32,
                        offset: offset_u32,
                        _pad: 0,
                    };
                    self.uniform_buffer(&params, "runmat-linspace-params-f64")
                }
                NumericPrecision::F32 => {
                    let params = crate::backend::wgpu::params::LinspaceParamsF32 {
                        start: start as f32,
                        step: step as f32,
                        stop: stop as f32,
                        _pad0: 0.0,
                        total: total_u32,
                        chunk: chunk_u32,
                        offset: offset_u32,
                        _pad1: 0,
                    };
                    self.uniform_buffer(&params, "runmat-linspace-params-f32")
                }
            };

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-linspace-bind"),
                    layout: &self.pipelines.linspace.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.linspace.pipeline,
                &bind_group,
                workgroups,
                "runmat-linspace-encoder",
                "runmat-linspace-pass",
            );

            offset += chunk_len;
        }

        Ok(self.register_existing_buffer(out_buffer, shape, count))
    }
    pub(crate) fn diag_from_vector_exec(
        &self,
        vector: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(vector)?;
        diag_ensure_shape(&entry.shape)?;
        let (rows, cols) = diag_rows_cols(&entry.shape);
        ensure!(
            diag_is_vector_like(rows, cols, entry.shape.len()),
            "diag: input must be a vector"
        );

        let len = entry.len;
        if len == 0 {
            return Err(anyhow!("diag: empty vector fallback"));
        }
        let (size, total) = diag_matrix_size_checked(len, offset)?;
        ensure!(
            len <= u32::MAX as usize,
            "diag: vector is too large for GPU dispatch"
        );
        ensure!(
            size <= u32::MAX as usize,
            "diag: result dimension exceeds GPU dispatch limits"
        );
        ensure!(
            total <= u32::MAX as usize,
            "diag: result size exceeds GPU dispatch limits"
        );
        let offset_i32 = i32::try_from(offset)
            .map_err(|_| anyhow!("diag: offset magnitude exceeds GPU limits"))?;

        let out_buffer = self.create_storage_buffer(total, "runmat-diag-vec-out");
        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-diag-vec-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-diag-vec-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.diag_from_vector.pipeline);
            drop(pass);
            self.submit(enc);
        }

        let params = crate::backend::wgpu::params::DiagFromVectorParams {
            len: len as u32,
            size: size as u32,
            offset: offset_i32,
            _pad: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-diag-vec-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-diag-vec-bind"),
                layout: &self.pipelines.diag_from_vector.layout,
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
        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::diag::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.diag_from_vector.pipeline,
            &bind_group,
            workgroups,
            "runmat-diag-vec-pass",
        );

        Ok(self.register_existing_buffer(out_buffer, vec![size, size], total))
    }
    pub(crate) fn diag_extract_exec(
        &self,
        matrix: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(matrix)?;
        diag_ensure_shape(&entry.shape)?;
        let (rows, cols) = diag_rows_cols(&entry.shape);
        ensure!(
            !diag_is_vector_like(rows, cols, entry.shape.len()),
            "diag: matrix input required"
        );
        let diag_len = diag_length(rows, cols, offset);
        if diag_len == 0 {
            return Err(anyhow!("diag: empty diagonal fallback"));
        }
        ensure!(
            diag_len <= u32::MAX as usize,
            "diag: diagonal length exceeds GPU dispatch limits"
        );
        ensure!(
            rows <= u32::MAX as usize && cols <= u32::MAX as usize,
            "diag: matrix dimensions exceed GPU dispatch limits"
        );
        let offset_i32 = i32::try_from(offset)
            .map_err(|_| anyhow!("diag: offset magnitude exceeds GPU limits"))?;

        let out_buffer = self.create_storage_buffer(diag_len, "runmat-diag-extract-out");
        {
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-diag-extract-noop"),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-diag-extract-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.diag_extract.pipeline);
            drop(pass);
            self.submit(enc);
        }

        let params = crate::backend::wgpu::params::DiagExtractParams {
            rows: rows as u32,
            cols: cols as u32,
            offset: offset_i32,
            diag_len: diag_len as u32,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-diag-extract-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-diag-extract-bind"),
                layout: &self.pipelines.diag_extract.layout,
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
        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            diag_len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::diag::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.diag_extract.pipeline,
            &bind_group,
            workgroups,
            "runmat-diag-extract-pass",
        );

        Ok(self.register_existing_buffer(out_buffer, vec![diag_len, 1], diag_len))
    }

    pub(crate) fn gather_linear_exec(
        &self,
        source: &GpuTensorHandle,
        indices: &[u32],
        output_shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(source)?;
        let expected = product_checked(output_shape)
            .ok_or_else(|| anyhow!("gather_linear: output shape product overflow"))?;
        let _span = info_span!(
            "gpu.gather_linear",
            source_len = entry.len,
            index_count = indices.len(),
            output_size = expected
        )
        .entered();
        ensure!(
            expected == indices.len(),
            "gather_linear: index count {} does not match output size {}",
            indices.len(),
            expected
        );
        if expected == 0 {
            let out = self.create_storage_buffer(0, "runmat-gather-linear-empty");
            return Ok(self.register_existing_buffer(out, output_shape.to_vec(), 0));
        }
        ensure!(
            indices.len() <= u32::MAX as usize,
            "gather_linear: index count exceeds GPU limits"
        );
        let indices_len_bytes = std::mem::size_of_val(indices) as u64;
        let indices_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-gather-linear-indices"),
            size: indices_len_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        if !indices.is_empty() {
            self.queue
                .write_buffer(indices_buffer.as_ref(), 0, cast_slice(indices));
        }
        log::trace!(
            "gather_linear begin source_buffer={} ptr=0x{:x} out_shape={:?} count={}",
            source.buffer_id,
            entry.buffer.as_ref() as *const wgpu::Buffer as usize,
            output_shape,
            indices.len()
        );

        let out_buffer =
            self.create_storage_buffer_checked(expected, "runmat-gather-linear-out")?;
        let params = LinearGatherParams {
            count: indices.len() as u32,
            _pad: [0; 3],
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-gather-linear-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-gather-linear-bind"),
                layout: &self.pipelines.gather_linear.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: indices_buffer.as_ref().as_entire_binding(),
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
        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            indices.len() as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::creation::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.gather_linear.pipeline,
            &bind_group,
            workgroups,
            "runmat-gather-linear-encoder",
            "runmat-gather-linear-pass",
        );
        log::trace!(
            "gather_linear complete source_buffer={} out_ptr=0x{:x} count={}",
            source.buffer_id,
            out_buffer.as_ref() as *const wgpu::Buffer as usize,
            indices.len()
        );

        Ok(self.register_existing_buffer(out_buffer, output_shape.to_vec(), expected))
    }

    pub(crate) fn scatter_linear_exec(
        &self,
        target: &GpuTensorHandle,
        indices: &[u32],
        values: &GpuTensorHandle,
    ) -> Result<()> {
        if indices.is_empty() {
            return Ok(());
        }
        ensure!(
            indices.len() <= u32::MAX as usize,
            "scatter_linear: index count exceeds GPU limits"
        );
        let target_entry = self.get_entry(target)?;
        let values_entry = self.get_entry(values)?;
        let _span = info_span!(
            "gpu.scatter_linear",
            target_len = target_entry.len,
            index_count = indices.len(),
            values_len = values_entry.len
        )
        .entered();
        ensure!(
            values_entry.len == indices.len(),
            "scatter_linear: values length {} does not match indices length {}",
            values_entry.len,
            indices.len()
        );
        ensure!(
            indices.iter().all(|&idx| (idx as usize) < target_entry.len),
            "scatter_linear: index out of bounds for target tensor"
        );
        let indices_len_bytes = std::mem::size_of_val(indices) as u64;
        let indices_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-scatter-linear-indices"),
            size: indices_len_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.queue
            .write_buffer(indices_buffer.as_ref(), 0, cast_slice(indices));
        log::trace!(
            "scatter_linear begin target_buffer={} target_ptr=0x{:x} values_buffer={} values_ptr=0x{:x} count={}",
            target.buffer_id,
            target_entry.buffer.as_ref() as *const wgpu::Buffer as usize,
            values.buffer_id,
            values_entry.buffer.as_ref() as *const wgpu::Buffer as usize,
            indices.len()
        );
        let params = LinearScatterParams {
            count: indices.len() as u32,
            _pad: [0; 3],
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-scatter-linear-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-scatter-linear-bind"),
                layout: &self.pipelines.scatter_linear.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: target_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: values_entry.buffer.as_ref().as_entire_binding(),
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
        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            indices.len() as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::creation::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.scatter_linear.pipeline,
            &bind_group,
            workgroups,
            "runmat-scatter-linear-encoder",
            "runmat-scatter-linear-pass",
        );
        log::trace!(
            "scatter_linear complete target_buffer={} values_buffer={} count={}",
            target.buffer_id,
            values.buffer_id,
            indices.len()
        );
        Ok(())
    }

    pub(crate) fn binary_op_exec(
        &self,
        op: crate::backend::wgpu::types::BinaryOpCode,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        if std::env::var("RUNMAT_DISABLE_BINARY").is_ok() {
            return Err(anyhow!("binary ops disabled via RUNMAT_DISABLE_BINARY"));
        }
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            // Attempt general N-D broadcasted binary op
            return self.binary_op_broadcast_exec(op, a, b);
        }
        let len = entry_a.len;
        if len == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-binary-out");
            return Ok(self.register_existing_buffer(out_buffer, entry_a.shape, entry_a.len));
        }
        if len > (u32::MAX as usize) {
            return Err(anyhow!("tensor too large for GPU buffer"));
        }
        let start = Instant::now();
        {
            let mut enc =
                self.device_ref()
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
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-binary-flush-gap"),
                });
            self.submit(enc);
        }
        let out_buffer = self.create_storage_buffer_checked(len, "runmat-binary-out")?;
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < len {
            let remaining = len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::LenOpParams {
                len: chunk_len as u32,
                op: op as u32,
                offset: offset as u32,
                total: len as u32,
            };
            let params_buffer = self.kernel_resources.uniform_buffer(
                self.device_ref(),
                UniformBufferKey::LenOpParams,
                std::mem::size_of::<crate::backend::wgpu::params::LenOpParams>() as u64,
                "runmat-binary-params",
            );
            self.queue
                .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
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
            let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::elementwise::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.binary.pipeline,
                &bind_group,
                groups,
            );
            offset += chunk_len;
        }
        let handle = self.register_existing_buffer(out_buffer, entry_a.shape, len);
        if let Some(info) = runmat_accelerate_api::handle_transpose_info(a) {
            runmat_accelerate_api::record_handle_transpose(&handle, info.base_rows, info.base_cols);
        }
        self.telemetry
            .record_fused_elementwise_duration(start.elapsed());
        Ok(handle)
    }
    fn binary_op_broadcast_exec(
        &self,
        op: crate::backend::wgpu::types::BinaryOpCode,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        use crate::backend::wgpu::params::{AlignedU32, BinaryBroadcastParams, BCAST_MAX_RANK};
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        // Compute broadcasted output shape
        let mut shape_a = entry_a.shape.clone();
        let mut shape_b = entry_b.shape.clone();
        let rank = shape_a.len().max(shape_b.len());
        if rank > BCAST_MAX_RANK {
            return Err(anyhow!("broadcast rank exceeds limit"));
        }
        if shape_a.len() < rank {
            let pad = rank - shape_a.len();
            let mut v = vec![1usize; pad];
            v.extend_from_slice(&shape_a);
            shape_a = v;
        }
        if shape_b.len() < rank {
            let pad = rank - shape_b.len();
            let mut v = vec![1usize; pad];
            v.extend_from_slice(&shape_b);
            shape_b = v;
        }
        let mut out_shape: Vec<usize> = vec![1; rank];
        for i in 0..rank {
            let da = shape_a[i];
            let db = shape_b[i];
            if da == db {
                out_shape[i] = da;
            } else if da == 1 {
                out_shape[i] = db;
            } else if db == 1 {
                out_shape[i] = da;
            } else {
                return Err(anyhow!("shape mismatch for broadcast"));
            }
        }
        let len: usize = out_shape
            .iter()
            .copied()
            .fold(1usize, |a, b| a.saturating_mul(b));
        if len == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-binary-bcast-out");
            return Ok(self.register_existing_buffer(out_buffer, out_shape, 0));
        }
        if len > (u32::MAX as usize) {
            return Err(anyhow!("tensor too large for GPU buffer"));
        }
        // Compute strides (column-major)
        let mut stride_a: Vec<u32> = vec![0; rank];
        let mut stride_b: Vec<u32> = vec![0; rank];
        let mut s: u64 = 1;
        for i in 0..rank {
            stride_a[i] = if shape_a[i] == 1 { 0 } else { s as u32 };
            s = s.saturating_mul(shape_a[i] as u64);
        }
        s = 1;
        for i in 0..rank {
            stride_b[i] = if shape_b[i] == 1 { 0 } else { s as u32 };
            s = s.saturating_mul(shape_b[i] as u64);
        }

        // Create output buffer
        let out_buffer = self.create_storage_buffer_checked(len, "runmat-binary-bcast-out")?;
        // Prepare params buffer and bind group once; update params per chunk
        let params_size = std::mem::size_of::<BinaryBroadcastParams>() as u64;
        let params_buffer = self.kernel_resources.uniform_buffer(
            self.device_ref(),
            UniformBufferKey::BinaryBroadcastParams,
            params_size,
            "runmat-binary-bcast-params",
        );
        let bind_group_layout = &self.pipelines.binary_broadcast.layout;
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-binary-bcast-bind"),
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.get_entry(a)?.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.get_entry(b)?.buffer.as_ref().as_entire_binding(),
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

        // Dispatch in chunks
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        let start = Instant::now();
        while offset < len {
            let remaining = len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            // Pack params
            let mut params = BinaryBroadcastParams {
                len: chunk_len as u32,
                offset: offset as u32,
                rank: rank as u32,
                op: op as u32,
                out_shape: [AlignedU32::new(0); BCAST_MAX_RANK],
                a_shape: [AlignedU32::new(0); BCAST_MAX_RANK],
                b_shape: [AlignedU32::new(0); BCAST_MAX_RANK],
                a_strides: [AlignedU32::new(0); BCAST_MAX_RANK],
                b_strides: [AlignedU32::new(0); BCAST_MAX_RANK],
            };
            for i in 0..rank {
                params.out_shape[i] = AlignedU32::new(out_shape[i] as u32);
                params.a_shape[i] = AlignedU32::new(shape_a[i] as u32);
                params.b_shape[i] = AlignedU32::new(shape_b[i] as u32);
                params.a_strides[i] = AlignedU32::new(stride_a[i]);
                params.b_strides[i] = AlignedU32::new(stride_b[i]);
            }
            self.queue_ref()
                .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
            let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::elementwise::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.binary_broadcast.pipeline,
                &bind_group,
                groups,
            );
            offset += chunk_len;
        }
        let handle = self.register_existing_buffer(out_buffer, out_shape, len);
        self.telemetry
            .record_fused_elementwise_duration(start.elapsed());
        Ok(handle)
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

    pub(crate) fn elem_eq_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_eq: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-elem-eq-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_EQ_SHADER_F64,
                NumericPrecision::F32 => ELEM_EQ_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn elem_ne_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_ne: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-elem-ne-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_NE_SHADER_F64,
                NumericPrecision::F32 => ELEM_NE_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }
    pub(crate) fn elem_lt_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_lt: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-elem-lt-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_LT_SHADER_F64,
                NumericPrecision::F32 => ELEM_LT_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn elem_le_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_le: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-elem-le-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_LE_SHADER_F64,
                NumericPrecision::F32 => ELEM_LE_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn elem_gt_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_gt: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-elem-gt-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_GT_SHADER_F64,
                NumericPrecision::F32 => ELEM_GT_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn elem_ge_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("elem_ge: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-elem-ge-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => ELEM_GE_SHADER_F64,
                NumericPrecision::F32 => ELEM_GE_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn logical_and_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("logical_and: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-logical-and-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_AND_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_AND_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn logical_or_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("logical_or: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-logical-or-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_OR_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_OR_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }
    pub(crate) fn logical_xor_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape != entry_b.shape {
            return Err(anyhow!("logical_xor: shape mismatch between inputs"));
        }
        let len = entry_a.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-logical-xor-empty");
            self.register_existing_buffer(out, entry_a.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_XOR_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_XOR_SHADER_F32,
            };
            self.fused_elementwise(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn logical_not_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-logical-not-empty");
            self.register_existing_buffer(out, entry.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_NOT_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_NOT_SHADER_F32,
            };
            self.fused_elementwise(shader, std::slice::from_ref(a), &entry.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn logical_isfinite_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-logical-isfinite-empty");
            self.register_existing_buffer(out, entry.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_ISFINITE_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_ISFINITE_SHADER_F32,
            };
            self.fused_elementwise(shader, std::slice::from_ref(a), &entry.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn logical_isnan_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-logical-isnan-empty");
            self.register_existing_buffer(out, entry.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_ISNAN_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_ISNAN_SHADER_F32,
            };
            self.fused_elementwise(shader, std::slice::from_ref(a), &entry.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }

    pub(crate) fn logical_isinf_exec(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        let len = entry.len;
        let handle = if len == 0 {
            let out = self.create_storage_buffer(0, "runmat-logical-isinf-empty");
            self.register_existing_buffer(out, entry.shape, 0)
        } else {
            let shader = match self.precision {
                NumericPrecision::F64 => LOGICAL_ISINF_SHADER_F64,
                NumericPrecision::F32 => LOGICAL_ISINF_SHADER_F32,
            };
            self.fused_elementwise(shader, std::slice::from_ref(a), &entry.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }
    pub(crate) fn unary_op_exec(
        &self,
        op: crate::backend::wgpu::types::UnaryOpCode,
        a: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        if std::env::var("RUNMAT_DISABLE_UNARY").is_ok() {
            return Err(anyhow!("unary ops disabled via RUNMAT_DISABLE_UNARY"));
        }
        let entry_a = self.get_entry(a)?;
        let len = entry_a.len;
        let out_buffer = self.create_storage_buffer_checked(len, "runmat-unary-out")?;
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry_a.shape, entry_a.len));
        }
        if len > (u32::MAX as usize) {
            return Err(anyhow!("tensor too large for GPU buffer"));
        }
        let start = Instant::now();
        {
            let mut enc =
                self.device_ref()
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
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-unary-flush-gap"),
                });
            self.submit(enc);
        }
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < len {
            let remaining = len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::LenOpParams {
                len: chunk_len as u32,
                op: op as u32,
                offset: offset as u32,
                total: len as u32,
            };
            let params_buffer = self.kernel_resources.uniform_buffer(
                self.device_ref(),
                UniformBufferKey::LenOpParams,
                std::mem::size_of::<crate::backend::wgpu::params::LenOpParams>() as u64,
                "runmat-unary-params",
            );
            self.queue
                .write_buffer(params_buffer.as_ref(), 0, bytes_of(&params));
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
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
            let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::elementwise::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.unary.pipeline,
                &bind_group,
                groups,
            );
            offset += chunk_len;
        }
        let handle = self.register_existing_buffer(out_buffer, entry_a.shape, len);
        self.telemetry
            .record_fused_elementwise_duration(start.elapsed());
        Ok(handle)
    }
    pub(crate) fn scalar_op_exec(
        &self,
        op: crate::backend::wgpu::types::ScalarOpCode,
        a: &GpuTensorHandle,
        scalar: f64,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let len = entry_a.len;
        let out_buffer = self.create_storage_buffer_checked(len, "runmat-scalar-out")?;
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry_a.shape, entry_a.len));
        }
        if len > (u32::MAX as usize) {
            return Err(anyhow!("tensor too large for GPU buffer"));
        }
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        let start = Instant::now();
        while offset < len {
            let remaining = len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params_buffer = match self.precision() {
                runmat_accelerate_api::ProviderPrecision::F64 => {
                    let params = crate::backend::wgpu::params::ScalarParamsF64 {
                        len: chunk_len as u32,
                        op: op as u32,
                        offset: offset as u32,
                        total: len as u32,
                        scalar,
                        _pad_scalar: 0.0,
                        _pad_tail: 0.0,
                        _pad_tail2: 0.0,
                        _pad_tail3: 0.0,
                        _pad_tail4: 0.0,
                    };
                    let buf = self.kernel_resources.uniform_buffer(
                        self.device_ref(),
                        UniformBufferKey::ScalarParamsF64,
                        std::mem::size_of::<crate::backend::wgpu::params::ScalarParamsF64>() as u64,
                        "runmat-scalar-params-f64",
                    );
                    self.queue.write_buffer(buf.as_ref(), 0, bytes_of(&params));
                    buf
                }
                _ => {
                    let params = crate::backend::wgpu::params::ScalarParamsF32 {
                        len: chunk_len as u32,
                        op: op as u32,
                        offset: offset as u32,
                        total: len as u32,
                        scalar: scalar as f32,
                        _pad_scalar: [0.0; 3],
                        _pad_tail: [0.0; 4],
                        _pad_tail2: [0.0; 4],
                    };
                    let buf = self.kernel_resources.uniform_buffer(
                        self.device_ref(),
                        UniformBufferKey::ScalarParamsF32,
                        std::mem::size_of::<crate::backend::wgpu::params::ScalarParamsF32>() as u64,
                        "runmat-scalar-params-f32",
                    );
                    self.queue.write_buffer(buf.as_ref(), 0, bytes_of(&params));
                    buf
                }
            };
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
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
            let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::elementwise::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.scalar.pipeline,
                &bind_group,
                groups,
            );
            offset += chunk_len;
        }
        let handle = self.register_existing_buffer(out_buffer, entry_a.shape, len);
        self.telemetry
            .record_fused_elementwise_duration(start.elapsed());
        Ok(handle)
    }

    pub(crate) fn reduce_global_exec(
        &self,
        a: &GpuTensorHandle,
        op: crate::backend::wgpu::types::GlobalReduceOp,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            log::debug!(
                "[reduce-global] in ptr={:p} len={} op={}",
                entry.buffer.as_ref(),
                entry.len,
                op as u32
            );
        }
        if std::env::var("RUNMAT_DISABLE_REDUCE_GLOBAL").is_ok() {
            return Err(anyhow!(
                "reduce_global disabled via RUNMAT_DISABLE_REDUCE_GLOBAL"
            ));
        }
        if entry.len == 0 {
            let default = match op {
                crate::backend::wgpu::types::GlobalReduceOp::Sum => 0.0,
                crate::backend::wgpu::types::GlobalReduceOp::Prod => 1.0,
                crate::backend::wgpu::types::GlobalReduceOp::Min => f64::INFINITY,
                crate::backend::wgpu::types::GlobalReduceOp::Max => f64::NEG_INFINITY,
                crate::backend::wgpu::types::GlobalReduceOp::CountNonZero => 0.0,
            };
            let data = [default];
            let shape = [1usize, 1usize];
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            return self.upload(&view);
        }
        let mut current = if std::env::var("RUNMAT_PROVIDER_REDUCTION_SNAPSHOT").is_ok() {
            let size_bytes = (entry.len * self.element_size) as u64;
            let snap = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-global-input-snapshot"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-reduce-global-input-snapshot-copy"),
                    });
            enc.copy_buffer_to_buffer(entry.buffer.as_ref(), 0, snap.as_ref(), 0, size_bytes);
            self.submit(enc);
            snap
        } else {
            entry.buffer.clone()
        };
        let mut current_len = entry.len;
        while current_len > 1 {
            let wg = crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE as usize;
            let max_groups = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize;
            let elems_per_group = 2 * wg;
            let max_input_per_pass = max_groups * elems_per_group;

            let output_len_total = current_len.div_ceil(elems_per_group).max(1);
            // Metal-safe (opt-in): snapshot input buffer for this pass to avoid any SR/SRW conflicts
            // Enabled only if RUNMAT_FORCE_REDUCE_SNAPSHOT is set to avoid perf impact by default.
            let mut input_for_pass = current.clone();
            if self.adapter_info.backend == wgpu::Backend::Metal
                && std::env::var("RUNMAT_FORCE_REDUCE_SNAPSHOT").is_ok()
            {
                let size_bytes = (current_len * self.element_size) as u64;
                let snap = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-reduce-pass-input-snapshot"),
                    size: size_bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                let mut enc =
                    self.device_ref()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("runmat-reduce-pass-input-snapshot-copy"),
                        });
                enc.copy_buffer_to_buffer(current.as_ref(), 0, snap.as_ref(), 0, size_bytes);
                self.submit(enc);
                input_for_pass = snap;
            }
            let mut out_buffer =
                self.create_storage_buffer_checked(output_len_total, "runmat-reduce-pass")?;
            // Prevent aliasing: output buffer must not be the same as input buffer
            if std::ptr::eq(out_buffer.as_ref(), input_for_pass.as_ref()) {
                if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                    log::debug!(
                        "reduce_global_exec: alias detected; current_ptr={:p} out_ptr={:p} len={} out_total={}",
                        input_for_pass.as_ref(),
                        out_buffer.as_ref(),
                        current_len,
                        output_len_total
                    );
                    eprintln!(
                        "[reduction] alias current={:p} out={:p} len={} out_total={}",
                        input_for_pass.as_ref(),
                        out_buffer.as_ref(),
                        current_len,
                        output_len_total
                    );
                }
                let size_bytes = (output_len_total * self.element_size) as u64;
                ensure!(
                    size_bytes <= self.adapter_limits.max_buffer_size,
                    "runmat-reduce-pass-unique: requested {} bytes exceeds device max {}",
                    size_bytes,
                    self.adapter_limits.max_buffer_size
                );
                out_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-reduce-pass-unique"),
                    size: size_bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }

            let mut in_offset_elems = 0usize;
            let mut _out_offset_elems = 0usize;
            while in_offset_elems < current_len {
                let remain = current_len - in_offset_elems;
                let chunk_in = remain.min(max_input_per_pass);
                let chunk_out = chunk_in.div_ceil(elems_per_group).max(1);

                let params = crate::backend::wgpu::params::ReduceGlobalParams {
                    len: chunk_in as u32,
                    op: op as u32,
                    offset: in_offset_elems as u32,
                    total: current_len as u32,
                };
                let params_buffer = self.uniform_buffer(&params, "runmat-reduce-global-params");

                let bind_group = self
                    .device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-reduce-global-bind"),
                        layout: &self.pipelines.reduce_global.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input_for_pass.as_ref().as_entire_binding(),
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
                let groups = crate::backend::wgpu::dispatch::common::dispatch_size_reduce(
                    chunk_in as u32,
                    crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE,
                );
                if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                    log::debug!(
                        "reduce_global_exec: dispatch groups={} current_ptr={:p} out_ptr={:p}",
                        groups,
                        input_for_pass.as_ref(),
                        out_buffer.as_ref()
                    );
                    eprintln!(
                        "[reduction] dispatch groups={} in={:p} out={:p}",
                        groups,
                        input_for_pass.as_ref(),
                        out_buffer.as_ref()
                    );
                }
                crate::backend::wgpu::dispatch::reduction::run_single_pass(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.reduce_global.pipeline,
                    &bind_group,
                    groups,
                );
                in_offset_elems += chunk_in;
                _out_offset_elems += chunk_out;
            }

            current = out_buffer;
            current_len = output_len_total;
        }
        Ok(self.register_existing_buffer(current, vec![1, 1], 1))
    }
    pub(crate) fn reduce_dim_sum_mean_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        op: crate::backend::wgpu::types::DimReduceOp,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(a)?;
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            eprintln!(
                "[reduce-dim-sum-mean begin] in ptr={:p} shape={:?} dim={} op={}",
                entry.buffer.as_ref(),
                entry.shape,
                dim,
                op as u32
            );
        }
        if std::env::var("RUNMAT_DISABLE_REDUCE_DIM").is_ok() {
            return Err(anyhow!("reduce_dim disabled via RUNMAT_DISABLE_REDUCE_DIM"));
        }
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            eprintln!(
                "[reduce-dim-sum-mean begin] in ptr={:p} shape={:?} dim={} op={}",
                entry.buffer.as_ref(),
                entry.shape,
                dim,
                op as u32
            );
        }
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
        // Optional snapshot of input
        let in_buf = if std::env::var("RUNMAT_PROVIDER_REDUCTION_SNAPSHOT").is_ok() {
            let size_bytes = (entry.len * self.element_size) as u64;
            let snap = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-dim-input-snapshot"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            let mut enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-reduce-dim-input-snapshot-copy"),
                    });
            enc.copy_buffer_to_buffer(entry.buffer.as_ref(), 0, snap.as_ref(), 0, size_bytes);
            self.submit(enc);
            snap
        } else {
            entry.buffer.clone()
        };
        let mut out_buffer =
            self.create_storage_buffer_checked(out_len, "runmat-reduce-dim-out")?;
        // Prevent aliasing: output must not be identical to input buffer
        if std::ptr::eq(out_buffer.as_ref(), entry.buffer.as_ref()) {
            if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                log::debug!(
                    "reduce_dim_sum_mean_exec: alias detected; in_ptr={:p} out_ptr={:p} rows={} cols={} out_len={}",
                    entry.buffer.as_ref(),
                    out_buffer.as_ref(),
                    rows,
                    cols,
                    out_len
                );
            }
            let size_bytes = (out_len * self.element_size) as u64;
            ensure!(
                size_bytes <= self.adapter_limits.max_buffer_size,
                "runmat-reduce-dim-out-unique: requested {} bytes exceeds device max {}",
                size_bytes,
                self.adapter_limits.max_buffer_size
            );
            out_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-dim-out-unique"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if out_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, out_len));
        }
        let params = crate::backend::wgpu::params::ReduceDimParams {
            rows: rows as u32,
            cols: cols as u32,
            dim: reduce_dim as u32,
            op: op as u32,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-reduce-dim-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-reduce-dim-bind"),
                layout: &self.pipelines.reduce_dim_sum_mean.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_buf.as_ref().as_entire_binding(),
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
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            out_len as u32,
            crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE,
        );
        if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
            eprintln!(
                "[reduce-dim-sum-mean] in ptr={:p} out ptr={:p} rows={} cols={} dim={} groups={}",
                entry.buffer.as_ref(),
                out_buffer.as_ref(),
                rows,
                cols,
                reduce_dim,
                groups
            );
        }
        crate::backend::wgpu::dispatch::reduction::run_single_pass(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.reduce_dim_sum_mean.pipeline,
            &bind_group,
            groups,
        );
        Ok(self.register_existing_buffer(out_buffer, out_shape, out_len))
    }
    pub(crate) fn reduce_dim_minmax_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        op: crate::backend::wgpu::types::DimReduceExtrema,
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
        let mut values_buffer =
            self.create_storage_buffer_checked(out_len, "runmat-reduce-dim-ext-values")?;
        let mut indices_buffer =
            self.create_storage_buffer_checked(out_len, "runmat-reduce-dim-ext-indices")?;
        // Prevent aliasing either output with input buffer
        if std::ptr::eq(values_buffer.as_ref(), entry.buffer.as_ref()) {
            let size_bytes = (out_len * self.element_size) as u64;
            ensure!(
                size_bytes <= self.adapter_limits.max_buffer_size,
                "runmat-reduce-dim-ext-values-unique: requested {} bytes exceeds device max {}",
                size_bytes,
                self.adapter_limits.max_buffer_size
            );
            values_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-dim-ext-values-unique"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if std::ptr::eq(indices_buffer.as_ref(), entry.buffer.as_ref()) {
            let size_bytes = (out_len * self.element_size) as u64;
            ensure!(
                size_bytes <= self.adapter_limits.max_buffer_size,
                "runmat-reduce-dim-ext-indices-unique: requested {} bytes exceeds device max {}",
                size_bytes,
                self.adapter_limits.max_buffer_size
            );
            indices_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-dim-ext-indices-unique"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if out_len == 0 {
            let values_handle =
                self.register_existing_buffer(values_buffer, out_shape.clone(), out_len);
            let indices_handle = self.register_existing_buffer(indices_buffer, out_shape, out_len);
            return Ok(ReduceDimResult {
                values: values_handle,
                indices: indices_handle,
            });
        }
        let params = crate::backend::wgpu::params::ReduceDimParams {
            rows: rows as u32,
            cols: cols as u32,
            dim: reduce_dim as u32,
            op: op as u32,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-reduce-dim-ext-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
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
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            out_len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::reduction::run_single_pass(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.reduce_dim_minmax.pipeline,
            &bind_group,
            groups,
        );
        let values_handle =
            self.register_existing_buffer(values_buffer, out_shape.clone(), out_len);
        let indices_handle = self.register_existing_buffer(indices_buffer, out_shape, out_len);
        Ok(ReduceDimResult {
            values: values_handle,
            indices: indices_handle,
        })
    }
    pub(crate) fn reduce_std_dim_exec(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        if matches!(nan_mode, ProviderNanMode::Omit) {
            return Err(anyhow!(
                "reduce_std_dim: omitnan is not supported by the wgpu provider"
            ));
        }
        let entry = self.get_entry(a)?;
        if entry.shape.len() != 2 {
            return Err(anyhow!("reduce_std_dim: only 2D tensors supported"));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        let reduce_len = match dim {
            0 => rows,
            1 => cols,
            _ => return Err(anyhow!("reduce_std_dim: only dims 0 or 1 supported")),
        };
        let out_shape = if dim == 0 {
            vec![1, cols]
        } else {
            vec![rows, 1]
        };
        if reduce_len == 0 {
            return self.fill_exec(&out_shape, f64::NAN);
        }

        let inv_len = 1.0 / (reduce_len as f64);

        let sum_handle =
            self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Sum)?;
        let squared = self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, a, a)?;
        let sum_sq_handle = self.reduce_dim_sum_mean_exec(
            &squared,
            dim,
            crate::backend::wgpu::types::DimReduceOp::Sum,
        )?;
        let _ = self.free(&squared);

        let sum_squared = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &sum_handle,
            &sum_handle,
        )?;
        let _ = self.free(&sum_handle);

        let sum_shape = self.get_entry(&sum_squared)?.shape.clone();
        let scale_tensor = self.fill_exec(&sum_shape, inv_len)?;
        let sum_squared_scaled = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &sum_squared,
            &scale_tensor,
        )?;
        let _ = self.free(&scale_tensor);
        let variance_numer = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Sub,
            &sum_sq_handle,
            &sum_squared_scaled,
        )?;
        let _ = self.free(&sum_sq_handle);
        let _ = self.free(&sum_squared);
        let _ = self.free(&sum_squared_scaled);

        let denom = match normalization {
            ProviderStdNormalization::Sample => {
                if reduce_len > 1 {
                    (reduce_len - 1) as f64
                } else {
                    1.0
                }
            }
            ProviderStdNormalization::Population => reduce_len as f64,
        };
        if denom == 0.0 {
            let _ = self.free(&variance_numer);
            return self.fill_exec(&out_shape, f64::NAN);
        }

        let variance_shape = self.get_entry(&variance_numer)?.shape.clone();
        let denom_tensor = self.fill_exec(&variance_shape, denom)?;
        let variance = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Div,
            &variance_numer,
            &denom_tensor,
        )?;
        let _ = self.free(&denom_tensor);
        let _ = self.free(&variance_numer);

        let abs_variance =
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Abs, &variance)?;
        let variance_plus_abs = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Add,
            &variance,
            &abs_variance,
        )?;
        let _ = self.free(&abs_variance);
        let _ = self.free(&variance);

        let half_shape = self.get_entry(&variance_plus_abs)?.shape.clone();
        let half_tensor = self.fill_exec(&half_shape, 0.5)?;
        let clamped = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &variance_plus_abs,
            &half_tensor,
        )?;
        let _ = self.free(&half_tensor);
        let _ = self.free(&variance_plus_abs);

        let std_handle =
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, &clamped)?;
        let _ = self.free(&clamped);

        Ok(std_handle)
    }

    pub(crate) fn reduce_std_exec(
        &self,
        a: &GpuTensorHandle,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        if matches!(nan_mode, ProviderNanMode::Omit) {
            return Err(anyhow!(
                "reduce_std: omitnan is not supported by the wgpu provider"
            ));
        }
        let entry = self.get_entry(a)?;
        let len = entry.len;
        let scalar_shape = [1usize, 1usize];
        if len == 0 {
            return self.fill_exec(&scalar_shape, f64::NAN);
        }

        let inv_len = 1.0 / (len as f64);

        let sum_handle =
            self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Sum)?;
        let squared = self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, a, a)?;
        let sum_sq_handle =
            self.reduce_global_exec(&squared, crate::backend::wgpu::types::GlobalReduceOp::Sum)?;
        let _ = self.free(&squared);

        let sum_squared = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &sum_handle,
            &sum_handle,
        )?;
        let _ = self.free(&sum_handle);

        let sum_shape = self.get_entry(&sum_squared)?.shape.clone();
        let scale_tensor = self.fill_exec(&sum_shape, inv_len)?;
        let sum_squared_scaled = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &sum_squared,
            &scale_tensor,
        )?;
        let _ = self.free(&scale_tensor);
        let variance_numer = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Sub,
            &sum_sq_handle,
            &sum_squared_scaled,
        )?;
        let _ = self.free(&sum_sq_handle);
        let _ = self.free(&sum_squared);
        let _ = self.free(&sum_squared_scaled);

        let denom = match normalization {
            ProviderStdNormalization::Sample => {
                if len > 1 {
                    (len - 1) as f64
                } else {
                    1.0
                }
            }
            ProviderStdNormalization::Population => len as f64,
        };
        if denom == 0.0 {
            let _ = self.free(&variance_numer);
            return self.fill_exec(&scalar_shape, f64::NAN);
        }

        let variance_shape = self.get_entry(&variance_numer)?.shape.clone();
        let denom_tensor = self.fill_exec(&variance_shape, denom)?;
        let variance = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Div,
            &variance_numer,
            &denom_tensor,
        )?;
        let _ = self.free(&denom_tensor);
        let _ = self.free(&variance_numer);

        let abs_variance =
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Abs, &variance)?;
        let variance_plus_abs = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Add,
            &variance,
            &abs_variance,
        )?;
        let _ = self.free(&abs_variance);
        let _ = self.free(&variance);

        let half_shape = self.get_entry(&variance_plus_abs)?.shape.clone();
        let half_tensor = self.fill_exec(&half_shape, 0.5)?;
        let clamped = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &variance_plus_abs,
            &half_tensor,
        )?;
        let _ = self.free(&half_tensor);
        let _ = self.free(&variance_plus_abs);

        let std_handle =
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, &clamped)?;
        let _ = self.free(&clamped);

        Ok(std_handle)
    }
    pub(crate) async fn fft_dim_exec(
        &self,
        handle: &GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        let HostTensorOwned { data, mut shape } =
            <Self as AccelProvider>::download(self, handle).await?;
        let mut complex_axis = false;
        if shape.last() == Some(&2) {
            complex_axis = true;
            shape.pop();
        }
        if shape.is_empty() {
            if complex_axis {
                let inferred = data.len() / 2;
                shape = vec![inferred];
            } else if data.is_empty() {
                shape = vec![0];
            } else {
                shape = vec![data.len()];
            }
        }
        let origin_rank = shape.len();
        while shape.len() <= dim {
            shape.push(1);
        }
        let current_len = shape.get(dim).copied().unwrap_or(0);
        let target_len = len.unwrap_or(current_len);

        let inner_stride = shape[..dim]
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let outer_stride = shape[dim + 1..]
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let num_slices = inner_stride.saturating_mul(outer_stride);

        let copy_len = current_len.min(target_len);

        let mut out_shape = shape.clone();
        if dim < out_shape.len() {
            out_shape[dim] = target_len;
        }

        if target_len == 0 || num_slices == 0 {
            fft_trim_trailing_ones(&mut out_shape, origin_rank.max(dim + 1));
            let mut packed_shape = out_shape.clone();
            packed_shape.push(2);
            let buffer = self.create_storage_buffer(0, "runmat-fft-empty");
            return Ok(self.register_existing_buffer(buffer, packed_shape, 0));
        }

        let total_elems = shape
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let mut input = Vec::with_capacity(total_elems);
        if complex_axis {
            for idx in 0..total_elems {
                let base = idx * 2;
                let re = data.get(base).copied().unwrap_or(0.0);
                let im = data.get(base + 1).copied().unwrap_or(0.0);
                input.push(Complex::new(re, im));
            }
        } else {
            for idx in 0..total_elems {
                let re = data.get(idx).copied().unwrap_or(0.0);
                input.push(Complex::new(re, 0.0));
            }
        }

        let mut planner = FftPlanner::<f64>::new();
        let fft_plan = if target_len > 1 {
            Some(planner.plan_fft_forward(target_len))
        } else {
            None
        };

        let mut buffer_line = vec![Complex::new(0.0, 0.0); target_len];
        let mut output = vec![Complex::new(0.0, 0.0); target_len.saturating_mul(num_slices)];

        for outer in 0..outer_stride {
            let base_in = outer.saturating_mul(current_len.saturating_mul(inner_stride));
            let base_out = outer.saturating_mul(target_len.saturating_mul(inner_stride));
            for inner in 0..inner_stride {
                buffer_line.fill(Complex::new(0.0, 0.0));
                for (k, slot) in buffer_line.iter_mut().enumerate().take(copy_len) {
                    let src_idx = base_in + inner + k * inner_stride;
                    if src_idx < input.len() {
                        *slot = input[src_idx];
                    }
                }
                if let Some(plan) = &fft_plan {
                    plan.process(&mut buffer_line);
                }
                for (k, value) in buffer_line.iter().enumerate().take(target_len) {
                    let dst_idx = base_out + inner + k * inner_stride;
                    if dst_idx < output.len() {
                        output[dst_idx] = *value;
                    }
                }
            }
        }

        fft_trim_trailing_ones(&mut out_shape, origin_rank.max(dim + 1));
        let mut packed_shape = out_shape.clone();
        packed_shape.push(2);

        let mut packed = Vec::with_capacity(output.len() * 2);
        for complex in output {
            packed.push(complex.re);
            packed.push(complex.im);
        }

        let view = HostTensorView {
            data: &packed,
            shape: &packed_shape,
        };
        let result = self.upload(&view)?;
        Ok(result)
    }
    pub(crate) async fn ifft_dim_exec(
        &self,
        handle: &GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        let HostTensorOwned { data, mut shape } =
            <Self as AccelProvider>::download(self, handle).await?;
        let mut complex_axis = false;
        if shape.last() == Some(&2) {
            complex_axis = true;
            shape.pop();
        }
        if shape.is_empty() {
            if complex_axis {
                let inferred = data.len() / 2;
                shape = vec![inferred];
            } else if data.is_empty() {
                shape = vec![0];
            } else {
                shape = vec![data.len()];
            }
        }
        let origin_rank = shape.len();
        while shape.len() <= dim {
            shape.push(1);
        }
        let current_len = shape.get(dim).copied().unwrap_or(0);
        let target_len = len.unwrap_or(current_len);

        let inner_stride = shape[..dim]
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let outer_stride = shape[dim + 1..]
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let num_slices = inner_stride.saturating_mul(outer_stride);
        let copy_len = current_len.min(target_len);

        let mut out_shape = shape.clone();
        if dim < out_shape.len() {
            out_shape[dim] = target_len;
        }

        if target_len == 0 || num_slices == 0 {
            fft_trim_trailing_ones(&mut out_shape, origin_rank.max(dim + 1));
            let mut packed_shape = out_shape.clone();
            packed_shape.push(2);
            let buffer = self.create_storage_buffer(0, "runmat-ifft-empty");
            return Ok(self.register_existing_buffer(buffer, packed_shape, 0));
        }

        let total_elems = shape
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let mut input = Vec::with_capacity(total_elems);
        if complex_axis {
            for idx in 0..total_elems {
                let base = idx * 2;
                let re = data.get(base).copied().unwrap_or(0.0);
                let im = data.get(base + 1).copied().unwrap_or(0.0);
                input.push(Complex::new(re, im));
            }
        } else {
            for idx in 0..total_elems {
                let re = data.get(idx).copied().unwrap_or(0.0);
                input.push(Complex::new(re, 0.0));
            }
        }

        let mut planner = FftPlanner::<f64>::new();
        let plan = if target_len > 1 {
            Some(planner.plan_fft_inverse(target_len))
        } else {
            None
        };

        let mut buffer_line = vec![Complex::new(0.0, 0.0); target_len];
        let mut output = vec![Complex::new(0.0, 0.0); target_len.saturating_mul(num_slices)];
        let scale = 1.0 / (target_len as f64);

        for outer in 0..outer_stride {
            let base_in = outer.saturating_mul(current_len.saturating_mul(inner_stride));
            let base_out = outer.saturating_mul(target_len.saturating_mul(inner_stride));
            for inner in 0..inner_stride {
                buffer_line.fill(Complex::new(0.0, 0.0));
                for (k, slot) in buffer_line.iter_mut().enumerate().take(copy_len) {
                    let src_idx = base_in + inner + k * inner_stride;
                    if src_idx < input.len() {
                        *slot = input[src_idx];
                    }
                }
                if let Some(plan) = &plan {
                    plan.process(&mut buffer_line);
                }
                for (k, value) in buffer_line.iter().enumerate().take(target_len) {
                    let dst_idx = base_out + inner + k * inner_stride;
                    if dst_idx < output.len() {
                        output[dst_idx] = *value * scale;
                    }
                }
            }
        }

        fft_trim_trailing_ones(&mut out_shape, origin_rank.max(dim + 1));
        let mut packed_shape = out_shape.clone();
        packed_shape.push(2);

        let mut packed = Vec::with_capacity(output.len() * 2);
        for value in output {
            packed.push(value.re);
            packed.push(value.im);
        }

        let view = HostTensorView {
            data: &packed,
            shape: &packed_shape,
        };
        let result = self.upload(&view)?;
        Ok(result)
    }
    pub(crate) fn find_exec(
        &self,
        a: &GpuTensorHandle,
        limit: Option<usize>,
        direction: FindDirection,
    ) -> Result<ProviderFindResult> {
        let entry = self.get_entry(a)?;
        let total = entry.len;
        if total == 0 {
            let shape = vec![0, 1];
            let indices = self.create_storage_buffer(0, "runmat-find-empty-indices");
            let rows = self.create_storage_buffer(0, "runmat-find-empty-rows");
            let cols = self.create_storage_buffer(0, "runmat-find-empty-cols");
            let values = self.create_storage_buffer(0, "runmat-find-empty-values");
            let linear = self.register_existing_buffer(indices, shape.clone(), 0);
            let rows_handle = self.register_existing_buffer(rows, shape.clone(), 0);
            let cols_handle = self.register_existing_buffer(cols, shape.clone(), 0);
            let values_handle = self.register_existing_buffer(values, shape, 0);
            return Ok(ProviderFindResult {
                linear,
                rows: rows_handle,
                cols: cols_handle,
                values: Some(values_handle),
            });
        }

        ensure!(
            total <= u32::MAX as usize,
            "find: tensor length exceeds GPU support"
        );

        let rows_extent = entry.shape.first().copied().unwrap_or(1).max(1);
        ensure!(
            rows_extent <= u32::MAX as usize,
            "find: row extent exceeds GPU support"
        );

        let raw_cap = match direction {
            FindDirection::First => limit.unwrap_or(total),
            FindDirection::Last => limit.unwrap_or(1),
        };
        let cap = raw_cap.min(total);

        if cap == 0 {
            let shape = vec![0, 1];
            let indices = self.create_storage_buffer(0, "runmat-find-zero-limit-indices");
            let rows = self.create_storage_buffer(0, "runmat-find-zero-limit-rows");
            let cols = self.create_storage_buffer(0, "runmat-find-zero-limit-cols");
            let values = self.create_storage_buffer(0, "runmat-find-zero-limit-values");
            let linear = self.register_existing_buffer(indices, shape.clone(), 0);
            let rows_handle = self.register_existing_buffer(rows, shape.clone(), 0);
            let cols_handle = self.register_existing_buffer(cols, shape.clone(), 0);
            let values_handle = self.register_existing_buffer(values, shape, 0);
            return Ok(ProviderFindResult {
                linear,
                rows: rows_handle,
                cols: cols_handle,
                values: Some(values_handle),
            });
        }

        ensure!(cap <= u32::MAX as usize, "find: limit exceeds GPU support");

        let indices_buffer = self.create_storage_buffer(cap, "runmat-find-indices");
        let rows_buffer = self.create_storage_buffer(cap, "runmat-find-rows");
        let cols_buffer = self.create_storage_buffer(cap, "runmat-find-cols");
        let values_buffer = self.create_storage_buffer(cap, "runmat-find-values");

        let count_storage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-find-count-storage"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&count_storage, 0, bytemuck::cast_slice(&[0u32, 0u32]));
        let count_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-find-count-staging"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = crate::backend::wgpu::params::FindParams {
            len: total as u32,
            limit: cap as u32,
            rows: rows_extent as u32,
            direction: match direction {
                FindDirection::First => 0,
                FindDirection::Last => 1,
            },
            include_values: 1,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-find-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-find-bind"),
                layout: &self.pipelines.find.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: indices_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: rows_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: cols_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: values_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: count_storage.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        crate::backend::wgpu::dispatch::find::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.find.pipeline,
            &bind_group,
        );

        let mut copy_encoder =
            self.device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-find-copy-count"),
                });
        copy_encoder.copy_buffer_to_buffer(&count_storage, 0, &count_staging, 0, 8);
        self.submit(copy_encoder);
        let bytes = self.map_readback_bytes_sync(count_staging, 8, "find")?;
        let counts: &[u32] = cast_slice(&bytes);
        let count = counts.first().copied().unwrap_or(0) as usize;

        let shape = vec![count, 1];
        let linear = self.register_existing_buffer(indices_buffer, shape.clone(), count);
        let rows_handle = self.register_existing_buffer(rows_buffer, shape.clone(), count);
        let cols_handle = self.register_existing_buffer(cols_buffer, shape.clone(), count);
        let values_handle = self.register_existing_buffer(values_buffer, shape, count);

        Ok(ProviderFindResult {
            linear,
            rows: rows_handle,
            cols: cols_handle,
            values: Some(values_handle),
        })
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
                },
                indices: HostTensorOwned {
                    data: indices,
                    shape,
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
                },
                indices: HostTensorOwned {
                    data: indices,
                    shape: indices_shape,
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

    fn unique<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        options: &'a UniqueOptions,
    ) -> AccelProviderFuture<'a, UniqueResult> {
        Box::pin(async move {
            let host = <Self as AccelProvider>::download(self, handle).await?;
            let HostTensorOwned { data, shape } = host;
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
            let HostTensorOwned {
                data: lhs_data,
                shape: lhs_shape,
            } = <Self as AccelProvider>::download(self, lhs).await?;
            let HostTensorOwned {
                data: rhs_data,
                shape: rhs_shape,
            } = <Self as AccelProvider>::download(self, rhs).await?;

            let lhs_tensor =
                Tensor::new(lhs_data, lhs_shape).map_err(|e| anyhow!("linsolve: {e}"))?;
            let rhs_tensor =
                Tensor::new(rhs_data, rhs_shape).map_err(|e| anyhow!("linsolve: {e}"))?;

            let (solution, rcond) =
                linsolve_host_real_for_provider(&lhs_tensor, &rhs_tensor, options)
                    .map_err(|e| anyhow!("{e}"))?;

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
            let HostTensorOwned { data, shape } =
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
            let HostTensorOwned { data, shape } =
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
            let HostTensorOwned { data, shape } =
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
            let HostTensorOwned { data, shape } =
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
            let HostTensorOwned { data, shape } =
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
            let HostTensorOwned { data, shape } =
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
            let HostTensorOwned {
                data: lhs_data,
                shape: lhs_shape,
            } = <Self as AccelProvider>::download(self, lhs).await?;
            let HostTensorOwned {
                data: rhs_data,
                shape: rhs_shape,
            } = <Self as AccelProvider>::download(self, rhs).await?;

            let lhs_tensor =
                Tensor::new(lhs_data, lhs_shape).map_err(|e| anyhow!("mldivide: {e}"))?;
            let rhs_tensor =
                Tensor::new(rhs_data, rhs_shape).map_err(|e| anyhow!("mldivide: {e}"))?;

            let result = mldivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
                .map_err(|e| anyhow!("{e}"))?;

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
            let HostTensorOwned {
                data: lhs_data,
                shape: lhs_shape,
            } = <Self as AccelProvider>::download(self, lhs).await?;
            let HostTensorOwned {
                data: rhs_data,
                shape: rhs_shape,
            } = <Self as AccelProvider>::download(self, rhs).await?;

            let lhs_tensor =
                Tensor::new(lhs_data, lhs_shape).map_err(|e| anyhow!("mrdivide: {e}"))?;
            let rhs_tensor =
                Tensor::new(rhs_data, rhs_shape).map_err(|e| anyhow!("mrdivide: {e}"))?;

            let result = mrdivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
                .map_err(|e| anyhow!("{e}"))?;

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
        ensure!(
            bytes <= self.adapter_limits.max_buffer_size,
            "upload: requested {} bytes exceeds device max {}",
            bytes,
            self.adapter_limits.max_buffer_size
        );
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

                    Ok(HostTensorOwned { data: out, shape })
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
        let mut guard = self.buffers.lock().expect("buffer mutex poisoned");
        if let Some(entry) = guard.remove(&h.buffer_id) {
            if entry.len > 0 {
                if Arc::strong_count(&entry.buffer) == 1 {
                    let buffer_ptr = entry.buffer.as_ref() as *const wgpu::Buffer as usize;
                    self.bind_group_cache.invalidate_buffer(buffer_ptr);
                    self.buffer_residency
                        .release(entry.usage, entry.len, entry.buffer.clone());
                } else {
                    log::trace!(
                        "buffer_residency: not pooling buffer id={} len={} due to outstanding views",
                        h.buffer_id,
                        entry.len
                    );
                }
            }
        }
        self.kernel_resources.clear_matmul_source(h.buffer_id);
        runmat_accelerate_api::clear_handle_logical(h);
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
impl WgpuProvider {
    async fn reduce_nd_mean_exec(
        &self,
        a: &GpuTensorHandle,
        dims_zero_based: &[usize],
    ) -> Result<GpuTensorHandle> {
        // If input is a known square of a base tensor, reuse or compute ex2 via moments
        if let Ok(map) = self.pow2_of.lock() {
            if let Some(&base_id) = map.get(&a.buffer_id) {
                drop(map);
                // Try cache
                if let Ok(cache) = self.moments_cache.lock() {
                    if let Some((_mean_h, ex2_h)) = cache.get(&(base_id, dims_zero_based.to_vec()))
                    {
                        return Ok(ex2_h.clone());
                    }
                }
                // Build a handle for the base buffer id
                let base_entry = {
                    let guard = self.buffers.lock().expect("buffer mutex poisoned");
                    guard.get(&base_id).cloned()
                };
                if let Some(entry) = base_entry {
                    let base_handle = GpuTensorHandle {
                        shape: entry.shape.clone(),
                        device_id: self.runtime_device_id,
                        buffer_id: base_id,
                    };
                    let moments = self.reduce_moments_nd_exec(&base_handle, dims_zero_based)?;
                    if let Ok(mut cache2) = self.moments_cache.lock() {
                        cache2.insert(
                            (base_id, dims_zero_based.to_vec()),
                            (moments.mean.clone(), moments.ex2.clone()),
                        );
                    }
                    return Ok(moments.ex2);
                }
            }
        }
        // Prefer computing moments once and caching ex2 for future reuse
        if let Ok(cache) = self.moments_cache.lock() {
            let key = (a.buffer_id, dims_zero_based.to_vec());
            if let Some((mean_h, _ex2_h)) = cache.get(&key) {
                return Ok(mean_h.clone());
            }
            // Compute moments and store
            drop(cache);
            let moments = self.reduce_moments_nd_exec(a, dims_zero_based)?;
            if let Ok(mut cache2) = self.moments_cache.lock() {
                cache2.insert(
                    (a.buffer_id, dims_zero_based.to_vec()),
                    (moments.mean.clone(), moments.ex2.clone()),
                );
            }
            return Ok(moments.mean);
        }
        let entry = self.get_entry(a)?;
        let rank = entry.shape.len();
        ensure!(rank > 0, "reduce_mean_nd: rank must be > 0");
        let mut reduce: Vec<usize> = dims_zero_based
            .iter()
            .copied()
            .filter(|&d| d < rank)
            .collect();
        reduce.sort_unstable();
        reduce.dedup();
        ensure!(
            !reduce.is_empty(),
            "reduce_mean_nd: no valid dims to reduce"
        );
        let kept: Vec<usize> = (0..rank).filter(|i| !reduce.contains(i)).collect();

        // Compute strides (MATLAB/column-major)
        let mut strides: Vec<usize> = vec![0; rank];
        let mut s = 1usize;
        for (i, stride_slot) in strides.iter_mut().enumerate().take(rank) {
            *stride_slot = s;
            s = s
                .checked_mul(entry.shape[i])
                .ok_or_else(|| anyhow!("reduce_mean_nd: shape too large"))?;
        }

        let kept_sizes: Vec<u32> = kept.iter().map(|&i| entry.shape[i] as u32).collect();
        let reduce_sizes: Vec<u32> = reduce.iter().map(|&i| entry.shape[i] as u32).collect();
        let kept_strides: Vec<u32> = kept.iter().map(|&i| strides[i] as u32).collect();
        let reduce_strides: Vec<u32> = reduce.iter().map(|&i| strides[i] as u32).collect();

        let rows: usize = reduce
            .iter()
            .fold(1usize, |acc, &i| acc.saturating_mul(entry.shape[i]));
        let cols: usize = kept
            .iter()
            .fold(1usize, |acc, &i| acc.saturating_mul(entry.shape[i]));
        ensure!(rows > 0 && cols > 0, "reduce_mean_nd: empty tensor");
        if rows as u64 > u32::MAX as u64 || cols as u64 > u32::MAX as u64 {
            return Err(anyhow!("reduce_mean_nd: tensor exceeds GPU limits"));
        }

        // Heuristic fallback: for very large row extents, prefer sequenced dim-reductions.
        if rows >= self.two_pass_threshold() {
            let mut current = a.clone();
            let mut owned = false;
            for &d in &reduce {
                let next = self.reduce_mean_dim(&current, d).await?;
                if owned {
                    let _ = self.free(&current);
                }
                current = next;
                owned = true;
            }
            return Ok(current);
        }

        let mut out_buffer = self.create_storage_buffer(cols, "runmat-reduce-nd-mean-out");
        // Prevent aliasing: output must not be the same as input buffer
        if std::ptr::eq(out_buffer.as_ref(), entry.buffer.as_ref()) {
            let size_bytes = (cols * self.element_size) as u64;
            out_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-nd-mean-out-unique"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let mut out_shape = entry.shape.clone();
        for &d in &reduce {
            out_shape[d] = 1;
        }

        match entry.precision {
            NumericPrecision::F64 => {
                let mut params = crate::backend::wgpu::params::ReduceNdParams {
                    rank: rank as u32,
                    kept_count: kept.len() as u32,
                    reduce_count: reduce.len() as u32,
                    _pad: 0,
                    rows: rows as u32,
                    cols: cols as u32,
                    _pad2: [0; 2],
                    kept_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    kept_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                };
                for (i, v) in kept_sizes.iter().enumerate() {
                    params.kept_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_sizes.iter().enumerate() {
                    params.reduce_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in kept_strides.iter().enumerate() {
                    params.kept_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_strides.iter().enumerate() {
                    params.reduce_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                let pbuf = self.uniform_buffer(&params, "runmat-reduce-nd-mean-params-f64");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-reduce-nd-mean-bind-f64"),
                    layout: &self.pipelines.reduce_nd_mean.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: pbuf.as_entire_binding(),
                        },
                    ],
                });
                if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                    eprintln!(
                        "[reduce-nd-mean f64] in ptr={:p} out ptr={:p} rows={} cols={}",
                        entry.buffer.as_ref(),
                        out_buffer.as_ref(),
                        rows,
                        cols
                    );
                }
                // One workgroup per output column (kept slice)
                let groups_x = cols as u32;
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.reduce_nd_mean.pipeline,
                    &bind_group,
                    groups_x,
                );
            }
            NumericPrecision::F32 => {
                let mut params = crate::backend::wgpu::params::ReduceNdParams {
                    rank: rank as u32,
                    kept_count: kept.len() as u32,
                    reduce_count: reduce.len() as u32,
                    _pad: 0,
                    rows: rows as u32,
                    cols: cols as u32,
                    _pad2: [0; 2],
                    kept_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    kept_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                };
                for (i, v) in kept_sizes.iter().enumerate() {
                    params.kept_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_sizes.iter().enumerate() {
                    params.reduce_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in kept_strides.iter().enumerate() {
                    params.kept_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_strides.iter().enumerate() {
                    params.reduce_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                let pbuf = self.uniform_buffer(&params, "runmat-reduce-nd-mean-params-f32");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-reduce-nd-mean-bind-f32"),
                    layout: &self.pipelines.reduce_nd_mean.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: pbuf.as_entire_binding(),
                        },
                    ],
                });
                if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                    eprintln!(
                        "[reduce-nd-mean f32] in ptr={:p} out ptr={:p} rows={} cols={}",
                        entry.buffer.as_ref(),
                        out_buffer.as_ref(),
                        rows,
                        cols
                    );
                }
                // One workgroup per output column (kept slice)
                let groups_x = cols as u32;
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.reduce_nd_mean.pipeline,
                    &bind_group,
                    groups_x,
                );
            }
        }

        Ok(self.register_existing_buffer(out_buffer, out_shape, cols))
    }

    fn reduce_moments_nd_exec(
        &self,
        a: &GpuTensorHandle,
        dims_zero_based: &[usize],
    ) -> Result<runmat_accelerate_api::ProviderMoments2> {
        let entry = self.get_entry(a)?;
        let rank = entry.shape.len();
        ensure!(rank > 0, "reduce_moments_nd: rank must be > 0");
        let mut reduce: Vec<usize> = dims_zero_based
            .iter()
            .copied()
            .filter(|&d| d < rank)
            .collect();
        reduce.sort_unstable();
        reduce.dedup();
        ensure!(
            !reduce.is_empty(),
            "reduce_moments_nd: no valid dims to reduce"
        );
        let kept: Vec<usize> = (0..rank).filter(|i| !reduce.contains(i)).collect();

        // Strides in column-major
        let mut strides: Vec<usize> = vec![0; rank];
        let mut s = 1usize;
        for (i, stride_slot) in strides.iter_mut().enumerate().take(rank) {
            *stride_slot = s;
            s = s
                .checked_mul(entry.shape[i])
                .ok_or_else(|| anyhow!("reduce_moments_nd: shape too large"))?;
        }

        let kept_sizes: Vec<u32> = kept.iter().map(|&i| entry.shape[i] as u32).collect();
        let reduce_sizes: Vec<u32> = reduce.iter().map(|&i| entry.shape[i] as u32).collect();
        let kept_strides: Vec<u32> = kept.iter().map(|&i| strides[i] as u32).collect();
        let reduce_strides: Vec<u32> = reduce.iter().map(|&i| strides[i] as u32).collect();

        let rows: usize = reduce
            .iter()
            .fold(1usize, |acc, &i| acc.saturating_mul(entry.shape[i]));
        let cols: usize = kept
            .iter()
            .fold(1usize, |acc, &i| acc.saturating_mul(entry.shape[i]));
        ensure!(rows > 0 && cols > 0, "reduce_moments_nd: empty tensor");
        if rows as u64 > u32::MAX as u64 || cols as u64 > u32::MAX as u64 {
            return Err(anyhow!("reduce_moments_nd: tensor exceeds GPU limits"));
        }

        // Allocate outputs
        let mean_out = self.create_storage_buffer(cols, "runmat-reduce-nd-moments-mean");
        let ex2_out = self.create_storage_buffer(cols, "runmat-reduce-nd-moments-ex2");
        let mut out_shape = entry.shape.clone();
        for &d in &reduce {
            out_shape[d] = 1;
        }

        match entry.precision {
            NumericPrecision::F64 => {
                let mut params = crate::backend::wgpu::params::ReduceNdParams {
                    rank: rank as u32,
                    kept_count: kept.len() as u32,
                    reduce_count: reduce.len() as u32,
                    _pad: 0,
                    rows: rows as u32,
                    cols: cols as u32,
                    _pad2: [0; 2],
                    kept_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    kept_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                };
                for (i, v) in kept_sizes.iter().enumerate() {
                    params.kept_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_sizes.iter().enumerate() {
                    params.reduce_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in kept_strides.iter().enumerate() {
                    params.kept_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_strides.iter().enumerate() {
                    params.reduce_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                let pbuf = self.uniform_buffer(&params, "runmat-reduce-nd-moments-params-f64");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-reduce-nd-moments-bind-f64"),
                    layout: &self.pipelines.reduce_nd_moments.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: mean_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: ex2_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: pbuf.as_entire_binding(),
                        },
                    ],
                });
                // One workgroup per output column (kept slice)
                let groups_x = cols as u32;
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.reduce_nd_moments.pipeline,
                    &bind_group,
                    groups_x,
                );
            }
            NumericPrecision::F32 => {
                let mut params = crate::backend::wgpu::params::ReduceNdParams {
                    rank: rank as u32,
                    kept_count: kept.len() as u32,
                    reduce_count: reduce.len() as u32,
                    _pad: 0,
                    rows: rows as u32,
                    cols: cols as u32,
                    _pad2: [0; 2],
                    kept_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    kept_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                };
                for (i, v) in kept_sizes.iter().enumerate() {
                    params.kept_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_sizes.iter().enumerate() {
                    params.reduce_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in kept_strides.iter().enumerate() {
                    params.kept_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_strides.iter().enumerate() {
                    params.reduce_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                let pbuf = self.uniform_buffer(&params, "runmat-reduce-nd-moments-params-f32");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-reduce-nd-moments-bind-f32"),
                    layout: &self.pipelines.reduce_nd_moments.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: mean_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: ex2_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: pbuf.as_entire_binding(),
                        },
                    ],
                });
                let groups_x = cols as u32;
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.reduce_nd_moments.pipeline,
                    &bind_group,
                    groups_x,
                );
            }
        }

        let mean_handle = self.register_existing_buffer(mean_out, out_shape.clone(), cols);
        let ex2_handle = self.register_existing_buffer(ex2_out, out_shape, cols);
        Ok(runmat_accelerate_api::ProviderMoments2 {
            mean: mean_handle,
            ex2: ex2_handle,
        })
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
