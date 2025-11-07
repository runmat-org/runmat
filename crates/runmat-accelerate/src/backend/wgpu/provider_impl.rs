use anyhow::{anyhow, ensure, Result};
use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use log::{info, warn};
use num_complex::Complex;
use once_cell::sync::OnceCell;
use pollster::block_on;
use runmat_accelerate_api::{
    AccelProvider, ApiDeviceInfo, CorrcoefNormalization, CorrcoefOptions, CorrcoefRows,
    CovNormalization, CovRows, CovarianceOptions, FindDirection, FspecialRequest, GpuTensorHandle,
    HostTensorOwned, HostTensorView, ImfilterOptions, ImfilterPadding, IsMemberOptions,
    IsMemberResult, MeshgridAxisView, PagefunOp, PagefunRequest, ProviderBandwidth,
    ProviderCholResult, ProviderCondNorm, ProviderConv1dOptions, ProviderConvMode,
    ProviderConvOrientation, ProviderCummaxResult, ProviderCumminResult, ProviderEigResult,
    ProviderFindResult, ProviderHermitianKind, ProviderIirFilterOptions, ProviderIirFilterResult,
    ProviderInvOptions, ProviderLinsolveOptions, ProviderLinsolveResult, ProviderLuResult,
    ProviderMeshgridResult, ProviderNanMode, ProviderNormOrder, ProviderPinvOptions,
    ProviderPolyderQuotient, ProviderPolyfitResult, ProviderPolyvalOptions, ProviderPrecision,
    ProviderQrOptions, ProviderQrPivot, ProviderQrResult, ProviderScanDirection,
    ProviderStdNormalization, ProviderSymmetryKind, ReduceDimResult, SetdiffOptions, SetdiffResult,
    SortComparison, SortOrder, SortResult, SortRowsColumnSpec, UnionOptions, UnionResult,
    UniqueOptions, UniqueResult,
};
use runmat_builtins::{Tensor, Value};
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
use rustfft::FftPlanner;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::atomic::AtomicU64;
use std::sync::{mpsc, Arc, Mutex};
use wgpu::util::DeviceExt;

use crate::backend::wgpu::cache::{
    bind_group::BindGroupCache, key as cache_key, persist as cache_persist,
};
use crate::backend::wgpu::config::{DEFAULT_REDUCTION_WG, DEFAULT_TWO_PASS_THRESHOLD};
use crate::backend::wgpu::params::{
    BandwidthParams, Conv1dParams, CummaxParams, CumminParams, CumprodParams, CumsumParams,
    DiffParams, FilterParams, ImageNormalizeUniforms, SymmetryParamsF32, SymmetryParamsF64,
    SyrkParams, IMAGE_NORMALIZE_FLAG_BIAS, IMAGE_NORMALIZE_FLAG_GAIN, IMAGE_NORMALIZE_FLAG_GAMMA,
    SYRK_FLAG_ACCUMULATE, SYRK_FLAG_FILL_BOTH,
};
use crate::backend::wgpu::pipelines::WgpuPipelines;
use crate::backend::wgpu::types::NumericPrecision;
use crate::host_lu::{lu_factor_host, LuHostFactors};
use crate::sortrows_host::{sort_rows_host, SortRowsHostOutputs};
use crate::telemetry::AccelTelemetry;

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

// Core WGPU provider state (device, caches, pipelines)
pub struct WgpuProvider {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    adapter_limits: wgpu::Limits,
    buffers: Mutex<HashMap<u64, BufferEntry>>, // in-memory handle table
    buffer_pool: Mutex<HashMap<usize, Vec<Arc<wgpu::Buffer>>>>, // reuse storage buffers by element count
    next_id: AtomicU64,
    pipelines: WgpuPipelines,
    device_id: u32,
    precision: NumericPrecision,
    element_size: usize,
    fused_pipeline_cache: Mutex<HashMap<u64, Arc<wgpu::ComputePipeline>>>,
    bind_group_layout_cache: Mutex<HashMap<String, Arc<wgpu::BindGroupLayout>>>,
    bind_group_cache: BindGroupCache,
    metrics: crate::backend::wgpu::metrics::WgpuMetrics,
    telemetry: AccelTelemetry,
    reduction_two_pass_threshold: usize,
    reduction_workgroup_size_default: u32,
    pipeline_cache_dir: Option<std::path::PathBuf>,
    // Optimization caches
    pow2_of: Mutex<HashMap<u64, u64>>, // squared_buffer_id -> base_buffer_id
    moments_cache: Mutex<HashMap<(u64, Vec<usize>), (GpuTensorHandle, GpuTensorHandle)>>, // (base_buffer_id, dims) -> (mean, ex2)
}

#[derive(Clone)]
struct BufferEntry {
    buffer: Arc<wgpu::Buffer>,
    len: usize,
    shape: Vec<usize>,
    precision: NumericPrecision,
}

#[derive(Clone, Copy)]
struct MatrixOperandView {
    rows: usize,
    cols: usize,
    lda: u32,
    transpose: bool,
}

fn build_matrix_operand_view(
    handle: &GpuTensorHandle,
    entry: &BufferEntry,
) -> Result<MatrixOperandView> {
    if entry.shape.len() != 2 {
        return Err(anyhow!(
            "matrix operand requires 2D tensor (buffer {} shape {:?})",
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
    let cond = (lhs != 0.0) && (rhs != 0.0);
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
    let cond = (lhs != f64(0.0)) && (rhs != f64(0.0));
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
    let cond = lhs != rhs;
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
    let cond = lhs != rhs;
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
    let cond = (lhs != 0.0) || (rhs != 0.0);
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
    let cond = (lhs != f64(0.0)) || (rhs != f64(0.0));
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
    let cond = ((lhs != 0.0) != (rhs != 0.0));
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
    let cond = ((lhs != f64(0.0)) != (rhs != f64(0.0)));
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
        0 => vec![1, 1],
        1 => {
            let n = shape[0];
            vec![n, n]
        }
        _ => shape.to_vec(),
    }
}

fn normalize_concat_shape(mut shape: Vec<usize>, dim_zero: usize) -> Vec<usize> {
    if shape.is_empty() {
        return shape;
    }
    let min_len = ((dim_zero + 1).max(2)).min(shape.len());
    while shape.len() > min_len && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
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
        base.extend(std::iter::repeat(1).take(dim_idx + 1 - base.len()));
    }
    if !base.is_empty() {
        base[dim_idx] = state_len;
    }
    base
}

fn host_tensor_from_value(label: &str, value: Value) -> Result<Tensor> {
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
    if shape.is_empty() {
        shape.push(1);
    }
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
const RNG_SHIFT: u32 = 11;
const RNG_SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
const RNG_MIN_UNIFORM: f64 = 1.0e-16;
const TWO_PI: f64 = PI * 2.0;

fn next_uniform_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(RNG_MULTIPLIER)
        .wrapping_add(RNG_INCREMENT);
    let bits = *state >> RNG_SHIFT;
    (bits as f64) * RNG_SCALE
}

fn uniform_sequence_f64(state: &mut u64, len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(next_uniform_f64(state));
    }
    out
}
fn normal_sequence_f64(state: &mut u64, len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let mut u1 = next_uniform_f64(state);
        if u1 <= 0.0 {
            u1 = RNG_MIN_UNIFORM;
        }
        let u2 = next_uniform_f64(state);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = TWO_PI * u2;
        out.push(radius * angle.cos());
        if out.len() < len {
            out.push(radius * angle.sin());
        }
    }
    out
}
fn integer_sequence_f64(
    state: &mut u64,
    len: usize,
    lower: i64,
    upper: i64,
    span: u64,
) -> Result<Vec<f64>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if span == 1 {
        return Ok(vec![lower as f64; len]);
    }
    let lower_i128 = lower as i128;
    let upper_i128 = upper as i128;
    let span_f = span as f64;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let mut offset = (next_uniform_f64(state) * span_f).floor() as u64;
        if offset >= span {
            offset = span - 1;
        }
        let mut value = lower_i128
            .checked_add(offset as i128)
            .ok_or_else(|| anyhow!("randi: integer overflow while sampling"))?;
        if value > upper_i128 {
            value = upper_i128;
        }
        out.push(value as f64);
    }
    Ok(out)
}
fn randperm_sequence_f64(state: &mut u64, n: usize, k: usize) -> Vec<f64> {
    let mut values: Vec<f64> = if n == 0 {
        Vec::new()
    } else {
        (1..=n).map(|v| v as f64).collect()
    };
    if k > 0 {
        let uniforms = uniform_sequence_f64(state, k);
        for (i, u) in uniforms.into_iter().enumerate() {
            if i >= k || i >= n {
                break;
            }
            let span = n - i;
            if span == 0 {
                break;
            }
            let mut offset = (u * span as f64).floor() as usize;
            if offset >= span {
                offset = span - 1;
            }
            let j = i + offset;
            values.swap(i, j);
        }
    }
    if values.len() > k {
        values.truncate(k);
    }
    values
}

fn rng_state() -> &'static Mutex<u64> {
    static RNG: OnceCell<Mutex<u64>> = OnceCell::new();
    RNG.get_or_init(|| Mutex::new(RNG_DEFAULT_SEED))
}
impl WgpuProvider {
    const BUFFER_POOL_MAX_PER_SIZE: usize = 8;

    fn create_storage_buffer_checked(&self, len: usize, label: &str) -> Result<Arc<wgpu::Buffer>> {
        // Centralised guard + warning for oversized allocations
        let size_bytes = (len as u64) * self.element_size as u64;
        if size_bytes >= (256u64 << 20) {
            log::warn!(
                "{}: large GPU allocation ({} bytes) len={} elems",
                label,
                size_bytes,
                len
            );
        }
        if size_bytes > (self.adapter_limits.max_buffer_size as u64) {
            return Err(anyhow!(
                "{}: requested {} bytes exceeds device max {}",
                label,
                size_bytes,
                self.adapter_limits.max_buffer_size
            ));
        }
        Ok(self.create_storage_buffer(len, label))
    }
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub(crate) fn device_ref(&self) -> &wgpu::Device {
        &self.device
    }
    pub(crate) fn queue_ref(&self) -> &wgpu::Queue {
        &self.queue
    }

    fn warmup_from_disk(&self) {
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

    fn image_normalize_cpu_fallback(
        &self,
        input: &GpuTensorHandle,
        desc: &runmat_accelerate_api::ImageNormalizeDescriptor,
    ) -> Result<GpuTensorHandle> {
        let mut host = self.download(input)?;
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

    pub fn new(opts: WgpuProviderOptions) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: opts.power_preference,
            force_fallback_adapter: opts.force_fallback_adapter,
            compatible_surface: None,
        }))
        .ok_or_else(|| anyhow!("wgpu: no compatible adapter found"))?;

        let adapter_features = adapter.features();
        let forced_precision = std::env::var("RUNMAT_WGPU_FORCE_PRECISION")
            .ok()
            .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
                "f32" | "float32" | "32" => Some(NumericPrecision::F32),
                "f64" | "float64" | "64" => Some(NumericPrecision::F64),
                _ => None,
            });

        let mut precision = if adapter_features.contains(wgpu::Features::SHADER_F64) {
            NumericPrecision::F64
        } else {
            NumericPrecision::F32
        };

        if let Some(requested) = forced_precision {
            if requested == NumericPrecision::F64
                && !adapter_features.contains(wgpu::Features::SHADER_F64)
            {
                warn!(
                    "RunMat Accelerate: requested f64 precision but adapter lacks SHADER_F64; falling back to f32"
                );
                precision = NumericPrecision::F32;
            } else {
                precision = requested;
            }
        }

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
            base.join("runmat")
                .join("pipelines")
                .join(format!("device-{}", device_id))
        } else {
            // Fallback to local target/tmp
            std::path::PathBuf::from("target")
                .join("tmp")
                .join(format!("wgpu-pipeline-cache-{}", device_id))
        };

        Ok(Self {
            device,
            queue,
            adapter_info,
            adapter_limits: limits.clone(),
            buffers: Mutex::new(HashMap::new()),
            buffer_pool: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            pipelines,
            device_id,
            precision,
            element_size,
            fused_pipeline_cache: Mutex::new(HashMap::new()),
            bind_group_layout_cache: Mutex::new(HashMap::new()),
            bind_group_cache: BindGroupCache::new(),
            metrics: crate::backend::wgpu::metrics::WgpuMetrics::new(),
            telemetry: AccelTelemetry::new(),
            reduction_two_pass_threshold: two_pass_threshold,
            reduction_workgroup_size_default: reduction_wg_default,
            pipeline_cache_dir: Some(cache_dir),
            pow2_of: Mutex::new(HashMap::new()),
            moments_cache: Mutex::new(HashMap::new()),
        })
    }

    fn register_existing_buffer(
        &self,
        buffer: Arc<wgpu::Buffer>,
        shape: Vec<usize>,
        len: usize,
    ) -> GpuTensorHandle {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        log::trace!("wgpu register id={} len={} shape={:?}", id, len, &shape);
        let handle = GpuTensorHandle {
            shape,
            device_id: self.device_id,
            buffer_id: id,
        };
        runmat_accelerate_api::set_handle_logical(&handle, false);
        runmat_accelerate_api::clear_handle_transpose(&handle);
        handle
    }

    fn trim_polynomial_handle(
        &self,
        handle: GpuTensorHandle,
        orientation: PolynomialOrientation,
    ) -> Result<GpuTensorHandle> {
        let host = self.download(&handle)?;
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

    fn create_storage_buffer(&self, len: usize, label: &str) -> Arc<wgpu::Buffer> {
        if len > 0 {
            if let Ok(mut pool) = self.buffer_pool.lock() {
                if let Some(list) = pool.get_mut(&len) {
                    if let Some(buf) = list.pop() {
                        return buf;
                    }
                }
            }
        }
        let size_bytes = (len.max(1) as u64) * self.element_size as u64;
        Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
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

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| anyhow!("bandwidth: map_async callback dropped"))?
            .map_err(|e: wgpu::BufferAsyncError| anyhow!(e))?;
        let mapped = slice.get_mapped_range();
        let words: &[u32] = cast_slice(&mapped);
        let lower = words.get(0).copied().unwrap_or(0);
        let upper = words.get(1).copied().unwrap_or(0);
        drop(mapped);
        staging.unmap();

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
        let output_buffer = self.create_storage_buffer(len, "runmat-fusion-output");
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

        // Dispatch in chunks to satisfy 65535 limit
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset_elems = 0usize;
        while offset_elems < len {
            let remaining = len - offset_elems;
            let chunk_len = remaining.min(chunk_capacity);

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
            crate::backend::wgpu::dispatch::elementwise::run(
                self.device_ref(),
                self.queue_ref(),
                &pipeline,
                bind_group.as_ref(),
                workgroups,
            );

            offset_elems += chunk_len;
        }
        Ok(self.register_existing_buffer(output_buffer, output_shape.to_vec(), len))
    }
    pub(crate) fn fused_reduction_exec(
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

        let workgroup_size = if workgroup_size == 0 {
            self.default_reduction_workgroup_size()
        } else {
            workgroup_size
        };
        let two_pass = reduce_len > self.two_pass_threshold() as usize;
        if !two_pass {
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
                let out_buffer = self.create_storage_buffer(out_len, "runmat-reduction-out");
                return Ok(self.register_existing_buffer(
                    out_buffer,
                    output_shape.to_vec(),
                    out_len,
                ));
            }
            let key = self.compute_pipeline_hash_bytes(
                shader.as_bytes(),
                layout_tag,
                Some(workgroup_size),
            );
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
            let flush_enc =
                self.device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-flush-single-pass-gap"),
                    });
            self.submit(flush_enc);
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
            // Build entries dynamically: N inputs, 1 output, 1 params
            let mut entries_vec: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(inputs.len() + 2);
            let mut input_bufs: Vec<Arc<wgpu::Buffer>> = Vec::with_capacity(inputs.len());
            for h in inputs.iter() {
                let buf_arc = self.get_entry(h)?.buffer.clone();
                input_bufs.push(buf_arc);
            }
            for (i, buf_arc) in input_bufs.iter().enumerate() {
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
            let bg = self
                .bind_group_cache
                .get_or_create(bgl.as_ref(), &entries_vec, || {
                    Arc::new(
                        self.device_ref()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-reduction-bg"),
                                layout: bgl.as_ref(),
                                entries: &entries_vec,
                            }),
                    )
                });
            let groups = (num_slices as u32).max(1);
            crate::backend::wgpu::dispatch::reduction::run_single_pass(
                self.device_ref(),
                self.queue_ref(),
                &pipeline,
                bg.as_ref(),
                groups,
            );
            return Ok(self.register_existing_buffer(out_buffer, output_shape.to_vec(), out_len));
        }

        let scalar_ty = match self.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => "f64",
            _ => "f32",
        };
        let flags = if shader.contains("const OMITNAN: bool = true") {
            1u32
        } else {
            0u32
        };
        let chunks = ((reduce_len as u32) + workgroup_size - 1) / workgroup_size;
        let partials_len = num_slices.max(1) * (chunks as usize);
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
            let out_buffer = self.create_storage_buffer(out_len, "runmat-reduction-out");
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
        let partials_buffer = self.create_storage_buffer(partials_len, "runmat-reduction-partials");
        let out_buffer = self.create_storage_buffer(num_slices.max(1), "runmat-reduction-out");
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
        let p1u = P1 {
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
        let p1_buf = Arc::new(self.uniform_buffer(&p1u, "runmat-reduction-p1-params"));
        let p2_buf = Arc::new(self.uniform_buffer(&p2u, "runmat-reduction-p2-params"));
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
        let g1 = chunks.max(1);
        crate::backend::wgpu::dispatch::reduction::run_two_pass(
            self.device_ref(),
            self.queue_ref(),
            &pipeline_p1,
            &pipeline_p2,
            bg1.as_ref(),
            bg2.as_ref(),
            g0,
            g1,
        );
        Ok(self.register_existing_buffer(out_buffer, output_shape.to_vec(), num_slices.max(1)))
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

        let output_buffer = self.create_storage_buffer(len, "runmat-sub2ind-out");
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
        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| anyhow!("failed to receive map_async result"))?
            .map_err(|e: wgpu::BufferAsyncError| anyhow!(e))?;
        let mapped = slice.get_mapped_range();
        let words: &[u32] = cast_slice(&mapped);
        let code = words.get(0).copied().unwrap_or(0);
        let dim_word = words.get(1).copied().unwrap_or(0);
        let extra = words.get(2).copied().unwrap_or(0);
        drop(mapped);
        staging.unmap();

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

        Ok(self.register_existing_buffer(output_buffer, output_shape.to_vec(), len))
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
            output_buffers.push(self.create_storage_buffer(len, "runmat-ind2sub-out"));
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
        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| anyhow!("failed to receive map_async result"))?
            .map_err(|e: wgpu::BufferAsyncError| anyhow!(e))?;
        let mapped = slice.get_mapped_range();
        let words: &[u32] = cast_slice(&mapped);
        let code = words.get(0).copied().unwrap_or(0);
        drop(mapped);
        staging.unmap();

        if code != 0 {
            let err = match code {
                1 | 2 | 3 => anyhow!("Linear indices must be positive integers."),
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
            src_shape.extend(std::iter::repeat(1usize).take(rank - src_shape.len()));
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
            ext_shape.extend(std::iter::repeat(1usize).take(shifts.len() - ext_shape.len()));
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

        Ok(self.register_existing_buffer(out_buffer, out_shape, entry.len))
    }

    pub(crate) fn tril_exec(
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
            return self.tril_exec_fallback(handle, offset);
        }
        if entry.len % plane != 0 {
            return self.tril_exec_fallback(handle, offset);
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
            return self.tril_exec_fallback(handle, offset);
        }

        let diag_offset = if offset > i32::MAX as isize {
            i32::MAX
        } else if offset < -(i32::MAX as isize) {
            -(i32::MAX as i32)
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

        Ok(self.register_existing_buffer(out_buffer, out_shape, entry.len))
    }

    fn tril_exec_fallback(
        &self,
        handle: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let HostTensorOwned { mut data, shape } = self.download(handle)?;
        apply_tril_mask_host(&mut data, &shape, offset)?;
        let view = HostTensorView {
            data: &data,
            shape: &shape,
        };
        <Self as AccelProvider>::upload(self, &view)
    }

    pub(crate) fn triu_exec(
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
            return self.triu_exec_fallback(handle, offset);
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
            return self.triu_exec_fallback(handle, offset);
        }

        let diag_offset = if offset > i32::MAX as isize {
            i32::MAX
        } else if offset < -(i32::MAX as isize) {
            -(i32::MAX as i32)
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

        Ok(self.register_existing_buffer(out_buffer, out_shape, entry.len))
    }

    fn triu_exec_fallback(
        &self,
        handle: &GpuTensorHandle,
        offset: isize,
    ) -> Result<GpuTensorHandle> {
        let HostTensorOwned { mut data, shape } = self.download(handle)?;
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
                ext_shape.extend(std::iter::repeat(1usize).take(needed - ext_shape.len()));
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

        Ok(self.register_existing_buffer(out_buffer, out_shape, entry.len))
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
        let out_buffer = self.create_storage_buffer(output_len, "runmat-conv1d-out");

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

        Ok(self.register_existing_buffer(out_buffer, out_shape, output_len))
    }
    pub(crate) fn iir_filter_exec(
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

        let b_host = self.download(b)?;
        let a_host = self.download(a)?;
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
            shape_ext.extend(std::iter::repeat(1).take(dim + 1 - shape_ext.len()));
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

        let out_buffer = self.create_storage_buffer(new_total, "runmat-repmat-out");
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

        let out_buffer = self.create_storage_buffer(total_len, "runmat-cat-out");
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

        let out_buffer = self.create_storage_buffer(len_out, "runmat-kron-out");
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

        let out_buffer = self.create_storage_buffer_checked(len, "runmat-syrk-out")?;
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, 0));
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

        let mut offset = 0usize;
        let mut first_chunk = true;
        while offset < rows {
            let remaining = rows - offset;
            let chunk_rows = remaining;
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
            let params_buffer = self.uniform_buffer(&params, "runmat-syrk-params");
            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-syrk-bind"),
                    layout: &self.pipelines.syrk.layout,
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

            crate::backend::wgpu::dispatch::syrk::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.syrk.pipeline,
                &bind_group,
                groups_x,
                groups_y,
            );

            offset += chunk_rows;
            first_chunk = false;
        }

        Ok(self.register_existing_buffer(out_buffer, out_shape, len))
    }

    pub(crate) fn matmul_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
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

        let out_shape = vec![m, n];
        let len = m * n;
        if len == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-matmul-out");
            return Ok(self.register_existing_buffer(out_buffer, out_shape, 0));
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
        let use_vec4 = can_vec4 && k < K_CHUNK_SWITCH;
        let enable_chunk = !view_a.transpose && !view_b.transpose && k >= K_CHUNK_SWITCH;

        let start = std::time::Instant::now();

        if enable_chunk {
            self.prepare_matmul_pipeline();
            self.device_ref().poll(wgpu::Maintain::Poll);
            let lda_u32 = view_a.lda;
            let ldb_u32 = view_b.lda;
            // Accumulator handle across chunks
            let mut acc: Option<GpuTensorHandle> = None;
            let mut k_off: usize = 0;
            while k_off < k {
                let k_sub = std::cmp::min(K_CHUNK, k - k_off);
                // Create partial output buffer and bind group
                let partial_buffer =
                    self.create_storage_buffer_checked(len, "runmat-matmul-partial")?;
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
                let params_buffer = self.uniform_buffer(&params, "runmat-matmul-params");
                let bg = self
                    .device_ref()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
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
                                resource: partial_buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
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
                    &bg,
                    groups_x,
                    groups_y,
                );
                // Wrap partial buffer into handle
                let partial = self.register_existing_buffer(partial_buffer, out_shape.clone(), len);
                acc = match acc {
                    None => Some(partial),
                    Some(prev) => Some(self.elem_add(&prev, &partial)?),
                };
                k_off += k_sub;
            }
            let handle = acc.expect("matmul chunking produced no output");
            self.telemetry.record_matmul_duration(start.elapsed());
            return Ok(handle);
        }

        // Default single-dispatch path
        let out_buffer = self.create_storage_buffer_checked(len, "runmat-matmul-out")?;
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
        let params_buffer = self.uniform_buffer(&params, "runmat-matmul-params");
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
        let bg = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-matmul-bind"),
                layout,
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
            &bg,
            groups_x,
            groups_y,
        );
        let handle = self.register_existing_buffer(out_buffer, out_shape, len);
        self.telemetry.record_matmul_duration(start.elapsed());
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
        let out_buffer = self.create_storage_buffer(total_len, "runmat-pagefun-mtimes-out");

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

        let start = std::time::Instant::now();

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

        Ok(self.register_existing_buffer(out_buffer, request.output_shape.clone(), total_len))
    }
    pub(crate) fn covariance_exec(
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
        let ones = self.fill_exec(&[rows, 1], 1.0)?;
        let means_full = self.matmul_exec(&ones, &means)?;
        let _ = self.free(&ones);
        let centered = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Sub,
            matrix,
            &means_full,
        )?;
        let _ = self.free(&means);
        let _ = self.free(&means_full);
        let centered_t = self.transpose_exec(&centered)?;
        let covariance = self.matmul_exec(&centered_t, &centered)?;
        let _ = self.free(&centered);
        let _ = self.free(&centered_t);

        let inv_cov = self.fill_exec(&covariance.shape, 1.0 / denom)?;
        let covariance_scaled = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Mul,
            &covariance,
            &inv_cov,
        )?;
        let _ = self.free(&covariance);
        let _ = self.free(&inv_cov);

        let mut host_cov = self.download(&covariance_scaled)?;
        let rows_out = host_cov.shape.get(0).copied().unwrap_or(cols);
        let cols_out = host_cov.shape.get(1).copied().unwrap_or(rows_out);
        let diag_len = rows_out.min(cols_out);
        for i in 0..diag_len {
            let idx = i + i * rows_out;
            if let Some(value) = host_cov.data.get_mut(idx) {
                if *value < 0.0 && *value > -1.0e-12 {
                    *value = 0.0;
                }
            }
        }
        let view = HostTensorView {
            data: &host_cov.data,
            shape: &host_cov.shape,
        };
        let adjusted = self.upload(&view)?;
        let _ = self.free(&covariance_scaled);
        Ok(adjusted)
    }

    pub(crate) fn corrcoef_exec(
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
        let mut host_variance = self.download(&variance)?;
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
        let out_buffer = self.create_storage_buffer(total_len, "runmat-eye-out");
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
        let out_buffer = self.create_storage_buffer(total_len, "runmat-fill-out");
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
    pub(crate) fn imfilter_exec(
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
            return self.imfilter_exec_fallback(image, kernel, options);
        }
        let image_entry = self.get_entry(image)?;
        let kernel_host = self.download(kernel)?;
        let kernel_tensor = Tensor::new(kernel_host.data.clone(), kernel_host.shape.clone())
            .map_err(|e| anyhow!("imfilter: {e}"))?;

        let image_shape = if image_entry.shape.is_empty() {
            vec![1usize]
        } else {
            image_entry.shape.clone()
        };

        let plan = match build_imfilter_plan(&image_shape, &kernel_tensor, options) {
            Ok(plan) => plan,
            Err(err) => return Err(anyhow!(err)),
        };

        if plan.rank > crate::backend::wgpu::params::IMFILTER_MAX_RANK {
            return self.imfilter_exec_fallback(image, kernel, options);
        }

        let image_ext_product = plan
            .image_shape_ext
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("imfilter: image dimensions exceed GPU limits"))?;
        if image_ext_product != image_entry.len {
            return self.imfilter_exec_fallback(image, kernel, options);
        }

        let output_len = plan
            .output_shape_ext
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("imfilter: output dimensions exceed GPU limits"))?;
        if output_len > u32::MAX as usize || image_entry.len > u32::MAX as usize {
            return self.imfilter_exec_fallback(image, kernel, options);
        }

        let kernel_points_len = plan.kernel_points.len();
        if kernel_points_len > u32::MAX as usize {
            return self.imfilter_exec_fallback(image, kernel, options);
        }

        let mut kernel_offsets = Vec::with_capacity(kernel_points_len * plan.rank);
        let mut kernel_values_f64 = Vec::with_capacity(kernel_points_len);
        for point in &plan.kernel_points {
            if point.offsets.len() != plan.rank {
                return self.imfilter_exec_fallback(image, kernel, options);
            }
            for &offset in &point.offsets {
                if offset < i32::MIN as isize || offset > i32::MAX as isize {
                    return self.imfilter_exec_fallback(image, kernel, options);
                }
                kernel_offsets.push(offset as i32);
            }
            kernel_values_f64.push(point.value);
        }

        if kernel_offsets.len() > u32::MAX as usize {
            return self.imfilter_exec_fallback(image, kernel, options);
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

        let out_buffer = self.create_storage_buffer(output_len, "runmat-imfilter-out");
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
            .get(0)
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

        for k in 0..nz {
            let z_value = z_axis.map(|axis| axis[k]);
            for col in 0..nx {
                let x_value = x_axis[col];
                for row in 0..ny {
                    x_data.push(x_value);
                    y_data.push(y_axis[row]);
                    if let (Some(ref mut z_vec), Some(val)) = (z_data.as_mut(), z_value) {
                        z_vec.push(val);
                    }
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

    fn imfilter_exec_fallback(
        &self,
        image: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: &ImfilterOptions,
    ) -> Result<GpuTensorHandle> {
        let image_host = self.download(image)?;
        let kernel_host = self.download(kernel)?;

        let image_tensor = Tensor::new(image_host.data.clone(), image_host.shape.clone())
            .map_err(|e| anyhow!("imfilter: {e}"))?;
        let kernel_tensor = Tensor::new(kernel_host.data.clone(), kernel_host.shape.clone())
            .map_err(|e| anyhow!("imfilter: {e}"))?;

        let result = runtime_apply_imfilter_tensor(&image_tensor, &kernel_tensor, options)
            .map_err(|e| anyhow!(e))?;
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
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-rng-uniform-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "rand: tensor length too large"
        );
        let values = {
            let mut guard = rng_state()
                .lock()
                .map_err(|_| anyhow!("rand: provider RNG mutex poisoned"))?;
            let mut state = *guard;
            let samples = uniform_sequence_f64(&mut state, total_len);
            *guard = state;
            samples
        };
        let upload_bytes = (total_len as u64) * (self.element_size as u64);
        match self.precision {
            NumericPrecision::F64 => {
                self.queue
                    .write_buffer(out_buffer.as_ref(), 0, cast_slice(&values));
            }
            NumericPrecision::F32 => {
                let data: Vec<f32> = values.iter().map(|v| *v as f32).collect();
                self.queue
                    .write_buffer(out_buffer.as_ref(), 0, cast_slice(&data));
            }
        }
        self.telemetry.record_upload_bytes(upload_bytes);
        Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), total_len))
    }

    pub(crate) fn random_normal_exec(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let total_len = product_checked(shape)
            .ok_or_else(|| anyhow!("randn: tensor size exceeds GPU limits"))?;
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-rng-normal-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "randn: tensor length too large"
        );
        let values = {
            let mut guard = rng_state()
                .lock()
                .map_err(|_| anyhow!("randn: provider RNG mutex poisoned"))?;
            let mut state = *guard;
            let samples = normal_sequence_f64(&mut state, total_len);
            *guard = state;
            samples
        };
        let upload_bytes = (total_len as u64) * (self.element_size as u64);
        match self.precision {
            NumericPrecision::F64 => {
                self.queue
                    .write_buffer(out_buffer.as_ref(), 0, cast_slice(&values));
            }
            NumericPrecision::F32 => {
                let data: Vec<f32> = values.iter().map(|v| *v as f32).collect();
                self.queue
                    .write_buffer(out_buffer.as_ref(), 0, cast_slice(&data));
            }
        }
        self.telemetry.record_upload_bytes(upload_bytes);
        Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), total_len))
    }
    pub(crate) fn fspecial_exec(&self, request: &FspecialRequest) -> Result<GpuTensorHandle> {
        let spec =
            runtime_fspecial_spec_from_request(&request.filter).map_err(|e: String| anyhow!(e))?;

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
        let out_buffer = self.create_storage_buffer(total_len, "runmat-fspecial-out");
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
        let values = {
            let mut guard = rng_state()
                .lock()
                .map_err(|_| anyhow!("randi: provider RNG mutex poisoned"))?;
            let mut state = *guard;
            let samples =
                integer_sequence_f64(&mut state, total_len, lower, upper, span_i128 as u64)?;
            *guard = state;
            samples
        };
        let upload_bytes = (total_len as u64) * (self.element_size as u64);
        match self.precision {
            NumericPrecision::F64 => {
                self.queue
                    .write_buffer(out_buffer.as_ref(), 0, cast_slice(&values));
            }
            NumericPrecision::F32 => {
                let data: Vec<f32> = values.iter().map(|v| *v as f32).collect();
                self.queue
                    .write_buffer(out_buffer.as_ref(), 0, cast_slice(&data));
            }
        }
        self.telemetry.record_upload_bytes(upload_bytes);
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
        let values = {
            let mut guard = rng_state()
                .lock()
                .map_err(|_| anyhow!("randperm: provider RNG mutex poisoned"))?;
            let mut state = *guard;
            let samples = randperm_sequence_f64(&mut state, n, effective_k);
            *guard = state;
            samples
        };
        let upload_bytes = (effective_k as u64) * (self.element_size as u64);
        match self.precision {
            NumericPrecision::F64 => {
                self.queue
                    .write_buffer(out_buffer.as_ref(), 0, cast_slice(&values));
            }
            NumericPrecision::F32 => {
                let data: Vec<f32> = values.iter().map(|v| *v as f32).collect();
                self.queue
                    .write_buffer(out_buffer.as_ref(), 0, cast_slice(&data));
            }
        }
        self.telemetry.record_upload_bytes(upload_bytes);
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

        let out_buffer = self.create_storage_buffer(len, "runmat-polyval-out");
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
        let out_buffer = self.create_storage_buffer(output_len, "runmat-polyint-out");
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

    pub(crate) fn polyder_exec(&self, polynomial: &GpuTensorHandle) -> Result<GpuTensorHandle> {
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
        let out_buffer = self.create_storage_buffer(output_len, "runmat-polyder-out");
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
        self.trim_polynomial_handle(handle, orientation)
    }

    pub(crate) fn polyder_product_exec(
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

        let dp = self.polyder_exec(p)?;
        let dq = self.polyder_exec(q)?;
        let options = ProviderConv1dOptions {
            mode: ProviderConvMode::Full,
            orientation: conv_orientation,
        };
        let term1 = self.conv1d_exec(&dp, q, options)?;
        let term2 = self.conv1d_exec(p, &dq, options)?;
        let result = self.elem_add(&term1, &term2)?;
        self.free(&dp).ok();
        self.free(&dq).ok();
        self.free(&term1).ok();
        self.free(&term2).ok();
        self.trim_polynomial_handle(result, orientation)
    }

    pub(crate) fn polyder_quotient_exec(
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

        let du = self.polyder_exec(u)?;
        let dv = self.polyder_exec(v)?;
        let term1 = self.conv1d_exec(&du, v, options_num)?;
        let term2 = self.conv1d_exec(u, &dv, options_num)?;
        let numerator_handle = self.elem_sub(&term1, &term2)?;
        let denominator_handle = self.conv1d_exec(v, v, options_den)?;
        self.free(&du).ok();
        self.free(&dv).ok();
        self.free(&term1).ok();
        self.free(&term2).ok();

        let numerator = self.trim_polynomial_handle(numerator_handle, orientation_u)?;
        let denominator = self.trim_polynomial_handle(denominator_handle, orientation_v)?;
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
    pub(crate) fn binary_op_exec(
        &self,
        op: crate::backend::wgpu::types::BinaryOpCode,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
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
        let start = std::time::Instant::now();
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
        let out_buffer = self.create_storage_buffer(len, "runmat-binary-out");
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
            let params_buffer = self.uniform_buffer(&params, "runmat-binary-params");
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
        let out_buffer = self.create_storage_buffer(len, "runmat-binary-bcast-out");
        // Prepare params buffer and bind group once; update params per chunk
        let params_size = std::mem::size_of::<BinaryBroadcastParams>() as u64;
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-binary-bcast-params"),
            size: params_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
        let start = std::time::Instant::now();
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
            self.fused_elementwise(shader, &[a.clone()], &entry.shape, len)?
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
            self.fused_elementwise(shader, &[a.clone()], &entry.shape, len)?
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
            self.fused_elementwise(shader, &[a.clone()], &entry.shape, len)?
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
            self.fused_elementwise(shader, &[a.clone()], &entry.shape, len)?
        };
        runmat_accelerate_api::set_handle_logical(&handle, true);
        Ok(handle)
    }
    pub(crate) fn unary_op_exec(
        &self,
        op: crate::backend::wgpu::types::UnaryOpCode,
        a: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let entry_a = self.get_entry(a)?;
        let len = entry_a.len;
        let out_buffer = self.create_storage_buffer(len, "runmat-unary-out");
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry_a.shape, entry_a.len));
        }
        if len > (u32::MAX as usize) {
            return Err(anyhow!("tensor too large for GPU buffer"));
        }
        let start = std::time::Instant::now();
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
            let params_buffer = self.uniform_buffer(&params, "runmat-unary-params");
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
        let out_buffer = self.create_storage_buffer(len, "runmat-scalar-out");
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry_a.shape, entry_a.len));
        }
        if len > (u32::MAX as usize) {
            return Err(anyhow!("tensor too large for GPU buffer"));
        }
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        let start = std::time::Instant::now();
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
                    self.uniform_buffer(&params, "runmat-scalar-params")
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
                    self.uniform_buffer(&params, "runmat-scalar-params")
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
        let mut current = entry.buffer.clone();
        let mut current_len = entry.len;
        while current_len > 1 {
            let wg = crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE as usize;
            let max_groups = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize;
            let elems_per_group = 2 * wg;
            let max_input_per_pass = max_groups * elems_per_group;

            let output_len_total = ((current_len + elems_per_group - 1) / elems_per_group).max(1);
            let out_buffer = self.create_storage_buffer(output_len_total, "runmat-reduce-pass");

            let mut in_offset_elems = 0usize;
            let mut _out_offset_elems = 0usize;
            while in_offset_elems < current_len {
                let remain = current_len - in_offset_elems;
                let chunk_in = remain.min(max_input_per_pass);
                let chunk_out = ((chunk_in + elems_per_group - 1) / elems_per_group).max(1);

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
                let groups = crate::backend::wgpu::dispatch::common::dispatch_size_reduce(
                    chunk_in as u32,
                    crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE,
                );
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
            out_len as u32,
            crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE,
        );
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
    pub(crate) fn fft_dim_exec(
        &self,
        handle: &GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        let HostTensorOwned { data, mut shape } = self.download(handle)?;
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
                for k in 0..copy_len {
                    let src_idx = base_in + inner + k * inner_stride;
                    if src_idx < input.len() {
                        buffer_line[k] = input[src_idx];
                    }
                }
                if let Some(plan) = &fft_plan {
                    plan.process(&mut buffer_line);
                }
                for k in 0..target_len {
                    let dst_idx = base_out + inner + k * inner_stride;
                    if dst_idx < output.len() {
                        output[dst_idx] = buffer_line[k];
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
    pub(crate) fn ifft_dim_exec(
        &self,
        handle: &GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        let HostTensorOwned { data, mut shape } = self.download(handle)?;
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
                for k in 0..copy_len {
                    let src_idx = base_in + inner + k * inner_stride;
                    if src_idx < input.len() {
                        buffer_line[k] = input[src_idx];
                    }
                }
                if let Some(plan) = &plan {
                    plan.process(&mut buffer_line);
                }
                for k in 0..target_len {
                    let dst_idx = base_out + inner + k * inner_stride;
                    if dst_idx < output.len() {
                        output[dst_idx] = buffer_line[k] * scale;
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
        let slice = count_staging.slice(..8);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| anyhow!("map_async callback dropped"))?
            .map_err(|e: wgpu::BufferAsyncError| anyhow!(e))?;
        let mapped = slice.get_mapped_range();
        let counts: &[u32] = cast_slice(&mapped);
        let count = counts.get(0).copied().unwrap_or(0) as usize;
        drop(mapped);
        count_staging.unmap();

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

    fn fspecial(&self, request: &FspecialRequest) -> Result<GpuTensorHandle> {
        self.fspecial_exec(request)
    }

    fn imfilter(
        &self,
        image: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: &ImfilterOptions,
    ) -> Result<GpuTensorHandle> {
        self.imfilter_exec(image, kernel, options)
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

    fn polyfit(
        &self,
        x: &GpuTensorHandle,
        y: &GpuTensorHandle,
        degree: usize,
        weights: Option<&GpuTensorHandle>,
    ) -> Result<ProviderPolyfitResult> {
        let x_host = self.download(x)?;
        let y_host = self.download(y)?;
        ensure!(
            x_host.data.len() == y_host.data.len(),
            "polyfit: X and Y vectors must match in length"
        );
        let weights_host = match weights {
            Some(handle) => Some(self.download(handle)?),
            None => None,
        };
        let weights_slice = weights_host.as_ref().map(|w| w.data.as_slice());
        let host_result =
            polyfit_host_real_for_provider(&x_host.data, &y_host.data, degree, weights_slice)
                .map_err(|e| anyhow!(e))?;
        Ok(ProviderPolyfitResult {
            coefficients: host_result.coefficients,
            r_matrix: host_result.r_matrix,
            normr: host_result.normr,
            df: host_result.df,
            mu: host_result.mu,
        })
    }

    fn polyder_single(&self, polynomial: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.polyder_exec(polynomial)
    }

    fn polyder_product(&self, p: &GpuTensorHandle, q: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.polyder_product_exec(p, q)
    }

    fn polyder_quotient(
        &self,
        u: &GpuTensorHandle,
        v: &GpuTensorHandle,
    ) -> Result<ProviderPolyderQuotient> {
        self.polyder_quotient_exec(u, v)
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

    fn tril(&self, matrix: &GpuTensorHandle, offset: isize) -> Result<GpuTensorHandle> {
        self.tril_exec(matrix, offset)
    }

    fn triu(&self, matrix: &GpuTensorHandle, offset: isize) -> Result<GpuTensorHandle> {
        self.triu_exec(matrix, offset)
    }

    fn reduce_mean_nd(
        &self,
        a: &GpuTensorHandle,
        dims_zero_based: &[usize],
    ) -> Result<GpuTensorHandle> {
        self.reduce_nd_mean_exec(a, dims_zero_based)
    }

    fn reduce_moments_nd(
        &self,
        a: &GpuTensorHandle,
        dims_zero_based: &[usize],
    ) -> Result<runmat_accelerate_api::ProviderMoments2> {
        self.reduce_moments_nd_exec(a, dims_zero_based)
    }

    fn elem_add(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Add, a, b)
    }

    fn elem_mul(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, a, b)
    }

    fn elem_sub(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, a, b)
    }

    fn elem_max(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Max, a, b)
    }

    fn elem_min(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Min, a, b)
    }

    fn elem_div(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Div, a, b)
    }

    fn elem_pow(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Pow, a, b)
    }

    fn elem_ge(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.elem_ge_exec(a, b)
    }

    fn elem_le(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.elem_le_exec(a, b)
    }

    fn elem_lt(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.elem_lt_exec(a, b)
    }

    fn elem_gt(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.elem_gt_exec(a, b)
    }

    fn elem_eq(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.elem_eq_exec(a, b)
    }

    fn elem_ne(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.elem_ne_exec(a, b)
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

    fn elem_hypot(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Hypot, a, b)
    }

    fn elem_atan2(&self, y: &GpuTensorHandle, x: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Atan2, y, x)
    }

    fn unary_sin(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sin, a)
    }

    fn unary_gamma(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Gamma, a)
    }

    fn unary_factorial(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Factorial, a)
    }

    fn unary_asinh(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Asinh, a)
    }

    fn unary_sinh(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sinh, a)
    }

    fn unary_cosh(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Cosh, a)
    }

    fn unary_asin(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Asin, a)
    }

    fn unary_acos(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Acos, a)
    }

    fn unary_acosh(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Acosh, a)
    }

    fn unary_tan(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Tan, a)
    }

    fn unary_tanh(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Tanh, a)
    }

    fn unary_atan(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Atan, a)
    }
    fn unary_atanh(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Atanh, a)
    }

    fn unary_ceil(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Ceil, a)
    }

    fn unary_floor(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Floor, a)
    }

    fn unary_fix(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Fix, a)
    }

    fn unary_cos(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Cos, a)
    }

    fn unary_abs(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Abs, a)
    }

    fn unary_conj(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Conj, a)
    }

    fn unary_exp(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Exp, a)
    }

    fn unary_log(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Log, a)
    }

    fn unary_log1p(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Log1p, a)
    }

    fn unary_sqrt(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, a)
    }

    fn unary_double(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        if self.precision != NumericPrecision::F64 {
            return Err(anyhow!(
                "wgpu provider: shader-f64 unavailable; cannot materialise double precision"
            ));
        }
        let entry = self.get_entry(a)?;
        Ok(self.register_existing_buffer(entry.buffer, entry.shape, entry.len))
    }

    fn unary_single(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Single, a)
    }

    fn unary_pow2(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let out = self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Pow2, a)?;
        // Record squared->base mapping for later reduction fusion (moments reuse)
        if let Ok(mut map) = self.pow2_of.lock() {
            map.insert(out.buffer_id, a.buffer_id);
        }
        Ok(out)
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
        let result = self.elem_mul(mantissa, &pow);
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

    fn sort_dim(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        order: SortOrder,
        comparison: SortComparison,
    ) -> Result<SortResult> {
        let host = self.download(a)?;
        let shape = host.shape.clone();
        let (values, indices) = sort_host_tensor(&host.data, &host.shape, dim, order, comparison)?;
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
    }
    fn sort_rows(
        &self,
        a: &GpuTensorHandle,
        columns: &[SortRowsColumnSpec],
        comparison: SortComparison,
    ) -> Result<SortResult> {
        let host = self.download(a)?;
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
    fn iir_filter(
        &self,
        b: &GpuTensorHandle,
        a: &GpuTensorHandle,
        x: &GpuTensorHandle,
        options: ProviderIirFilterOptions,
    ) -> Result<ProviderIirFilterResult> {
        self.iir_filter_exec(b, a, x, options)
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

    fn fft_dim(
        &self,
        handle: &GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        self.fft_dim_exec(handle, len, dim)
    }

    fn ifft_dim(
        &self,
        handle: &GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        self.ifft_dim_exec(handle, len, dim)
    }

    fn unique(&self, handle: &GpuTensorHandle, options: &UniqueOptions) -> Result<UniqueResult> {
        let host = self.download(handle)?;
        let HostTensorOwned { data, shape } = host;
        let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("unique: {e}"))?;
        let eval =
            runmat_runtime::builtins::array::sorting_sets::unique::unique_numeric_from_tensor(
                tensor, options,
            )
            .map_err(|e| anyhow!("unique: {e}"))?;
        eval.into_numeric_unique_result()
            .map_err(|e| anyhow!("unique: {e}"))
    }

    fn ismember(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        options: &IsMemberOptions,
    ) -> Result<IsMemberResult> {
        let host_a = self.download(a)?;
        let host_b = self.download(b)?;
        let tensor_a =
            Tensor::new(host_a.data, host_a.shape).map_err(|e| anyhow!("ismember: {e}"))?;
        let tensor_b =
            Tensor::new(host_b.data, host_b.shape).map_err(|e| anyhow!("ismember: {e}"))?;
        runmat_runtime::builtins::array::sorting_sets::ismember::ismember_numeric_from_tensors(
            tensor_a,
            tensor_b,
            options.rows,
        )
        .and_then(|eval| eval.into_numeric_ismember_result())
        .map_err(|e| anyhow!("ismember: {e}"))
    }

    fn union(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        options: &UnionOptions,
    ) -> Result<UnionResult> {
        let host_a = self.download(a)?;
        let host_b = self.download(b)?;
        let tensor_a = Tensor::new(host_a.data, host_a.shape).map_err(|e| anyhow!("union: {e}"))?;
        let tensor_b = Tensor::new(host_b.data, host_b.shape).map_err(|e| anyhow!("union: {e}"))?;
        let eval =
            runmat_runtime::builtins::array::sorting_sets::union::union_numeric_from_tensors(
                tensor_a, tensor_b, options,
            )
            .map_err(|e| anyhow!("union: {e}"))?;
        eval.into_numeric_union_result()
            .map_err(|e| anyhow!("union: {e}"))
    }

    fn setdiff(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        options: &SetdiffOptions,
    ) -> Result<SetdiffResult> {
        let host_a = self.download(a)?;
        let host_b = self.download(b)?;
        let tensor_a =
            Tensor::new(host_a.data, host_a.shape).map_err(|e| anyhow!("setdiff: {e}"))?;
        let tensor_b =
            Tensor::new(host_b.data, host_b.shape).map_err(|e| anyhow!("setdiff: {e}"))?;
        runmat_runtime::builtins::array::sorting_sets::setdiff::setdiff_numeric_from_tensors(
            tensor_a, tensor_b, options,
        )
        .and_then(|eval| eval.into_numeric_setdiff_result())
        .map_err(|e| anyhow!("setdiff: {e}"))
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

    fn lu(&self, a: &GpuTensorHandle) -> Result<ProviderLuResult> {
        let host = self.download(a)?;
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
    }

    fn chol(&self, a: &GpuTensorHandle, lower: bool) -> Result<ProviderCholResult> {
        let host = self.download(a)?;
        let tensor =
            Tensor::new(host.data.clone(), host.shape.clone()).map_err(|e| anyhow!("chol: {e}"))?;
        let mut args = Vec::new();
        if lower {
            args.push(Value::from("lower"));
        }
        let eval = runmat_runtime::builtins::math::linalg::factor::chol::evaluate(
            Value::Tensor(tensor),
            &args,
        )
        .map_err(|e| anyhow!("chol: {e}"))?;
        let factor_tensor = host_tensor_from_value("chol", eval.factor())?;
        let factor = self.upload(&HostTensorView {
            data: &factor_tensor.data,
            shape: &factor_tensor.shape,
        })?;
        Ok(ProviderCholResult {
            factor,
            info: eval.flag_index() as u32,
        })
    }
    fn qr(&self, handle: &GpuTensorHandle, options: ProviderQrOptions) -> Result<ProviderQrResult> {
        let host = self.download(handle)?;
        let tensor =
            Tensor::new(host.data.clone(), host.shape.clone()).map_err(|e| anyhow!("qr: {e}"))?;
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
        .map_err(|e| anyhow!("qr: {e}"))?;

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
    fn matmul(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.matmul_exec(a, b)
    }

    fn syrk(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.syrk_exec(a)
    }

    fn matmul_epilogue(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        ep: &runmat_accelerate_api::MatmulEpilogue,
    ) -> Result<GpuTensorHandle> {
        use runmat_accelerate_api::ProviderPrecision;
        let entry_a = self.get_entry(a)?;
        let entry_b = self.get_entry(b)?;
        if entry_a.shape.len() != 2 || entry_b.shape.len() != 2 {
            return Err(anyhow!("matmul_epilogue: only 2D tensors supported"));
        }
        let view_a =
            build_matrix_operand_view(a, &entry_a).map_err(|e| anyhow!("matmul_epilogue: {e}"))?;
        let view_b =
            build_matrix_operand_view(b, &entry_b).map_err(|e| anyhow!("matmul_epilogue: {e}"))?;

        if view_a.cols != view_b.rows {
            return Err(anyhow!("matmul_epilogue: inner dimensions must match"));
        }
        let m = view_a.rows;
        let n = view_b.cols;
        let k = view_a.cols;

        let out_shape = vec![m, n];
        let len = m * n;
        let out_buffer = self.create_storage_buffer_checked(len, "runmat-matmul-epilogue-out")?;
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, len));
        }

        let start = std::time::Instant::now();

        // Ensure pipelines exist
        self.prepare_matmul_pipeline();
        self.device_ref().poll(wgpu::Maintain::Poll);

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
        let groups_x = crate::backend::wgpu::dispatch::common::dispatch_size_dim(n as u32, tile);
        let groups_y = crate::backend::wgpu::dispatch::common::dispatch_size_dim(m as u32, tile);

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
        let key = self.compute_pipeline_hash_bytes(shader_src.as_bytes(), &layout_tag, Some(tile));
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

        self.telemetry.record_matmul_duration(start.elapsed());

        Ok(self.register_existing_buffer(out_buffer, out_shape, len))
    }
    fn pagefun(&self, request: &PagefunRequest) -> Result<GpuTensorHandle> {
        self.pagefun_exec(request)
    }
    fn image_normalize(
        &self,
        input: &GpuTensorHandle,
        desc: &runmat_accelerate_api::ImageNormalizeDescriptor,
    ) -> Result<GpuTensorHandle> {
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
            return self.image_normalize_cpu_fallback(input, desc);
        }

        match self.precision {
            NumericPrecision::F64 => self.image_normalize_cpu_fallback(input, desc),
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

                let uniforms = ImageNormalizeUniforms {
                    batches: batches_u32,
                    height: height_u32,
                    width: width_u32,
                    plane: plane_u32,
                    stride_h: stride_h_u32,
                    stride_w: stride_w_u32,
                    flags,
                    _pad0: 0,
                    epsilon: desc.epsilon as f32,
                    gain: desc.gain.unwrap_or(1.0) as f32,
                    bias: desc.bias.unwrap_or(0.0) as f32,
                    gamma: desc.gamma.unwrap_or(1.0) as f32,
                };

                let out_buffer =
                    self.create_storage_buffer_checked(entry.len, "runmat-image-normalize-out")?;
                let uniform_buf = self.uniform_buffer(&uniforms, "runmat-image-normalize-uniform");

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-image-normalize-bind"),
                    layout: &self.pipelines.image_normalize.layout,
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
                            resource: uniform_buf.as_entire_binding(),
                        },
                    ],
                });

                crate::backend::wgpu::dispatch::image_normalize::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.image_normalize,
                    &bind_group,
                    batches_u32,
                );

                Ok(self.register_existing_buffer(out_buffer, entry.shape.clone(), entry.len))
            }
        }
    }
    fn matmul_power_step(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        epilogue: &runmat_accelerate_api::PowerStepEpilogue,
    ) -> Result<GpuTensorHandle> {
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
        let norms = self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, &sum_sq)?;
        let _ = self.free(&sum_sq);
        let normalized = self.binary_op_exec(
            crate::backend::wgpu::types::BinaryOpCode::Div,
            &product,
            &norms,
        )?;
        let _ = self.free(&product);
        let _ = self.free(&norms);
        Ok(normalized)
    }
    fn covariance(
        &self,
        matrix: &GpuTensorHandle,
        second: Option<&GpuTensorHandle>,
        weights: Option<&GpuTensorHandle>,
        options: &CovarianceOptions,
    ) -> Result<GpuTensorHandle> {
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

        let result = (|| {
            let source = combined.as_ref().unwrap_or(matrix);
            self.covariance_exec(source, options)
        })();

        if let Some(handle) = combined {
            let _ = self.free(&handle);
        }

        result
    }

    fn corrcoef(
        &self,
        matrix: &GpuTensorHandle,
        options: &CorrcoefOptions,
    ) -> Result<GpuTensorHandle> {
        self.corrcoef_exec(matrix, options)
    }

    fn linsolve(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        options: &ProviderLinsolveOptions,
    ) -> Result<ProviderLinsolveResult> {
        let HostTensorOwned {
            data: lhs_data,
            shape: lhs_shape,
        } = self.download(lhs)?;
        let HostTensorOwned {
            data: rhs_data,
            shape: rhs_shape,
        } = self.download(rhs)?;

        let lhs_tensor = Tensor::new(lhs_data, lhs_shape).map_err(|e| anyhow!("linsolve: {e}"))?;
        let rhs_tensor = Tensor::new(rhs_data, rhs_shape).map_err(|e| anyhow!("linsolve: {e}"))?;

        let (solution, rcond) = linsolve_host_real_for_provider(&lhs_tensor, &rhs_tensor, options)
            .map_err(|e| anyhow!("{e}"))?;

        let handle = self.upload(&HostTensorView {
            data: &solution.data,
            shape: &solution.shape,
        })?;

        Ok(ProviderLinsolveResult {
            solution: handle,
            reciprocal_condition: rcond,
        })
    }

    fn inv(
        &self,
        matrix: &GpuTensorHandle,
        _options: ProviderInvOptions,
    ) -> Result<GpuTensorHandle> {
        let HostTensorOwned { data, shape } = self.download(matrix)?;
        let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("inv: {e}"))?;
        let result = inv_host_real_for_provider(&tensor).map_err(|e| anyhow!("{e}"))?;
        self.upload(&HostTensorView {
            data: &result.data,
            shape: &result.shape,
        })
    }

    fn pinv(
        &self,
        matrix: &GpuTensorHandle,
        options: ProviderPinvOptions,
    ) -> Result<GpuTensorHandle> {
        let HostTensorOwned { data, shape } = self.download(matrix)?;
        let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("pinv: {e}"))?;
        let result =
            pinv_host_real_for_provider(&tensor, options.tolerance).map_err(|e| anyhow!("{e}"))?;
        self.upload(&HostTensorView {
            data: &result.data,
            shape: &result.shape,
        })
    }

    fn cond(&self, matrix: &GpuTensorHandle, norm: ProviderCondNorm) -> Result<GpuTensorHandle> {
        let HostTensorOwned { data, shape } = self.download(matrix)?;
        let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("cond: {e}"))?;
        let cond_value = cond_host_real_for_provider(&tensor, norm).map_err(|e| anyhow!("{e}"))?;
        let scalar = [cond_value];
        let shape = [1usize, 1usize];
        self.upload(&HostTensorView {
            data: &scalar,
            shape: &shape,
        })
    }

    fn norm(&self, tensor: &GpuTensorHandle, order: ProviderNormOrder) -> Result<GpuTensorHandle> {
        let HostTensorOwned { data, shape } = self.download(tensor)?;
        let host_tensor = Tensor::new(data, shape).map_err(|e| anyhow!("norm: {e}"))?;
        let value = norm_host_real_for_provider(&host_tensor, order).map_err(|e| anyhow!("{e}"))?;
        let scalar = [value];
        let shape = [1usize, 1usize];
        self.upload(&HostTensorView {
            data: &scalar,
            shape: &shape,
        })
    }

    fn rank(&self, matrix: &GpuTensorHandle, tolerance: Option<f64>) -> Result<GpuTensorHandle> {
        let HostTensorOwned { data, shape } = self.download(matrix)?;
        let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("rank: {e}"))?;
        let rank =
            rank_host_real_for_provider(&tensor, tolerance).map_err(|e| anyhow!("{e}"))? as f64;
        let scalar = [rank];
        let shape = [1usize, 1usize];
        self.upload(&HostTensorView {
            data: &scalar,
            shape: &shape,
        })
    }

    fn rcond(&self, matrix: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let HostTensorOwned { data, shape } = self.download(matrix)?;
        let tensor = Tensor::new(data, shape).map_err(|e| anyhow!("rcond: {e}"))?;
        let estimate = rcond_host_real_for_provider(&tensor).map_err(|e| anyhow!("{e}"))?;
        let scalar = [estimate];
        let shape = [1usize, 1usize];
        self.upload(&HostTensorView {
            data: &scalar,
            shape: &shape,
        })
    }

    fn mldivide(&self, lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let HostTensorOwned {
            data: lhs_data,
            shape: lhs_shape,
        } = self.download(lhs)?;
        let HostTensorOwned {
            data: rhs_data,
            shape: rhs_shape,
        } = self.download(rhs)?;

        let lhs_tensor = Tensor::new(lhs_data, lhs_shape).map_err(|e| anyhow!("mldivide: {e}"))?;
        let rhs_tensor = Tensor::new(rhs_data, rhs_shape).map_err(|e| anyhow!("mldivide: {e}"))?;

        let result = mldivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
            .map_err(|e| anyhow!("{e}"))?;

        let handle = self.upload(&HostTensorView {
            data: &result.data,
            shape: &result.shape,
        })?;
        Ok(handle)
    }

    fn mrdivide(&self, lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let HostTensorOwned {
            data: lhs_data,
            shape: lhs_shape,
        } = self.download(lhs)?;
        let HostTensorOwned {
            data: rhs_data,
            shape: rhs_shape,
        } = self.download(rhs)?;

        let lhs_tensor = Tensor::new(lhs_data, lhs_shape).map_err(|e| anyhow!("mrdivide: {e}"))?;
        let rhs_tensor = Tensor::new(rhs_data, rhs_shape).map_err(|e| anyhow!("mrdivide: {e}"))?;

        let result = mrdivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
            .map_err(|e| anyhow!("{e}"))?;

        let handle = self.upload(&HostTensorView {
            data: &result.data,
            shape: &result.shape,
        })?;
        Ok(handle)
    }

    fn dot(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        self.dot_exec(lhs, rhs, dim)
    }
    fn eig(&self, handle: &GpuTensorHandle, compute_left: bool) -> Result<ProviderEigResult> {
        let host = self.download(handle)?;
        let tensor =
            Tensor::new(host.data.clone(), host.shape.clone()).map_err(|e| anyhow!("eig: {e}"))?;
        let eval = runmat_runtime::builtins::math::linalg::factor::eig::evaluate(
            Value::Tensor(tensor),
            &[],
            compute_left,
        )
        .map_err(|e| anyhow!("eig: {e}"))?;

        let eigenvalues_tensor = host_tensor_from_value("eig", eval.eigenvalues())?;
        let diagonal_tensor = host_tensor_from_value("eig", eval.diagonal_matrix())?;
        let right_tensor = host_tensor_from_value("eig", eval.right())?;

        let left_value = if compute_left {
            Some(eval.left().map_err(|e| anyhow!("eig: {e}"))?)
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
    }

    fn reduce_sum_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<GpuTensorHandle> {
        self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Sum)
    }

    fn reduce_nnz_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<GpuTensorHandle> {
        self.reduce_dim_sum_mean_exec(
            a,
            dim,
            crate::backend::wgpu::types::DimReduceOp::CountNonZero,
        )
    }

    fn reduce_prod_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<GpuTensorHandle> {
        self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Prod)
    }

    fn reduce_mean_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<GpuTensorHandle> {
        self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Mean)
    }
    fn reduce_any_dim(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> Result<GpuTensorHandle> {
        let op = if omit_nan {
            crate::backend::wgpu::types::DimReduceOp::AnyOmit
        } else {
            crate::backend::wgpu::types::DimReduceOp::AnyInclude
        };
        self.reduce_dim_sum_mean_exec(a, dim, op)
    }
    fn reduce_any(&self, a: &GpuTensorHandle, omit_nan: bool) -> Result<GpuTensorHandle> {
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
    }

    fn reduce_all_dim(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> Result<GpuTensorHandle> {
        let op = if omit_nan {
            crate::backend::wgpu::types::DimReduceOp::AllOmit
        } else {
            crate::backend::wgpu::types::DimReduceOp::AllInclude
        };
        self.reduce_dim_sum_mean_exec(a, dim, op)
    }

    fn reduce_all(&self, a: &GpuTensorHandle, omit_nan: bool) -> Result<GpuTensorHandle> {
        let op = if omit_nan {
            crate::backend::wgpu::types::DimReduceOp::AllOmit
        } else {
            crate::backend::wgpu::types::DimReduceOp::AllInclude
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
    }

    fn reduce_median(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let host = self.download(a)?;
        let median = median_from_slice(&host.data);
        let data = [median];
        let shape = [1usize, 1usize];
        self.upload(&HostTensorView {
            data: &data,
            shape: &shape,
        })
    }

    fn reduce_median_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<GpuTensorHandle> {
        let host = self.download(a)?;
        if host.shape.len() != 2 {
            return Err(anyhow!("reduce_median_dim: only 2D supported"));
        }
        let rows = host.shape[0];
        let cols = host.shape[1];
        let mut scratch = Vec::<f64>::with_capacity(rows.max(cols));
        let (out, shape) = if dim <= 1 {
            let mut values = vec![f64::NAN; cols];
            for c in 0..cols {
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
                values[c] = if saw_nan || scratch.is_empty() {
                    f64::NAN
                } else {
                    compute_median_inplace(&mut scratch)
                };
            }
            (values, vec![1usize, cols])
        } else {
            let mut values = vec![f64::NAN; rows];
            for r in 0..rows {
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
                values[r] = if saw_nan || scratch.is_empty() {
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
    }

    fn reduce_sum(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Sum)
    }

    fn reduce_nnz(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::CountNonZero)
    }

    fn reduce_prod(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Prod)
    }

    fn reduce_mean(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
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
    }
    fn reduce_std(
        &self,
        a: &GpuTensorHandle,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        self.reduce_std_exec(a, normalization, nan_mode)
    }

    fn reduce_std_dim(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        self.reduce_std_dim_exec(a, dim, normalization, nan_mode)
    }

    fn reduce_min(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Min)
    }

    fn reduce_max(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Max)
    }

    fn reduce_min_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<ReduceDimResult> {
        self.reduce_dim_minmax_exec(a, dim, crate::backend::wgpu::types::DimReduceExtrema::Min)
    }

    fn reduce_max_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<ReduceDimResult> {
        self.reduce_dim_minmax_exec(a, dim, crate::backend::wgpu::types::DimReduceExtrema::Max)
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

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| anyhow!("issymmetric: map_async callback dropped"))?
            .map_err(|e: wgpu::BufferAsyncError| anyhow!(e))?;
        let mapped = slice.get_mapped_range();
        let words: &[u32] = cast_slice(&mapped);
        let flag = words.get(0).copied().unwrap_or(0);
        drop(mapped);
        staging.unmap();

        Ok(flag != 0)
    }

    fn ishermitian(
        &self,
        matrix: &GpuTensorHandle,
        kind: ProviderHermitianKind,
        tolerance: f64,
    ) -> Result<bool> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(anyhow!(
                "ishermitian: tolerance must be finite and non-negative"
            ));
        }
        let host = self.download(matrix)?;
        let skew = matches!(kind, ProviderHermitianKind::Skew);
        ishermitian_host_real_data(&host.shape, &host.data, skew, tolerance).map_err(|e| anyhow!(e))
    }

    fn bandwidth(&self, matrix: &GpuTensorHandle) -> Result<ProviderBandwidth> {
        self.bandwidth_exec(matrix)
    }

    fn sym_rcm(&self, matrix: &GpuTensorHandle) -> Result<Vec<usize>> {
        let host = self.download(matrix)?;
        symrcm_host_real_data(&host.shape, &host.data).map_err(|e| anyhow!(e))
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
        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| anyhow!("read_scalar: map_async dropped"))?
            .map_err(|e: wgpu::BufferAsyncError| anyhow!(e))?;
        let mapped = slice.get_mapped_range();
        let value = match entry.precision {
            NumericPrecision::F64 => {
                let words: &[f64] = cast_slice(&mapped);
                words.get(0).copied().unwrap_or(0.0)
            }
            NumericPrecision::F32 => {
                let words: &[f32] = cast_slice(&mapped);
                words.get(0).copied().unwrap_or(0.0) as f64
            }
        };
        drop(mapped);
        staging.unmap();
        Ok(value)
    }

    fn fused_elementwise(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
    ) -> Result<GpuTensorHandle> {
        let start = std::time::Instant::now();
        let result = self.fused_elementwise_exec(shader, inputs, output_shape, len);
        if result.is_ok() {
            let elapsed = start.elapsed();
            self.telemetry.record_fused_elementwise_duration(elapsed);
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
        self.fused_elementwise(shader, &[a.clone()], &entry.shape, len)
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
        self.fused_elementwise(shader, &[a.clone()], &entry.shape, len)
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
        let start = std::time::Instant::now();
        let result = self.fused_reduction_exec(
            shader,
            inputs,
            output_shape,
            reduce_len,
            num_slices,
            workgroup_size,
        );
        if result.is_ok() {
            let elapsed = start.elapsed();
            self.telemetry.record_fused_reduction_duration(elapsed);
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

        let start = std::time::Instant::now();
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
        self.telemetry
            .snapshot(fusion_hits, fusion_misses, bind_hits, bind_misses)
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
        let bytes = (len as u64).saturating_mul(self.element_size as u64);
        self.telemetry.record_upload_bytes(bytes);
        Ok(self.register_existing_buffer(buffer, shape, len))
    }
    fn download(&self, h: &GpuTensorHandle) -> Result<HostTensorOwned> {
        log::trace!("wgpu download id={} shape={:?}", h.buffer_id, &h.shape);
        let entry = self.get_entry(h)?;
        if entry.len == 0 {
            return Ok(HostTensorOwned {
                data: Vec::new(),
                shape: h.shape.clone(),
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

        Ok(HostTensorOwned { data: out, shape })
    }
    fn free(&self, h: &GpuTensorHandle) -> Result<()> {
        // Remove from handle table and return buffer to pool for reuse
        log::trace!("wgpu free id={}", h.buffer_id);
        let mut guard = self.buffers.lock().expect("buffer mutex poisoned");
        if let Some(entry) = guard.remove(&h.buffer_id) {
            if entry.len > 0 {
                if Arc::strong_count(&entry.buffer) == 1 {
                    let mut pool = self.buffer_pool.lock().expect("buffer pool mutex poisoned");
                    let list = pool.entry(entry.len).or_default();
                    if list.len() < Self::BUFFER_POOL_MAX_PER_SIZE {
                        list.push(entry.buffer.clone());
                    } else {
                        // Drop buffer instead of pooling to cap memory footprint
                        log::trace!(
                            "buffer_pool: dropping buffer (len={} elems) due to pool cap",
                            entry.len
                        );
                    }
                } else {
                    log::trace!(
                        "buffer_pool: not pooling buffer id={} len={} due to outstanding views",
                        h.buffer_id,
                        entry.len
                    );
                }
            }
        }
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
            device_id: self.device_id,
            name: self.adapter_info.name.clone(),
            vendor: canonical_vendor_name(&self.adapter_info),
            memory_bytes,
            backend: Some(backend),
        }
    }
}
impl WgpuProvider {
    fn reduce_nd_mean_exec(
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
                        device_id: self.device_id,
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
        for i in 0..rank {
            strides[i] = s;
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
                let next = self.reduce_mean_dim(&current, d)?;
                if owned {
                    let _ = self.free(&current);
                }
                current = next;
                owned = true;
            }
            return Ok(current);
        }

        let out_buffer = self.create_storage_buffer(cols, "runmat-reduce-nd-mean-out");
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
        for i in 0..rank {
            strides[i] = s;
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
