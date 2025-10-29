use anyhow::{anyhow, ensure, Result};
use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use log::{info, warn};
use once_cell::sync::OnceCell;
use pollster::block_on;
use runmat_accelerate_api::{
    AccelProvider, ApiDeviceInfo, CorrcoefNormalization, CorrcoefOptions, CorrcoefRows,
    CovNormalization, CovRows, CovarianceOptions, FindDirection, FspecialRequest, GpuTensorHandle,
    HostTensorOwned, HostTensorView, ImfilterOptions, ImfilterPadding, IsMemberOptions,
    IsMemberResult, PagefunOp, PagefunRequest, ProviderFindResult, ProviderPrecision,
    ReduceDimResult, SetdiffOptions, SetdiffResult, SortComparison, SortOrder, SortResult,
    SortRowsColumnSpec, UnionOptions, UnionResult, UniqueOptions, UniqueResult,
};
use runmat_builtins::Tensor;
use runmat_runtime::builtins::image::filters::fspecial::{
    spec_from_request as runtime_fspecial_spec_from_request, FspecialFilterSpec,
};
use runmat_runtime::builtins::image::filters::imfilter::{
    apply_imfilter_tensor as runtime_apply_imfilter_tensor, build_imfilter_plan,
};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{mpsc, Arc, Mutex};
use wgpu::util::DeviceExt;

use crate::backend::wgpu::cache::{key as cache_key, persist as cache_persist};
use crate::backend::wgpu::config::{DEFAULT_REDUCTION_WG, DEFAULT_TWO_PASS_THRESHOLD};
use crate::backend::wgpu::pipelines::WgpuPipelines;
use crate::backend::wgpu::types::NumericPrecision;
use crate::sortrows_host::{sort_rows_host, SortRowsHostOutputs};

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
    next_id: AtomicU64,
    pipelines: WgpuPipelines,
    device_id: u32,
    precision: NumericPrecision,
    element_size: usize,
    fused_pipeline_cache: Mutex<HashMap<u64, Arc<wgpu::ComputePipeline>>>,
    metrics: crate::backend::wgpu::metrics::WgpuMetrics,
    reduction_two_pass_threshold: usize,
    reduction_workgroup_size_default: u32,
    pipeline_cache_dir: Option<std::path::PathBuf>,
}

#[derive(Clone)]
struct BufferEntry {
    buffer: Arc<wgpu::Buffer>,
    len: usize,
    shape: Vec<usize>,
    precision: NumericPrecision,
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

@compute @workgroup_size(256)
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

const RNG_MULTIPLIER: u64 = 6364136223846793005;
const RNG_INCREMENT: u64 = 1;
const RNG_SHIFT: u32 = 11;
const RNG_SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
const RNG_DEFAULT_SEED: u64 = 0x9e3779b97f4a7c15;
const MAX_SAFE_INTEGER: u64 = 1 << 53;

fn rng_state() -> &'static Mutex<u64> {
    static RNG: OnceCell<Mutex<u64>> = OnceCell::new();
    RNG.get_or_init(|| Mutex::new(RNG_DEFAULT_SEED))
}

fn next_uniform(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(RNG_MULTIPLIER)
        .wrapping_add(RNG_INCREMENT);
    let bits = *state >> RNG_SHIFT;
    (bits as f64) * RNG_SCALE
}

fn next_normal_pair(state: &mut u64) -> (f64, f64) {
    let mut u1 = next_uniform(state);
    if u1 <= 0.0 {
        u1 = f64::MIN_POSITIVE;
    }
    let u2 = next_uniform(state);
    let radius = (-2.0 * u1.ln()).sqrt();
    let angle = 2.0 * std::f64::consts::PI * u2;
    (radius * angle.cos(), radius * angle.sin())
}

impl WgpuProvider {
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub(crate) fn device_ref(&self) -> &wgpu::Device {
        &self.device
    }
    pub(crate) fn queue_ref(&self) -> &wgpu::Queue {
        &self.queue
    }

    // removed unused helper build_bgl_for_layout_tag; call bindings::build_bgl_for_layout_tag directly where needed

    fn warmup_from_disk(&self) {
        crate::backend::wgpu::warmup::warmup_from_disk(
            &self.device,
            self.pipeline_cache_dir.as_deref(),
            |bytes, tag, wg| self.compute_pipeline_hash_bytes(bytes, tag, wg),
            |key, pl, module, label, src, tag, wg| {
                self.get_or_create_pipeline(key, pl, module, label, src, tag, wg)
            },
            |pipeline| {
                crate::backend::wgpu::warmup::noop_after_create(&self.device, &self.queue, pipeline)
            },
        );
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
            next_id: AtomicU64::new(1),
            pipelines,
            device_id,
            precision,
            element_size,
            fused_pipeline_cache: Mutex::new(HashMap::new()),
            metrics: crate::backend::wgpu::metrics::WgpuMetrics::new(),
            reduction_two_pass_threshold: two_pass_threshold,
            reduction_workgroup_size_default: reduction_wg_default,
            pipeline_cache_dir: Some(cache_dir),
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
        let handle = GpuTensorHandle {
            shape,
            device_id: self.device_id,
            buffer_id: id,
        };
        runmat_accelerate_api::set_handle_logical(&handle, false);
        handle
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
        let bind_group_layout =
            crate::backend::wgpu::bindings::build_fusion_bgl(self.device_ref(), inputs.len());
        let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-fusion-pipeline-layout",
            &bind_group_layout,
        );
        let layout_tag = {
            let mut tag = String::from("runmat-fusion-layout-");
            tag.push_str(&inputs.len().to_string());
            tag
        };
        let shader_hash = self.compute_pipeline_hash_bytes(
            shader.as_bytes(),
            &layout_tag,
            Some(crate::backend::wgpu::config::WORKGROUP_SIZE),
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
            Some(crate::backend::wgpu::config::WORKGROUP_SIZE),
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
        let params = crate::backend::wgpu::params::FusionParams {
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
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-fusion-bind-group"),
                layout: &bind_group_layout,
                entries: &bind_entries,
            });
        {
            crate::backend::wgpu::dispatch::elementwise::warmup_noop(
                self.device_ref(),
                self.queue_ref(),
                &pipeline,
            );
        }
        self.device_ref().poll(wgpu::Maintain::Poll);
        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::elementwise::run(
            self.device_ref(),
            self.queue_ref(),
            &pipeline,
            &bind_group,
            workgroups,
        );
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
            let layout_tag = "runmat-reduction-bgl";
            let module = crate::backend::wgpu::pipelines::create_shader_module(
                self.device_ref(),
                "runmat-fused-reduction-module",
                shader,
            );
            let bgl = crate::backend::wgpu::bindings::build_bgl_for_layout_tag(
                self.device_ref(),
                layout_tag,
            )
            .expect("reduction bgl");
            let pl = crate::backend::wgpu::pipelines::create_pipeline_layout(
                self.device_ref(),
                "runmat-reduction-pl",
                &bgl,
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
            let bg = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-reduction-bg"),
                    layout: &bgl,
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
            let groups = (num_slices as u32).max(1);
            crate::backend::wgpu::dispatch::reduction::run_single_pass(
                self.device_ref(),
                self.queue_ref(),
                &pipeline,
                &bg,
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
        let bgl1 = crate::backend::wgpu::bindings::build_bgl_for_layout_tag(
            self.device_ref(),
            "runmat-reduction-p1-bgl",
        )
        .expect("p1 bgl");
        let bgl2 = crate::backend::wgpu::bindings::build_bgl_for_layout_tag(
            self.device_ref(),
            "runmat-reduction-p2-bgl",
        )
        .expect("p2 bgl");
        let pl1 = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-reduction-p1-pl",
            &bgl1,
        );
        let pl2 = crate::backend::wgpu::pipelines::create_pipeline_layout(
            self.device_ref(),
            "runmat-reduction-p2-pl",
            &bgl2,
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
        let bg1 = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
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
        let bg2 = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
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
        let g0 = (num_slices as u32).max(1);
        let g1 = chunks.max(1);
        crate::backend::wgpu::dispatch::reduction::run_two_pass(
            self.device_ref(),
            self.queue_ref(),
            &pipeline_p1,
            &pipeline_p2,
            &bg1,
            &bg2,
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
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(cols as u32, 256);
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

        let mut src_shape_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::PERMUTE_MAX_RANK];
        let mut dst_shape_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::PERMUTE_MAX_RANK];
        let mut order_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::PERMUTE_MAX_RANK];
        let mut strides_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::PERMUTE_MAX_RANK];
        for i in 0..rank {
            src_shape_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(src_shape[i] as u32);
            dst_shape_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(dst_shape[i] as u32);
            order_arr[i] = crate::backend::wgpu::params::PackedU32::from_scalar(order[i] as u32);
            strides_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(src_strides[i] as u32);
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

        let mut shape_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::CIRCSHIFT_MAX_RANK];
        let mut strides_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::CIRCSHIFT_MAX_RANK];
        let mut shifts_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::CIRCSHIFT_MAX_RANK];
        for axis in 0..rank {
            shape_arr[axis] =
                crate::backend::wgpu::params::PackedU32::from_scalar(ext_shape[axis] as u32);
            strides_arr[axis] =
                crate::backend::wgpu::params::PackedU32::from_scalar(strides[axis] as u32);
            let denom = ext_shape[axis].max(1);
            shifts_arr[axis] = crate::backend::wgpu::params::PackedU32::from_scalar(
                (normalized[axis] % denom) as u32,
            );
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

        let mut shape_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::FLIP_MAX_RANK];
        let mut strides_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::FLIP_MAX_RANK];
        let mut flags_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::FLIP_MAX_RANK];
        for i in 0..rank {
            shape_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(ext_shape[i] as u32);
            strides_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(strides[i] as u32);
            flags_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(if flags[i] { 1 } else { 0 });
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

        let mut base_shape_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::REPMAT_MAX_RANK];
        let mut new_shape_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::REPMAT_MAX_RANK];
        let mut strides_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::REPMAT_MAX_RANK];
        for i in 0..rank {
            base_shape_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(base_shape[i] as u32);
            new_shape_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(new_shape[i] as u32);
            strides_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(base_strides[i] as u32);
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

        let mut shape_a_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::KRON_MAX_RANK];
        let mut shape_b_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::KRON_MAX_RANK];
        let mut shape_out_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::KRON_MAX_RANK];
        let mut stride_a_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::KRON_MAX_RANK];
        let mut stride_b_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::KRON_MAX_RANK];
        for i in 0..rank {
            shape_a_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(shape_a[i] as u32);
            shape_b_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(shape_b[i] as u32);
            shape_out_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(shape_out[i] as u32);
            stride_a_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(strides_a[i] as u32);
            stride_b_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(strides_b[i] as u32);
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
        let out_shape = vec![cols, rows];
        let out_buffer = self.create_storage_buffer(len, "runmat-transpose-out");
        if len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, out_shape, len));
        }
        if len > (u32::MAX as usize) {
            return Err(anyhow!("tensor too large for GPU buffer"));
        }
        {
            let mut enc =
                self.device_ref()
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
        self.device_ref().poll(wgpu::Maintain::Poll);
        {
            let enc = self
                .device_ref()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-transpose-flush-gap"),
                });
            self.submit(enc);
        }
        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let mut offset = 0usize;
        while offset < len {
            let remaining = len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let params = crate::backend::wgpu::params::TransposeParams {
                rows: rows as u32,
                cols: cols as u32,
                len: chunk_len as u32,
                offset: offset as u32,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-transpose-params");
            let bg = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
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
            let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_len as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::transpose::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.transpose.pipeline,
                &bg,
                groups,
            );
            offset += chunk_len;
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
        self.prepare_matmul_pipeline();
        self.device_ref().poll(wgpu::Maintain::Poll);
        let params = crate::backend::wgpu::params::MatmulParams {
            m: m as u32,
            n: n as u32,
            k: k_a as u32,
            lda: entry_a.shape[0] as u32,
            ldb: entry_b.shape[0] as u32,
            ldc: m as u32,
            offset_a: 0,
            offset_b: 0,
            offset_out: 0,
            _pad: 0,
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
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        let groups_x = crate::backend::wgpu::dispatch::common::dispatch_size_dim(
            n as u32,
            crate::backend::wgpu::config::MATMUL_TILE,
        );
        let groups_y = crate::backend::wgpu::dispatch::common::dispatch_size_dim(
            m as u32,
            crate::backend::wgpu::config::MATMUL_TILE,
        );
        crate::backend::wgpu::dispatch::matmul::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.matmul.pipeline,
            &bg,
            groups_x,
            groups_y,
        );
        Ok(self.register_existing_buffer(out_buffer, out_shape, len))
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

        let canonical_a = canonical_matrix_shape(&entry_a.shape);
        let canonical_b = canonical_matrix_shape(&entry_b.shape);
        ensure!(
            canonical_a.len() >= 2 && canonical_b.len() >= 2,
            "pagefun: @mtimes operands must be at least 2-D"
        );

        let rows = canonical_a[0];
        let k_a = canonical_a[1];
        let k_b = canonical_b[0];
        let cols = canonical_b[1];
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

        let lda = m_u32;
        let ldb = k_u32;
        let ldc = m_u32;

        let groups_x = crate::backend::wgpu::dispatch::common::dispatch_size_dim(
            n_u32,
            crate::backend::wgpu::config::MATMUL_TILE,
        );
        let groups_y = crate::backend::wgpu::dispatch::common::dispatch_size_dim(
            m_u32,
            crate::backend::wgpu::config::MATMUL_TILE,
        );

        self.prepare_matmul_pipeline();
        self.device_ref().poll(wgpu::Maintain::Poll);

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
                _pad: 0,
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

        let params_buffer = match self.precision {
            NumericPrecision::F64 => {
                let params = crate::backend::wgpu::params::FillParamsF64 {
                    value,
                    len: total_len as u32,
                    _pad: [0, 0, 0],
                };
                self.uniform_buffer(&params, "runmat-fill-params-f64")
            }
            NumericPrecision::F32 => {
                let params = crate::backend::wgpu::params::FillParamsF32 {
                    value: value as f32,
                    len: total_len as u32,
                    _pad: [0, 0],
                };
                self.uniform_buffer(&params, "runmat-fill-params-f32")
            }
        };

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-fill-bind"),
                layout: &self.pipelines.fill.layout,
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
            &self.pipelines.fill.pipeline,
            &bind_group,
            workgroups,
            "runmat-fill-encoder",
            "runmat-fill-pass",
        );

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

        let mut image_shape_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::IMFILTER_MAX_RANK];
        let mut image_strides_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::IMFILTER_MAX_RANK];
        let mut output_shape_arr = [crate::backend::wgpu::params::PackedU32::default();
            crate::backend::wgpu::params::IMFILTER_MAX_RANK];
        let mut base_offset_arr = [crate::backend::wgpu::params::PackedI32::default();
            crate::backend::wgpu::params::IMFILTER_MAX_RANK];

        for i in 0..plan.rank {
            let dim = plan.image_shape_ext[i];
            ensure!(
                dim <= u32::MAX as usize,
                "imfilter: image dimension exceeds GPU limits"
            );
            image_shape_arr[i] = crate::backend::wgpu::params::PackedU32::from_scalar(dim as u32);

            let stride = plan.image_strides[i];
            ensure!(
                stride <= u32::MAX as usize,
                "imfilter: image stride exceeds GPU limits"
            );
            image_strides_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(stride as u32);

            let out_dim = plan.output_shape_ext[i];
            ensure!(
                out_dim <= u32::MAX as usize,
                "imfilter: output dimension exceeds GPU limits"
            );
            output_shape_arr[i] =
                crate::backend::wgpu::params::PackedU32::from_scalar(out_dim as u32);

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
                        _pad_tail: crate::backend::wgpu::params::PackedU32::default(),
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
                        _pad_tail: crate::backend::wgpu::params::PackedU32::default(),
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
        let mut data = Vec::with_capacity(total_len);
        if total_len > 0 {
            let mut guard = rng_state()
                .lock()
                .map_err(|_| anyhow!("rand: provider RNG mutex poisoned"))?;
            for _ in 0..total_len {
                data.push(next_uniform(&mut *guard));
            }
            drop(guard);
        }
        let view = HostTensorView { data: &data, shape };
        let handle = <Self as AccelProvider>::upload(self, &view)?;
        Ok(handle)
    }

    pub(crate) fn random_normal_exec(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let total_len = product_checked(shape)
            .ok_or_else(|| anyhow!("randn: tensor size exceeds GPU limits"))?;
        let mut data = Vec::with_capacity(total_len);
        if total_len > 0 {
            let mut guard = rng_state()
                .lock()
                .map_err(|_| anyhow!("randn: provider RNG mutex poisoned"))?;
            while data.len() < total_len {
                let (z0, z1) = next_normal_pair(&mut *guard);
                data.push(z0);
                if data.len() < total_len {
                    data.push(z1);
                }
            }
            drop(guard);
        }
        let view = HostTensorView { data: &data, shape };
        let handle = <Self as AccelProvider>::upload(self, &view)?;
        Ok(handle)
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
        ensure!(
            lower <= upper,
            "randi: lower bound must be <= upper bound for wgpu provider"
        );
        let span_i128 = (upper as i128)
            .checked_sub(lower as i128)
            .and_then(|delta| delta.checked_add(1))
            .ok_or_else(|| anyhow!("randi: integer range overflow"))?;
        ensure!(span_i128 > 0, "randi: integer range must be non-empty");
        ensure!(
            span_i128 <= (1i128 << 53),
            "randi: integer range exceeds 2^53 and cannot be represented exactly"
        );
        let span = span_i128 as u64;

        let total_len = product_checked(shape)
            .ok_or_else(|| anyhow!("randi: tensor size exceeds GPU limits"))?;
        let mut data = Vec::with_capacity(total_len);
        if total_len > 0 {
            if span == 1 {
                data.resize(total_len, lower as f64);
            } else {
                let mut guard = rng_state()
                    .lock()
                    .map_err(|_| anyhow!("randi: provider RNG mutex poisoned"))?;
                let span_f64 = span as f64;
                for _ in 0..total_len {
                    let mut u = next_uniform(&mut *guard);
                    if u >= 1.0 {
                        u = 0.9999999999999999;
                    }
                    let mut offset = (u * span_f64).floor() as u64;
                    if offset >= span {
                        offset = span - 1;
                    }
                    let value = (lower as i128 + offset as i128) as f64;
                    data.push(value);
                }
                drop(guard);
            }
        }

        let view = HostTensorView { data: &data, shape };
        let handle = <Self as AccelProvider>::upload(self, &view)?;
        Ok(handle)
    }

    pub(crate) fn randperm_exec(&self, n: usize, k: usize) -> Result<GpuTensorHandle> {
        ensure!(
            k <= n,
            "randperm: K must satisfy 0 <= K <= N for wgpu provider"
        );
        ensure!(
            (n as u64) <= MAX_SAFE_INTEGER,
            "randperm: N exceeds 2^53 and cannot be represented exactly"
        );
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
        if effective_k == 0 {
            let buffer = self.create_storage_buffer(0, "runmat-randperm-out");
            return Ok(self.register_existing_buffer(buffer, shape_vec, 0));
        }

        let mut values: Vec<f64> = (1..=n).map(|v| v as f64).collect();
        if effective_k > 0 {
            let mut guard = rng_state()
                .lock()
                .map_err(|_| anyhow!("randperm: provider RNG mutex poisoned"))?;
            for i in 0..effective_k {
                let span = n - i;
                if span == 0 {
                    break;
                }
                let mut u = next_uniform(&mut *guard);
                if u >= 1.0 {
                    u = 0.9999999999999999;
                }
                let mut offset = (u * span as f64).floor() as usize;
                if offset >= span {
                    offset = span - 1;
                }
                let j = i + offset;
                values.swap(i, j);
            }
            drop(guard);
        }

        if values.len() > effective_k {
            values.truncate(effective_k);
        }

        let view = HostTensorView {
            data: &values,
            shape: &shape_vec,
        };
        let handle = <Self as AccelProvider>::upload(self, &view)?;
        Ok(handle)
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
        Ok(self.register_existing_buffer(out_buffer, entry_a.shape, len))
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
            self.fused_elementwise_exec(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
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
            self.fused_elementwise_exec(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
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
            self.fused_elementwise_exec(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
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
            self.fused_elementwise_exec(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
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
            self.fused_elementwise_exec(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
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
            self.fused_elementwise_exec(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
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
            self.fused_elementwise_exec(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
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
            self.fused_elementwise_exec(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
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
            self.fused_elementwise_exec(shader, &[a.clone(), b.clone()], &entry_a.shape, len)?
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
            self.fused_elementwise_exec(shader, &[a.clone()], &entry.shape, len)?
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
            self.fused_elementwise_exec(shader, &[a.clone()], &entry.shape, len)?
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
            self.fused_elementwise_exec(shader, &[a.clone()], &entry.shape, len)?
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
            self.fused_elementwise_exec(shader, &[a.clone()], &entry.shape, len)?
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
        Ok(self.register_existing_buffer(out_buffer, entry_a.shape, len))
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
        Ok(self.register_existing_buffer(out_buffer, entry_a.shape, len))
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
                crate::backend::wgpu::types::GlobalReduceOp::Min => f64::INFINITY,
                crate::backend::wgpu::types::GlobalReduceOp::Max => f64::NEG_INFINITY,
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
            let output_len = ((current_len
                + (crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE as usize * 2)
                - 1)
                / (crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE as usize * 2))
                .max(1);
            let out_buffer = self.create_storage_buffer(output_len, "runmat-reduce-pass");
            let params = crate::backend::wgpu::params::ReduceGlobalParams {
                len: current_len as u32,
                op: op as u32,
                _pad0: 0,
                _pad1: 0,
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
                current_len as u32,
                crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::reduction::run_single_pass(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.reduce_global.pipeline,
                &bind_group,
                groups,
            );
            current = out_buffer;
            current_len = output_len;
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
            crate::backend::wgpu::config::WORKGROUP_SIZE,
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

        let count_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runmat-find-count"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&count_buffer, 0, bytemuck::cast_slice(&[0u32, 0u32]));

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
                        resource: count_buffer.as_entire_binding(),
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

        let slice = count_buffer.slice(..8);
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
        count_buffer.unmap();

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

    fn elem_add(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Add, a, b)
    }

    fn elem_mul(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Mul, a, b)
    }

    fn elem_sub(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Sub, a, b)
    }

    fn elem_div(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Div, a, b)
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

    fn unary_sin(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sin, a)
    }

    fn unary_cos(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Cos, a)
    }

    fn unary_abs(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Abs, a)
    }

    fn unary_exp(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Exp, a)
    }

    fn unary_log(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Log, a)
    }

    fn unary_sqrt(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, a)
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

    fn circshift(&self, handle: &GpuTensorHandle, shifts: &[isize]) -> Result<GpuTensorHandle> {
        self.circshift_exec(handle, shifts)
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

    fn matmul(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.matmul_exec(a, b)
    }
    fn pagefun(&self, request: &PagefunRequest) -> Result<GpuTensorHandle> {
        self.pagefun_exec(request)
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

    fn reduce_sum_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<GpuTensorHandle> {
        self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Sum)
    }

    fn reduce_mean_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<GpuTensorHandle> {
        self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Mean)
    }

    fn reduce_sum(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Sum)
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

    fn fused_elementwise(
        &self,
        shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
    ) -> Result<GpuTensorHandle> {
        self.fused_elementwise_exec(shader, inputs, output_shape, len)
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
        self.fused_reduction_exec(
            shader,
            inputs,
            output_shape,
            reduce_len,
            num_slices,
            workgroup_size,
        )
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
        let ms = start.elapsed().as_millis() as u64;
        self.metrics.set_last_warmup_millis(ms);
    }

    fn fused_cache_counters(&self) -> (u64, u64) {
        self.metrics.counters()
    }

    fn last_warmup_millis(&self) -> Option<u64> {
        Some(self.metrics.last_warmup_millis())
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
        runmat_accelerate_api::clear_handle_logical(h);
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
