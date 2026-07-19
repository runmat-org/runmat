pub const CUMSUM_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    segment_len: u32,
    segments: u32,
    stride_before: u32,
    block: u32,
    flags: u32,
    total_len: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn is_nan_f64(value: f64) -> bool {
    return value != value;
}

fn nan_f64() -> f64 {
    var bits: u64 = 0x7ff8000000000000u;
    return bitcast<f64>(bits);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let segment = gid.x;
    if segment >= params.segments {
        return;
    }
    let stride_before = params.stride_before;
    if stride_before == 0u {
        return;
    }
    let before = segment % stride_before;
    let after = segment / stride_before;
    let base = after * params.block;
    let is_reverse = (params.flags & 1u) != 0u;
    let omit_nan = (params.flags & 2u) != 0u;

    if is_reverse {
        var sum: f64 = 0.0;
        var has_nan = false;
        var offset = params.segment_len;
        loop {
            if offset == 0u {
                break;
            }
            offset = offset - 1u;
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                continue;
            }
            let value = Input.data[index];
            if omit_nan {
                if !is_nan_f64(value) {
                    sum = sum + value;
                }
                Output.data[index] = sum;
            } else {
                if has_nan {
                    Output.data[index] = nan_f64();
                } else if is_nan_f64(value) {
                    has_nan = true;
                    Output.data[index] = nan_f64();
                } else {
                    sum = sum + value;
                    Output.data[index] = sum;
                }
            }
        }
    } else {
        var sum: f64 = 0.0;
        var has_nan = false;
        var offset: u32 = 0u;
        loop {
            if offset >= params.segment_len {
                break;
            }
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                offset = offset + 1u;
                continue;
            }
            let value = Input.data[index];
            if omit_nan {
                if !is_nan_f64(value) {
                    sum = sum + value;
                }
                Output.data[index] = sum;
            } else {
                if has_nan {
                    Output.data[index] = nan_f64();
                } else if is_nan_f64(value) {
                    has_nan = true;
                    Output.data[index] = nan_f64();
                } else {
                    sum = sum + value;
                    Output.data[index] = sum;
                }
            }
            offset = offset + 1u;
        }
    }
}
"#;

pub const CUMSUM_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    segment_len: u32,
    segments: u32,
    stride_before: u32,
    block: u32,
    flags: u32,
    total_len: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn is_nan_f32(value: f32) -> bool {
    return value != value;
}

fn nan_f32() -> f32 {
    var bits: u32 = 0x7fc00000u;
    return bitcast<f32>(bits);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let segment = gid.x;
    if segment >= params.segments {
        return;
    }
    let stride_before = params.stride_before;
    if stride_before == 0u {
        return;
    }
    let before = segment % stride_before;
    let after = segment / stride_before;
    let base = after * params.block;
    let is_reverse = (params.flags & 1u) != 0u;
    let omit_nan = (params.flags & 2u) != 0u;

    if is_reverse {
        var sum: f32 = 0.0;
        var has_nan = false;
        var offset = params.segment_len;
        loop {
            if offset == 0u {
                break;
            }
            offset = offset - 1u;
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                continue;
            }
            let value = Input.data[index];
            if omit_nan {
                if !is_nan_f32(value) {
                    sum = sum + value;
                }
                Output.data[index] = sum;
            } else {
                if has_nan {
                    Output.data[index] = nan_f32();
                } else if is_nan_f32(value) {
                    has_nan = true;
                    Output.data[index] = nan_f32();
                } else {
                    sum = sum + value;
                    Output.data[index] = sum;
                }
            }
        }
    } else {
        var sum: f32 = 0.0;
        var has_nan = false;
        var offset: u32 = 0u;
        loop {
            if offset >= params.segment_len {
                break;
            }
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                offset = offset + 1u;
                continue;
            }
            let value = Input.data[index];
            if omit_nan {
                if !is_nan_f32(value) {
                    sum = sum + value;
                }
                Output.data[index] = sum;
            } else {
                if has_nan {
                    Output.data[index] = nan_f32();
                } else if is_nan_f32(value) {
                    has_nan = true;
                    Output.data[index] = nan_f32();
                } else {
                    sum = sum + value;
                    Output.data[index] = sum;
                }
            }
            offset = offset + 1u;
        }
    }
}
"#;

pub const TRAPEZOID_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    segment_len: u32,
    segments: u32,
    stride_before: u32,
    block: u32,
    total_len: u32,
    output_len: u32,
    spacing_kind: u32,
    mode: u32,
    spacing_scalar: f64,
    _pad0: f64,
    _pad1: f64,
    _pad2: f64,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> Spacing: Tensor;

fn interval_width(idx0: u32, idx1: u32, k: u32) -> f64 {
    if params.spacing_kind == 0u {
        return 1.0;
    }
    if params.spacing_kind == 1u {
        return params.spacing_scalar;
    }
    if params.spacing_kind == 2u {
        return Spacing.data[0u];
    }
    if params.spacing_kind == 3u {
        return Spacing.data[k + 1u] - Spacing.data[k];
    }
    return Spacing.data[idx1] - Spacing.data[idx0];
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let segment = gid.x;
    if segment >= params.segments || params.stride_before == 0u {
        return;
    }

    let before = segment % params.stride_before;
    let after = segment / params.stride_before;
    let base = after * params.block;
    let out_base = after * params.stride_before + before;

    if params.mode == 1u {
        let first_idx = base + before;
        if first_idx < params.total_len {
            Output.data[first_idx] = 0.0;
        }
    }

    var acc: f64 = 0.0;
    var k: u32 = 0u;
    loop {
        if k + 1u >= params.segment_len {
            break;
        }
        let idx0 = base + before + k * params.stride_before;
        let idx1 = idx0 + params.stride_before;
        if idx1 < params.total_len {
            let width = interval_width(idx0, idx1, k);
            acc = acc + 0.5 * width * (Input.data[idx0] + Input.data[idx1]);
            if params.mode == 1u {
                Output.data[idx1] = acc;
            }
        }
        k = k + 1u;
    }

    if params.mode == 0u && out_base < params.output_len {
        Output.data[out_base] = acc;
    }
}
"#;

pub const TRAPEZOID_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    meta0: vec4<u32>,
    meta1: vec4<u32>,
    scalar: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> Spacing: Tensor;

fn segment_len() -> u32 { return params.meta0.x; }
fn segments() -> u32 { return params.meta0.y; }
fn stride_before() -> u32 { return params.meta0.z; }
fn block() -> u32 { return params.meta0.w; }
fn total_len() -> u32 { return params.meta1.x; }
fn output_len() -> u32 { return params.meta1.y; }
fn spacing_kind() -> u32 { return params.meta1.z; }
fn mode() -> u32 { return params.meta1.w; }

fn interval_width(idx0: u32, idx1: u32, k: u32) -> f32 {
    if spacing_kind() == 0u {
        return 1.0;
    }
    if spacing_kind() == 1u {
        return params.scalar.x;
    }
    if spacing_kind() == 2u {
        return Spacing.data[0u];
    }
    if spacing_kind() == 3u {
        return Spacing.data[k + 1u] - Spacing.data[k];
    }
    return Spacing.data[idx1] - Spacing.data[idx0];
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let segment = gid.x;
    if segment >= segments() || stride_before() == 0u {
        return;
    }

    let before = segment % stride_before();
    let after = segment / stride_before();
    let base = after * block();
    let out_base = after * stride_before() + before;

    if mode() == 1u {
        let first_idx = base + before;
        if first_idx < total_len() {
            Output.data[first_idx] = 0.0;
        }
    }

    var acc: f32 = 0.0;
    var k: u32 = 0u;
    loop {
        if k + 1u >= segment_len() {
            break;
        }
        let idx0 = base + before + k * stride_before();
        let idx1 = idx0 + stride_before();
        if idx1 < total_len() {
            let width = interval_width(idx0, idx1, k);
            acc = acc + 0.5 * width * (Input.data[idx0] + Input.data[idx1]);
            if mode() == 1u {
                Output.data[idx1] = acc;
            }
        }
        k = k + 1u;
    }

    if mode() == 0u && out_base < output_len() {
        Output.data[out_base] = acc;
    }
}
"#;

pub const CUMPROD_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    segment_len: u32,
    segments: u32,
    stride_before: u32,
    block: u32,
    flags: u32,
    total_len: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn is_nan_f64(value: f64) -> bool {
    return value != value;
}

fn nan_f64() -> f64 {
    var bits: u64 = 0x7ff8000000000000u;
    return bitcast<f64>(bits);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let segment = gid.x;
    if segment >= params.segments {
        return;
    }
    let stride_before = params.stride_before;
    if stride_before == 0u {
        return;
    }
    let before = segment % stride_before;
    let after = segment / stride_before;
    let base = after * params.block;
    let is_reverse = (params.flags & 1u) != 0u;
    let omit_nan = (params.flags & 2u) != 0u;

    if is_reverse {
        var prod: f64 = 1.0;
        var has_nan = false;
        var offset = params.segment_len;
        loop {
            if offset == 0u {
                break;
            }
            offset = offset - 1u;
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                continue;
            }
            let value = Input.data[index];
            if omit_nan {
                if !is_nan_f64(value) {
                    prod = prod * value;
                }
                Output.data[index] = prod;
            } else {
                if has_nan {
                    Output.data[index] = nan_f64();
                } else if is_nan_f64(value) {
                    has_nan = true;
                    Output.data[index] = nan_f64();
                } else {
                    prod = prod * value;
                    Output.data[index] = prod;
                }
            }
        }
    } else {
        var prod: f64 = 1.0;
        var has_nan = false;
        var offset: u32 = 0u;
        loop {
            if offset >= params.segment_len {
                break;
            }
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                offset = offset + 1u;
                continue;
            }
            let value = Input.data[index];
            if omit_nan {
                if !is_nan_f64(value) {
                    prod = prod * value;
                }
                Output.data[index] = prod;
            } else {
                if has_nan {
                    Output.data[index] = nan_f64();
                } else if is_nan_f64(value) {
                    has_nan = true;
                    Output.data[index] = nan_f64();
                } else {
                    prod = prod * value;
                    Output.data[index] = prod;
                }
            }
            offset = offset + 1u;
        }
    }
}
"#;

pub const CUMPROD_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    segment_len: u32,
    segments: u32,
    stride_before: u32,
    block: u32,
    flags: u32,
    total_len: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn is_nan_f32(value: f32) -> bool {
    return value != value;
}

fn nan_f32() -> f32 {
    var bits: u32 = 0x7fc00000u;
    return bitcast<f32>(bits);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let segment = gid.x;
    if segment >= params.segments {
        return;
    }
    let stride_before = params.stride_before;
    if stride_before == 0u {
        return;
    }
    let before = segment % stride_before;
    let after = segment / stride_before;
    let base = after * params.block;
    let is_reverse = (params.flags & 1u) != 0u;
    let omit_nan = (params.flags & 2u) != 0u;

    if is_reverse {
        var prod: f32 = 1.0;
        var has_nan = false;
        var offset = params.segment_len;
        loop {
            if offset == 0u {
                break;
            }
            offset = offset - 1u;
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                continue;
            }
            let value = Input.data[index];
            if omit_nan {
                if !is_nan_f32(value) {
                    prod = prod * value;
                }
                Output.data[index] = prod;
            } else {
                if has_nan {
                    Output.data[index] = nan_f32();
                } else if is_nan_f32(value) {
                    has_nan = true;
                    Output.data[index] = nan_f32();
                } else {
                    prod = prod * value;
                    Output.data[index] = prod;
                }
            }
        }
    } else {
        var prod: f32 = 1.0;
        var has_nan = false;
        var offset: u32 = 0u;
        loop {
            if offset >= params.segment_len {
                break;
            }
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                offset = offset + 1u;
                continue;
            }
            let value = Input.data[index];
            if omit_nan {
                if !is_nan_f32(value) {
                    prod = prod * value;
                }
                Output.data[index] = prod;
            } else {
                if has_nan {
                    Output.data[index] = nan_f32();
                } else if is_nan_f32(value) {
                    has_nan = true;
                    Output.data[index] = nan_f32();
                } else {
                    prod = prod * value;
                    Output.data[index] = prod;
                }
            }
            offset = offset + 1u;
        }
    }
}
"#;

pub const CUMMIN_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Indices {
    data: array<f64>,
};

struct Params {
    segment_len: u32,
    segments: u32,
    stride_before: u32,
    block: u32,
    flags: u32,
    total_len: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> OutputVals: Tensor;
@group(0) @binding(2) var<storage, read_write> OutputIdx: Indices;
@group(0) @binding(3) var<uniform> params: Params;

fn is_nan_f64(value: f64) -> bool {
    return value != value;
}

fn nan_f64() -> f64 {
    var bits: u64 = 0x7ff8000000000000u;
    return bitcast<f64>(bits);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let segment = gid.x;
    if segment >= params.segments {
        return;
    }
    let stride_before = params.stride_before;
    if stride_before == 0u {
        return;
    }
    let before = segment % stride_before;
    let after = segment / stride_before;
    let base = after * params.block;
    let is_reverse = (params.flags & 1u) != 0u;
    let omit_nan = (params.flags & 2u) != 0u;

    if is_reverse {
        var current: f64 = 0.0;
        var current_idx: u32 = 1u;
        var has_value: bool = false;
        var nan_fixed: bool = false;
        var nan_idx: u32 = 0u;
        var offset = params.segment_len;
        loop {
            if offset == 0u {
                break;
            }
            offset = offset - 1u;
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                continue;
            }
            let value = Input.data[index];
            let pos = offset + 1u;
            if omit_nan {
                if is_nan_f64(value) {
                    if has_value {
                        OutputVals.data[index] = current;
                        OutputIdx.data[index] = f64(current_idx);
                    } else {
                        OutputVals.data[index] = nan_f64();
                        OutputIdx.data[index] = nan_f64();
                    }
                } else {
                    if !has_value || value < current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f64(current_idx);
                }
            } else {
                if nan_fixed {
                    OutputVals.data[index] = nan_f64();
                    OutputIdx.data[index] = f64(nan_idx);
                } else if is_nan_f64(value) {
                    nan_fixed = true;
                    nan_idx = pos;
                    OutputVals.data[index] = nan_f64();
                    OutputIdx.data[index] = f64(pos);
                } else {
                    if !has_value || value < current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f64(current_idx);
                }
            }
        }
    } else {
        var current: f64 = 0.0;
        var current_idx: u32 = 1u;
        var has_value: bool = false;
        var nan_fixed: bool = false;
        var nan_idx: u32 = 0u;
        var offset: u32 = 0u;
        loop {
            if offset >= params.segment_len {
                break;
            }
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                offset = offset + 1u;
                continue;
            }
            let value = Input.data[index];
            let pos = offset + 1u;
            if omit_nan {
                if is_nan_f64(value) {
                    if has_value {
                        OutputVals.data[index] = current;
                        OutputIdx.data[index] = f64(current_idx);
                    } else {
                        OutputVals.data[index] = nan_f64();
                        OutputIdx.data[index] = nan_f64();
                    }
                } else {
                    if !has_value || value < current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f64(current_idx);
                }
            } else {
                if nan_fixed {
                    OutputVals.data[index] = nan_f64();
                    OutputIdx.data[index] = f64(nan_idx);
                } else if is_nan_f64(value) {
                    nan_fixed = true;
                    nan_idx = pos;
                    OutputVals.data[index] = nan_f64();
                    OutputIdx.data[index] = f64(pos);
                } else {
                    if !has_value || value < current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f64(current_idx);
                }
            }
            offset = offset + 1u;
        }
}
}
"#;

pub const CUMMAX_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Indices {
    data: array<f64>,
};

struct Params {
    segment_len: u32,
    segments: u32,
    stride_before: u32,
    block: u32,
    flags: u32,
    total_len: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> OutputVals: Tensor;
@group(0) @binding(2) var<storage, read_write> OutputIdx: Indices;
@group(0) @binding(3) var<uniform> params: Params;

fn is_nan_f64(value: f64) -> bool {
    return value != value;
}

fn nan_f64() -> f64 {
    var bits: u64 = 0x7ff8000000000000u;
    return bitcast<f64>(bits);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let segment = gid.x;
    if segment >= params.segments {
        return;
    }
    let stride_before = params.stride_before;
    if stride_before == 0u {
        return;
    }
    let before = segment % stride_before;
    let after = segment / stride_before;
    let base = after * params.block;
    let is_reverse = (params.flags & 1u) != 0u;
    let omit_nan = (params.flags & 2u) != 0u;

    if is_reverse {
        var current: f64 = 0.0;
        var current_idx: u32 = 1u;
        var has_value: bool = false;
        var nan_fixed: bool = false;
        var nan_idx: u32 = 0u;
        var offset = params.segment_len;
        loop {
            if offset == 0u {
                break;
            }
            offset = offset - 1u;
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                continue;
            }
            let value = Input.data[index];
            let pos = offset + 1u;
            if omit_nan {
                if is_nan_f64(value) {
                    if has_value {
                        OutputVals.data[index] = current;
                        OutputIdx.data[index] = f64(current_idx);
                    } else {
                        OutputVals.data[index] = nan_f64();
                        OutputIdx.data[index] = nan_f64();
                    }
                } else {
                    if !has_value || value > current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f64(current_idx);
                }
            } else {
                if nan_fixed {
                    OutputVals.data[index] = nan_f64();
                    OutputIdx.data[index] = f64(nan_idx);
                } else if is_nan_f64(value) {
                    nan_fixed = true;
                    nan_idx = pos;
                    OutputVals.data[index] = nan_f64();
                    OutputIdx.data[index] = f64(pos);
                } else {
                    if !has_value || value > current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f64(current_idx);
                }
            }
        }
    } else {
        var current: f64 = 0.0;
        var current_idx: u32 = 1u;
        var has_value: bool = false;
        var nan_fixed: bool = false;
        var nan_idx: u32 = 0u;
        var offset: u32 = 0u;
        loop {
            if offset >= params.segment_len {
                break;
            }
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                offset = offset + 1u;
                continue;
            }
            let value = Input.data[index];
            let pos = offset + 1u;
            if omit_nan {
                if is_nan_f64(value) {
                    if has_value {
                        OutputVals.data[index] = current;
                        OutputIdx.data[index] = f64(current_idx);
                    } else {
                        OutputVals.data[index] = nan_f64();
                        OutputIdx.data[index] = nan_f64();
                    }
                } else {
                    if !has_value || value > current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f64(current_idx);
                }
            } else {
                if nan_fixed {
                    OutputVals.data[index] = nan_f64();
                    OutputIdx.data[index] = f64(nan_idx);
                } else if is_nan_f64(value) {
                    nan_fixed = true;
                    nan_idx = pos;
                    OutputVals.data[index] = nan_f64();
                    OutputIdx.data[index] = f64(pos);
                } else {
                    if !has_value || value > current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f64(current_idx);
                }
            }
            offset = offset + 1u;
        }
    }
}
"#;

pub const CUMMIN_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Indices {
    data: array<f32>,
};

struct Params {
    segment_len: u32,
    segments: u32,
    stride_before: u32,
    block: u32,
    flags: u32,
    total_len: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> OutputVals: Tensor;
@group(0) @binding(2) var<storage, read_write> OutputIdx: Indices;
@group(0) @binding(3) var<uniform> params: Params;

fn is_nan_f32(value: f32) -> bool {
    return value != value;
}

fn nan_f32() -> f32 {
    var bits: u32 = 0x7fc00000u;
    return bitcast<f32>(bits);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let segment = gid.x;
    if segment >= params.segments {
        return;
    }
    let stride_before = params.stride_before;
    if stride_before == 0u {
        return;
    }
    let before = segment % stride_before;
    let after = segment / stride_before;
    let base = after * params.block;
    let is_reverse = (params.flags & 1u) != 0u;
    let omit_nan = (params.flags & 2u) != 0u;

    if is_reverse {
        var current: f32 = 0.0;
        var current_idx: u32 = 1u;
        var has_value: bool = false;
        var nan_fixed: bool = false;
        var nan_idx: u32 = 0u;
        var offset = params.segment_len;
        loop {
            if offset == 0u {
                break;
            }
            offset = offset - 1u;
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                continue;
            }
            let value = Input.data[index];
            let pos = offset + 1u;
            if omit_nan {
                if is_nan_f32(value) {
                    if has_value {
                        OutputVals.data[index] = current;
                        OutputIdx.data[index] = f32(current_idx);
                    } else {
                        OutputVals.data[index] = nan_f32();
                        OutputIdx.data[index] = nan_f32();
                    }
                } else {
                    if !has_value || value < current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f32(current_idx);
                }
            } else {
                if nan_fixed {
                    OutputVals.data[index] = nan_f32();
                    OutputIdx.data[index] = f32(nan_idx);
                } else if is_nan_f32(value) {
                    nan_fixed = true;
                    nan_idx = pos;
                    OutputVals.data[index] = nan_f32();
                    OutputIdx.data[index] = f32(pos);
                } else {
                    if !has_value || value < current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f32(current_idx);
                }
            }
        }
    } else {
        var current: f32 = 0.0;
        var current_idx: u32 = 1u;
        var has_value: bool = false;
        var nan_fixed: bool = false;
        var nan_idx: u32 = 0u;
        var offset: u32 = 0u;
        loop {
            if offset >= params.segment_len {
                break;
            }
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                offset = offset + 1u;
                continue;
            }
            let value = Input.data[index];
            let pos = offset + 1u;
            if omit_nan {
                if is_nan_f32(value) {
                    if has_value {
                        OutputVals.data[index] = current;
                        OutputIdx.data[index] = f32(current_idx);
                    } else {
                        OutputVals.data[index] = nan_f32();
                        OutputIdx.data[index] = nan_f32();
                    }
                } else {
                    if !has_value || value < current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f32(current_idx);
                }
            } else {
                if nan_fixed {
                    OutputVals.data[index] = nan_f32();
                    OutputIdx.data[index] = f32(nan_idx);
                } else if is_nan_f32(value) {
                    nan_fixed = true;
                    nan_idx = pos;
                    OutputVals.data[index] = nan_f32();
                    OutputIdx.data[index] = f32(pos);
                } else {
                    if !has_value || value < current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f32(current_idx);
                }
            }
            offset = offset + 1u;
        }
}
}
"#;

pub const CUMMAX_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Indices {
    data: array<f32>,
};

struct Params {
    segment_len: u32,
    segments: u32,
    stride_before: u32,
    block: u32,
    flags: u32,
    total_len: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> OutputVals: Tensor;
@group(0) @binding(2) var<storage, read_write> OutputIdx: Indices;
@group(0) @binding(3) var<uniform> params: Params;

fn is_nan_f32(value: f32) -> bool {
    return value != value;
}

fn nan_f32() -> f32 {
    var bits: u32 = 0x7fc00000u;
    return bitcast<f32>(bits);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let segment = gid.x;
    if segment >= params.segments {
        return;
    }
    let stride_before = params.stride_before;
    if stride_before == 0u {
        return;
    }
    let before = segment % stride_before;
    let after = segment / stride_before;
    let base = after * params.block;
    let is_reverse = (params.flags & 1u) != 0u;
    let omit_nan = (params.flags & 2u) != 0u;

    if is_reverse {
        var current: f32 = 0.0;
        var current_idx: u32 = 1u;
        var has_value: bool = false;
        var nan_fixed: bool = false;
        var nan_idx: u32 = 0u;
        var offset = params.segment_len;
        loop {
            if offset == 0u {
                break;
            }
            offset = offset - 1u;
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                continue;
            }
            let value = Input.data[index];
            let pos = offset + 1u;
            if omit_nan {
                if is_nan_f32(value) {
                    if has_value {
                        OutputVals.data[index] = current;
                        OutputIdx.data[index] = f32(current_idx);
                    } else {
                        OutputVals.data[index] = nan_f32();
                        OutputIdx.data[index] = nan_f32();
                    }
                } else {
                    if !has_value || value > current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f32(current_idx);
                }
            } else {
                if nan_fixed {
                    OutputVals.data[index] = nan_f32();
                    OutputIdx.data[index] = f32(nan_idx);
                } else if is_nan_f32(value) {
                    nan_fixed = true;
                    nan_idx = pos;
                    OutputVals.data[index] = nan_f32();
                    OutputIdx.data[index] = f32(pos);
                } else {
                    if !has_value || value > current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f32(current_idx);
                }
            }
        }
    } else {
        var current: f32 = 0.0;
        var current_idx: u32 = 1u;
        var has_value: bool = false;
        var nan_fixed: bool = false;
        var nan_idx: u32 = 0u;
        var offset: u32 = 0u;
        loop {
            if offset >= params.segment_len {
                break;
            }
            let index = base + before + offset * stride_before;
            if index >= params.total_len {
                offset = offset + 1u;
                continue;
            }
            let value = Input.data[index];
            let pos = offset + 1u;
            if omit_nan {
                if is_nan_f32(value) {
                    if has_value {
                        OutputVals.data[index] = current;
                        OutputIdx.data[index] = f32(current_idx);
                    } else {
                        OutputVals.data[index] = nan_f32();
                        OutputIdx.data[index] = nan_f32();
                    }
                } else {
                    if !has_value || value > current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f32(current_idx);
                }
            } else {
                if nan_fixed {
                    OutputVals.data[index] = nan_f32();
                    OutputIdx.data[index] = f32(nan_idx);
                } else if is_nan_f32(value) {
                    nan_fixed = true;
                    nan_idx = pos;
                    OutputVals.data[index] = nan_f32();
                    OutputIdx.data[index] = f32(pos);
                } else {
                    if !has_value || value > current {
                        has_value = true;
                        current = value;
                        current_idx = pos;
                    }
                    OutputVals.data[index] = current;
                    OutputIdx.data[index] = f32(current_idx);
                }
            }
            offset = offset + 1u;
        }
    }
}
"#;
