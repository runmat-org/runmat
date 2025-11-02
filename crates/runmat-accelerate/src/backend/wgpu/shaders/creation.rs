pub const EYE_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct EyeParams {
    rows: u32,
    cols: u32,
    diag_len: u32,
    slices: u32,
    stride_slice: u32,
    diag_total: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: EyeParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if params.diag_total == 0u {
        return;
    }
    let idx = gid.x;
    if idx >= params.diag_total {
        return;
    }
    let diag = idx % params.diag_len;
    let slice = idx / params.diag_len;
    let base = slice * params.stride_slice + diag + diag * params.rows;
    Out.data[base] = 1.0;
}
"#;

pub const EYE_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct EyeParams {
    rows: u32,
    cols: u32,
    diag_len: u32,
    slices: u32,
    stride_slice: u32,
    diag_total: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: EyeParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if params.diag_total == 0u {
        return;
    }
    let idx = gid.x;
    if idx >= params.diag_total {
        return;
    }
    let diag = idx % params.diag_len;
    let slice = idx / params.diag_len;
    let base = slice * params.stride_slice + diag + diag * params.rows;
    Out.data[base] = 1.0;
}
"#;

pub const FILL_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct FillParams {
    value: f64,
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: FillParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    Out.data[idx] = params.value;
}
"#;

pub const FILL_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct FillParams {
    value: f32,
    len: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: FillParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    Out.data[idx] = params.value;
}
"#;

pub const LINSPACE_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct LinspaceParams {
    start: f64,
    step: f64,
    stop: f64,
    total: u32,
    chunk: u32,
    offset: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: LinspaceParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if params.chunk == 0u || params.total == 0u {
        return;
    }
    let local = gid.x;
    if local >= params.chunk {
        return;
    }
    let idx = params.offset + local;
    if idx >= params.total {
        return;
    }
    var value = params.start + f64(idx) * params.step;
    if params.total == 1u || idx == params.total - 1u {
        value = params.stop;
    }
    Out.data[idx] = value;
}
"#;

pub const LINSPACE_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct LinspaceParams {
    start: f32,
    step: f32,
    stop: f32,
    _pad0: f32,
    total: u32,
    chunk: u32,
    offset: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: LinspaceParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if params.chunk == 0u || params.total == 0u {
        return;
    }
    let local = gid.x;
    if local >= params.chunk {
        return;
    }
    let idx = params.offset + local;
    if idx >= params.total {
        return;
    }
    var value = params.start + f32(idx) * params.step;
    if params.total == 1u || idx == params.total - 1u {
        value = params.stop;
    }
    Out.data[idx] = value;
}
"#;

pub const FSPECIAL_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct FsParams {
    rows: u32,
    cols: u32,
    kind: u32,
    len: u32,
    sigma: f64,
    alpha: f64,
    norm: f64,
    center_x: f64,
    center_y: f64,
    extra0: f64,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: FsParams;

fn laplacian_base(row: u32, col: u32, a: f64, b: f64) -> f64 {
    if col == 0u {
        if row == 0u { return a; }
        if row == 1u { return b; }
        return a;
    }
    if col == 1u {
        if row == 0u { return b; }
        if row == 1u { return -1.0; }
        return b;
    }
    if row == 0u { return a; }
    if row == 1u { return b; }
    return a;
}

fn sobel_base(row: u32, col: u32) -> f64 {
    if col == 0u {
        if row == 0u { return 1.0; }
        if row == 1u { return 2.0; }
        return 1.0;
    }
    if col == 1u {
        return 0.0;
    }
    if row == 0u { return -1.0; }
    if row == 1u { return -2.0; }
    return -1.0;
}

fn unsharp_base(row: u32, col: u32, alpha: f64) -> f64 {
    if col == 0u {
        if row == 1u { return alpha - 1.0; }
        return -alpha;
    }
    if col == 1u {
        if row == 0u || row == 2u { return alpha - 1.0; }
        return alpha + 5.0;
    }
    if row == 1u { return alpha - 1.0; }
    return -alpha;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    let rows = params.rows;
    let row = idx % rows;
    let col = idx / rows;
    var value: f64 = 0.0;
    switch params.kind {
        case 0u: { // average
            value = params.norm;
        }
        case 1u: { // gaussian
            if params.sigma > 0.0 {
                let dx = f64(col) - params.center_x;
                let dy = f64(row) - params.center_y;
                let denom = 2.0 * params.sigma * params.sigma;
                value = params.norm * exp(-((dx * dx + dy * dy) / denom));
            } else {
                value = 0.0;
            }
        }
        case 2u: { // laplacian
            let a = params.alpha / 4.0;
            let b = (1.0 - params.alpha) / 4.0;
            value = laplacian_base(row, col, a, b) * params.norm;
        }
        case 3u: { // prewitt
            if col == 0u {
                value = 1.0;
            } else if col == 1u {
                value = 0.0;
            } else {
                value = -1.0;
            }
        }
        case 4u: { // sobel
            value = sobel_base(row, col);
        }
        case 5u: { // unsharp
            value = unsharp_base(row, col, params.alpha) * params.norm;
        }
        default: {
            value = 0.0;
        }
    }
    Out.data[idx] = value;
}
"#;

pub const FSPECIAL_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct FsParams {
    rows: u32,
    cols: u32,
    kind: u32,
    len: u32,
    sigma: f32,
    alpha: f32,
    norm: f32,
    _pad0: f32,
    center_x: f32,
    center_y: f32,
    _pad1: vec2<f32>,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: FsParams;

fn laplacian_base(row: u32, col: u32, a: f32, b: f32) -> f32 {
    if col == 0u {
        if row == 0u { return a; }
        if row == 1u { return b; }
        return a;
    }
    if col == 1u {
        if row == 0u { return b; }
        if row == 1u { return -1.0; }
        return b;
    }
    if row == 0u { return a; }
    if row == 1u { return b; }
    return a;
}

fn sobel_base(row: u32, col: u32) -> f32 {
    if col == 0u {
        if row == 0u { return 1.0; }
        if row == 1u { return 2.0; }
        return 1.0;
    }
    if col == 1u {
        return 0.0;
    }
    if row == 0u { return -1.0; }
    if row == 1u { return -2.0; }
    return -1.0;
}

fn unsharp_base(row: u32, col: u32, alpha: f32) -> f32 {
    if col == 0u {
        if row == 1u { return alpha - 1.0; }
        return -alpha;
    }
    if col == 1u {
        if row == 0u || row == 2u { return alpha - 1.0; }
        return alpha + 5.0;
    }
    if row == 1u { return alpha - 1.0; }
    return -alpha;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    let rows = params.rows;
    let row = idx % rows;
    let col = idx / rows;
    var value: f32 = 0.0;
    switch params.kind {
        case 0u: {
            value = params.norm;
        }
        case 1u: {
            if params.sigma > 0.0 {
                let dx = f32(col) - params.center_x;
                let dy = f32(row) - params.center_y;
                let denom = 2.0 * params.sigma * params.sigma;
                value = params.norm * exp(-((dx * dx + dy * dy) / denom));
            } else {
                value = 0.0;
            }
        }
        case 2u: {
            let a = params.alpha / 4.0;
            let b = (1.0 - params.alpha) / 4.0;
            value = laplacian_base(row, col, a, b) * params.norm;
        }
        case 3u: {
            if col == 0u {
                value = 1.0;
            } else if col == 1u {
                value = 0.0;
            } else {
                value = -1.0;
            }
        }
        case 4u: {
            value = sobel_base(row, col);
        }
        case 5u: {
            value = unsharp_base(row, col, params.alpha) * params.norm;
        }
        default: {
            value = 0.0;
        }
    }
    Out.data[idx] = value;
}
"#;

pub const RANDOM_INT_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct RandomIntParams {
    lower: f64,
    upper: f64,
    span: f64,
    span_minus_one: f64,
    offset: u32,
    chunk: u32,
    seed: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: RandomIntParams;

const LCG_MULT: u32 = 1664525u;
const LCG_INC: u32 = 1013904223u;
const INV_POW53: f64 = 1.0 / 9007199254740992.0;

fn lcg_next(state: ptr<function, u32>) -> u32 {
    let x = (*state) * LCG_MULT + LCG_INC;
    *state = x;
    return x;
}

fn uniform_f64(state: ptr<function, u32>) -> f64 {
    let hi = lcg_next(state) >> 5u;
   let lo = lcg_next(state) >> 6u;
   let combined = f64(hi) * 67108864.0 + f64(lo);
    return combined * INV_POW53;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.chunk {
        return;
    }
    let global_idx = params.offset + idx;
    var state = (params.seed ^ global_idx) | 1u;
    var u = uniform_f64(&state);
    if u >= 1.0 {
        u = 0.9999999999999999;
    }
    var offset = floor(u * params.span);
    if offset > params.span_minus_one {
        offset = params.span_minus_one;
    }
    var value = params.lower + offset;
    if value > params.upper {
        value = params.upper;
    }
    Out.data[global_idx] = value;
}
"#;

pub const RANDOM_INT_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct RandomIntParams {
    lower: f32,
    upper: f32,
    span: f32,
    span_minus_one: f32,
    offset: u32,
    chunk: u32,
    seed: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: RandomIntParams;

const LCG_MULT: u32 = 1664525u;
const LCG_INC: u32 = 1013904223u;
const INV_POW23: f32 = 1.0 / 8388608.0;

fn lcg_next(state: ptr<function, u32>) -> u32 {
    let x = (*state) * LCG_MULT + LCG_INC;
    *state = x;
    return x;
}

fn uniform_f32(state: ptr<function, u32>) -> f32 {
    let bits = lcg_next(state) >> 9u;
    return f32(bits) * INV_POW23;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.chunk {
        return;
    }
    let global_idx = params.offset + idx;
    var state = (params.seed ^ global_idx) | 1u;
    var u = uniform_f32(&state);
    if u >= 1.0 {
        u = 0.99999994;
    }
    var offset = floor(u * params.span);
    if offset > params.span_minus_one {
        offset = params.span_minus_one;
    }
    var value = params.lower + offset;
    if value > params.upper {
        value = params.upper;
    }
    Out.data[global_idx] = value;
}
"#;

pub const RANDOM_UNIFORM_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct RandomScalarParams {
    offset: u32,
    chunk: u32,
    seed: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: RandomScalarParams;

const LCG_MULT: u32 = 1664525u;
const LCG_INC: u32 = 1013904223u;
const INV_POW53: f64 = 1.0 / 9007199254740992.0;

fn lcg_next(state: ptr<function, u32>) -> u32 {
    let x = (*state) * LCG_MULT + LCG_INC;
    *state = x;
    return x;
}

fn uniform_f64(state: ptr<function, u32>) -> f64 {
    let hi = lcg_next(state) >> 5u;
    let lo = lcg_next(state) >> 6u;
    let combined = f64(hi) * 67108864.0 + f64(lo);
    return combined * INV_POW53;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.chunk {
        return;
    }
    let global_idx = params.offset + idx;
    var state = (params.seed ^ global_idx) | 1u;
    var u = uniform_f64(&state);
    if u >= 1.0 {
        u = 0.9999999999999999;
    }
    Out.data[global_idx] = u;
}
"#;

pub const RANDOM_UNIFORM_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct RandomScalarParams {
    offset: u32,
    chunk: u32,
    seed: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: RandomScalarParams;

const LCG_MULT: u32 = 1664525u;
const LCG_INC: u32 = 1013904223u;
const INV_U32: f32 = 1.0 / 4294967296.0;
const ALMOST_ONE: f32 = 0.99999994;

fn lcg_next(state: ptr<function, u32>) -> u32 {
    let x = (*state) * LCG_MULT + LCG_INC;
    *state = x;
    return x;
}

fn uniform_f32(state: ptr<function, u32>) -> f32 {
    let bits = lcg_next(state);
    let sample = (f32(bits) + 0.5) * INV_U32;
    return min(sample, ALMOST_ONE);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.chunk {
        return;
    }
    let global_idx = params.offset + idx;
    var state = (params.seed ^ global_idx) | 1u;
    let u = uniform_f32(&state);
    Out.data[global_idx] = u;
}
"#;

pub const RANDOM_NORMAL_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct RandomScalarParams {
    offset: u32,
    chunk: u32,
    seed: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: RandomScalarParams;

const LCG_MULT: u32 = 1664525u;
const LCG_INC: u32 = 1013904223u;
const INV_POW53: f64 = 1.0 / 9007199254740992.0;
const TWO_PI: f64 = 6.283185307179586;
const MIN_UNIFORM: f64 = 1.0e-16;

fn lcg_next(state: ptr<function, u32>) -> u32 {
    let x = (*state) * LCG_MULT + LCG_INC;
    *state = x;
    return x;
}

fn uniform_f64(state: ptr<function, u32>) -> f64 {
    let hi = lcg_next(state) >> 5u;
    let lo = lcg_next(state) >> 6u;
    let combined = f64(hi) * 67108864.0 + f64(lo);
    return combined * INV_POW53;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.chunk {
        return;
    }
    let global_idx = params.offset + idx;
    var state = (params.seed ^ global_idx) | 1u;
    var u1 = uniform_f64(&state);
    if u1 <= 0.0 {
        u1 = MIN_UNIFORM;
    }
    let u2 = uniform_f64(&state);
    let radius = sqrt(-2.0 * log(u1));
    let angle = TWO_PI * u2;
    let sample = radius * cos(angle);
    Out.data[global_idx] = sample;
}
"#;

pub const RANDOM_NORMAL_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct RandomScalarParams {
    offset: u32,
    chunk: u32,
    seed: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: RandomScalarParams;

const LCG_MULT: u32 = 1664525u;
const LCG_INC: u32 = 1013904223u;
const INV_U32: f32 = 1.0 / 4294967296.0;
const TWO_PI: f32 = 6.2831855;
const MIN_UNIFORM: f32 = 1.0e-8;
const ALMOST_ONE: f32 = 0.99999994;

fn lcg_next(state: ptr<function, u32>) -> u32 {
    let x = (*state) * LCG_MULT + LCG_INC;
    *state = x;
    return x;
}

fn uniform_f32(state: ptr<function, u32>) -> f32 {
    let bits = lcg_next(state);
    let sample = (f32(bits) + 0.5) * INV_U32;
    return clamp(sample, MIN_UNIFORM, ALMOST_ONE);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.chunk {
        return;
    }
    let global_idx = params.offset + idx;
    var state = (params.seed ^ global_idx) | 1u;
    let u1 = uniform_f32(&state);
    let u2 = uniform_f32(&state);
    let radius = sqrt(-2.0 * log(u1));
    let angle = TWO_PI * u2;
    let sample = radius * cos(angle);
    Out.data[global_idx] = sample;
}
"#;

pub const RANDPERM_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct RandPermParams {
    n: u32,
    k: u32,
    seed: u32,
    _pad: u32,
};

const LCG_MULT: u32 = 1664525u;
const LCG_INC: u32 = 1013904223u;
const INV_POW53: f64 = 1.0 / 9007199254740992.0;
const ALMOST_ONE: f64 = 0.9999999999999999;

fn lcg_next(state: ptr<function, u32>) -> u32 {
    let next = (*state) * LCG_MULT + LCG_INC;
    *state = next;
    return next;
}

fn uniform_f64(state: ptr<function, u32>) -> f64 {
    let hi = lcg_next(state) >> 5u;
    let lo = lcg_next(state) >> 6u;
    let combined = f64(hi) * 67108864.0 + f64(lo);
    return combined * INV_POW53;
}

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: RandPermParams;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x > 0u {
        return;
    }
    let n = params.n;
    var k = params.k;
    if n == 0u || k == 0u {
        return;
    }
    if k > n {
        k = n;
    }

    var state = params.seed | 1u;

    var i: u32 = 0u;
    loop {
        if i >= k {
            break;
        }
        Out.data[i] = f64(i + 1u);
        i = i + 1u;
    }

    var t: u32 = k;
    loop {
        if t >= n {
            break;
        }
        let total = t + 1u;
        var u = uniform_f64(&state);
        if u >= 1.0 {
            u = ALMOST_ONE;
        }
        let span = total;
        var raw = u * f64(span);
        var offset = u32(floor(raw));
        if offset >= span {
            offset = span - 1u;
        }
        if offset < k {
            Out.data[offset] = f64(total);
        }
        t = total;
    }

    var idx: u32 = 0u;
    loop {
        if idx >= k {
            break;
        }
        let span = k - idx;
        var u = uniform_f64(&state);
        if u >= 1.0 {
            u = ALMOST_ONE;
        }
        var raw = u * f64(span);
        var offset = u32(floor(raw));
        if offset >= span {
            offset = span - 1u;
        }
        let swap_idx = idx + offset;
        let tmp = Out.data[idx];
        Out.data[idx] = Out.data[swap_idx];
        Out.data[swap_idx] = tmp;
        idx = idx + 1u;
    }
}
"#;

pub const RANDPERM_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct RandPermParams {
    n: u32,
    k: u32,
    seed: u32,
    _pad: u32,
};

const ALMOST_ONE: f32 = 0.99999994;
const INV_U32: f32 = 1.0 / 4294967296.0;

fn lcg_next(state: ptr<function, u32>) -> u32 {
    let next = (*state) * 1664525u + 1013904223u;
    *state = next;
    return next;
}

fn uniform_f32(state: ptr<function, u32>) -> f32 {
    let bits = lcg_next(state);
    let sample = (f32(bits) + 0.5) * INV_U32;
    if sample >= 1.0 {
        return ALMOST_ONE;
    }
    return sample;
}

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: RandPermParams;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x > 0u {
        return;
    }
    let n = params.n;
    var k = params.k;
    if n == 0u || k == 0u {
        return;
    }
    if k > n {
        k = n;
    }

    var state = params.seed | 1u;

    var i: u32 = 0u;
    loop {
        if i >= k {
            break;
        }
        Out.data[i] = f32(i + 1u);
        i = i + 1u;
    }

    var t: u32 = k;
    loop {
        if t >= n {
            break;
        }
        let total = t + 1u;
        var u = uniform_f32(&state);
        if u >= 1.0 {
            u = ALMOST_ONE;
        }
        let span = total;
        let span_f = f32(span);
        var raw = u * span_f;
        var offset = u32(floor(raw));
        if offset >= span {
            offset = span - 1u;
        }
        if offset < k {
            Out.data[offset] = f32(total);
        }
        t = total;
    }

    var idx: u32 = 0u;
    loop {
        if idx >= k {
            break;
        }
        let span = k - idx;
        var u = uniform_f32(&state);
        if u >= 1.0 {
            u = ALMOST_ONE;
        }
        let span_f = f32(span);
        var raw = u * span_f;
        var offset = u32(floor(raw));
        if offset >= span {
            offset = span - 1u;
        }
        let swap_idx = idx + offset;
        let tmp = Out.data[idx];
        Out.data[idx] = Out.data[swap_idx];
        Out.data[swap_idx] = tmp;
        idx = idx + 1u;
    }
}
"#;
