pub const BINARY_SHADER_F64: &str = r#"
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

fn hypot(a: f64, b: f64) -> f64 {
    return sqrt((a * a) + (b * b));
}

fn apply(a: f64, b: f64) -> f64 {
    switch params.op {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        case 4u: { return hypot(a, b); }
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

pub const BINARY_SHADER_F32: &str = r#"
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

fn hypot(a: f32, b: f32) -> f32 {
    return sqrt((a * a) + (b * b));
}

fn apply(a: f32, b: f32) -> f32 {
    switch params.op {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        case 4u: { return hypot(a, b); }
        default: { return pow(a, b); }
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

pub const UNARY_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    op: u32,
    offset: u32,
    total: u32,
};

fn expm1_precise(a: f64) -> f64 {
    let abs_a = abs(a);
    if abs_a < 1.0e-6 {
        let a2 = a * a;
        let a3 = a2 * a;
        let a4 = a2 * a2;
        let a5 = a4 * a;
        return (((a5 * (1.0 / 120.0)) + (a4 * (1.0 / 24.0)) + (a3 * (1.0 / 6.0))) + (a2 * 0.5)) + a;
    }
    return exp(a) - 1.0;
}

fn log1p_precise(a: f64) -> f64 {
    let abs_a = abs(a);
    if abs_a < 1.0e-6 {
        let a2 = a * a;
        let a3 = a2 * a;
        let a4 = a2 * a2;
        let a5 = a4 * a;
        return ((((a5 * (1.0 / 5.0)) - (a4 * 0.25)) + (a3 * (1.0 / 3.0))) - (a2 * 0.5)) + a;
    }
    return log(1.0 + a);
}

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
        case 6u: {
            if (a > 0.0) {
                return 1.0;
            }
            if (a < 0.0) {
                return -1.0;
            }
            if (a == 0.0) {
                return 0.0;
            }
            return a;
        }
        case 7u: { return a; }
        case 8u: { return 0.0; }
        case 9u: { return a; }
        case 10u: { return atan2(f64(0.0), a); }
        case 11u: { return expm1_precise(a); }
        case 12u: { return log1p_precise(a); }
        case 13u: { return log(a) * 0.4342944819032518; }
        case 14u: { return log(a) * 1.4426950408889634; }
        case 15u: { return exp(a * 0.6931471805599453); }
        case 16u: { return floor(a); }
        case 17u: { return ceil(a); }
        case 18u: {
            let t = trunc(a);
            if (t == 0.0) {
                return 0.0;
            }
            return t;
        }
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

pub const UNARY_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    op: u32,
    offset: u32,
    total: u32,
};

fn expm1_precise(a: f32) -> f32 {
    let abs_a = abs(a);
    if abs_a < 1.0e-4 {
        let a2 = a * a;
        let a3 = a2 * a;
        let a4 = a2 * a2;
        let a5 = a4 * a;
        return (((a5 * (1.0 / 120.0)) + (a4 * (1.0 / 24.0)) + (a3 * (1.0 / 6.0))) + (a2 * 0.5)) + a;
    }
    return exp(a) - 1.0;
}

fn log1p_precise(a: f32) -> f32 {
    let abs_a = abs(a);
    if abs_a < 1.0e-4 {
        let a2 = a * a;
        let a3 = a2 * a;
        let a4 = a2 * a2;
        let a5 = a4 * a;
        return ((((a5 * (1.0 / 5.0)) - (a4 * 0.25)) + (a3 * (1.0 / 3.0))) - (a2 * 0.5)) + a;
    }
    return log(1.0 + a);
}

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
        case 6u: {
            if (a > 0.0) {
                return 1.0;
            }
            if (a < 0.0) {
                return -1.0;
            }
            if (a == 0.0) {
                return 0.0;
            }
            return a;
        }
        case 7u: { return a; }
        case 8u: { return 0.0; }
        case 9u: { return a; }
        case 10u: { return atan2(0.0, a); }
        case 11u: { return expm1_precise(a); }
        case 12u: { return log1p_precise(a); }
        case 13u: { return log(a) * 0.4342944819; }
        case 14u: { return log(a) * 1.4426950409; }
        case 15u: { return exp(a * 0.6931472); }
        case 16u: { return floor(a); }
        case 17u: { return ceil(a); }
        case 18u: {
            let t = trunc(a);
            if (t == 0.0) {
                return 0.0;
            }
            return t;
        }
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

pub const SCALAR_SHADER_F64: &str = r#"
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
    scalar_pad2: f64,
    scalar_pad3: f64,
    scalar_pad4: f64,
    scalar_pad5: f64,
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

pub const SCALAR_SHADER_F32: &str = r#"
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
    scalar_pad2: vec4<f32>,
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
