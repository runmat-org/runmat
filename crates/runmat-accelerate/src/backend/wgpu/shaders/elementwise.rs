use crate::backend::wgpu::types::NumericPrecision;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ComplexUnaryOp {
    Real,
    Imag,
    Abs,
    Conj,
    Angle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ComplexBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl ComplexBinaryOp {
    pub(crate) fn try_from_binary_op(
        op: crate::backend::wgpu::types::BinaryOpCode,
    ) -> Option<Self> {
        match op {
            crate::backend::wgpu::types::BinaryOpCode::Add => Some(Self::Add),
            crate::backend::wgpu::types::BinaryOpCode::Sub => Some(Self::Sub),
            crate::backend::wgpu::types::BinaryOpCode::Mul => Some(Self::Mul),
            crate::backend::wgpu::types::BinaryOpCode::Div => Some(Self::Div),
            _ => None,
        }
    }
}

pub(crate) fn complex_unary_shader(op: ComplexUnaryOp, precision: NumericPrecision) -> String {
    let ty = match precision {
        NumericPrecision::F64 => "f64",
        NumericPrecision::F32 => "f32",
    };
    let expression = match op {
        ComplexUnaryOp::Real => "A.data[idx * 2u]",
        ComplexUnaryOp::Imag => "A.data[idx * 2u + 1u]",
        ComplexUnaryOp::Abs => {
            "sqrt((A.data[idx * 2u] * A.data[idx * 2u]) + (A.data[idx * 2u + 1u] * A.data[idx * 2u + 1u]))"
        }
        ComplexUnaryOp::Conj => {
            "select(A.data[idx], -A.data[idx], (idx % 2u) == 1u)"
        }
        ComplexUnaryOp::Angle => "atan2(A.data[idx * 2u + 1u], A.data[idx * 2u])",
    };
    format!(
        r#"
struct Tensor {{
    data: array<{ty}>,
}};

struct Params {{
    len: u32,
    offset: u32,
    _pad0: u32,
    _pad1: u32,
}};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    if idx >= params.len + params.offset {{
        return;
    }}
    Out.data[idx] = {expression};
}}
"#,
        ty = ty,
        expression = expression,
    )
}

pub(crate) fn complex_from_real_shader(precision: NumericPrecision) -> String {
    let ty = match precision {
        NumericPrecision::F64 => "f64",
        NumericPrecision::F32 => "f32",
    };
    format!(
        r#"
struct Tensor {{
    data: array<{ty}>,
}};

struct Params {{
    len: u32,
    offset: u32,
    _pad0: u32,
    _pad1: u32,
}};

@group(0) @binding(0) var<storage, read> Real: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    if idx >= params.len + params.offset {{
        return;
    }}
    let elem = idx / 2u;
    if (idx % 2u) == 0u {{
        Out.data[idx] = Real.data[elem];
    }} else {{
        Out.data[idx] = 0.0;
    }}
}}
"#,
        ty = ty,
    )
}

pub(crate) fn complex_from_real_imag_shader(
    precision: NumericPrecision,
    real_scalar: bool,
    imag_scalar: bool,
) -> String {
    let ty = match precision {
        NumericPrecision::F64 => "f64",
        NumericPrecision::F32 => "f32",
    };
    let real_index = if real_scalar { "0u" } else { "elem" };
    let imag_index = if imag_scalar { "0u" } else { "elem" };
    format!(
        r#"
struct Tensor {{
    data: array<{ty}>,
}};

struct Params {{
    len: u32,
    offset: u32,
    _pad0: u32,
    _pad1: u32,
}};

@group(0) @binding(0) var<storage, read> Real: Tensor;
@group(0) @binding(1) var<storage, read> Imag: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    if idx >= params.len + params.offset {{
        return;
    }}
    let elem = idx / 2u;
    if (idx % 2u) == 0u {{
        Out.data[idx] = Real.data[{real_index}];
    }} else {{
        Out.data[idx] = Imag.data[{imag_index}];
    }}
}}
"#,
        ty = ty,
        real_index = real_index,
        imag_index = imag_index,
    )
}

pub(crate) fn complex_binary_shader(
    op: ComplexBinaryOp,
    precision: NumericPrecision,
    lhs_complex: bool,
    rhs_complex: bool,
) -> String {
    let ty = match precision {
        NumericPrecision::F64 => "f64",
        NumericPrecision::F32 => "f32",
    };
    let lhs_real = if lhs_complex {
        "A.data[elem * 2u]"
    } else {
        "A.data[elem]"
    };
    let lhs_imag = if lhs_complex {
        "A.data[elem * 2u + 1u]"
    } else {
        "0.0"
    };
    let rhs_real = if rhs_complex {
        "B.data[elem * 2u]"
    } else {
        "B.data[elem]"
    };
    let rhs_imag = if rhs_complex {
        "B.data[elem * 2u + 1u]"
    } else {
        "0.0"
    };
    let body = match op {
        ComplexBinaryOp::Add => "let out_re = ar + br;\n    let out_im = ai + bi;",
        ComplexBinaryOp::Sub => "let out_re = ar - br;\n    let out_im = ai - bi;",
        ComplexBinaryOp::Mul => {
            "let out_re = (ar * br) - (ai * bi);\n    let out_im = (ar * bi) + (ai * br);"
        }
        ComplexBinaryOp::Div => {
            "let denom = (br * br) + (bi * bi);\n    let out_re = ((ar * br) + (ai * bi)) / denom;\n    let out_im = ((ai * br) - (ar * bi)) / denom;"
        }
    };
    format!(
        r#"
struct Tensor {{
    data: array<{ty}>,
}};

struct Params {{
    len: u32,
    offset: u32,
    _pad0: u32,
    _pad1: u32,
}};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.len {{
        return;
    }}
    let idx = params.offset + local;
    if idx >= params.len + params.offset {{
        return;
    }}
    let elem = idx / 2u;
    let ar = {lhs_real};
    let ai = {lhs_imag};
    let br = {rhs_real};
    let bi = {rhs_imag};
    {body}
    Out.data[idx] = select(out_re, out_im, (idx % 2u) == 1u);
}}
"#,
        ty = ty,
        lhs_real = lhs_real,
        lhs_imag = lhs_imag,
        rhs_real = rhs_real,
        rhs_imag = rhs_imag,
        body = body,
    )
}

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
        case 5u: { return atan2(a, b); }
        case 6u: { return pow(a, b); }
        case 7u: { return max(a, b); }
        case 8u: { return min(a, b); }
        default: { return a; }
    }
}

@compute @workgroup_size(512)
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
        case 5u: { return atan2(a, b); }
        case 6u: { return pow(a, b); }
        case 7u: { return max(a, b); }
        case 8u: { return min(a, b); }
        default: { return a; }
    }
}

@compute @workgroup_size(512)
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

// Broadcast-aware binary shader (N-D implicit expansion)
pub const BINARY_BROADCAST_SHADER_F64: &str = r#"
const MAX_RANK: u32 = 128u;

struct PackedValue {
    value: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

alias PackedArray = array<PackedValue, MAX_RANK>;

struct Tensor { data: array<f64>, };

struct Params {
    len: u32,
    offset: u32,
    rank: u32,
    op: u32,
    out_shape: PackedArray,
    a_shape: PackedArray,
    b_shape: PackedArray,
    a_stride: PackedArray,
    b_stride: PackedArray,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

fn hypot(a: f64, b: f64) -> f64 { return sqrt((a * a) + (b * b)); }

fn apply(a: f64, b: f64) -> f64 {
    switch params.op {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        case 4u: { return hypot(a, b); }
        case 5u: { return atan2(a, b); }
        case 6u: { return pow(a, b); }
        case 7u: { return max(a, b); }
        case 8u: { return min(a, b); }
        default: { return a; }
    }
}

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len { return; }
    let idx = params.offset + local;

    // Compute N-D coordinates from linear index (column-major order)
    var coord: array<u32, MAX_RANK>;
    var tmp: u32 = idx;
    var d: u32 = 0u;
    loop {
        if d >= params.rank { break; }
        let dim = params.out_shape[d].value;
        if dim == 0u { coord[d] = 0u; }
        else { coord[d] = tmp % dim; tmp = tmp / dim; }
        d = d + 1u;
    }

    // Map to A and B indices with broadcasting
    var ia: u32 = 0u;
    var ib: u32 = 0u;
    d = 0u;
    loop {
        if d >= params.rank { break; }
        let ad = params.a_shape[d].value;
        let bd = params.b_shape[d].value;
        let ca = select(coord[d], 0u, ad == 1u);
        let cb = select(coord[d], 0u, bd == 1u);
        ia = ia + ca * params.a_stride[d].value;
        ib = ib + cb * params.b_stride[d].value;
        d = d + 1u;
    }

    Out.data[idx] = apply(A.data[ia], B.data[ib]);
}
"#;

pub const BINARY_BROADCAST_SHADER_F32: &str = r#"
const MAX_RANK: u32 = 128u;

struct PackedValue {
    value: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

alias PackedArray = array<PackedValue, MAX_RANK>;

struct Tensor { data: array<f32>, };

struct Params {
    len: u32,
    offset: u32,
    rank: u32,
    op: u32,
    out_shape: PackedArray,
    a_shape: PackedArray,
    b_shape: PackedArray,
    a_stride: PackedArray,
    b_stride: PackedArray,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

fn hypot(a: f32, b: f32) -> f32 { return sqrt((a * a) + (b * b)); }

fn apply(a: f32, b: f32) -> f32 {
    switch params.op {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        case 4u: { return hypot(a, b); }
        case 5u: { return atan2(a, b); }
        case 6u: { return pow(a, b); }
        case 7u: { return max(a, b); }
        case 8u: { return min(a, b); }
        default: { return a; }
    }
}

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len { return; }
    let idx = params.offset + local;

    var coord: array<u32, MAX_RANK>;
    var tmp: u32 = idx;
    var d: u32 = 0u;
    loop {
        if d >= params.rank { break; }
        let dim = params.out_shape[d].value;
        if dim == 0u { coord[d] = 0u; }
        else { coord[d] = tmp % dim; tmp = tmp / dim; }
        d = d + 1u;
    }

    var ia: u32 = 0u;
    var ib: u32 = 0u;
    d = 0u;
    loop {
        if d >= params.rank { break; }
        let ad = params.a_shape[d].value;
        let bd = params.b_shape[d].value;
        let ca = select(coord[d], 0u, ad == 1u);
        let cb = select(coord[d], 0u, bd == 1u);
        ia = ia + ca * params.a_stride[d].value;
        ib = ib + cb * params.b_stride[d].value;
        d = d + 1u;
    }

    Out.data[idx] = apply(A.data[ia], B.data[ib]);
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

const PI: f64 = 3.141592653589793;
const SQRT_TWO_PI: f64 = 2.5066282746310002;
const LANCZOS_G: f64 = 7.0;
const EPSILON: f64 = 1.0e-12;
const FACTORIAL_MAX: u32 = 170u;
const FACTORIAL_INT_TOL: f64 = 1.0e-10;
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

fn lanczos_gamma(z: f64) -> f64 {
    let z_minus_one = z - 1.0;
    var sum = 0.99999999999980993;
    sum = sum + 676.5203681218851 / (z_minus_one + 1.0);
    sum = sum + -1259.1392167224028 / (z_minus_one + 2.0);
    sum = sum + 771.3234287776531 / (z_minus_one + 3.0);
    sum = sum + -176.6150291621406 / (z_minus_one + 4.0);
    sum = sum + 12.507343278686905 / (z_minus_one + 5.0);
    sum = sum + -0.13857109526572012 / (z_minus_one + 6.0);
    sum = sum + 0.000009984369578019572 / (z_minus_one + 7.0);
    sum = sum + 0.00000015056327351493116 / (z_minus_one + 8.0);
    let t = z_minus_one + (LANCZOS_G + 0.5);
    return SQRT_TWO_PI * pow(t, z_minus_one + 0.5) * exp(-t) * sum;
}

fn is_non_positive_integer(x: f64) -> bool {
    if (x > 0.0) {
        return false;
    }
    let nearest = round(x);
    return abs(x - nearest) <= EPSILON * (1.0 + abs(x));
}

fn is_nan64(x: f64) -> bool {
    let bits = bitcast<u64>(x);
    return (bits & 0x7ff0000000000000u) == 0x7ff0000000000000u &&
        (bits & 0x000fffffffffffffu) != 0u;
}

fn pos_inf_f64() -> f64 {
    var bits: u64 = 0x7ff0000000000000u;
    return bitcast<f64>(bits);
}

fn neg_inf_f64() -> f64 {
    var bits: u64 = 0xfff0000000000000u;
    return bitcast<f64>(bits);
}

fn nan_f64() -> f64 {
    var bits: u64 = 0x7ff8000000000000u;
    return bitcast<f64>(bits);
}

fn is_inf64(x: f64) -> bool {
    let inf = pos_inf_f64();
    let neg_inf = neg_inf_f64();
    return x == inf || x == neg_inf;
}

fn gamma_real(a: f64) -> f64 {
    if (is_nan64(a)) {
        return a;
    }
    if (is_inf64(a)) {
        if (a > 0.0) {
            return pos_inf_f64();
        }
        return nan_f64();
    }
    if (is_non_positive_integer(a)) {
        return pos_inf_f64();
    }
    if (a < 0.5) {
        let sin_term = sin(PI * a);
        if (abs(sin_term) <= EPSILON) {
            return pos_inf_f64();
        }
        let gamma_one_minus = lanczos_gamma(1.0 - a);
        return PI / (sin_term * gamma_one_minus);
    }
    return lanczos_gamma(a);
}

fn factorial_real(a: f64) -> f64 {
    if (is_nan64(a)) {
        return a;
    }
    if (a == 0.0) {
        return 1.0;
    }
    if (is_inf64(a)) {
        if (a > 0.0) {
            return pos_inf_f64();
        }
        return nan_f64();
    }
    if (a < 0.0) {
        return nan_f64();
    }
    let rounded = round(a);
    if (abs(a - rounded) > FACTORIAL_INT_TOL) {
        return nan_f64();
    }
    if (rounded < 0.0) {
        return nan_f64();
    }
    if (rounded > f64(FACTORIAL_MAX)) {
        return pos_inf_f64();
    }
    let n = i32(rounded);
    if (n <= 1) {
        return 1.0;
    }
    let limit = u32(n);
    var acc: f64 = 1.0;
    var i: u32 = 2u;
    loop {
        if (i > limit) {
            break;
        }
        acc = acc * f64(i);
        i = i + 1u;
    }
    return acc;
}

fn sinc_real(a: f64) -> f64 {
    if (a == 0.0) {
        return 1.0;
    }
    let abs_a = abs(a);
    if (!is_nan64(a) && !is_inf64(a) && floor(abs_a) == abs_a) {
        return 0.0;
    }
    let scaled = PI * a;
    return sin(scaled) / scaled;
}

fn heaviside_real(a: f64) -> f64 {
    if (is_nan64(a)) {
        return a;
    }
    if (a > 0.0) {
        return 1.0;
    }
    if (a < 0.0) {
        return 0.0;
    }
    return 0.5;
}

fn apply(a: f64) -> f64 {
    switch params.op {
        case 0u: { return sin(a); }
        case 1u: { return cos(a); }
        case 19u: { return tan(a); }
        case 20u: { return asin(a); }
        case 21u: { return acos(a); }
        case 22u: { return atan(a); }
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
        case 23u: { return sinh(a); }
        case 24u: { return cosh(a); }
        case 25u: { return tanh(a); }
        case 26u: { return asinh(a); }
        case 27u: { return acosh(a); }
        case 28u: { return atanh(a); }
        case 29u: { return gamma_real(a); }
        case 30u: { return factorial_real(a); }
        case 31u: { return f64(f32(a)); }
        case 32u: {
            let aa = abs(a);
            if (aa == 0.0) {
                return 0.0;
            }
            return ceil(log2(aa));
        }
        case 33u: { return sinc_real(a); }
        case 34u: { return heaviside_real(a); }
        default: { return a; }
    }
}

@compute @workgroup_size(512)
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

const PI: f32 = 3.1415927;
const SQRT_TWO_PI: f32 = 2.5066283;
const LANCZOS_G: f32 = 7.0;
const EPSILON: f32 = 1.0e-5;
const FACTORIAL_MAX_F32: u32 = 170u;
const FACTORIAL_INT_TOL_F32: f32 = 1.0e-4;
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

fn lanczos_gamma(z: f32) -> f32 {
    let z_minus_one = z - 1.0;
    var sum: f32 = 0.99999994;
    sum = sum + 676.5204 / (z_minus_one + 1.0);
    sum = sum + -1259.1393 / (z_minus_one + 2.0);
    sum = sum + 771.3234 / (z_minus_one + 3.0);
    sum = sum + -176.61502 / (z_minus_one + 4.0);
    sum = sum + 12.507343 / (z_minus_one + 5.0);
    sum = sum + -0.1385711 / (z_minus_one + 6.0);
    sum = sum + 0.00000998437 / (z_minus_one + 7.0);
    sum = sum + 0.00000015056327 / (z_minus_one + 8.0);
    let t = z_minus_one + (LANCZOS_G + 0.5);
    return SQRT_TWO_PI * pow(t, z_minus_one + 0.5) * exp(-t) * sum;
}

fn is_non_positive_integer(x: f32) -> bool {
    if (x > 0.0) {
        return false;
    }
    let nearest = round(x);
    return abs(x - nearest) <= EPSILON * (1.0 + abs(x));
}

fn is_nan32(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7f800000u) == 0x7f800000u &&
        (bits & 0x007fffffu) != 0u;
}

fn pos_inf_f32() -> f32 {
    var bits: u32 = 0x7f800000u;
    return bitcast<f32>(bits);
}

fn neg_inf_f32() -> f32 {
    var bits: u32 = 0xff800000u;
    return bitcast<f32>(bits);
}

fn nan_f32() -> f32 {
    var bits: u32 = 0x7fc00000u;
    return bitcast<f32>(bits);
}

fn is_inf32(x: f32) -> bool {
    let inf = pos_inf_f32();
    let neg_inf = neg_inf_f32();
    return x == inf || x == neg_inf;
}

fn gamma_real(a: f32) -> f32 {
    if (is_nan32(a)) {
        return a;
    }
    if (is_inf32(a)) {
        if (a > 0.0) {
            return pos_inf_f32();
        }
        return nan_f32();
    }
    if (is_non_positive_integer(a)) {
        return pos_inf_f32();
    }
    if (a < 0.5) {
        let sin_term = sin(PI * a);
        if (abs(sin_term) <= EPSILON) {
            return pos_inf_f32();
        }
        let gamma_one_minus = lanczos_gamma(1.0 - a);
        return PI / (sin_term * gamma_one_minus);
    }
    return lanczos_gamma(a);
}

fn factorial_real(a: f32) -> f32 {
    if (is_nan32(a)) {
        return a;
    }
    if (a == 0.0) {
        return 1.0;
    }
    if (is_inf32(a)) {
        if (a > 0.0) {
            return pos_inf_f32();
        }
        return nan_f32();
    }
    if (a < 0.0) {
        return nan_f32();
    }
    let rounded = round(a);
    if (abs(a - rounded) > FACTORIAL_INT_TOL_F32) {
        return nan_f32();
    }
    if (rounded < 0.0) {
        return nan_f32();
    }
    if (rounded > f32(FACTORIAL_MAX_F32)) {
        return pos_inf_f32();
    }
    let n = i32(rounded);
    if (n <= 1) {
        return 1.0;
    }
    let limit = u32(n);
    var acc: f32 = 1.0;
    var i: u32 = 2u;
    loop {
        if (i > limit) {
            break;
        }
        acc = acc * f32(i);
        i = i + 1u;
    }
    return acc;
}

fn sinc_real(a: f32) -> f32 {
    if (a == 0.0) {
        return 1.0;
    }
    let abs_a = abs(a);
    if (!is_nan32(a) && !is_inf32(a) && floor(abs_a) == abs_a) {
        return 0.0;
    }
    let scaled = PI * a;
    return sin(scaled) / scaled;
}

fn heaviside_real(a: f32) -> f32 {
    if (is_nan32(a)) {
        return a;
    }
    if (a > 0.0) {
        return 1.0;
    }
    if (a < 0.0) {
        return 0.0;
    }
    return 0.5;
}

fn apply(a: f32) -> f32 {
    switch params.op {
        case 0u: { return sin(a); }
        case 1u: { return cos(a); }
        case 19u: { return tan(a); }
        case 20u: { return asin(a); }
        case 21u: { return acos(a); }
        case 22u: { return atan(a); }
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
        case 23u: { return sinh(a); }
        case 24u: { return cosh(a); }
        case 25u: { return tanh(a); }
        case 26u: { return asinh(a); }
        case 27u: { return acosh(a); }
        case 28u: { return atanh(a); }
        case 29u: { return gamma_real(a); }
        case 30u: { return factorial_real(a); }
        case 31u: { return a; }
        case 32u: {
            let aa = abs(a);
            if (aa == 0.0) {
                return 0.0;
            }
            return ceil(log2(aa));
        }
        case 33u: { return sinc_real(a); }
        case 34u: { return heaviside_real(a); }
        default: { return a; }
    }
}

@compute @workgroup_size(512)
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

@compute @workgroup_size(512)
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
        case 6u: { result = max(a, scalar); }
        case 7u: { result = min(a, scalar); }
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

@compute @workgroup_size(512)
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
        case 6u: { result = max(a, s); }
        case 7u: { result = min(a, s); }
        default: { result = a; }
    }
    Out.data[idx] = result;
}
"#;
