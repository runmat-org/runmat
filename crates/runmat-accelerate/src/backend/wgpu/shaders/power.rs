//! Broadcasted real-input power and its exact paired-domain probe.
use crate::backend::wgpu::types::NumericPrecision;
use runmat_accelerate_api::PowerResultDomain;

pub(crate) enum PowerShader {
    Classify,
    Evaluate(PowerResultDomain),
}

pub(crate) fn shader(precision: NumericPrecision, mode: PowerShader) -> String {
    let ty = precision.as_str();
    let bits = match precision {
        NumericPrecision::F32 => {
            r#"
fn finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
fn nan(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }
fn negative(v: f32) -> bool { return (bitcast<u32>(v) & 0x80000000u) != 0u; }
"#
        }
        NumericPrecision::F64 => {
            r#"
fn finite(v: f64) -> bool { return (bitcast<vec2<u32>>(v).y & 0x7ff00000u) != 0x7ff00000u; }
fn nan(v: f64) -> bool {
    let b = bitcast<vec2<u32>>(v);
    return (b.y & 0x7ff00000u) == 0x7ff00000u && ((b.y & 0x000fffffu) != 0u || b.x != 0u);
}
fn negative(v: f64) -> bool { return (bitcast<vec2<u32>>(v).y & 0x80000000u) != 0u; }
"#
        }
    };
    let (width, body) = match mode {
        PowerShader::Classify => (
            1,
            "Out.data[idx] = select(T(0.0), T(1.0), needs_complex(a, b));",
        ),
        PowerShader::Evaluate(PowerResultDomain::Real) => (1, "Out.data[idx] = real_power(a, b);"),
        PowerShader::Evaluate(PowerResultDomain::Complex) => (
            2,
            r#"
    var re = real_power(a, b);
    var im = T(0.0);
    if needs_complex(a, b) {
        let magnitude = exp(b * log(abs(a)));
        let phase = b * T(3.1415926535897932384626433832795);
        re = magnitude * cos(phase);
        im = magnitude * sin(phase);
    }
    Out.data[idx] = select(re, im, (idx % 2u) == 1u);
"#,
        ),
    };
    let max_rank = crate::backend::wgpu::params::BCAST_MAX_RANK;
    format!(
        r#"
alias T = {ty};
const MAX_RANK: u32 = {max_rank}u;
struct PackedValue {{ value: u32, _pad0: u32, _pad1: u32, _pad2: u32 }};
alias PackedArray = array<PackedValue, MAX_RANK>;
struct Tensor {{ data: array<T> }};
struct Params {{
    len: u32, offset: u32, rank: u32, op: u32,
    out_shape: PackedArray, a_shape: PackedArray, a_stride: PackedArray,
    b_shape: PackedArray, b_stride: PackedArray,
}};
@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
{bits}
fn needs_complex(a: T, b: T) -> bool {{
    return finite(a) && a < T(0.0) && finite(b) && trunc(b) != b;
}}
fn real_power(a: T, b: T) -> T {{
    let infinity = T(bitcast<f32>(0x7f800000u));
    let not_number = T(bitcast<f32>(0x7fc00000u));
    if nan(a) || nan(b) {{ return not_number; }}
    if b == T(0.0) {{ return T(1.0); }}
    let magnitude = abs(a);
    if !finite(b) {{
        if magnitude == T(1.0) {{ return T(1.0); }}
        return select(T(0.0), infinity, (magnitude > T(1.0)) == (b > T(0.0)));
    }}
    let odd = trunc(b) == b && abs(b - T(2.0) * trunc(b / T(2.0))) == T(1.0);
    var result: T;
    if magnitude == T(0.0) {{ result = select(T(0.0), infinity, b < T(0.0)); }}
    else if !finite(a) {{ result = select(T(0.0), infinity, b > T(0.0)); }}
    else {{ result = pow(magnitude, b); }}
    if negative(a) && odd {{ return -result; }}
    if needs_complex(a, b) {{ return not_number; }}
    return result;
}}
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if gid.x >= params.len {{ return; }}
    let idx = params.offset + gid.x;
    var remaining = idx / {width}u;
    var ia = 0u;
    var ib = 0u;
    for (var d = 0u; d < params.rank; d = d + 1u) {{
        let extent = params.out_shape[d].value;
        let coordinate = remaining % extent;
        remaining = remaining / extent;
        if params.a_shape[d].value != 1u {{ ia = ia + coordinate * params.a_stride[d].value; }}
        if params.b_shape[d].value != 1u {{ ib = ib + coordinate * params.b_stride[d].value; }}
    }}
    let a = A.data[ia];
    let b = B.data[ib];
    {body}
}}
"#
    )
}
