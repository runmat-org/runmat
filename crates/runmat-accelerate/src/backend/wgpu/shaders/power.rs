//! Broadcasted real-input power and its exact paired-domain probe.
use crate::backend::wgpu::types::NumericPrecision;
use runmat_accelerate_api::PowerResultDomain;

pub(crate) enum PowerShader {
    Classify,
    Evaluate(PowerResultDomain),
}

pub(crate) fn shader(precision: NumericPrecision, mode: PowerShader) -> String {
    let ty = precision.as_str();
    let precision_api = match precision {
        NumericPrecision::F32 => runmat_accelerate_api::ProviderPrecision::F32,
        NumericPrecision::F64 => runmat_accelerate_api::ProviderPrecision::F64,
    };
    let math = crate::kernel_math::power::functions(precision_api);
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
    let name = crate::kernel_math::power::entrypoint(precision_api);
    let body = body
        .replace("real_power(", &format!("{name}("))
        .replace("needs_complex(", &format!("{name}_needs_complex("));
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
{math}
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
