//! Broadcasted floating remainder using the shared direct/fusion semantics.
use crate::backend::wgpu::types::NumericPrecision;
use runmat_types::FloatingRemainderKind;

pub(crate) fn shader(precision: NumericPrecision, kind: FloatingRemainderKind) -> String {
    let ty = precision.as_str();
    let precision = match precision {
        NumericPrecision::F32 => runmat_accelerate_api::ProviderPrecision::F32,
        NumericPrecision::F64 => runmat_accelerate_api::ProviderPrecision::F64,
    };
    let helpers = crate::kernel_math::floating_remainder::functions(precision);
    let name = crate::kernel_math::floating_remainder::entrypoint(precision);
    let modulus = kind == FloatingRemainderKind::Modulus;
    let rank = crate::backend::wgpu::params::BCAST_MAX_RANK;
    let source = format!(
        r#"
struct PackedValue {{ value: u32, _pad0: u32, _pad1: u32, _pad2: u32 }};
alias PackedArray = array<PackedValue, {rank}>;
struct Tensor {{ data: array<{ty}> }};
struct Params {{
    len: u32, offset: u32, rank: u32, op: u32,
    out_shape: PackedArray, a_shape: PackedArray, a_stride: PackedArray,
    b_shape: PackedArray, b_stride: PackedArray,
}};
@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
{helpers}
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if gid.x >= params.len {{ return; }}
    let idx = params.offset + gid.x;
    var remaining = idx;
    var left = 0u;
    var right = 0u;
    for (var dimension = 0u; dimension < params.rank; dimension = dimension + 1u) {{
        let coordinate = remaining % params.out_shape[dimension].value;
        remaining = remaining / params.out_shape[dimension].value;
        if params.a_shape[dimension].value != 1u {{ left = left + coordinate * params.a_stride[dimension].value; }}
        if params.b_shape[dimension].value != 1u {{ right = right + coordinate * params.b_stride[dimension].value; }}
    }}
    Out.data[idx] = {name}(A.data[left], B.data[right], {modulus});
}}
"#
    );
    if precision == runmat_accelerate_api::ProviderPrecision::F64 {
        super::ieee_words::inject(&source)
    } else {
        source
    }
}
