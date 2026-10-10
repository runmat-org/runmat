//! Ternary element selection. Only the chosen branch is loaded and copied.
use crate::backend::wgpu::types::NumericPrecision;

pub(crate) fn shader(precision: NumericPrecision) -> String {
    let ty = precision.as_str();
    let max_rank = crate::backend::wgpu::params::BCAST_MAX_RANK;
    format!(
        r#"
alias T = {ty};
const MAX_RANK: u32 = {max_rank}u;
struct PackedValue {{ value: u32, _pad0: u32, _pad1: u32, _pad2: u32 }};
alias PackedArray = array<PackedValue, MAX_RANK>;
struct Tensor {{ data: array<T> }};
struct Params {{
    len: u32, offset: u32, rank: u32, _pad: u32,
    out_shape: PackedArray,
    in0_shape: PackedArray, in0_stride: PackedArray,
    in1_shape: PackedArray, in1_stride: PackedArray,
    in2_shape: PackedArray, in2_stride: PackedArray,
}};
@group(0) @binding(0) var<storage, read> Mask: Tensor;
@group(0) @binding(1) var<storage, read> Then: Tensor;
@group(0) @binding(2) var<storage, read> Else: Tensor;
@group(0) @binding(3) var<storage, read_write> Out: Tensor;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if gid.x >= params.len {{ return; }}
    let index = params.offset + gid.x;
    var remainder = index;
    var i0 = 0u;
    var i1 = 0u;
    var i2 = 0u;
    for (var axis = 0u; axis < params.rank; axis = axis + 1u) {{
        let extent = params.out_shape[axis].value;
        let coordinate = remainder % extent;
        remainder = remainder / extent;
        if params.in0_shape[axis].value != 1u {{ i0 = i0 + coordinate * params.in0_stride[axis].value; }}
        if params.in1_shape[axis].value != 1u {{ i1 = i1 + coordinate * params.in1_stride[axis].value; }}
        if params.in2_shape[axis].value != 1u {{ i2 = i2 + coordinate * params.in2_stride[axis].value; }}
    }}
    if Mask.data[i0] != T(0.0) {{
        Out.data[index] = Then.data[i1];
    }} else {{
        Out.data[index] = Else.data[i2];
    }}
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn shader_has_three_broadcast_operands_and_no_arithmetic_blending() {
        for precision in [NumericPrecision::F32, NumericPrecision::F64] {
            let source = shader(precision);
            assert!(source.contains("in2_stride: PackedArray"));
            assert!(source.contains("Out.data[index] = Then.data[i1]"));
            assert!(source.contains("Out.data[index] = Else.data[i2]"));
            assert!(!source.contains("Mask.data[i0] *"));
            let required = match precision {
                NumericPrecision::F32 => RequiredCaps::None,
                NumericPrecision::F64 => RequiredCaps::Float64,
            };
            validate_generated(&format!("select-{precision:?}"), &source, 64, 8, required);
        }
    }
}
