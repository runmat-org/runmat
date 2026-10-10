//! Broadcasted real-input power and its exact paired-domain probe.
use crate::backend::wgpu::types::NumericPrecision;
use runmat_accelerate_api::PowerResultDomain;

pub(crate) enum PowerShader {
    Classify,
    Evaluate(PowerResultDomain),
}

pub(crate) fn shader(precision: NumericPrecision, mode: PowerShader) -> String {
    shader_with_input_precision(precision, precision, mode)
}

pub(crate) fn classification_shader(
    input_precision: NumericPrecision,
    output_precision: NumericPrecision,
) -> String {
    shader_with_input_precision(output_precision, input_precision, PowerShader::Classify)
}

fn shader_with_input_precision(
    precision: NumericPrecision,
    input_precision: NumericPrecision,
    mode: PowerShader,
) -> String {
    let ty = precision.as_str();
    let input_ty = input_precision.as_str();
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
    let source = format!(
        r#"
alias T = {ty};
const MAX_RANK: u32 = {max_rank}u;
struct PackedValue {{ value: u32, _pad0: u32, _pad1: u32, _pad2: u32 }};
alias PackedArray = array<PackedValue, MAX_RANK>;
struct Tensor {{ data: array<T> }};
struct InputTensor {{ data: array<{input_ty}> }};
struct Params {{
    len: u32, offset: u32, rank: u32, op: u32,
    out_shape: PackedArray, a_shape: PackedArray, a_stride: PackedArray,
    b_shape: PackedArray, b_stride: PackedArray,
}};
@group(0) @binding(0) var<storage, read> A: InputTensor;
@group(0) @binding(1) var<storage, read> B: InputTensor;
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
    let a = T(A.data[ia]);
    let b = T(B.data[ib]);
    {body}
}}
"#
    );
    if precision == NumericPrecision::F64 {
        super::ieee_words::inject(&source)
    } else {
        source
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};

    #[test]
    fn classification_shader_identity_contains_input_and_flag_precision() {
        let native = classification_shader(NumericPrecision::F64, NumericPrecision::F64);
        let typed = classification_shader(NumericPrecision::F32, NumericPrecision::F64);
        let single = classification_shader(NumericPrecision::F32, NumericPrecision::F32);
        assert_ne!(native, typed);
        assert_ne!(single, typed);
        assert!(typed.contains("alias T = f64;"));
        assert!(typed.contains("struct InputTensor { data: array<f32> }"));
        assert!(typed.contains("let a = T(A.data[ia]);"));
        assert!(typed.contains("params.a_stride[d].value"));
        assert!(typed.contains("rm_power_f64_needs_complex(a, b)"));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn generated_power_shaders_validate_without_constant_nonfinite_bitcasts() {
        for precision in [NumericPrecision::F32, NumericPrecision::F64] {
            let caps = if precision == NumericPrecision::F32 {
                RequiredCaps::None
            } else {
                RequiredCaps::Float64Int64
            };
            for (name, source) in [
                (
                    "classification",
                    classification_shader(precision, precision),
                ),
                (
                    "real",
                    shader(precision, PowerShader::Evaluate(PowerResultDomain::Real)),
                ),
                (
                    "complex",
                    shader(precision, PowerShader::Evaluate(PowerResultDomain::Complex)),
                ),
            ] {
                assert!(!source.contains("bitcast<f32>(0x"), "{name} {precision:?}");
                validate_generated(name, &source, 64, 8, caps);
            }
        }
    }
}
