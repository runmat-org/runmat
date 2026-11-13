//! Classify stored IEEE encodings without floating-point comparisons.
//! Shader finite-math assumptions must not erase NaN or infinity detection.
use crate::backend::wgpu::types::NumericPrecision;
use runmat_builtins::NumericClassificationPredicate;

pub(crate) fn classification_shader(
    precision: NumericPrecision,
    predicate: NumericClassificationPredicate,
) -> String {
    let (scalar, bits) = match precision {
        NumericPrecision::F32 => (
            "f32",
            "let magnitude = input0.data[idx] & 0x7fffffffu;\nlet infinite = magnitude == 0x7f800000u;\nlet nan = magnitude > 0x7f800000u;",
        ),
        NumericPrecision::F64 => (
            "f64",
            "let low = input0.data[2u * idx];\nlet high = input0.data[2u * idx + 1u] & 0x7fffffffu;\nlet infinite = high == 0x7ff00000u && low == 0u;\nlet nan = high > 0x7ff00000u || (high == 0x7ff00000u && low != 0u);",
        ),
    };
    let condition = match predicate {
        NumericClassificationPredicate::Nan => "nan",
        NumericClassificationPredicate::Infinite => "infinite",
        NumericClassificationPredicate::Finite => "!nan && !infinite",
    };
    format!(
        r#"struct Words {{ data: array<u32> }};
struct Output {{ data: array<{scalar}> }};
struct Params {{ len: u32, _p0: u32, _p1: u32, _p2: u32 }};
@group(0) @binding(0) var<storage, read> input0: Words;
@group(0) @binding(1) var<storage, read_write> output: Output;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.len) {{ return; }}
    {bits}
    output.data[idx] = select({scalar}(0.0), {scalar}(1.0), {condition});
}}
"#
    )
}
