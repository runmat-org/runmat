// NaN handling on stored IEEE words. Non-NaN lanes never enter floating-point
// arithmetic, so signed zero and subnormal payloads retain their exact bits.

use crate::backend::wgpu::types::NumericPrecision;

pub(crate) fn map_source(precision: NumericPrecision) -> String {
    let source = match precision {
        NumericPrecision::F32 => NAN_TO_ZERO_SHADER_F32,
        NumericPrecision::F64 => NAN_TO_ZERO_SHADER_F64,
    };
    super::nan_predicate::inject_words(source, precision)
}

pub(crate) fn mask_source(precision: NumericPrecision) -> String {
    let source = match precision {
        NumericPrecision::F32 => NOT_NAN_MASK_SHADER_F32,
        NumericPrecision::F64 => NOT_NAN_MASK_SHADER_F64,
    };
    super::nan_predicate::inject_words(source, precision)
}

const NAN_TO_ZERO_SHADER_F64: &str = r#"struct Words { data: array<u32> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Words;
@group(0) @binding(1) var<storage, read_write> output: Words;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let word = 2u * idx;
  let low = input0.data[word];
  let high = input0.data[word + 1u];
  let nan = @IEEE_NAN_WORD_TEST@;
  output.data[word] = select(low, 0u, nan);
  output.data[word + 1u] = select(high, 0u, nan);
}
"#;

const NAN_TO_ZERO_SHADER_F32: &str = r#"struct Words { data: array<u32> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Words;
@group(0) @binding(1) var<storage, read_write> output: Words;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let bits = input0.data[idx];
  let nan = @IEEE_NAN_WORD_TEST@;
  output.data[idx] = select(bits, 0u, nan);
}
"#;

const NOT_NAN_MASK_SHADER_F64: &str = r#"struct Words { data: array<u32> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Words;
@group(0) @binding(1) var<storage, read_write> output: Words;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let word = 2u * idx;
  let low = input0.data[word];
  let high = input0.data[word + 1u];
  let nan = @IEEE_NAN_WORD_TEST@;
  output.data[word] = 0u;
  output.data[word + 1u] = select(0x3ff00000u, 0u, nan);
}
"#;

const NOT_NAN_MASK_SHADER_F32: &str = r#"struct Words { data: array<u32> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Words;
@group(0) @binding(1) var<storage, read_write> output: Words;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let bits = input0.data[idx];
  let nan = @IEEE_NAN_WORD_TEST@;
  output.data[idx] = select(0x3f800000u, 0u, nan);
}
"#;
