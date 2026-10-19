//! One invocation per output slice, with the reduced axis visited in host
//! source order. Input and output storage classes are selected separately;
//! accumulation and division always use Double.

use crate::backend::wgpu::types::NumericPrecision;

pub(crate) const LANES: u32 = 64;

pub(crate) fn source(input: NumericPrecision, output: NumericPrecision) -> String {
    let input_type = match input {
        NumericPrecision::F32 => "f32",
        NumericPrecision::F64 => "f64",
    };
    let output_type = match output {
        NumericPrecision::F32 => "f32",
        NumericPrecision::F64 => "f64",
    };
    const SHADER: &str = r#"
const MAX_RANK: u32 = 128u;
struct PackedValue { value: u32, _pad0: u32, _pad1: u32, _pad2: u32, };
alias PackedArray = array<PackedValue, MAX_RANK>;
struct InTensor { data: array<@INPUT@>, };
struct OutTensor { data: array<@OUTPUT@>, };
struct Params {
  rank: u32,
  kept_count: u32,
  reduce_count: u32,
  omit_nan: u32,
  rows: u32,
  cols: u32,
  _pad2: vec2<u32>,
  kept_sizes: PackedArray,
  reduce_sizes: PackedArray,
  kept_strides: PackedArray,
  reduce_strides: PackedArray,
};
@group(0) @binding(0) var<storage, read> InBuf: InTensor;
@group(0) @binding(1) var<storage, read_write> OutBuf: OutTensor;
@group(0) @binding(2) var<uniform> params: Params;

fn nan_f64() -> f64 {
  return runmat_f64_from_words(0u, 0x7ff80000u);
}

fn isNan(value: f64) -> bool { @IEEE_NAN_PREDICATE@ }

fn map_col_to_base(col: u32) -> u32 {
  var rem = col;
  var base = 0u;
  for (var j = 0u; j < params.kept_count; j = j + 1u) {
    let size = params.kept_sizes[j].value;
    let coord = rem % size;
    rem = rem / size;
    base = base + coord * params.kept_strides[j].value;
  }
  return base;
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let col = wid.x * 64u + lid.x;
  if (col >= params.cols) { return; }
  let base = map_col_to_base(col);
  let stride = params.reduce_strides[0u].value;
  var sum = f64(0.0);
  var count = 0u;
  var saw_nan = false;
  for (var k = 0u; k < params.rows; k = k + 1u) {
    let value = f64(InBuf.data[base + k * stride]);
    if (isNan(value)) {
      if (params.omit_nan == 0u) { saw_nan = true; break; }
    } else {
      sum = sum + value;
      count = count + 1u;
    }
  }
  var result = nan_f64();
  if (!saw_nan) {
    if (params.omit_nan == 0u && params.rows > 0u) {
      result = sum / f64(params.rows);
    }
    if (params.omit_nan != 0u && count > 0u) {
      result = sum / f64(count);
    }
  }
  OutBuf.data[col] = @OUTPUT@(result);
}
"#;
    super::super::nan_predicate::inject(
        &SHADER
            .replace("@INPUT@", input_type)
            .replace("@OUTPUT@", output_type),
        NumericPrecision::F64,
    )
}
