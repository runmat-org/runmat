//! Resident deep-learning kernels. These do not change generic reductions.

pub const POOL_WORKGROUP_SIZE: u32 = 256;

const STABLE_POOL_1D: &str = r#"
struct Tensor { data: array<$SCALAR$>, };
struct Params {
  pool_size: u32,
  chunk_output_count: u32,
  inner_stride: u32,
  output_offset: u32,
};
@group(0) @binding(0) var<storage, read> input: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn is_special(value: $SCALAR$) -> bool {
  // x - x is NaN exactly for infinities; x != x detects NaNs.
  return value != value || (value - value) != (value - value);
}

@compute @workgroup_size($WG$)
fn main(@builtin(local_invocation_id) lane: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let local_column = group.x * $WG$u + lane.x;
  if (local_column >= params.chunk_output_count) { return; }
  let column = params.output_offset + local_column;
  let inner = column % params.inner_stride;
  let outer = column / params.inner_stride;
  let base = inner + outer * params.inner_stride * params.pool_size;
  // Each invocation owns one output and follows the host sum order. A
  // max-scaled parallel tree loses representable cancellation residuals such
  // as [MAX, -MAX, subnormal], even though its finite range is attractive.
  var sum: $SCALAR$ = 0.0;
  var overflow = false;
  var index = 0u;
  while (index < params.pool_size) {
    let value = input.data[base + index * params.inner_stride];
    let next = sum + value;
    if (!is_special(sum) && !is_special(value) && is_special(next)) {
      overflow = true;
      break;
    }
    sum = next;
    index = index + 1u;
  }
  if (!overflow) {
    output.data[column] = sum / $SCALAR$(params.pool_size);
    return;
  }
  // Match the host overflow branch in the input precision, without a
  // widened mirror. This recurrence never materializes an overflowing sum.
  var mean: $SCALAR$ = 0.0;
  index = 0u;
  while (index < params.pool_size) {
    let count = $SCALAR$(index + 1u);
    let previous_weight = $SCALAR$(index) / count;
    mean = mean * previous_weight + input.data[base + index * params.inner_stride] / count;
    index = index + 1u;
  }
  output.data[column] = mean;
}
"#;

pub fn stable_pool_1d(precision: super::super::types::NumericPrecision) -> String {
    let scalar = match precision {
        super::super::types::NumericPrecision::F32 => "f32",
        super::super::types::NumericPrecision::F64 => "f64",
    };
    STABLE_POOL_1D
        .replace("$SCALAR$", scalar)
        .replace("$WG$", &POOL_WORKGROUP_SIZE.to_string())
}
