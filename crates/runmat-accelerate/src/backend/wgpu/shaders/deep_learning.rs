//! Resident deep-learning kernels. These do not change generic reductions.

pub const POOL_WORKGROUP_SIZE: u32 = 256;

const POOL_1D_ADJOINT: &str = r#"
struct Tensor { data: array<$SCALAR$>, };
struct Params {
  pool_size: u32,
  chunk_output_count: u32,
  inner_stride: u32,
  output_offset: u32,
};
@group(0) @binding(0) var<storage, read> upstream: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size($WG$)
fn main(@builtin(local_invocation_id) lane: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let local = group.x * $WG$u + lane.x;
  if (local >= params.chunk_output_count) { return; }
  let index = params.output_offset + local;
  let block = params.inner_stride * params.pool_size;
  let upstream_index = (index / block) * params.inner_stride + index % params.inner_stride;
  output.data[index] = upstream.data[upstream_index] / $SCALAR$(params.pool_size);
}
"#;

pub fn pool_1d_adjoint(precision: super::super::types::NumericPrecision) -> String {
    let scalar = match precision {
        super::super::types::NumericPrecision::F32 => "f32",
        super::super::types::NumericPrecision::F64 => "f64",
    };
    POOL_1D_ADJOINT
        .replace("$SCALAR$", scalar)
        .replace("$WG$", &POOL_WORKGROUP_SIZE.to_string())
}

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
  $SPECIAL$
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
    let same_sign = (sum > 0.0 && value > 0.0) || (sum < 0.0 && value < 0.0);
    if (!is_special(sum) && !is_special(value) && same_sign &&
        abs(sum) > $MAX_FINITE$ - abs(value)) {
      overflow = true;
      break;
    }
    sum = sum + value;
    index = index + 1u;
  }
  if (!overflow) {
    output.data[column] = sum / $SCALAR$(params.pool_size);
    return;
  }
  // Recompute a weighted mean in the input precision, without a widened
  // mirror. Explicit fused arithmetic avoids an overflowing numerator;
  // backend rounding can differ from the host's separate multiply/add.
  var mean: $SCALAR$ = 0.0;
  index = 0u;
  while (index < params.pool_size) {
    let count = $SCALAR$(index + 1u);
    let previous_weight = $SCALAR$(index) / count;
    mean = fma(mean, previous_weight, input.data[base + index * params.inner_stride] / count);
    index = index + 1u;
  }
  output.data[column] = mean;
}
"#;

pub fn stable_pool_1d(precision: super::super::types::NumericPrecision) -> String {
    let (scalar, special, max_finite) = match precision {
        super::super::types::NumericPrecision::F32 => (
            "f32",
            "return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;",
            "3.4028234663852886e38",
        ),
        super::super::types::NumericPrecision::F64 => (
            "f64",
            "return (runmat_f64_high(value) & 0x7ff00000u) == 0x7ff00000u;",
            "1.7976931348623157e308",
        ),
    };
    let source = STABLE_POOL_1D
        .replace("$SCALAR$", scalar)
        .replace("$SPECIAL$", special)
        .replace("$MAX_FINITE$", max_finite)
        .replace("$WG$", &POOL_WORKGROUP_SIZE.to_string());
    if precision == super::super::types::NumericPrecision::F64 {
        super::ieee_words::inject(&source)
    } else {
        source
    }
}
