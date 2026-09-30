//! Resident deep-learning kernels. These do not change generic reductions.

const STABLE_POOL_1D: &str = r#"
struct Tensor { data: array<$SCALAR$>, };
struct Params {
  pool_size: u32,
  output_count: u32,
  inner_stride: u32,
  output_offset: u32,
};
@group(0) @binding(0) var<storage, read> input: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> maxima: array<$SCALAR$, 256>;
var<workgroup> specials: array<$SCALAR$, 256>;
var<workgroup> special_counts: array<u32, 256>;
var<workgroup> sums: array<$SCALAR$, 256>;

fn is_special(value: $SCALAR$) -> bool {
  // x - x is NaN exactly for infinities; x != x detects NaNs.
  return value != value || (value - value) != (value - value);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lane: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let column = group.x + params.output_offset;
  if (column >= params.output_count) { return; }
  let inner = column % params.inner_stride;
  let outer = column / params.inner_stride;
  let base = inner + outer * params.inner_stride * params.pool_size;
  var local_max: $SCALAR$ = 0.0;
  var local_special: $SCALAR$ = 0.0;
  var local_special_count: u32 = 0u;
  var index = lane.x;
  while (index < params.pool_size) {
    let value = input.data[base + index * params.inner_stride];
    if (is_special(value)) {
      local_special = local_special + value;
      local_special_count = local_special_count + 1u;
    } else {
      local_max = max(local_max, abs(value));
    }
    index = index + 256u;
  }
  maxima[lane.x] = local_max;
  specials[lane.x] = local_special;
  special_counts[lane.x] = local_special_count;
  workgroupBarrier();
  var offset = 128u;
  loop {
    if (offset == 0u) { break; }
    if (lane.x < offset) {
      maxima[lane.x] = max(maxima[lane.x], maxima[lane.x + offset]);
      specials[lane.x] = specials[lane.x] + specials[lane.x + offset];
      special_counts[lane.x] = special_counts[lane.x] + special_counts[lane.x + offset];
    }
    workgroupBarrier();
    offset = offset / 2u;
  }
  if (special_counts[0] != 0u) {
    if (lane.x == 0u) {
      output.data[column] = specials[0] / $SCALAR$(params.pool_size);
    }
    return;
  }
  let scale = maxima[0];
  if (scale == 0.0) {
    if (lane.x == 0u) { output.data[column] = 0.0; }
    return;
  }
  var local_sum: $SCALAR$ = 0.0;
  index = lane.x;
  while (index < params.pool_size) {
    local_sum = local_sum + input.data[base + index * params.inner_stride] / scale;
    index = index + 256u;
  }
  sums[lane.x] = local_sum;
  workgroupBarrier();
  offset = 128u;
  loop {
    if (offset == 0u) { break; }
    if (lane.x < offset) {
      sums[lane.x] = sums[lane.x] + sums[lane.x + offset];
    }
    workgroupBarrier();
    offset = offset / 2u;
  }
  if (lane.x == 0u) {
    output.data[column] = (sums[0] / $SCALAR$(params.pool_size)) * scale;
  }
}
"#;

pub fn stable_pool_1d(precision: super::super::types::NumericPrecision) -> String {
    let scalar = match precision {
        super::super::types::NumericPrecision::F32 => "f32",
        super::super::types::NumericPrecision::F64 => "f64",
    };
    STABLE_POOL_1D.replace("$SCALAR$", scalar)
}
