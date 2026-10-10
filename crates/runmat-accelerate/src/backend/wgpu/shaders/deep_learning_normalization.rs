//! Stable resident layer normalization. A statistics invocation owns one
//! normalization group; a second pass applies channel-affine parameters.

pub const NORMALIZATION_WORKGROUP_SIZE: u32 = 64;

const COMMON: &str = r#"
struct Tensor { data: array<$SCALAR$>, };
struct Params {
  shape0: u32, shape1: u32, shape2: u32, shape3: u32,
  rank: u32, reduce_mask: u32, group_count: u32, group_size: u32,
  channel_axis: u32, total: u32, chunk_offset: u32, chunk_count: u32,
  epsilon_low: u32, epsilon_high: u32,
};
@group(0) @binding($PARAM_BINDING$) var<uniform> params: Params;

fn extent(axis: u32) -> u32 {
  switch axis {
    case 0u: { return params.shape0; }
    case 1u: { return params.shape1; }
    case 2u: { return params.shape2; }
    default: { return params.shape3; }
  }
}

fn reduced(axis: u32) -> bool {
  return (params.reduce_mask & (1u << axis)) != 0u;
}

fn physical_index(group_index: u32, sample_index: u32) -> u32 {
  var group = group_index;
  var sample = sample_index;
  var stride = 1u;
  var linear = 0u;
  for (var axis = 0u; axis < params.rank; axis = axis + 1u) {
    let size = extent(axis);
    var coordinate: u32;
    if (reduced(axis)) {
      coordinate = sample % size;
      sample = sample / size;
    } else {
      coordinate = group % size;
      group = group / size;
    }
    linear = linear + coordinate * stride;
    stride = stride * size;
  }
  return linear;
}

fn group_and_channel(index: u32) -> vec2<u32> {
  var linear = index;
  var group_stride = 1u;
  var group = 0u;
  var channel = 0u;
  for (var axis = 0u; axis < params.rank; axis = axis + 1u) {
    let size = extent(axis);
    let coordinate = linear % size;
    linear = linear / size;
    if (!reduced(axis)) {
      group = group + coordinate * group_stride;
      group_stride = group_stride * size;
    }
    if (axis == params.channel_axis) { channel = coordinate; }
  }
  return vec2<u32>(group, channel);
}

fn epsilon() -> $SCALAR$ { $EPSILON$ }
fn quiet_nan() -> $SCALAR$ { $NAN$ }
fn is_special(value: $SCALAR$) -> bool { $SPECIAL$ }

// The divisor is finite, positive and at least sqrt(epsilon). Splitting
// its binary exponent keeps both scale factors normal. Explicit fused
// operations prevent a reciprocal of a large divisor from becoming a
// subnormal intermediate even when the quotient itself is normal.
fn scaled_ratio(value: $SCALAR$, divisor: $SCALAR$) -> $SCALAR$ {
  $RATIO_SCALE$
  let numerator = fma(fma(value, first, $SCALAR$(0.0)), second, $SCALAR$(0.0));
  let denominator = fma(fma(divisor, first, $SCALAR$(0.0)), second, $SCALAR$(0.0));
  return numerator / denominator;
}
"#;

const STATISTICS: &str = r#"
@group(0) @binding(0) var<storage, read> input: Tensor;
@group(0) @binding(1) var<storage, read_write> statistics: Tensor;
$COMMON$

@compute @workgroup_size($WG$)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let local = invocation.x;
  if (local >= params.chunk_count) { return; }
  let group = params.chunk_offset + local;
  let sqrt_epsilon = sqrt(epsilon());
  var maximum = sqrt_epsilon;
  var special = false;
  for (var sample = 0u; sample < params.group_size; sample = sample + 1u) {
    let value = input.data[physical_index(group, sample)];
    special = special || is_special(value);
    maximum = max(maximum, abs(value));
  }
  if (special) {
    statistics.data[3u * group] = maximum;
    statistics.data[3u * group + 1u] = quiet_nan();
    statistics.data[3u * group + 2u] = quiet_nan();
    return;
  }
  let count = $SCALAR$(params.group_size);
  var mean = $SCALAR$(0.0);
  for (var sample = 0u; sample < params.group_size; sample = sample + 1u) {
    mean = mean + scaled_ratio(input.data[physical_index(group, sample)], maximum);
  }
  mean = mean / count;
  var variance = $SCALAR$(0.0);
  for (var sample = 0u; sample < params.group_size; sample = sample + 1u) {
    let delta = scaled_ratio(input.data[physical_index(group, sample)], maximum) - mean;
    variance = variance + delta * delta;
  }
  let epsilon_ratio = scaled_ratio(sqrt_epsilon, maximum);
  let denominator = sqrt(variance / count + epsilon_ratio * epsilon_ratio);
  var inv_std_scaled = $SCALAR$(0.0);
  if (denominator != $SCALAR$(0.0)) {
    inv_std_scaled = $SCALAR$(1.0) / denominator;
  }
  statistics.data[3u * group] = maximum;
  statistics.data[3u * group + 1u] = mean;
  statistics.data[3u * group + 2u] = inv_std_scaled;
}
"#;

const AFFINE: &str = r#"
@group(0) @binding(0) var<storage, read> input: Tensor;
@group(0) @binding(1) var<storage, read> scale: Tensor;
@group(0) @binding(2) var<storage, read> offset: Tensor;
@group(0) @binding(3) var<storage, read> statistics: Tensor;
@group(0) @binding(4) var<storage, read_write> output: Tensor;
$COMMON$

@compute @workgroup_size($WG$)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let local = invocation.x;
  if (local >= params.chunk_count) { return; }
  let index = params.chunk_offset + local;
  let position = group_and_channel(index);
  let base = 3u * position.x;
  if (is_special(statistics.data[base]) || is_special(statistics.data[base + 1u])) {
    output.data[index] = quiet_nan();
    return;
  }
  let normalized = (scaled_ratio(input.data[index], statistics.data[base]) - statistics.data[base + 1u]) * statistics.data[base + 2u];
  output.data[index] = normalized * scale.data[position.y] + offset.data[position.y];
}
"#;

fn common(precision: super::super::types::NumericPrecision, binding: u32) -> String {
    let (scalar, epsilon, nan, special, ratio_scale) = match precision {
        super::super::types::NumericPrecision::F32 => (
            "f32",
            "return bitcast<f32>(params.epsilon_low);",
            "let nan_bits: u32 = 0x7fc00000u; return bitcast<f32>(nan_bits);",
            "return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;",
            r#"let exponent = i32((bitcast<u32>(divisor) >> 23u) & 0xffu) - 127;
  let half = exponent / 2;
  let first = bitcast<f32>(u32(127 - half) << 23u);
  let second = bitcast<f32>(u32(127 - (exponent - half)) << 23u);"#,
        ),
        super::super::types::NumericPrecision::F64 => (
            "f64",
            "return runmat_f64_from_words(params.epsilon_low, params.epsilon_high);",
            "return runmat_f64_from_words(0u, 0x7ff80000u);",
            "return (runmat_f64_high(value) & 0x7ff00000u) == 0x7ff00000u;",
            r#"let exponent = i32((runmat_f64_high(divisor) >> 20u) & 0x7ffu) - 1023;
  let half = exponent / 2;
  let first = runmat_f64_from_words(0u, u32(1023 - half) << 20u);
  let second = runmat_f64_from_words(0u, u32(1023 - (exponent - half)) << 20u);"#,
        ),
    };
    COMMON
        .replace("$SCALAR$", scalar)
        .replace("$PARAM_BINDING$", &binding.to_string())
        .replace("$EPSILON$", epsilon)
        .replace("$NAN$", nan)
        .replace("$SPECIAL$", special)
        .replace("$RATIO_SCALE$", ratio_scale)
}

pub fn statistics(precision: super::super::types::NumericPrecision) -> String {
    let source = STATISTICS
        .replace("$COMMON$", &common(precision, 2))
        .replace("$SCALAR$", precision_scalar(precision))
        .replace("$WG$", &NORMALIZATION_WORKGROUP_SIZE.to_string());
    if precision == super::super::types::NumericPrecision::F64 {
        super::ieee_words::inject(&source)
    } else {
        source
    }
}

pub fn affine(precision: super::super::types::NumericPrecision) -> String {
    let source = AFFINE
        .replace("$COMMON$", &common(precision, 5))
        .replace("$SCALAR$", precision_scalar(precision))
        .replace("$WG$", &NORMALIZATION_WORKGROUP_SIZE.to_string());
    if precision == super::super::types::NumericPrecision::F64 {
        super::ieee_words::inject(&source)
    } else {
        source
    }
}

fn precision_scalar(precision: super::super::types::NumericPrecision) -> &'static str {
    match precision {
        super::super::types::NumericPrecision::F32 => "f32",
        super::super::types::NumericPrecision::F64 => "f64",
    }
}
