//! Native-precision, learned multichannel 1-D cross-correlation. Reverse is
//! gather-owned: one invocation writes each gradient, with no float atomics.

pub const CONV_WORKGROUP_SIZE: u32 = 64;

const COMMON: &str = r#"
struct Tensor { data: array<$SCALAR$>, };
struct Params {
  input_shape: vec4<u32>, output_shape: vec4<u32>,
  input_strides: vec4<u32>, output_strides: vec4<u32>,
  rank: u32, span_axis: u32, channel_axis: u32, batch_axis: u32,
  time_axis: u32, input_len: u32, output_len: u32, input_channels: u32,
  output_channels: u32, filter_size: u32, batch_size: u32, time_size: u32,
  stride: u32, dilation: u32, padding_left: u32, padding_mode: u32,
  $PADDING_FIELDS$ chunk_offset: u32, chunk_count: u32,
  reflection_step: u32, reflection_inverse: u32,
  _reserved0: u32, _reserved1: u32,
};
@group(0) @binding($PARAM_BINDING$) var<uniform> params: Params;

fn input_offset(span: u32, channel: u32, batch: u32, time: u32) -> u32 {
  var index = span * params.input_strides[params.span_axis] +
      channel * params.input_strides[params.channel_axis];
  if (params.batch_axis != 0xffffffffu) {
    index = index + batch * params.input_strides[params.batch_axis];
  }
  if (params.time_axis != 0xffffffffu) {
    index = index + time * params.input_strides[params.time_axis];
  }
  return index;
}

fn output_offset(span: u32, channel: u32, batch: u32, time: u32) -> u32 {
  var index = span * params.output_strides[params.span_axis] +
      channel * params.output_strides[params.channel_axis];
  if (params.batch_axis != 0xffffffffu) {
    index = index + batch * params.output_strides[params.batch_axis];
  }
  if (params.time_axis != 0xffffffffu) {
    index = index + time * params.output_strides[params.time_axis];
  }
  return index;
}

fn input_coordinate(index: u32, axis: u32) -> u32 {
  return (index / params.input_strides[axis]) % params.input_shape[axis];
}
fn output_coordinate(index: u32, axis: u32) -> u32 {
  return (index / params.output_strides[axis]) % params.output_shape[axis];
}
fn weight_offset(tap: u32, in_channel: u32, out_channel: u32,
                 filter_size: u32, input_channels: u32) -> u32 {
  return tap + filter_size * (in_channel + input_channels * out_channel);
}
fn padding_value() -> $SCALAR$ { $PADDING_VALUE$ }

fn mapped_span(raw: i32) -> i32 {
  let len = i32(params.input_len);
  if (raw >= 0 && raw < len) { return raw; }
  switch params.padding_mode {
    case 0u: { return -1; }
    case 1u: { return clamp(raw, 0, len - 1); }
    case 2u: {
      let period = 2 * len;
      let index = i32(runmat_i32_positive_period_residue(raw, u32(period)));
      return select(period - index - 1, index, index < len);
    }
    default: {
      let period = 2 * (len - 1);
      let index = i32(runmat_i32_positive_period_residue(raw, u32(period)));
      return select(period - index, index, index < len);
    }
  }
}

"#;

const SOURCE_VALUE: &str = r#"
fn source_value(span: i32, channel: u32, batch: u32, time: u32) -> $SCALAR$ {
  if (span < 0) { return padding_value(); }
  return input.data[input_offset(u32(span), channel, batch, time)];
}
"#;

const FORWARD: &str = r#"
@group(0) @binding(0) var<storage, read> input: Tensor;
@group(0) @binding(1) var<storage, read> weights: Tensor;
@group(0) @binding(2) var<storage, read> bias: Tensor;
@group(0) @binding(3) var<storage, read_write> output: Tensor;
$COMMON$
$SOURCE_VALUE$
@compute @workgroup_size($WG$)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let local = invocation.x;
  if (local >= params.chunk_count) { return; }
  let index = params.chunk_offset + local;
  let span = output_coordinate(index, params.span_axis);
  let out_channel = output_coordinate(index, params.channel_axis);
  var batch = 0u;
  var time = 0u;
  if (params.batch_axis != 0xffffffffu) {
    batch = output_coordinate(index, params.batch_axis);
  }
  if (params.time_axis != 0xffffffffu) {
    time = output_coordinate(index, params.time_axis);
  }
  var sum = bias.data[out_channel];
  for (var in_channel = 0u; in_channel < params.input_channels; in_channel = in_channel + 1u) {
    for (var tap = 0u; tap < params.filter_size; tap = tap + 1u) {
      let raw = i32(span * params.stride + tap * params.dilation) - i32(params.padding_left);
      let sample = source_value(mapped_span(raw), in_channel, batch, time);
      let weight_index = weight_offset(tap, in_channel, out_channel,
                                       params.filter_size, params.input_channels);
      sum = sum + sample * weights.data[weight_index];
    }
  }
  output.data[index] = sum;
}
"#;

const ADJOINT_WEIGHT: &str = r#"
@group(0) @binding(0) var<storage, read> input: Tensor;
@group(0) @binding(1) var<storage, read> upstream: Tensor;
@group(0) @binding(2) var<storage, read_write> dweights: Tensor;
$COMMON$
$SOURCE_VALUE$
@compute @workgroup_size($WG$)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let local = invocation.x;
  if (local >= params.chunk_count) { return; }
  let index = params.chunk_offset + local;
  let tap = index % params.filter_size;
  let in_channel = (index / params.filter_size) % params.input_channels;
  let out_channel = index / (params.filter_size * params.input_channels);
  var sum = $SCALAR$(0.0);
  for (var time = 0u; time < params.time_size; time = time + 1u) {
    for (var batch = 0u; batch < params.batch_size; batch = batch + 1u) {
      for (var span = 0u; span < params.output_len; span = span + 1u) {
        let raw = i32(span * params.stride + tap * params.dilation) - i32(params.padding_left);
        let sample = source_value(mapped_span(raw), in_channel, batch, time);
        sum = sum + upstream.data[output_offset(span, out_channel, batch, time)] * sample;
      }
    }
  }
  dweights.data[index] = sum;
}
"#;

const ADJOINT_BIAS: &str = r#"
@group(0) @binding(0) var<storage, read> upstream: Tensor;
@group(0) @binding(1) var<storage, read_write> dbias: Tensor;
$COMMON$
@compute @workgroup_size($WG$)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let local = invocation.x;
  if (local >= params.chunk_count) { return; }
  let out_channel = params.chunk_offset + local;
  var sum = $SCALAR$(0.0);
  for (var time = 0u; time < params.time_size; time = time + 1u) {
    for (var batch = 0u; batch < params.batch_size; batch = batch + 1u) {
      for (var span = 0u; span < params.output_len; span = span + 1u) {
        sum = sum + upstream.data[output_offset(span, out_channel, batch, time)];
      }
    }
  }
  dbias.data[out_channel] = sum;
}
"#;

const ADJOINT_INPUT: &str = r#"
@group(0) @binding(0) var<storage, read> weights: Tensor;
@group(0) @binding(1) var<storage, read> upstream: Tensor;
@group(0) @binding(2) var<storage, read_write> dinput: Tensor;
$COMMON$

// One raw coordinate contributes at most one output position for a fixed
// filter after the stride equation is solved. Boundary modes enumerate only
// inverse images of the requested input span, never every output position.
fn coordinate_contribution(raw: i32, tap: u32, in_channel: u32,
                           batch: u32, time: u32) -> $SCALAR$ {
  let numerator = raw + i32(params.padding_left) - i32(tap * params.dilation);
  if (numerator < 0 || numerator % i32(params.stride) != 0) {
    return $SCALAR$(0.0);
  }
  let span = u32(numerator / i32(params.stride));
  if (span >= params.output_len) { return $SCALAR$(0.0); }
  var sum = $SCALAR$(0.0);
  for (var out_channel = 0u; out_channel < params.output_channels;
       out_channel = out_channel + 1u) {
    sum = sum + upstream.data[output_offset(span, out_channel, batch, time)] *
        weights.data[weight_offset(tap, in_channel, out_channel,
                                   params.filter_size, params.input_channels)];
  }
  return sum;
}

// All modular operands are below 2^30 after physical admission. The largest
// addition below is therefore below 2^31, so u32 arithmetic does not wrap.
fn multiply_mod(left: u32, right: u32, modulus: u32) -> u32 {
  if (modulus == 1u) { return 0u; }
  var factor = left % modulus;
  var remaining = right;
  var product = 0u;
  while (remaining != 0u) {
    if ((remaining & 1u) != 0u) {
      product = (product + factor) % modulus;
    }
    remaining = remaining >> 1u;
    factor = (factor + factor) % modulus;
  }
  return product;
}

fn residue_contribution(residue: i32, period: i32,
                        tap: u32, in_channel: u32, batch: u32, time: u32) -> $SCALAR$ {
  let period_u = u32(period);
  let step = params.reflection_step;
  let divisor = period_u / step;
  let congruence_target = residue + i32(params.padding_left) - i32(tap * params.dilation);
  let positive_target = runmat_i32_positive_period_residue(congruence_target, period_u);
  if (positive_target % divisor != 0u) { return $SCALAR$(0.0); }
  let first = multiply_mod(
      positive_target / divisor,
      params.reflection_inverse,
      step);
  var span = first;
  var sum = $SCALAR$(0.0);
  while (span < params.output_len) {
    let raw = i32(span * params.stride + tap * params.dilation) -
        i32(params.padding_left);
    sum = sum + coordinate_contribution(raw, tap, in_channel, batch, time);
    if (step > params.output_len - 1u - span) { break; }
    span = span + step;
  }
  return sum;
}

// Replication maps a contiguous raw interval to one boundary sample. Solve
// the stride congruence first, so runtime grows with contributing output
// positions rather than with the (possibly enormous) padding width.
fn replicated_interval(low: i32, high: i32, tap: u32, in_channel: u32,
                       batch: u32, time: u32) -> $SCALAR$ {
  if (high < low || params.output_len == 0u) { return $SCALAR$(0.0); }
  let stride = i32(params.stride);
  let base = i32(params.padding_left) - i32(tap * params.dilation);
  let low_numerator = low + base;
  let high_numerator = high + base;
  if (high_numerator < 0) { return $SCALAR$(0.0); }
  var first = 0u;
  if (low_numerator > 0) {
    first = u32((low_numerator - 1) / stride + 1);
  }
  let last = min(u32(high_numerator / stride), params.output_len - 1u);
  if (first > last) { return $SCALAR$(0.0); }
  var span = first;
  var sum = $SCALAR$(0.0);
  loop {
    let raw = i32(span * params.stride + tap * params.dilation) -
        i32(params.padding_left);
    sum = sum + coordinate_contribution(raw, tap, in_channel, batch, time);
    if (span == last) { break; }
    span = span + 1u;
  }
  return sum;
}

@compute @workgroup_size($WG$)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let local = invocation.x;
  if (local >= params.chunk_count) { return; }
  let index = params.chunk_offset + local;
  let input_span = input_coordinate(index, params.span_axis);
  let in_channel = input_coordinate(index, params.channel_axis);
  var batch = 0u;
  var time = 0u;
  if (params.batch_axis != 0xffffffffu) {
    batch = input_coordinate(index, params.batch_axis);
  }
  if (params.time_axis != 0xffffffffu) {
    time = input_coordinate(index, params.time_axis);
  }
  var sum = $SCALAR$(0.0);
  if (params.output_len > 0u) {
    let minimum = -i32(params.padding_left);
    let maximum = i32((params.output_len - 1u) * params.stride +
                      (params.filter_size - 1u) * params.dilation) -
                  i32(params.padding_left);
    for (var tap = 0u; tap < params.filter_size; tap = tap + 1u) {
      let source = i32(input_span);
      switch params.padding_mode {
        case 0u: {
          sum = sum + coordinate_contribution(source, tap, in_channel, batch, time);
        }
        case 1u: {
          if (params.input_len == 1u) {
            sum = sum + replicated_interval(minimum, maximum, tap,
                                            in_channel, batch, time);
          } else if (input_span == 0u) {
            sum = sum + replicated_interval(minimum, min(0, maximum), tap,
                                            in_channel, batch, time);
          } else if (input_span + 1u == params.input_len) {
            sum = sum + replicated_interval(max(source, minimum), maximum, tap,
                                            in_channel, batch, time);
          } else {
            sum = sum + coordinate_contribution(source, tap, in_channel, batch, time);
          }
        }
        case 2u: {
          let period = 2 * i32(params.input_len);
          sum = sum + residue_contribution(source, period,
                                           tap, in_channel, batch, time);
          sum = sum + residue_contribution(period - source - 1, period,
                                           tap, in_channel, batch, time);
        }
        default: {
          if (params.input_len == 1u) {
            sum = sum + coordinate_contribution(source, tap, in_channel, batch, time);
          } else {
            let period = 2 * (i32(params.input_len) - 1);
            sum = sum + residue_contribution(source, period,
                                             tap, in_channel, batch, time);
            if (input_span != 0u && input_span + 1u != params.input_len) {
              sum = sum + residue_contribution(period - source, period,
                                               tap, in_channel, batch, time);
            }
          }
        }
      }
    }
  }
  dinput.data[index] = sum;
}
"#;

fn shader(
    source: &str,
    precision: super::super::types::NumericPrecision,
    parameter_binding: u32,
) -> String {
    let (scalar, padding_fields, padding_value) = match precision {
        super::super::types::NumericPrecision::F32 => (
            "f32",
            "padding_low: u32, padding_high: u32,",
            "return bitcast<f32>(params.padding_low);",
        ),
        super::super::types::NumericPrecision::F64 => (
            "f64",
            "padding_scalar: f64,",
            "return params.padding_scalar;",
        ),
    };
    let common = COMMON
        .replace("$SCALAR$", scalar)
        .replace("$PARAM_BINDING$", &parameter_binding.to_string())
        .replace("$PADDING_FIELDS$", padding_fields)
        .replace("$PADDING_VALUE$", padding_value);
    let mut rendered = source
        .replace("$COMMON$", &common)
        .replace("$SOURCE_VALUE$", &SOURCE_VALUE.replace("$SCALAR$", scalar))
        .replace("$SCALAR$", scalar)
        .replace("$WG$", &CONV_WORKGROUP_SIZE.to_string());
    rendered.push_str(crate::wgsl::integer::POSITIVE_PERIOD_RESIDUE);
    rendered
}

pub fn forward(precision: super::super::types::NumericPrecision) -> String {
    shader(FORWARD, precision, 4)
}
pub fn adjoint_weight(precision: super::super::types::NumericPrecision) -> String {
    shader(ADJOINT_WEIGHT, precision, 3)
}
pub fn adjoint_bias(precision: super::super::types::NumericPrecision) -> String {
    shader(ADJOINT_BIAS, precision, 2)
}
pub fn adjoint_input(precision: super::super::types::NumericPrecision) -> String {
    shader(ADJOINT_INPUT, precision, 3)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use wgpu::naga;

    #[test]
    fn learned_convolution_shaders_parse_and_validate() {
        for precision in [
            super::super::super::types::NumericPrecision::F32,
            super::super::super::types::NumericPrecision::F64,
        ] {
            for (name, source) in [
                ("forward", forward(precision)),
                ("adjoint input", adjoint_input(precision)),
                ("adjoint weights", adjoint_weight(precision)),
                ("adjoint bias", adjoint_bias(precision)),
            ] {
                let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
                    panic!("{precision:?} {name}: {}", error.emit_to_string(&source))
                });
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap_or_else(|error| panic!("{precision:?} {name}: {error:?}"));
            }
        }
    }
}
