// Image shader with direct data->viewport mapping

struct DirectUniforms {
  data_min: vec2<f32>,
  data_max: vec2<f32>,
  viewport_min: vec2<f32>,
  viewport_max: vec2<f32>,
  viewport_px: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uDirect: DirectUniforms;
@group(1) @binding(0) var imgSampler: sampler;
@group(1) @binding(1) var imgTex: texture_2d<f32>;

struct VSIn {
  @location(0) position: vec3<f32>,
  @location(3) uv: vec2<f32>,
}

struct VSOut {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(input: VSIn) -> VSOut {
  var out: VSOut;
  let data_range = uDirect.data_max - uDirect.data_min;
  let viewport_range = uDirect.viewport_max - uDirect.viewport_min;
  let normalized = (input.position.xy - uDirect.data_min) / data_range;
  let ndc = uDirect.viewport_min + normalized * viewport_range;
  out.clip_position = vec4<f32>(ndc, 0.0, 1.0);
  out.uv = input.uv;
  return out;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
  let color = textureSample(imgTex, imgSampler, input.uv);
  return color;
}


