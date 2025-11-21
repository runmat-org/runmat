// NaN handling shaders: map NaN to zero, and emit numeric not-NaN mask (1.0/0.0)

pub const NAN_TO_ZERO_SHADER_F64: &str = r#"struct Tensor { data: array<f64> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
fn isnan64(x: f64) -> bool { return x != x; }
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let v = input0.data[idx];
  output.data[idx] = select(v, f64(0.0), isnan64(v));
}
"#;

pub const NAN_TO_ZERO_SHADER_F32: &str = r#"struct Tensor { data: array<f32> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
fn isnan32(x: f32) -> bool { return x != x; }
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let v = input0.data[idx];
  output.data[idx] = select(v, 0.0, isnan32(v));
}
"#;

pub const NOT_NAN_MASK_SHADER_F64: &str = r#"struct Tensor { data: array<f64> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
fn isnan64(x: f64) -> bool { return x != x; }
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let v = input0.data[idx];
  output.data[idx] = select(f64(0.0), f64(1.0), !isnan64(v));
}
"#;

pub const NOT_NAN_MASK_SHADER_F32: &str = r#"struct Tensor { data: array<f32> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
fn isnan32(x: f32) -> bool { return x != x; }
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let v = input0.data[idx];
  output.data[idx] = select(0.0, 1.0, !isnan32(v));
}
"#;
