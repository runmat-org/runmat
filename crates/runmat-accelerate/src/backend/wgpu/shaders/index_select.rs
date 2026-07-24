pub const LINEAR_GATHER_SHADER_F64: &str = r#"
struct Tensor {
  data: array<f64>
};
struct Indices {
  data: array<u32>
};
struct Params {
  count: u32,
  lane_factor: u32,
  _pad1: u32,
  _pad2: u32,
};
@group(0) @binding(0) var<storage, read> Source: Tensor;
@group(0) @binding(1) var<storage, read> Ind: Indices;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> Pm: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= Pm.count) {
    return;
  }
  let lin = Ind.data[idx];
  let src = lin * Pm.lane_factor;
  let dst = idx * Pm.lane_factor;
  var lane = 0u;
  loop {
    if (lane >= Pm.lane_factor) {
      break;
    }
    Out.data[dst + lane] = Source.data[src + lane];
    lane = lane + 1u;
  }
}
"#;

pub const LINEAR_GATHER_SHADER_F32: &str = r#"
struct Tensor {
  data: array<f32>
};
struct Indices {
  data: array<u32>
};
struct Params {
  count: u32,
  lane_factor: u32,
  _pad1: u32,
  _pad2: u32,
};
@group(0) @binding(0) var<storage, read> Source: Tensor;
@group(0) @binding(1) var<storage, read> Ind: Indices;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> Pm: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= Pm.count) {
    return;
  }
  let lin = Ind.data[idx];
  let src = lin * Pm.lane_factor;
  let dst = idx * Pm.lane_factor;
  var lane = 0u;
  loop {
    if (lane >= Pm.lane_factor) {
      break;
    }
    Out.data[dst + lane] = Source.data[src + lane];
    lane = lane + 1u;
  }
}
"#;

pub const LINEAR_SCATTER_SHADER_F64: &str = r#"
struct Tensor {
  data: array<f64>
};
struct Indices {
  data: array<u32>
};
struct Params {
  count: u32,
  lane_factor: u32,
  _pad1: u32,
  _pad2: u32,
};
@group(0) @binding(0) var<storage, read_write> Target: Tensor;
@group(0) @binding(1) var<storage, read> Values: Tensor;
@group(0) @binding(2) var<storage, read> Ind: Indices;
@group(0) @binding(3) var<uniform> Pm: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= Pm.count) {
    return;
  }
  let lin = Ind.data[idx];
  let dst = lin * Pm.lane_factor;
  let src = idx * Pm.lane_factor;
  var lane = 0u;
  loop {
    if (lane >= Pm.lane_factor) {
      break;
    }
    Target.data[dst + lane] = Values.data[src + lane];
    lane = lane + 1u;
  }
}
"#;

pub const LINEAR_SCATTER_SHADER_F32: &str = r#"
struct Tensor {
  data: array<f32>
};
struct Indices {
  data: array<u32>
};
struct Params {
  count: u32,
  lane_factor: u32,
  _pad1: u32,
  _pad2: u32,
};
@group(0) @binding(0) var<storage, read_write> Target: Tensor;
@group(0) @binding(1) var<storage, read> Values: Tensor;
@group(0) @binding(2) var<storage, read> Ind: Indices;
@group(0) @binding(3) var<uniform> Pm: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= Pm.count) {
    return;
  }
  let lin = Ind.data[idx];
  let dst = lin * Pm.lane_factor;
  let src = idx * Pm.lane_factor;
  var lane = 0u;
  loop {
    if (lane >= Pm.lane_factor) {
      break;
    }
    Target.data[dst + lane] = Values.data[src + lane];
    lane = lane + 1u;
  }
}
"#;
