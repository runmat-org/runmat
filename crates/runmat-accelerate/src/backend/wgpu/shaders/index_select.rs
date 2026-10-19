/// Bit-preserving linear indexing. Each logical element occupies one or more
/// physical u32 words (including native integer packing and complex storage);
/// the kernels never interpret or canonicalize the numeric payload.
pub const LINEAR_SCATTER_WORDS_SHADER: &str = r#"
struct Words { data: array<u32> };
struct Indices { data: array<u32> };
struct Params {
  count: u32,
  lane_factor: u32,
  _pad1: u32,
  _pad2: u32,
};
@group(0) @binding(0) var<storage, read_write> Target: Words;
@group(0) @binding(1) var<storage, read> Values: Words;
@group(0) @binding(2) var<storage, read> Ind: Indices;
@group(0) @binding(3) var<uniform> Pm: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= Pm.count) { return; }
  let dst = Ind.data[idx] * Pm.lane_factor;
  let src = idx * Pm.lane_factor;
  for (var lane = 0u; lane < Pm.lane_factor; lane = lane + 1u) {
    Target.data[dst + lane] = Values.data[src + lane];
  }
}
"#;

pub const LINEAR_GATHER_WORDS_SHADER: &str = r#"
struct Words { data: array<u32> };
struct Indices { data: array<u32> };
struct Params {
  count: u32,
  lane_factor: u32,
  _pad1: u32,
  _pad2: u32,
};
@group(0) @binding(0) var<storage, read> Source: Words;
@group(0) @binding(1) var<storage, read> Ind: Indices;
@group(0) @binding(2) var<storage, read_write> Out: Words;
@group(0) @binding(3) var<uniform> Pm: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= Pm.count) { return; }
  let src = Ind.data[idx] * Pm.lane_factor;
  let dst = idx * Pm.lane_factor;
  for (var lane = 0u; lane < Pm.lane_factor; lane = lane + 1u) {
    Out.data[dst + lane] = Source.data[src + lane];
  }
}
"#;
