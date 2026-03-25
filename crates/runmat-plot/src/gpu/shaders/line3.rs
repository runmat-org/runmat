pub const F32: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw { data: array<f32, 12u>, };
struct Line3Params { color: vec4<f32>, count: u32, line_style: u32, _pad0: u32, _pad1: u32, };

@group(0) @binding(0) var<storage, read> buf_x: array<f32>;
@group(0) @binding(1) var<storage, read> buf_y: array<f32>;
@group(0) @binding(2) var<storage, read> buf_z: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_vertices: array<VertexRaw>;
@group(0) @binding(4) var<uniform> params: Line3Params;

fn should_draw(segment: u32, style: u32) -> bool {
  switch(style) {
    case 0u: { return true; }
    case 1u: { return (segment % 4u) < 2u; }
    case 2u: { return (segment % 4u) < 2u; }
    case 3u: { let m = segment % 6u; return (m < 2u) || (m == 3u); }
    default: { return true; }
  }
}

fn write_vertex(index: u32, pos: vec3<f32>, color: vec4<f32>) {
  var vertex: VertexRaw;
  vertex.data[0u] = pos.x; vertex.data[1u] = pos.y; vertex.data[2u] = pos.z;
  vertex.data[3u] = color.x; vertex.data[4u] = color.y; vertex.data[5u] = color.z; vertex.data[6u] = color.w;
  vertex.data[7u] = 0.0; vertex.data[8u] = 0.0; vertex.data[9u] = 1.0; vertex.data[10u] = 0.0; vertex.data[11u] = 0.0;
  out_vertices[index] = vertex;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (params.count < 2u) { return; }
  let segments = params.count - 1u;
  let idx = gid.x;
  if (idx >= segments) { return; }
  var color = params.color;
  if (!should_draw(idx, params.line_style)) { color.w = 0.0; }
  let base = idx * 2u;
  write_vertex(base + 0u, vec3<f32>(buf_x[idx], buf_y[idx], buf_z[idx]), color);
  write_vertex(base + 1u, vec3<f32>(buf_x[idx + 1u], buf_y[idx + 1u], buf_z[idx + 1u]), color);
}
"#;

pub const F64: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw { data: array<f32, 12u>, };
struct Line3Params { color: vec4<f32>, count: u32, line_style: u32, _pad0: u32, _pad1: u32, };

@group(0) @binding(0) var<storage, read> buf_x: array<f64>;
@group(0) @binding(1) var<storage, read> buf_y: array<f64>;
@group(0) @binding(2) var<storage, read> buf_z: array<f64>;
@group(0) @binding(3) var<storage, read_write> out_vertices: array<VertexRaw>;
@group(0) @binding(4) var<uniform> params: Line3Params;

fn should_draw(segment: u32, style: u32) -> bool {
  switch(style) {
    case 0u: { return true; }
    case 1u: { return (segment % 4u) < 2u; }
    case 2u: { return (segment % 4u) < 2u; }
    case 3u: { let m = segment % 6u; return (m < 2u) || (m == 3u); }
    default: { return true; }
  }
}

fn write_vertex(index: u32, pos: vec3<f32>, color: vec4<f32>) {
  var vertex: VertexRaw;
  vertex.data[0u] = pos.x; vertex.data[1u] = pos.y; vertex.data[2u] = pos.z;
  vertex.data[3u] = color.x; vertex.data[4u] = color.y; vertex.data[5u] = color.z; vertex.data[6u] = color.w;
  vertex.data[7u] = 0.0; vertex.data[8u] = 0.0; vertex.data[9u] = 1.0; vertex.data[10u] = 0.0; vertex.data[11u] = 0.0;
  out_vertices[index] = vertex;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (params.count < 2u) { return; }
  let segments = params.count - 1u;
  let idx = gid.x;
  if (idx >= segments) { return; }
  var color = params.color;
  if (!should_draw(idx, params.line_style)) { color.w = 0.0; }
  let base = idx * 2u;
  write_vertex(base + 0u, vec3<f32>(f32(buf_x[idx]), f32(buf_y[idx]), f32(buf_z[idx])), color);
  write_vertex(base + 1u, vec3<f32>(f32(buf_x[idx + 1u]), f32(buf_y[idx + 1u]), f32(buf_z[idx + 1u])), color);
}
"#;
