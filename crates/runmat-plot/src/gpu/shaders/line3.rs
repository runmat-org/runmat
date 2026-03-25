pub const F32: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw { data: array<f32, 12u>, };
struct Line3Params { color: vec4<f32>, count: u32, half_width_data: f32, line_style: u32, thick: u32, _pad: u32, };

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

fn write_line_vertices(base: u32, p0: vec3<f32>, p1: vec3<f32>, color: vec4<f32>) {
  write_vertex(base + 0u, p0, color);
  write_vertex(base + 1u, p1, color);
}

fn write_thick_vertices(base: u32, p0: vec3<f32>, p1: vec3<f32>, color: vec4<f32>, half_width: f32) {
  var dir = normalize(p1 - p0);
  var normal = cross(dir, vec3<f32>(0.0, 0.0, 1.0));
  if (length(normal) < 0.0001) {
    normal = cross(dir, vec3<f32>(1.0, 0.0, 0.0));
  }
  normal = normalize(normal) * half_width;
  let v0 = p0 + normal;
  let v1 = p1 + normal;
  let v2 = p1 - normal;
  let v3 = p0 - normal;
  write_vertex(base + 0u, v0, color);
  write_vertex(base + 1u, v1, color);
  write_vertex(base + 2u, v2, color);
  write_vertex(base + 3u, v0, color);
  write_vertex(base + 4u, v2, color);
  write_vertex(base + 5u, v3, color);
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
  let p0 = vec3<f32>(buf_x[idx], buf_y[idx], buf_z[idx]);
  let p1 = vec3<f32>(buf_x[idx + 1u], buf_y[idx + 1u], buf_z[idx + 1u]);
  if (params.thick != 0u) {
    let base = idx * 6u;
    write_thick_vertices(base, p0, p1, color, params.half_width_data);
  } else {
    let base = idx * 2u;
    write_line_vertices(base, p0, p1, color);
  }
}
"#;

pub const F64: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw { data: array<f32, 12u>, };
struct Line3Params { color: vec4<f32>, count: u32, half_width_data: f32, line_style: u32, thick: u32, _pad: u32, };

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

fn write_line_vertices(base: u32, p0: vec3<f32>, p1: vec3<f32>, color: vec4<f32>) {
  write_vertex(base + 0u, p0, color);
  write_vertex(base + 1u, p1, color);
}

fn write_thick_vertices(base: u32, p0: vec3<f32>, p1: vec3<f32>, color: vec4<f32>, half_width: f32) {
  var dir = normalize(p1 - p0);
  var normal = cross(dir, vec3<f32>(0.0, 0.0, 1.0));
  if (length(normal) < 0.0001) {
    normal = cross(dir, vec3<f32>(1.0, 0.0, 0.0));
  }
  normal = normalize(normal) * half_width;
  let v0 = p0 + normal;
  let v1 = p1 + normal;
  let v2 = p1 - normal;
  let v3 = p0 - normal;
  write_vertex(base + 0u, v0, color);
  write_vertex(base + 1u, v1, color);
  write_vertex(base + 2u, v2, color);
  write_vertex(base + 3u, v0, color);
  write_vertex(base + 4u, v2, color);
  write_vertex(base + 5u, v3, color);
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
  let p0 = vec3<f32>(f32(buf_x[idx]), f32(buf_y[idx]), f32(buf_z[idx]));
  let p1 = vec3<f32>(f32(buf_x[idx + 1u]), f32(buf_y[idx + 1u]), f32(buf_z[idx + 1u]));
  if (params.thick != 0u) {
    let base = idx * 6u;
    write_thick_vertices(base, p0, p1, color, params.half_width_data);
  } else {
    let base = idx * 2u;
    write_line_vertices(base, p0, p1, color);
  }
}
"#;
