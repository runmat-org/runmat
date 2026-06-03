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
    case 2u: { return (segment % 4u) == 0u; }
    case 3u: { let m = segment % 6u; return (m < 2u) || (m == 3u); }
    default: { return true; }
  }
}

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
  let len = length(v);
  if (len < 0.000001) {
    return vec3<f32>(1.0, 0.0, 0.0);
  }
  return v / len;
}

fn point(i: u32) -> vec3<f32> {
  return vec3<f32>(buf_x[i], buf_y[i], buf_z[i]);
}

fn segment_dir(i0: u32, i1: u32) -> vec3<f32> {
  let d = point(i1) - point(i0);
  let len = length(d);
  if (len < 0.000001) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  return d / len;
}

fn side_at(i: u32) -> vec3<f32> {
  var prev = vec3<f32>(0.0, 0.0, 0.0);
  var next = vec3<f32>(0.0, 0.0, 0.0);
  var has_prev = false;
  var has_next = false;

  if (i > 0u) {
    let d = segment_dir(i - 1u, i);
    if (length(d) > 0.0) {
      prev = d;
      has_prev = true;
    }
  }

  if (i + 1u < params.count) {
    let d = segment_dir(i, i + 1u);
    if (length(d) > 0.0) {
      next = d;
      has_next = true;
    }
  }

  var tangent = vec3<f32>(1.0, 0.0, 0.0);
  if (has_prev && has_next) {
    let s = prev + next;
    if (length(s) > 0.000001) {
      tangent = normalize(s);
    } else {
      tangent = next;
    }
  } else if (has_prev) {
    tangent = prev;
  } else if (has_next) {
    tangent = next;
  }

  let ref_axis = select(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
    abs(tangent.z) < 0.95,
  );
  var side = cross(tangent, ref_axis);
  if (length(side) < 0.000001) {
    side = cross(tangent, vec3<f32>(0.0, 1.0, 0.0));
  }
  if (length(side) < 0.000001) {
    return vec3<f32>(0.0, 1.0, 0.0);
  }
  return normalize(side);
}

fn write_line_vertices(base: u32, p0: vec3<f32>, p1: vec3<f32>, color: vec4<f32>) {
  write_vertex(base + 0u, p0, color);
  write_vertex(base + 1u, p1, color);
}

fn write_thick_vertices(segment: u32, base: u32, p0: vec3<f32>, p1: vec3<f32>, color: vec4<f32>, half_width: f32) {
  let dir = safe_normalize(p1 - p0);
  let side0 = side_at(segment);
  let side1 = side_at(segment + 1u);
  let ext = dir * half_width;
  let a = p0 - ext;
  let b = p1 + ext;
  let v0 = a + side0 * half_width;
  let v1 = b + side1 * half_width;
  let v2 = b - side1 * half_width;
  let v3 = a - side0 * half_width;
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
  if (!should_draw(idx, params.line_style)) {
    color.w = 0.0;
  }

  let p0 = point(idx);
  let p1 = point(idx + 1u);
  if (distance(p0, p1) < 0.000001) {
    color.w = 0.0;
  }

  if (params.thick != 0u) {
    let base = idx * 6u;
    write_thick_vertices(idx, base, p0, p1, color, params.half_width_data);
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
    case 2u: { return (segment % 4u) == 0u; }
    case 3u: { let m = segment % 6u; return (m < 2u) || (m == 3u); }
    default: { return true; }
  }
}

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
  let len = length(v);
  if (len < 0.000001) {
    return vec3<f32>(1.0, 0.0, 0.0);
  }
  return v / len;
}

fn point(i: u32) -> vec3<f32> {
  return vec3<f32>(f32(buf_x[i]), f32(buf_y[i]), f32(buf_z[i]));
}

fn segment_dir(i0: u32, i1: u32) -> vec3<f32> {
  let d = point(i1) - point(i0);
  let len = length(d);
  if (len < 0.000001) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  return d / len;
}

fn side_at(i: u32) -> vec3<f32> {
  var prev = vec3<f32>(0.0, 0.0, 0.0);
  var next = vec3<f32>(0.0, 0.0, 0.0);
  var has_prev = false;
  var has_next = false;

  if (i > 0u) {
    let d = segment_dir(i - 1u, i);
    if (length(d) > 0.0) {
      prev = d;
      has_prev = true;
    }
  }

  if (i + 1u < params.count) {
    let d = segment_dir(i, i + 1u);
    if (length(d) > 0.0) {
      next = d;
      has_next = true;
    }
  }

  var tangent = vec3<f32>(1.0, 0.0, 0.0);
  if (has_prev && has_next) {
    let s = prev + next;
    if (length(s) > 0.000001) {
      tangent = normalize(s);
    } else {
      tangent = next;
    }
  } else if (has_prev) {
    tangent = prev;
  } else if (has_next) {
    tangent = next;
  }

  let ref_axis = select(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
    abs(tangent.z) < 0.95,
  );
  var side = cross(tangent, ref_axis);
  if (length(side) < 0.000001) {
    side = cross(tangent, vec3<f32>(0.0, 1.0, 0.0));
  }
  if (length(side) < 0.000001) {
    return vec3<f32>(0.0, 1.0, 0.0);
  }
  return normalize(side);
}

fn write_line_vertices(base: u32, p0: vec3<f32>, p1: vec3<f32>, color: vec4<f32>) {
  write_vertex(base + 0u, p0, color);
  write_vertex(base + 1u, p1, color);
}

fn write_thick_vertices(segment: u32, base: u32, p0: vec3<f32>, p1: vec3<f32>, color: vec4<f32>, half_width: f32) {
  let dir = safe_normalize(p1 - p0);
  let side0 = side_at(segment);
  let side1 = side_at(segment + 1u);
  let ext = dir * half_width;
  let a = p0 - ext;
  let b = p1 + ext;
  let v0 = a + side0 * half_width;
  let v1 = b + side1 * half_width;
  let v2 = b - side1 * half_width;
  let v3 = a - side0 * half_width;
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
  if (!should_draw(idx, params.line_style)) {
    color.w = 0.0;
  }

  let p0 = point(idx);
  let p1 = point(idx + 1u);
  if (distance(p0, p1) < 0.000001) {
    color.w = 0.0;
  }

  if (params.thick != 0u) {
    let base = idx * 6u;
    write_thick_vertices(idx, base, p0, p1, color, params.half_width_data);
  } else {
    let base = idx * 2u;
    write_line_vertices(base, p0, p1, color);
  }
}
"#;
