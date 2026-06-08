pub const F32: &str = r#"
const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw { data: array<f32, 12u>, };
struct Params { color: vec4<f32>, count: u32, line_style: u32, cap_half_width: f32, orientation: u32, };

@group(0) @binding(0) var<storage, read> buf_x: array<f32>;
@group(0) @binding(1) var<storage, read> buf_y: array<f32>;
@group(0) @binding(2) var<storage, read> buf_x_neg: array<f32>;
@group(0) @binding(3) var<storage, read> buf_x_pos: array<f32>;
@group(0) @binding(4) var<storage, read> buf_y_neg: array<f32>;
@group(0) @binding(5) var<storage, read> buf_y_pos: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_vertices: array<VertexRaw>;
@group(0) @binding(7) var<uniform> params: Params;

fn write_vertex(index: u32, pos: vec3<f32>, color: vec4<f32>) {
  var v: VertexRaw;
  v.data[0u] = pos.x; v.data[1u] = pos.y; v.data[2u] = 0.0;
  v.data[3u] = color.x; v.data[4u] = color.y; v.data[5u] = color.z; v.data[6u] = color.w;
  v.data[7u] = 0.0; v.data[8u] = 0.0; v.data[9u] = 1.0; v.data[10u] = 0.0; v.data[11u] = 0.0;
  out_vertices[index] = v;
}

fn include_segment(i:u32, style:u32)->bool { switch(style){ case 0u:{return true;} case 1u:{return (i%4u)<2u;} case 2u:{return (i%4u)==0u;} case 3u:{let m=i%6u; return (m<2u)||(m==3u);} default:{return true;} } }

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  if (!include_segment(i, params.line_style)) { return; }
  let x = buf_x[i];
  let y = buf_y[i];
  let y0 = y - buf_y_neg[i];
  let y1 = y + buf_y_pos[i];
  let x0 = x - buf_x_neg[i];
  let x1 = x + buf_x_pos[i];
  let base = i * (select(6u, 12u, params.orientation == 2u));
  if (params.orientation != 1u) {
    write_vertex(base + 0u, vec3<f32>(x, y0, 0.0), params.color);
    write_vertex(base + 1u, vec3<f32>(x, y1, 0.0), params.color);
    write_vertex(base + 2u, vec3<f32>(x - params.cap_half_width, y0, 0.0), params.color);
    write_vertex(base + 3u, vec3<f32>(x + params.cap_half_width, y0, 0.0), params.color);
    write_vertex(base + 4u, vec3<f32>(x - params.cap_half_width, y1, 0.0), params.color);
    write_vertex(base + 5u, vec3<f32>(x + params.cap_half_width, y1, 0.0), params.color);
  }
  if (params.orientation != 0u) {
    let off = select(0u, 6u, params.orientation == 2u);
    write_vertex(base + off + 0u, vec3<f32>(x0, y, 0.0), params.color);
    write_vertex(base + off + 1u, vec3<f32>(x1, y, 0.0), params.color);
    write_vertex(base + off + 2u, vec3<f32>(x0, y - params.cap_half_width, 0.0), params.color);
    write_vertex(base + off + 3u, vec3<f32>(x0, y + params.cap_half_width, 0.0), params.color);
    write_vertex(base + off + 4u, vec3<f32>(x1, y - params.cap_half_width, 0.0), params.color);
    write_vertex(base + off + 5u, vec3<f32>(x1, y + params.cap_half_width, 0.0), params.color);
  }
}
"#;

pub const F64: &str = r#"
const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw { data: array<f32, 12u>, };
struct Params { color: vec4<f32>, count: u32, line_style: u32, cap_half_width: f32, orientation: u32, };

@group(0) @binding(0) var<storage, read> buf_x: array<f64>;
@group(0) @binding(1) var<storage, read> buf_y: array<f64>;
@group(0) @binding(2) var<storage, read> buf_x_neg: array<f64>;
@group(0) @binding(3) var<storage, read> buf_x_pos: array<f64>;
@group(0) @binding(4) var<storage, read> buf_y_neg: array<f64>;
@group(0) @binding(5) var<storage, read> buf_y_pos: array<f64>;
@group(0) @binding(6) var<storage, read_write> out_vertices: array<VertexRaw>;
@group(0) @binding(7) var<uniform> params: Params;

fn write_vertex(index: u32, pos: vec3<f32>, color: vec4<f32>) {
  var v: VertexRaw;
  v.data[0u] = pos.x; v.data[1u] = pos.y; v.data[2u] = 0.0;
  v.data[3u] = color.x; v.data[4u] = color.y; v.data[5u] = color.z; v.data[6u] = color.w;
  v.data[7u] = 0.0; v.data[8u] = 0.0; v.data[9u] = 1.0; v.data[10u] = 0.0; v.data[11u] = 0.0;
  out_vertices[index] = v;
}

fn include_segment(i:u32, style:u32)->bool { switch(style){ case 0u:{return true;} case 1u:{return (i%4u)<2u;} case 2u:{return (i%4u)==0u;} case 3u:{let m=i%6u; return (m<2u)||(m==3u);} default:{return true;} } }

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  if (!include_segment(i, params.line_style)) { return; }
  let x = f32(buf_x[i]);
  let y = f32(buf_y[i]);
  let y0 = y - f32(buf_y_neg[i]);
  let y1 = y + f32(buf_y_pos[i]);
  let x0 = x - f32(buf_x_neg[i]);
  let x1 = x + f32(buf_x_pos[i]);
  let base = i * (select(6u, 12u, params.orientation == 2u));
  if (params.orientation != 1u) {
    write_vertex(base + 0u, vec3<f32>(x, y0, 0.0), params.color);
    write_vertex(base + 1u, vec3<f32>(x, y1, 0.0), params.color);
    write_vertex(base + 2u, vec3<f32>(x - params.cap_half_width, y0, 0.0), params.color);
    write_vertex(base + 3u, vec3<f32>(x + params.cap_half_width, y0, 0.0), params.color);
    write_vertex(base + 4u, vec3<f32>(x - params.cap_half_width, y1, 0.0), params.color);
    write_vertex(base + 5u, vec3<f32>(x + params.cap_half_width, y1, 0.0), params.color);
  }
  if (params.orientation != 0u) {
    let off = select(0u, 6u, params.orientation == 2u);
    write_vertex(base + off + 0u, vec3<f32>(x0, y, 0.0), params.color);
    write_vertex(base + off + 1u, vec3<f32>(x1, y, 0.0), params.color);
    write_vertex(base + off + 2u, vec3<f32>(x0, y - params.cap_half_width, 0.0), params.color);
    write_vertex(base + off + 3u, vec3<f32>(x0, y + params.cap_half_width, 0.0), params.color);
    write_vertex(base + off + 4u, vec3<f32>(x1, y - params.cap_half_width, 0.0), params.color);
    write_vertex(base + off + 5u, vec3<f32>(x1, y + params.cap_half_width, 0.0), params.color);
  }
}
"#;
