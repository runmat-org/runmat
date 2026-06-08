pub const F32: &str = r#"
const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;
struct VertexRaw { data: array<f32, 12u>, };
struct Params { color: vec4<f32>, count: u32, rows: u32, cols: u32, xy_mode: u32, scale: f32, head_size: f32, _pad: f32, };
@group(0) @binding(0) var<storage, read> buf_x: array<f32>;
@group(0) @binding(1) var<storage, read> buf_y: array<f32>;
@group(0) @binding(2) var<storage, read> buf_u: array<f32>;
@group(0) @binding(3) var<storage, read> buf_v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_vertices: array<VertexRaw>;
@group(0) @binding(5) var<uniform> params: Params;
fn write_vertex(index: u32, pos: vec3<f32>, color: vec4<f32>) { var v: VertexRaw; v.data[0u]=pos.x; v.data[1u]=pos.y; v.data[2u]=pos.z; v.data[3u]=color.x; v.data[4u]=color.y; v.data[5u]=color.z; v.data[6u]=color.w; v.data[7u]=0.0; v.data[8u]=0.0; v.data[9u]=1.0; v.data[10u]=0.0; v.data[11u]=0.0; out_vertices[index]=v; }
fn coord_x(i:u32)->f32 { if (params.xy_mode == 0u) { return buf_x[i]; } let col = i / params.rows; return buf_x[col]; }
fn coord_y(i:u32)->f32 { if (params.xy_mode == 0u) { return buf_y[i]; } let row = i % params.rows; return buf_y[row]; }
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let i = gid.x; if (i >= params.count) { return; } let x = coord_x(i); let y = coord_y(i); let dx = buf_u[i] * params.scale; let dy = buf_v[i] * params.scale; let tip = vec3<f32>(x + dx, y + dy, 0.0); let base = i * 6u; write_vertex(base + 0u, vec3<f32>(x, y, 0.0), params.color); write_vertex(base + 1u, tip, params.color); let len = sqrt(dx*dx + dy*dy); var left = tip; var right = tip; if (len > 0.0 && params.head_size > 0.0) { let hx = dx / len; let hy = dy / len; let px = -hy; let py = hx; let h = min(params.head_size, len * 0.5); left = vec3<f32>(tip.x - h*hx + 0.5*h*px, tip.y - h*hy + 0.5*h*py, 0.0); right = vec3<f32>(tip.x - h*hx - 0.5*h*px, tip.y - h*hy - 0.5*h*py, 0.0); } write_vertex(base + 2u, tip, params.color); write_vertex(base + 3u, left, params.color); write_vertex(base + 4u, tip, params.color); write_vertex(base + 5u, right, params.color); }
"#;

pub const F64: &str = r#"
const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;
struct VertexRaw { data: array<f32, 12u>, };
struct Params { color: vec4<f32>, count: u32, rows: u32, cols: u32, xy_mode: u32, scale: f32, head_size: f32, _pad: f32, };
@group(0) @binding(0) var<storage, read> buf_x: array<f64>;
@group(0) @binding(1) var<storage, read> buf_y: array<f64>;
@group(0) @binding(2) var<storage, read> buf_u: array<f64>;
@group(0) @binding(3) var<storage, read> buf_v: array<f64>;
@group(0) @binding(4) var<storage, read_write> out_vertices: array<VertexRaw>;
@group(0) @binding(5) var<uniform> params: Params;
fn write_vertex(index: u32, pos: vec3<f32>, color: vec4<f32>) { var v: VertexRaw; v.data[0u]=pos.x; v.data[1u]=pos.y; v.data[2u]=pos.z; v.data[3u]=color.x; v.data[4u]=color.y; v.data[5u]=color.z; v.data[6u]=color.w; v.data[7u]=0.0; v.data[8u]=0.0; v.data[9u]=1.0; v.data[10u]=0.0; v.data[11u]=0.0; out_vertices[index]=v; }
fn coord_x(i:u32)->f32 { if (params.xy_mode == 0u) { return f32(buf_x[i]); } let col = i / params.rows; return f32(buf_x[col]); }
fn coord_y(i:u32)->f32 { if (params.xy_mode == 0u) { return f32(buf_y[i]); } let row = i % params.rows; return f32(buf_y[row]); }
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let i = gid.x; if (i >= params.count) { return; } let x = coord_x(i); let y = coord_y(i); let dx = f32(buf_u[i]) * params.scale; let dy = f32(buf_v[i]) * params.scale; let tip = vec3<f32>(x + dx, y + dy, 0.0); let base = i * 6u; write_vertex(base + 0u, vec3<f32>(x, y, 0.0), params.color); write_vertex(base + 1u, tip, params.color); let len = sqrt(dx*dx + dy*dy); var left = tip; var right = tip; if (len > 0.0 && params.head_size > 0.0) { let hx = dx / len; let hy = dy / len; let px = -hy; let py = hx; let h = min(params.head_size, len * 0.5); left = vec3<f32>(tip.x - h*hx + 0.5*h*px, tip.y - h*hy + 0.5*h*py, 0.0); right = vec3<f32>(tip.x - h*hx - 0.5*h*px, tip.y - h*hy - 0.5*h*py, 0.0); } write_vertex(base + 2u, tip, params.color); write_vertex(base + 3u, left, params.color); write_vertex(base + 4u, tip, params.color); write_vertex(base + 5u, right, params.color); }
"#;
