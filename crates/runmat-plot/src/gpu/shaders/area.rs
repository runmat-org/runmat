pub const F32: &str = r#"
const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;
struct VertexRaw { data: array<f32, 12u>, };
struct Params { color: vec4<f32>, rows: u32, cols: u32, target_col: u32, baseline: f32, _pad0: f32, _pad1: f32, _pad2: f32, };
@group(0) @binding(0) var<storage, read> buf_x: array<f32>;
@group(0) @binding(1) var<storage, read> buf_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_vertices: array<VertexRaw>;
@group(0) @binding(3) var<uniform> params: Params;
fn write_vertex(index:u32, pos:vec3<f32>, color:vec4<f32>) { var v:VertexRaw; v.data[0u]=pos.x; v.data[1u]=pos.y; v.data[2u]=pos.z; v.data[3u]=color.x; v.data[4u]=color.y; v.data[5u]=color.z; v.data[6u]=color.w; v.data[7u]=0.0; v.data[8u]=0.0; v.data[9u]=1.0; v.data[10u]=0.0; v.data[11u]=0.0; out_vertices[index]=v; }
fn y_at(col:u32,row:u32)->f32 { return buf_y[col * params.rows + row]; }
fn lower_sum(row:u32)->f32 { var acc = params.baseline; var c:u32 = 0u; loop { if (c >= params.target_col) { break; } acc = acc + y_at(c,row); c = c + 1u; } return acc; }
fn upper_sum(row:u32)->f32 { return lower_sum(row) + y_at(params.target_col,row); }
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let i = gid.x; if (i + 1u >= params.rows) { return; } let x0 = buf_x[i]; let x1 = buf_x[i + 1u]; let l0 = lower_sum(i); let l1 = lower_sum(i + 1u); let u0 = upper_sum(i); let u1 = upper_sum(i + 1u); let base = i * 6u; write_vertex(base + 0u, vec3<f32>(x0,l0,0.0), params.color); write_vertex(base + 1u, vec3<f32>(x0,u0,0.0), params.color); write_vertex(base + 2u, vec3<f32>(x1,u1,0.0), params.color); write_vertex(base + 3u, vec3<f32>(x0,l0,0.0), params.color); write_vertex(base + 4u, vec3<f32>(x1,u1,0.0), params.color); write_vertex(base + 5u, vec3<f32>(x1,l1,0.0), params.color); }
"#;

pub const F64: &str = r#"
const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;
struct VertexRaw { data: array<f32, 12u>, };
struct Params { color: vec4<f32>, rows: u32, cols: u32, target_col: u32, baseline: f32, _pad0: f32, _pad1: f32, _pad2: f32, };
@group(0) @binding(0) var<storage, read> buf_x: array<f64>;
@group(0) @binding(1) var<storage, read> buf_y: array<f64>;
@group(0) @binding(2) var<storage, read_write> out_vertices: array<VertexRaw>;
@group(0) @binding(3) var<uniform> params: Params;
fn write_vertex(index:u32, pos:vec3<f32>, color:vec4<f32>) { var v:VertexRaw; v.data[0u]=pos.x; v.data[1u]=pos.y; v.data[2u]=pos.z; v.data[3u]=color.x; v.data[4u]=color.y; v.data[5u]=color.z; v.data[6u]=color.w; v.data[7u]=0.0; v.data[8u]=0.0; v.data[9u]=1.0; v.data[10u]=0.0; v.data[11u]=0.0; out_vertices[index]=v; }
fn y_at(col:u32,row:u32)->f32 { return f32(buf_y[col * params.rows + row]); }
fn lower_sum(row:u32)->f32 { var acc = params.baseline; var c:u32 = 0u; loop { if (c >= params.target_col) { break; } acc = acc + y_at(c,row); c = c + 1u; } return acc; }
fn upper_sum(row:u32)->f32 { return lower_sum(row) + y_at(params.target_col,row); }
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let i = gid.x; if (i + 1u >= params.rows) { return; } let x0 = f32(buf_x[i]); let x1 = f32(buf_x[i + 1u]); let l0 = lower_sum(i); let l1 = lower_sum(i + 1u); let u0 = upper_sum(i); let u1 = upper_sum(i + 1u); let base = i * 6u; write_vertex(base + 0u, vec3<f32>(x0,l0,0.0), params.color); write_vertex(base + 1u, vec3<f32>(x0,u0,0.0), params.color); write_vertex(base + 2u, vec3<f32>(x1,u1,0.0), params.color); write_vertex(base + 3u, vec3<f32>(x0,l0,0.0), params.color); write_vertex(base + 4u, vec3<f32>(x1,u1,0.0), params.color); write_vertex(base + 5u, vec3<f32>(x1,l1,0.0), params.color); }
"#;
