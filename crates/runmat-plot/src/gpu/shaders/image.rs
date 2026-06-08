pub const F32: &str = r#"
const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;
struct VertexRaw { data: array<f32, 12u>, };
struct Params { rows: u32, cols: u32, channels: u32, _pad: u32, };
@group(0) @binding(0) var<storage, read> buf_x: array<f32>;
@group(0) @binding(1) var<storage, read> buf_y: array<f32>;
@group(0) @binding(2) var<storage, read> buf_img: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_vertices: array<VertexRaw>;
@group(0) @binding(4) var<uniform> params: Params;
fn write_vertex(index:u32, pos:vec3<f32>, color:vec4<f32>) { var v:VertexRaw; v.data[0u]=pos.x; v.data[1u]=pos.y; v.data[2u]=pos.z; v.data[3u]=color.x; v.data[4u]=color.y; v.data[5u]=color.z; v.data[6u]=color.w; v.data[7u]=0.0; v.data[8u]=0.0; v.data[9u]=1.0; v.data[10u]=0.0; v.data[11u]=0.0; out_vertices[index]=v; }
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let idx = gid.x; let count = params.rows * params.cols; if (idx >= count) { return; } let row = idx % params.rows; let col = idx / params.rows; let plane = params.rows * params.cols; let base = row + params.rows * col; let r = buf_img[base]; let g = buf_img[base + plane]; let b = buf_img[base + 2u * plane]; let a = select(1.0, buf_img[base + 3u * plane], params.channels == 4u); write_vertex(idx, vec3<f32>(buf_x[row], buf_y[col], 0.0), vec4<f32>(r,g,b,a)); }
"#;

pub const F64: &str = r#"
const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;
struct VertexRaw { data: array<f32, 12u>, };
struct Params { rows: u32, cols: u32, channels: u32, _pad: u32, };
@group(0) @binding(0) var<storage, read> buf_x: array<f64>;
@group(0) @binding(1) var<storage, read> buf_y: array<f64>;
@group(0) @binding(2) var<storage, read> buf_img: array<f64>;
@group(0) @binding(3) var<storage, read_write> out_vertices: array<VertexRaw>;
@group(0) @binding(4) var<uniform> params: Params;
fn write_vertex(index:u32, pos:vec3<f32>, color:vec4<f32>) { var v:VertexRaw; v.data[0u]=pos.x; v.data[1u]=pos.y; v.data[2u]=pos.z; v.data[3u]=color.x; v.data[4u]=color.y; v.data[5u]=color.z; v.data[6u]=color.w; v.data[7u]=0.0; v.data[8u]=0.0; v.data[9u]=1.0; v.data[10u]=0.0; v.data[11u]=0.0; out_vertices[index]=v; }
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let idx = gid.x; let count = params.rows * params.cols; if (idx >= count) { return; } let row = idx % params.rows; let col = idx / params.rows; let plane = params.rows * params.cols; let base = row + params.rows * col; let r = f32(buf_img[base]); let g = f32(buf_img[base + plane]); let b = f32(buf_img[base + 2u * plane]); let a = select(1.0, f32(buf_img[base + 3u * plane]), params.channels == 4u); write_vertex(idx, vec3<f32>(f32(buf_x[row]), f32(buf_y[col]), 0.0), vec4<f32>(r,g,b,a)); }
"#;
