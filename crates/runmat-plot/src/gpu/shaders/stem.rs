pub const F32: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;
struct VertexRaw { data: array<f32, 12u>, };
struct StemParams { color: vec4<f32>, baseline_color: vec4<f32>, baseline: f32, min_x: f32, max_x: f32, point_count: u32, line_style: u32, baseline_visible: u32, };
@group(0) @binding(0) var<storage, read> buf_x: array<f32>;
@group(0) @binding(1) var<storage, read> buf_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_vertices: array<VertexRaw>;
@group(0) @binding(3) var<uniform> params: StemParams;
fn write_vertex(index: u32, pos: vec3<f32>, color: vec4<f32>) { var v: VertexRaw; v.data[0u]=pos.x; v.data[1u]=pos.y; v.data[2u]=0.0; v.data[3u]=color.x; v.data[4u]=color.y; v.data[5u]=color.z; v.data[6u]=color.w; v.data[7u]=0.0; v.data[8u]=0.0; v.data[9u]=1.0; v.data[10u]=0.0; v.data[11u]=0.0; out_vertices[index]=v; }
fn include_segment(i:u32, style:u32)->bool { switch(style){ case 0u:{return true;} case 1u:{return (i%4u)<2u;} case 2u:{return (i%4u)==0u;} case 3u:{let m=i%6u; return (m<2u)||(m==3u);} default:{return true;} } }
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
 let i = gid.x; if (i >= params.point_count) { return; }
 if (i == 0u && params.baseline_visible != 0u) { write_vertex(0u, vec3<f32>(params.min_x, params.baseline, 0.0), params.baseline_color); write_vertex(1u, vec3<f32>(params.max_x, params.baseline, 0.0), params.baseline_color); }
 if (!include_segment(i, params.line_style)) { return; }
 let base = 2u + i*2u; write_vertex(base, vec3<f32>(buf_x[i], params.baseline, 0.0), params.color); write_vertex(base+1u, vec3<f32>(buf_x[i], buf_y[i], 0.0), params.color);
}
"#;
pub const F64: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;
struct VertexRaw { data: array<f32, 12u>, };
struct StemParams { color: vec4<f32>, baseline_color: vec4<f32>, baseline: f32, min_x: f32, max_x: f32, point_count: u32, line_style: u32, baseline_visible: u32, };
@group(0) @binding(0) var<storage, read> buf_x: array<f64>;
@group(0) @binding(1) var<storage, read> buf_y: array<f64>;
@group(0) @binding(2) var<storage, read_write> out_vertices: array<VertexRaw>;
@group(0) @binding(3) var<uniform> params: StemParams;
fn write_vertex(index: u32, pos: vec3<f32>, color: vec4<f32>) { var v: VertexRaw; v.data[0u]=pos.x; v.data[1u]=pos.y; v.data[2u]=0.0; v.data[3u]=color.x; v.data[4u]=color.y; v.data[5u]=color.z; v.data[6u]=color.w; v.data[7u]=0.0; v.data[8u]=0.0; v.data[9u]=1.0; v.data[10u]=0.0; v.data[11u]=0.0; out_vertices[index]=v; }
fn include_segment(i:u32, style:u32)->bool { switch(style){ case 0u:{return true;} case 1u:{return (i%4u)<2u;} case 2u:{return (i%4u)==0u;} case 3u:{let m=i%6u; return (m<2u)||(m==3u);} default:{return true;} } }
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
 let i = gid.x; if (i >= params.point_count) { return; }
 if (i == 0u && params.baseline_visible != 0u) { write_vertex(0u, vec3<f32>(params.min_x, params.baseline, 0.0), params.baseline_color); write_vertex(1u, vec3<f32>(params.max_x, params.baseline, 0.0), params.baseline_color); }
 if (!include_segment(i, params.line_style)) { return; }
 let base = 2u + i*2u; write_vertex(base, vec3<f32>(f32(buf_x[i]), params.baseline, 0.0), params.color); write_vertex(base+1u, vec3<f32>(f32(buf_x[i]), f32(buf_y[i]), 0.0), params.color);
}
"#;
