pub const F32: &str = r#"override WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw {
    data: array<f32, 12u>;
};

struct Scatter3Params {
    color: vec4<f32>,
    point_size: f32,
    count: u32,
    _pad: vec2<u32>,
};

@group(0) @binding(0)
var<storage, read> buf_x: array<f32>;

@group(0) @binding(1)
var<storage, read> buf_y: array<f32>;

@group(0) @binding(2)
var<storage, read> buf_z: array<f32>;

@group(0) @binding(3)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(4)
var<uniform> params: Scatter3Params;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let px = buf_x[idx];
    let py = buf_y[idx];
    let pz = buf_z[idx];

    var vertex: VertexRaw;
    vertex.data[0u] = px;
    vertex.data[1u] = py;
    vertex.data[2u] = pz;
    vertex.data[3u] = params.color.x;
    vertex.data[4u] = params.color.y;
    vertex.data[5u] = params.color.z;
    vertex.data[6u] = params.color.w;
    vertex.data[7u] = 0.0;
    vertex.data[8u] = 0.0;
    vertex.data[9u] = params.point_size;
    vertex.data[10u] = 0.0;
    vertex.data[11u] = 0.0;

    out_vertices[idx] = vertex;
}
"#;

pub const F64: &str = r#"override WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw {
    data: array<f32, 12u>;
};

struct Scatter3Params {
    color: vec4<f32>,
    point_size: f32,
    count: u32,
    _pad: vec2<u32>,
};

@group(0) @binding(0)
var<storage, read> buf_x: array<f64>;

@group(0) @binding(1)
var<storage, read> buf_y: array<f64>;

@group(0) @binding(2)
var<storage, read> buf_z: array<f64>;

@group(0) @binding(3)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(4)
var<uniform> params: Scatter3Params;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let px = f32(buf_x[idx]);
    let py = f32(buf_y[idx]);
    let pz = f32(buf_z[idx]);

    var vertex: VertexRaw;
    vertex.data[0u] = px;
    vertex.data[1u] = py;
    vertex.data[2u] = pz;
    vertex.data[3u] = params.color.x;
    vertex.data[4u] = params.color.y;
    vertex.data[5u] = params.color.z;
    vertex.data[6u] = params.color.w;
    vertex.data[7u] = 0.0;
    vertex.data[8u] = 0.0;
    vertex.data[9u] = params.point_size;
    vertex.data[10u] = 0.0;
    vertex.data[11u] = 0.0;

    out_vertices[idx] = vertex;
}
"#;
