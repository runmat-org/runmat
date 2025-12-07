pub const F32: &str = r#"override WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw {
    data: array<f32, 12u>;
};

struct StairsParams {
    color: vec4<f32>,
    point_count: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0)
var<storage, read> buf_x: array<f32>;

@group(0) @binding(1)
var<storage, read> buf_y: array<f32>;

@group(0) @binding(2)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(3)
var<uniform> params: StairsParams;

fn encode_vertex(px: f32, py: f32) -> VertexRaw {
    var vertex: VertexRaw;
    vertex.data[0u] = px;
    vertex.data[1u] = py;
    vertex.data[2u] = 0.0;
    vertex.data[3u] = params.color.x;
    vertex.data[4u] = params.color.y;
    vertex.data[5u] = params.color.z;
    vertex.data[6u] = params.color.w;
    vertex.data[7u] = 0.0;
    vertex.data[8u] = 0.0;
    vertex.data[9u] = 1.0;
    vertex.data[10u] = 0.0;
    vertex.data[11u] = 0.0;
    return vertex;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seg = gid.x;
    if (seg + 1u >= params.point_count) {
        return;
    }
    let base = seg * 4u;
    let x0 = buf_x[seg];
    let y0 = buf_y[seg];
    let x1 = buf_x[seg + 1u];
    let y1 = buf_y[seg + 1u];

    out_vertices[base + 0u] = encode_vertex(x0, y0);
    out_vertices[base + 1u] = encode_vertex(x1, y0);
    out_vertices[base + 2u] = encode_vertex(x1, y0);
    out_vertices[base + 3u] = encode_vertex(x1, y1);
}
"#;

pub const F64: &str = r#"override WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw {
    data: array<f32, 12u>;
};

struct StairsParams {
    color: vec4<f32>,
    point_count: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0)
var<storage, read> buf_x: array<f64>;

@group(0) @binding(1)
var<storage, read> buf_y: array<f64>;

@group(0) @binding(2)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(3)
var<uniform> params: StairsParams;

fn encode_vertex(px: f32, py: f32) -> VertexRaw {
    var vertex: VertexRaw;
    vertex.data[0u] = px;
    vertex.data[1u] = py;
    vertex.data[2u] = 0.0;
    vertex.data[3u] = params.color.x;
    vertex.data[4u] = params.color.y;
    vertex.data[5u] = params.color.z;
    vertex.data[6u] = params.color.w;
    vertex.data[7u] = 0.0;
    vertex.data[8u] = 0.0;
    vertex.data[9u] = 1.0;
    vertex.data[10u] = 0.0;
    vertex.data[11u] = 0.0;
    return vertex;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seg = gid.x;
    if (seg + 1u >= params.point_count) {
        return;
    }
    let base = seg * 4u;
    let x0 = f32(buf_x[seg]);
    let y0 = f32(buf_y[seg]);
    let x1 = f32(buf_x[seg + 1u]);
    let y1 = f32(buf_y[seg + 1u]);

    out_vertices[base + 0u] = encode_vertex(x0, y0);
    out_vertices[base + 1u] = encode_vertex(x1, y0);
    out_vertices[base + 2u] = encode_vertex(x1, y0);
    out_vertices[base + 3u] = encode_vertex(x1, y1);
}
"#;
