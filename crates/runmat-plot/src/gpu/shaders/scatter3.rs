pub const F32: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw {
    data: array<f32, 12u>;
};

struct Scatter3Params {
    color: vec4<f32>,
    point_size: f32,
    count: u32,
    lod_stride: u32,
    has_sizes: u32,
    has_colors: u32,
    color_stride: u32,
};

struct IndirectArgs {
    vertex_count: atomic<u32>,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

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

@group(0) @binding(5)
var<storage, read> buf_sizes: array<f32>;

@group(0) @binding(6)
var<storage, read> buf_colors: array<f32>;

@group(0) @binding(7)
var<storage, read_write> indirect: IndirectArgs;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let stride = max(params.lod_stride, 1u);
    if ((idx % stride) != 0u) {
        return;
    }

    let px = buf_x[idx];
    let py = buf_y[idx];
    let pz = buf_z[idx];

    var vertex: VertexRaw;
    vertex.data[0u] = px;
    vertex.data[1u] = py;
    vertex.data[2u] = pz;

    var v_color = params.color;
    if (params.has_colors != 0u) {
        let base = idx * params.color_stride;
        let r = buf_colors[base];
        let g = buf_colors[base + 1u];
        let b = buf_colors[base + 2u];
        let a = if params.color_stride > 3u {
            buf_colors[base + 3u]
        } else {
            1.0
        };
        v_color = vec4<f32>(r, g, b, a);
    }

    let mut point_size = params.point_size;
    if (params.has_sizes != 0u) {
        point_size = buf_sizes[idx];
    }

    vertex.data[3u] = v_color.x;
    vertex.data[4u] = v_color.y;
    vertex.data[5u] = v_color.z;
    vertex.data[6u] = v_color.w;
    vertex.data[7u] = 0.0;
    vertex.data[8u] = 0.0;
    vertex.data[9u] = point_size;
    vertex.data[10u] = 0.0;
    vertex.data[11u] = 0.0;

    let out_idx = atomicAdd(&(indirect.vertex_count), 1u);
    out_vertices[out_idx] = vertex;
}
"#;

pub const F64: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw {
    data: array<f32, 12u>;
};

struct Scatter3Params {
    color: vec4<f32>,
    point_size: f32,
    count: u32,
    lod_stride: u32,
    has_sizes: u32,
    has_colors: u32,
    color_stride: u32,
};

struct IndirectArgs {
    vertex_count: atomic<u32>,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

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

@group(0) @binding(5)
var<storage, read> buf_sizes: array<f32>;

@group(0) @binding(6)
var<storage, read> buf_colors: array<f32>;

@group(0) @binding(7)
var<storage, read_write> indirect: IndirectArgs;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let stride = max(params.lod_stride, 1u);
    if ((idx % stride) != 0u) {
        return;
    }

    let px = f32(buf_x[idx]);
    let py = f32(buf_y[idx]);
    let pz = f32(buf_z[idx]);

    var vertex: VertexRaw;
    vertex.data[0u] = px;
    vertex.data[1u] = py;
    vertex.data[2u] = pz;

    var v_color = params.color;
    if (params.has_colors != 0u) {
        let base = idx * params.color_stride;
        let r = buf_colors[base];
        let g = buf_colors[base + 1u];
        let b = buf_colors[base + 2u];
        let a = if params.color_stride > 3u {
            buf_colors[base + 3u]
        } else {
            1.0
        };
        v_color = vec4<f32>(r, g, b, a);
    }

    let mut point_size = params.point_size;
    if (params.has_sizes != 0u) {
        point_size = buf_sizes[idx];
    }

    vertex.data[3u] = v_color.x;
    vertex.data[4u] = v_color.y;
    vertex.data[5u] = v_color.z;
    vertex.data[6u] = v_color.w;
    vertex.data[7u] = 0.0;
    vertex.data[8u] = 0.0;
    vertex.data[9u] = point_size;
    vertex.data[10u] = 0.0;
    vertex.data[11u] = 0.0;

    let out_idx = atomicAdd(&(indirect.vertex_count), 1u);
    out_vertices[out_idx] = vertex;
}
"#;
