pub const F32: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw {
    data: array<f32, 12u>;
};

struct LineParams {
    color: vec4<f32>,
    count: u32,
    line_width: f32,
    line_style: u32,
    _pad: u32,
};

struct IndirectArgs {
    vertex_count: atomic<u32>,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0)
var<storage, read> buf_x: array<f32>;

@group(0) @binding(1)
var<storage, read> buf_y: array<f32>;

@group(0) @binding(2)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(3)
var<storage, read_write> indirect: IndirectArgs;

@group(0) @binding(4)
var<uniform> params: LineParams;

fn should_draw(segment: u32, style: u32) -> bool {
    switch(style) {
        case 0u: { return true; } // Solid
        case 1u: { return (segment % 4u) < 2u; } // Dashed: on,on,off,off
        case 2u: { return (segment % 4u) < 2u; } // Dotted approximated via dashed pattern
        case 3u: {
            let m = segment % 6u;
            return (m < 2u) || (m == 3u); // DashDot: on,on,off,on,off,off
        }
        default: { return true; }
    }
}

fn write_vertex(index: u32, pos: vec2<f32>, color: vec4<f32>) {
    var vertex: VertexRaw;
    vertex.data[0u] = pos.x;
    vertex.data[1u] = pos.y;
    vertex.data[2u] = 0.0;
    vertex.data[3u] = color.x;
    vertex.data[4u] = color.y;
    vertex.data[5u] = color.z;
    vertex.data[6u] = color.w;
    vertex.data[7u] = 0.0;
    vertex.data[8u] = 0.0;
    vertex.data[9u] = 1.0;
    vertex.data[10u] = 0.0;
    vertex.data[11u] = 0.0;
    out_vertices[index] = vertex;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.count < 2u) {
        return;
    }
    let segments = params.count - 1u;
    let idx = gid.x;
    if (idx >= segments) {
        return;
    }
    if (!should_draw(idx, params.line_style)) {
        return;
    }

    let p0 = vec2<f32>(buf_x[idx], buf_y[idx]);
    let p1 = vec2<f32>(buf_x[idx + 1u], buf_y[idx + 1u]);
    let delta = p1 - p0;
    let len = length(delta);
    if (len == 0.0) {
        return;
    }

    let thick = params.line_width > 1.0;
    if (!thick) {
        let base = atomicAdd(&(indirect.vertex_count), 2u);
        write_vertex(base + 0u, p0, params.color);
        write_vertex(base + 1u, p1, params.color);
        return;
    }

    var half_width = params.line_width * 0.5;
    if (half_width < 0.0001) {
        half_width = 0.0001;
    }
    let dir = normalize(delta);
    let normal = vec2<f32>(-dir.y, dir.x);
    let offset = normal * half_width;
    let v0 = p0 + offset;
    let v1 = p1 + offset;
    let v2 = p1 - offset;
    let v3 = p0 - offset;

    let base = atomicAdd(&(indirect.vertex_count), 6u);
    write_vertex(base + 0u, v0, params.color);
    write_vertex(base + 1u, v1, params.color);
    write_vertex(base + 2u, v2, params.color);
    write_vertex(base + 3u, v0, params.color);
    write_vertex(base + 4u, v2, params.color);
    write_vertex(base + 5u, v3, params.color);
}
"#;

pub const F64: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw {
    data: array<f32, 12u>;
};

struct LineParams {
    color: vec4<f32>,
    count: u32,
    line_width: f32,
    line_style: u32,
    _pad: u32,
};

struct IndirectArgs {
    vertex_count: atomic<u32>,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0)
var<storage, read> buf_x: array<f64>;

@group(0) @binding(1)
var<storage, read> buf_y: array<f64>;

@group(0) @binding(2)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(3)
var<storage, read_write> indirect: IndirectArgs;

@group(0) @binding(4)
var<uniform> params: LineParams;

fn should_draw(segment: u32, style: u32) -> bool {
    switch(style) {
        case 0u: { return true; }
        case 1u: { return (segment % 4u) < 2u; }
        case 2u: { return (segment % 4u) < 2u; }
        case 3u: {
            let m = segment % 6u;
            return (m < 2u) || (m == 3u);
        }
        default: { return true; }
    }
}

fn write_vertex(index: u32, pos: vec2<f32>, color: vec4<f32>) {
    var vertex: VertexRaw;
    vertex.data[0u] = pos.x;
    vertex.data[1u] = pos.y;
    vertex.data[2u] = 0.0;
    vertex.data[3u] = color.x;
    vertex.data[4u] = color.y;
    vertex.data[5u] = color.z;
    vertex.data[6u] = color.w;
    vertex.data[7u] = 0.0;
    vertex.data[8u] = 0.0;
    vertex.data[9u] = 1.0;
    vertex.data[10u] = 0.0;
    vertex.data[11u] = 0.0;
    out_vertices[index] = vertex;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.count < 2u) {
        return;
    }
    let segments = params.count - 1u;
    let idx = gid.x;
    if (idx >= segments) {
        return;
    }
    if (!should_draw(idx, params.line_style)) {
        return;
    }

    let p0 = vec2<f32>(f32(buf_x[idx]), f32(buf_y[idx]));
    let p1 = vec2<f32>(f32(buf_x[idx + 1u]), f32(buf_y[idx + 1u]));
    let delta = p1 - p0;
    let len = length(delta);
    if (len == 0.0) {
        return;
    }

    let thick = params.line_width > 1.0;
    if (!thick) {
        let base = atomicAdd(&(indirect.vertex_count), 2u);
        write_vertex(base + 0u, p0, params.color);
        write_vertex(base + 1u, p1, params.color);
        return;
    }

    var half_width = params.line_width * 0.5;
    if (half_width < 0.0001) {
        half_width = 0.0001;
    }
    let dir = normalize(delta);
    let normal = vec2<f32>(-dir.y, dir.x);
    let offset = normal * half_width;
    let v0 = p0 + offset;
    let v1 = p1 + offset;
    let v2 = p1 - offset;
    let v3 = p0 - offset;

    let base = atomicAdd(&(indirect.vertex_count), 6u);
    write_vertex(base + 0u, v0, params.color);
    write_vertex(base + 1u, v1, params.color);
    write_vertex(base + 2u, v2, params.color);
    write_vertex(base + 3u, v0, params.color);
    write_vertex(base + 4u, v2, params.color);
    write_vertex(base + 5u, v3, params.color);
}
"#;

pub const MARKER_F32: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw {
    data: array<f32, 12u>;
};

struct MarkerParams {
    color: vec4<f32>,
    count: u32,
    size: f32,
    _pad: vec2<u32>,
};

@group(0) @binding(0)
var<storage, read> buf_x: array<f32>;

@group(0) @binding(1)
var<storage, read> buf_y: array<f32>;

@group(0) @binding(2)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(3)
var<uniform> params: MarkerParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let px = buf_x[idx];
    let py = buf_y[idx];

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
    vertex.data[9u] = params.size;
    vertex.data[10u] = 0.0;
    vertex.data[11u] = 0.0;

    out_vertices[idx] = vertex;
}
"#;

pub const MARKER_F64: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw {
    data: array<f32, 12u>;
};

struct MarkerParams {
    color: vec4<f32>,
    count: u32,
    size: f32,
    _pad: vec2<u32>,
};

@group(0) @binding(0)
var<storage, read> buf_x: array<f64>;

@group(0) @binding(1)
var<storage, read> buf_y: array<f64>;

@group(0) @binding(2)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(3)
var<uniform> params: MarkerParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let px = f32(buf_x[idx]);
    let py = f32(buf_y[idx]);

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
    vertex.data[9u] = params.size;
    vertex.data[10u] = 0.0;
    vertex.data[11u] = 0.0;

    out_vertices[idx] = vertex;
}
"#;
