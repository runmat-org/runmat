pub const F32: &str = r#"override WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

const VERTICES_PER_BAR: u32 = 6u;

struct VertexRaw {
    data: array<f32, 12u>;
};

struct BarParams {
    color: vec4<f32>,
    bar_width: f32,
    count: u32,
    group_index: u32,
    group_count: u32,
    orientation: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> values: array<f32>;

@group(0) @binding(1)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(2)
var<uniform> params: BarParams;

fn encode_vertex(position: vec3<f32>) -> VertexRaw {
    var vertex: VertexRaw;
    vertex.data[0u] = position.x;
    vertex.data[1u] = position.y;
    vertex.data[2u] = position.z;
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

fn write_vertices(base_index: u32, quad: array<vec3<f32>, 4u>) {
    out_vertices[base_index + 0u] = encode_vertex(quad[0u]);
    out_vertices[base_index + 1u] = encode_vertex(quad[1u]);
    out_vertices[base_index + 2u] = encode_vertex(quad[2u]);
    out_vertices[base_index + 3u] = encode_vertex(quad[0u]);
    out_vertices[base_index + 4u] = encode_vertex(quad[2u]);
    out_vertices[base_index + 5u] = encode_vertex(quad[3u]);
}

fn build_vertical_quad(idx: u32, value: f32, per_group_width: f32, local_offset: f32) -> array<vec3<f32>, 4u> {
    let center = (f32(idx) + 1.0) + local_offset;
    let half = per_group_width * 0.5;
    let left = center - half;
    let right = center + half;
    let bottom = 0.0;
    let top = value;

    return array<vec3<f32>, 4u>(
        vec3<f32>(left, bottom, 0.0),
        vec3<f32>(right, bottom, 0.0),
        vec3<f32>(right, top, 0.0),
        vec3<f32>(left, top, 0.0),
    );
}

fn build_horizontal_quad(idx: u32, value: f32, per_group_width: f32, local_offset: f32) -> array<vec3<f32>, 4u> {
    let center = (f32(idx) + 1.0) + local_offset;
    let half = per_group_width * 0.5;
    let bottom = center - half;
    let top = center + half;
    let left = 0.0;
    let right = value;

    return array<vec3<f32>, 4u>(
        vec3<f32>(left, bottom, 0.0),
        vec3<f32>(right, bottom, 0.0),
        vec3<f32>(right, top, 0.0),
        vec3<f32>(left, top, 0.0),
    );
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let safe_group_count = max(params.group_count, 1u);
    let per_group_width = max(params.bar_width / f32(safe_group_count), 0.01);
    let group_offset_start = -params.bar_width * 0.5;
    let local_offset = group_offset_start
        + per_group_width * f32(min(params.group_index, safe_group_count - 1u))
        + per_group_width * 0.5;

    let value = values[idx];
    let quad = if (params.orientation == 0u) {
        build_vertical_quad(idx, value, per_group_width, local_offset)
    } else {
        build_horizontal_quad(idx, value, per_group_width, local_offset)
    };

    let base_index = idx * VERTICES_PER_BAR;
    write_vertices(base_index, quad);
}
"#;

pub const F64: &str = r#"override WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

const VERTICES_PER_BAR: u32 = 6u;

struct VertexRaw {
    data: array<f32, 12u>;
};

struct BarParams {
    color: vec4<f32>,
    bar_width: f32,
    count: u32,
    group_index: u32,
    group_count: u32,
    orientation: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> values: array<f64>;

@group(0) @binding(1)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(2)
var<uniform> params: BarParams;

fn encode_vertex(position: vec3<f32>) -> VertexRaw {
    var vertex: VertexRaw;
    vertex.data[0u] = position.x;
    vertex.data[1u] = position.y;
    vertex.data[2u] = position.z;
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

fn write_vertices(base_index: u32, quad: array<vec3<f32>, 4u>) {
    out_vertices[base_index + 0u] = encode_vertex(quad[0u]);
    out_vertices[base_index + 1u] = encode_vertex(quad[1u]);
    out_vertices[base_index + 2u] = encode_vertex(quad[2u]);
    out_vertices[base_index + 3u] = encode_vertex(quad[0u]);
    out_vertices[base_index + 4u] = encode_vertex(quad[2u]);
    out_vertices[base_index + 5u] = encode_vertex(quad[3u]);
}

fn build_vertical_quad(idx: u32, value: f32, per_group_width: f32, local_offset: f32) -> array<vec3<f32>, 4u> {
    let center = (f32(idx) + 1.0) + local_offset;
    let half = per_group_width * 0.5;
    let left = center - half;
    let right = center + half;
    let bottom = 0.0;
    let top = value;

    return array<vec3<f32>, 4u>(
        vec3<f32>(left, bottom, 0.0),
        vec3<f32>(right, bottom, 0.0),
        vec3<f32>(right, top, 0.0),
        vec3<f32>(left, top, 0.0),
    );
}

fn build_horizontal_quad(idx: u32, value: f32, per_group_width: f32, local_offset: f32) -> array<vec3<f32>, 4u> {
    let center = (f32(idx) + 1.0) + local_offset;
    let half = per_group_width * 0.5;
    let bottom = center - half;
    let top = center + half;
    let left = 0.0;
    let right = value;

    return array<vec3<f32>, 4u>(
        vec3<f32>(left, bottom, 0.0),
        vec3<f32>(right, bottom, 0.0),
        vec3<f32>(right, top, 0.0),
        vec3<f32>(left, top, 0.0),
    );
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let safe_group_count = max(params.group_count, 1u);
    let per_group_width = max(params.bar_width / f32(safe_group_count), 0.01);
    let group_offset_start = -params.bar_width * 0.5;
    let local_offset = group_offset_start
        + per_group_width * f32(min(params.group_index, safe_group_count - 1u))
        + per_group_width * 0.5;

    let value = f32(values[idx]);
    let quad = if (params.orientation == 0u) {
        build_vertical_quad(idx, value, per_group_width, local_offset)
    } else {
        build_horizontal_quad(idx, value, per_group_width, local_offset)
    };

    let base_index = idx * VERTICES_PER_BAR;
    write_vertices(base_index, quad);
}
"#;
