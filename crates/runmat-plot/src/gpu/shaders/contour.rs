pub const F32: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

const VERTICES_PER_INVOCATION: u32 = 4u;

struct VertexRaw {
    data: array<f32, 12u>,
};

struct ContourParams {
    min_z: f32,
    max_z: f32,
    base_z: f32,
    level_count: u32,
    x_len: u32,
    y_len: u32,
    color_table_len: u32,
    cell_count: u32,
};

@group(0) @binding(0)
var<storage, read> buf_x: array<f32>;

@group(0) @binding(1)
var<storage, read> buf_y: array<f32>;

@group(0) @binding(2)
var<storage, read> buf_z: array<f32>;

@group(0) @binding(3)
var<storage, read> color_table: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(5)
var<uniform> params: ContourParams;

@group(0) @binding(6)
var<storage, read> level_values: array<f32>;

fn sample_color(t: f32) -> vec4<f32> {
    let table_len = params.color_table_len;
    if (table_len <= 1u) {
        return color_table[0u];
    }
    let clamped = clamp(t, 0.0, 1.0);
    let scaled = clamped * f32(table_len - 1u);
    let lower = u32(scaled);
    let upper = min(lower + 1u, table_len - 1u);
    let frac = scaled - f32(lower);
    return mix(color_table[lower], color_table[upper], frac);
}

fn encode_vertex(position: vec3<f32>, color: vec4<f32>) -> VertexRaw {
    var vertex: VertexRaw;
    vertex.data[0u] = position.x;
    vertex.data[1u] = position.y;
    vertex.data[2u] = position.z;
    vertex.data[3u] = color.x;
    vertex.data[4u] = color.y;
    vertex.data[5u] = color.z;
    vertex.data[6u] = color.w;
    vertex.data[7u] = 0.0;
    vertex.data[8u] = 0.0;
    vertex.data[9u] = 1.0;
    vertex.data[10u] = 0.0;
    vertex.data[11u] = 0.0;
    return vertex;
}

fn interpolate_edge(edge: u32, corners: array<vec2<f32>, 4>, values: array<f32, 4>, level: f32) -> vec2<f32> {
    var a: vec2<f32>;
    var b: vec2<f32>;
    var va: f32;
    var vb: f32;
    switch edge {
        case 0u: { a = corners[0u]; b = corners[1u]; va = values[0u]; vb = values[1u]; }
        case 1u: { a = corners[1u]; b = corners[2u]; va = values[1u]; vb = values[2u]; }
        case 2u: { a = corners[2u]; b = corners[3u]; va = values[2u]; vb = values[3u]; }
        default: { a = corners[3u]; b = corners[0u]; va = values[3u]; vb = values[0u]; }
    }
    let denom = max(abs(vb - va), 1e-6);
    let t = clamp((level - va) / denom, 0.0, 1.0);
    return mix(a, b, t);
}

fn write_vertex_range(base_index: u32, segment_points: array<vec2<f32>, 4>, segment_count: u32, color: vec4<f32>) {
    for (var i: u32 = 0u; i < VERTICES_PER_INVOCATION; i = i + 1u) {
        let idx = base_index + i;
        let vertex = if (i < segment_count * 2u) {
            let pt = segment_points[i];
            encode_vertex(vec3<f32>(pt, params.base_z), color)
        } else {
            encode_vertex(vec3<f32>(0.0, 0.0, params.base_z), vec4<f32>(color.xyz, 0.0))
        };
        out_vertices[idx] = vertex;
    }
}

fn add_segment(
    edge_a: u32,
    edge_b: u32,
    corners: array<vec2<f32>, 4>,
    values: array<f32, 4>,
    level: f32,
    io_segments: ptr<function, array<vec2<f32>, 4>>,
    io_count: ptr<function, u32>,
) {
    if (*io_count) >= 2u {
        return;
    }
    let idx = (*io_count) * 2u;
    (*io_segments)[idx] = interpolate_edge(edge_a, corners, values, level);
    (*io_segments)[idx + 1u] = interpolate_edge(edge_b, corners, values, level);
    *io_count = *io_count + 1u;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.cell_count * params.level_count;
    let invocation = gid.x;
    if (invocation >= total) {
        return;
    }

    let level_idx = invocation % params.level_count;
    let cell_idx = invocation / params.level_count;
    let cells_x = params.x_len - 1u;
    let row = cell_idx % cells_x;
    let col = cell_idx / cells_x;

    let base_index = row + col * params.x_len;
    let idx00 = base_index;
    let idx10 = idx00 + 1u;
    let idx01 = idx00 + params.x_len;
    let idx11 = idx01 + 1u;

    let x0 = buf_x[row];
    let x1 = buf_x[row + 1u];
    let y0 = buf_y[col];
    let y1 = buf_y[col + 1u];

    let z00 = buf_z[idx00];
    let z10 = buf_z[idx10];
    let z11 = buf_z[idx11];
    let z01 = buf_z[idx01];

    let corners = array<vec2<f32>, 4>(
        vec2<f32>(x0, y0),
        vec2<f32>(x1, y0),
        vec2<f32>(x1, y1),
        vec2<f32>(x0, y1)
    );
    let values = array<f32, 4>(z00, z10, z11, z01);

    let level = level_values[level_idx];

    var case_index: u32 = 0u;
    if (z00 > level) { case_index = case_index | 1u; }
    if (z10 > level) { case_index = case_index | 2u; }
    if (z11 > level) { case_index = case_index | 4u; }
    if (z01 > level) { case_index = case_index | 8u; }

    var segments: array<vec2<f32>, 4>;
    var segment_count: u32 = 0u;

    switch case_index {
        case 0u, 15u: {}
        case 1u, 14u: { add_segment(3u, 0u, corners, values, level, &segments, &segment_count); }
        case 2u, 13u: { add_segment(0u, 1u, corners, values, level, &segments, &segment_count); }
        case 3u, 12u: { add_segment(3u, 1u, corners, values, level, &segments, &segment_count); }
        case 4u, 11u: { add_segment(1u, 2u, corners, values, level, &segments, &segment_count); }
        case 5u: {
            add_segment(3u, 2u, corners, values, level, &segments, &segment_count);
            add_segment(0u, 1u, corners, values, level, &segments, &segment_count);
        }
        case 10u: {
            add_segment(0u, 1u, corners, values, level, &segments, &segment_count);
            add_segment(3u, 2u, corners, values, level, &segments, &segment_count);
        }
        case 6u, 9u: { add_segment(0u, 2u, corners, values, level, &segments, &segment_count); }
        case 7u, 8u: { add_segment(3u, 2u, corners, values, level, &segments, &segment_count); }
    }

    let norm = (level - params.min_z) / max(params.max_z - params.min_z, 1e-6);
    let color = sample_color(norm);
    let base_vertex = invocation * VERTICES_PER_INVOCATION;
    write_vertex_range(base_vertex, segments, segment_count, color);
}
"#;

pub const F64: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

const VERTICES_PER_INVOCATION: u32 = 4u;

struct VertexRaw {
    data: array<f32, 12u>,
};

struct ContourParams {
    min_z: f32,
    max_z: f32,
    base_z: f32,
    level_count: u32,
    x_len: u32,
    y_len: u32,
    color_table_len: u32,
    cell_count: u32,
};

@group(0) @binding(0)
var<storage, read> buf_x: array<f32>;

@group(0) @binding(1)
var<storage, read> buf_y: array<f32>;

@group(0) @binding(2)
var<storage, read> buf_z: array<f64>;

@group(0) @binding(3)
var<storage, read> color_table: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(5)
var<uniform> params: ContourParams;

@group(0) @binding(6)
var<storage, read> level_values: array<f32>;

fn sample_color(t: f32) -> vec4<f32> {
    let table_len = params.color_table_len;
    if (table_len <= 1u) {
        return color_table[0u];
    }
    let clamped = clamp(t, 0.0, 1.0);
    let scaled = clamped * f32(table_len - 1u);
    let lower = u32(scaled);
    let upper = min(lower + 1u, table_len - 1u);
    let frac = scaled - f32(lower);
    return mix(color_table[lower], color_table[upper], frac);
}

fn encode_vertex(position: vec3<f32>, color: vec4<f32>) -> VertexRaw {
    var vertex: VertexRaw;
    vertex.data[0u] = position.x;
    vertex.data[1u] = position.y;
    vertex.data[2u] = position.z;
    vertex.data[3u] = color.x;
    vertex.data[4u] = color.y;
    vertex.data[5u] = color.z;
    vertex.data[6u] = color.w;
    vertex.data[7u] = 0.0;
    vertex.data[8u] = 0.0;
    vertex.data[9u] = 1.0;
    vertex.data[10u] = 0.0;
    vertex.data[11u] = 0.0;
    return vertex;
}

fn interpolate_edge(edge: u32, corners: array<vec2<f32>, 4>, values: array<f32, 4>, level: f32) -> vec2<f32> {
    var a: vec2<f32>;
    var b: vec2<f32>;
    var va: f32;
    var vb: f32;
    switch edge {
        case 0u: { a = corners[0u]; b = corners[1u]; va = values[0u]; vb = values[1u]; }
        case 1u: { a = corners[1u]; b = corners[2u]; va = values[1u]; vb = values[2u]; }
        case 2u: { a = corners[2u]; b = corners[3u]; va = values[2u]; vb = values[3u]; }
        default: { a = corners[3u]; b = corners[0u]; va = values[3u]; vb = values[0u]; }
    }
    let denom = max(abs(vb - va), 1e-6);
    let t = clamp((level - va) / denom, 0.0, 1.0);
    return mix(a, b, t);
}

fn write_vertex_range(base_index: u32, segment_points: array<vec2<f32>, 4>, segment_count: u32, color: vec4<f32>) {
    for (var i: u32 = 0u; i < VERTICES_PER_INVOCATION; i = i + 1u) {
        let idx = base_index + i;
        let vertex = if (i < segment_count * 2u) {
            let pt = segment_points[i];
            encode_vertex(vec3<f32>(pt, params.base_z), color)
        } else {
            encode_vertex(vec3<f32>(0.0, 0.0, params.base_z), vec4<f32>(color.xyz, 0.0))
        };
        out_vertices[idx] = vertex;
    }
}

fn add_segment(
    edge_a: u32,
    edge_b: u32,
    corners: array<vec2<f32>, 4>,
    values: array<f32, 4>,
    level: f32,
    io_segments: ptr<function, array<vec2<f32>, 4>>,
    io_count: ptr<function, u32>,
) {
    if (*io_count) >= 2u {
        return;
    }
    let idx = (*io_count) * 2u;
    (*io_segments)[idx] = interpolate_edge(edge_a, corners, values, level);
    (*io_segments)[idx + 1u] = interpolate_edge(edge_b, corners, values, level);
    *io_count = *io_count + 1u;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.cell_count * params.level_count;
    let invocation = gid.x;
    if (invocation >= total) {
        return;
    }

    let level_idx = invocation % params.level_count;
    let cell_idx = invocation / params.level_count;
    let cells_x = params.x_len - 1u;
    let row = cell_idx % cells_x;
    let col = cell_idx / cells_x;

    let base_index = row + col * params.x_len;
    let idx00 = base_index;
    let idx10 = idx00 + 1u;
    let idx01 = idx00 + params.x_len;
    let idx11 = idx01 + 1u;

    let x0 = buf_x[row];
    let x1 = buf_x[row + 1u];
    let y0 = buf_y[col];
    let y1 = buf_y[col + 1u];

    let z00 = f32(buf_z[idx00]);
    let z10 = f32(buf_z[idx10]);
    let z11 = f32(buf_z[idx11]);
    let z01 = f32(buf_z[idx01]);

    let corners = array<vec2<f32>, 4>(
        vec2<f32>(x0, y0),
        vec2<f32>(x1, y0),
        vec2<f32>(x1, y1),
        vec2<f32>(x0, y1)
    );
    let values = array<f32, 4>(z00, z10, z11, z01);

    let level = level_values[level_idx];

    var case_index: u32 = 0u;
    if (z00 > level) { case_index = case_index | 1u; }
    if (z10 > level) { case_index = case_index | 2u; }
    if (z11 > level) { case_index = case_index | 4u; }
    if (z01 > level) { case_index = case_index | 8u; }

    var segments: array<vec2<f32>, 4>;
    var segment_count: u32 = 0u;

    switch case_index {
        case 0u, 15u: {}
        case 1u, 14u: { add_segment(3u, 0u, corners, values, level, &segments, &segment_count); }
        case 2u, 13u: { add_segment(0u, 1u, corners, values, level, &segments, &segment_count); }
        case 3u, 12u: { add_segment(3u, 1u, corners, values, level, &segments, &segment_count); }
        case 4u, 11u: { add_segment(1u, 2u, corners, values, level, &segments, &segment_count); }
        case 5u: {
            add_segment(3u, 2u, corners, values, level, &segments, &segment_count);
            add_segment(0u, 1u, corners, values, level, &segments, &segment_count);
        }
        case 10u: {
            add_segment(0u, 1u, corners, values, level, &segments, &segment_count);
            add_segment(3u, 2u, corners, values, level, &segments, &segment_count);
        }
        case 6u, 9u: { add_segment(0u, 2u, corners, values, level, &segments, &segment_count); }
        case 7u, 8u: { add_segment(3u, 2u, corners, values, level, &segments, &segment_count); }
    }

    let norm = (level - params.min_z) / max(params.max_z - params.min_z, 1e-6);
    let color = sample_color(norm);
    let base_vertex = invocation * VERTICES_PER_INVOCATION;
    write_vertex_range(base_vertex, segments, segment_count, color);
}
"#;
