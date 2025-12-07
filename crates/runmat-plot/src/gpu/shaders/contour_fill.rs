pub const F32: &str = r#"override WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

const VERTICES_PER_CELL: u32 = 6u;

struct VertexRaw {
    data: array<f32, 12u>;
}

struct ContourFillParams {
    base_z: f32,
    alpha: f32,
    x_len: u32,
    y_len: u32,
    color_table_len: u32,
    level_count: u32,
    cell_count: u32,
    _pad: u32,
}

@group(0) @binding(0)
var<storage, read> buf_x: array<f32>;

@group(0) @binding(1)
var<storage, read> buf_y: array<f32>;

@group(0) @binding(2)
var<storage, read> buf_z: array<f32>;

@group(0) @binding(3)
var<storage, read> color_table: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read> level_values: array<f32>;

@group(0) @binding(5)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(6)
var<uniform> params: ContourFillParams;

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

fn band_index(value: f32) -> u32 {
    var idx: u32 = 0u;
    let last = params.level_count - 1u;
    loop {
        if (idx >= last) {
            break;
        }
        if (value < level_values[idx + 1u]) {
            break;
        }
        idx = idx + 1u;
    }
    return min(idx, params.color_table_len - 1u);
}

fn sample_color(value: f32) -> vec4<f32> {
    if (params.color_table_len == 0u) {
        return vec4<f32>(1.0, 1.0, 1.0, params.alpha);
    }
    let idx = band_index(value);
    let color = color_table[idx];
    return vec4<f32>(color.xyz, color.w * params.alpha);
}

fn write_triangle(base_index: u32, positions: array<vec3<f32>, 3>, colors: array<vec4<f32>, 3>) {
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        out_vertices[base_index + i] = encode_vertex(positions[i], colors[i]);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_idx = gid.x;
    if (cell_idx >= params.cell_count) {
        return;
    }

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

    let colors = array<vec4<f32>, 4>(
        sample_color(z00),
        sample_color(z10),
        sample_color(z11),
        sample_color(z01),
    );

    let base_vertex = cell_idx * VERTICES_PER_CELL;

    let tri1_positions = array<vec3<f32>, 3>(
        vec3<f32>(x0, y0, params.base_z),
        vec3<f32>(x1, y0, params.base_z),
        vec3<f32>(x1, y1, params.base_z),
    );
    let tri1_colors = array<vec4<f32>, 3>(colors[0], colors[1], colors[2]);
    write_triangle(base_vertex, tri1_positions, tri1_colors);

    let tri2_positions = array<vec3<f32>, 3>(
        vec3<f32>(x0, y0, params.base_z),
        vec3<f32>(x1, y1, params.base_z),
        vec3<f32>(x0, y1, params.base_z),
    );
    let tri2_colors = array<vec4<f32>, 3>(colors[0], colors[2], colors[3]);
    write_triangle(base_vertex + 3u, tri2_positions, tri2_colors);
}
"#;

pub const F64: &str = r#"override WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

const VERTICES_PER_CELL: u32 = 6u;

struct VertexRaw {
    data: array<f32, 12u>;
}

struct ContourFillParams {
    base_z: f32,
    alpha: f32,
    x_len: u32,
    y_len: u32,
    color_table_len: u32,
    level_count: u32,
    cell_count: u32,
    _pad: u32,
}

@group(0) @binding(0)
var<storage, read> buf_x: array<f32>;

@group(0) @binding(1)
var<storage, read> buf_y: array<f32>;

@group(0) @binding(2)
var<storage, read> buf_z: array<f64>;

@group(0) @binding(3)
var<storage, read> color_table: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read> level_values: array<f32>;

@group(0) @binding(5)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(6)
var<uniform> params: ContourFillParams;

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

fn band_index(value: f32) -> u32 {
    var idx: u32 = 0u;
    let last = params.level_count - 1u;
    loop {
        if (idx >= last) {
            break;
        }
        if (value < level_values[idx + 1u]) {
            break;
        }
        idx = idx + 1u;
    }
    return min(idx, params.color_table_len - 1u);
}

fn sample_color(value: f32) -> vec4<f32> {
    if (params.color_table_len == 0u) {
        return vec4<f32>(1.0, 1.0, 1.0, params.alpha);
    }
    let idx = band_index(value);
    let color = color_table[idx];
    return vec4<f32>(color.xyz, color.w * params.alpha);
}

fn write_triangle(base_index: u32, positions: array<vec3<f32>, 3>, colors: array<vec4<f32>, 3>) {
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        out_vertices[base_index + i] = encode_vertex(positions[i], colors[i]);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_idx = gid.x;
    if (cell_idx >= params.cell_count) {
        return;
    }

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

    let colors = array<vec4<f32>, 4>(
        sample_color(z00),
        sample_color(z10),
        sample_color(z11),
        sample_color(z01),
    );

    let base_vertex = cell_idx * VERTICES_PER_CELL;

    let tri1_positions = array<vec3<f32>, 3>(
        vec3<f32>(x0, y0, params.base_z),
        vec3<f32>(x1, y0, params.base_z),
        vec3<f32>(x1, y1, params.base_z),
    );
    let tri1_colors = array<vec4<f32>, 3>(colors[0], colors[1], colors[2]);
    write_triangle(base_vertex, tri1_positions, tri1_colors);

    let tri2_positions = array<vec3<f32>, 3>(
        vec3<f32>(x0, y0, params.base_z),
        vec3<f32>(x1, y1, params.base_z),
        vec3<f32>(x0, y1, params.base_z),
    );
    let tri2_colors = array<vec4<f32>, 3>(colors[0], colors[2], colors[3]);
    write_triangle(base_vertex + 3u, tri2_positions, tri2_colors);
}
"#;
