pub const F32: &str = r#"override WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw {
    data: array<f32, 12u>;
};

struct SurfaceParams {
    min_z: f32,
    max_z: f32,
    alpha: f32,
    flatten: u32,
    x_len: u32,
    y_len: u32,
    color_table_len: u32,
    _pad: u32,
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
var<uniform> params: SurfaceParams;

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

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x_len = params.x_len;
    let y_len = params.y_len;
    let total = x_len * y_len;

    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let row = idx % x_len;
    let col = idx / x_len;

    let px = buf_x[row];
    let py = buf_y[col];
    let raw_z = buf_z[idx];
    let z_extent = max(params.max_z - params.min_z, 1e-6);
    let norm_z = (raw_z - params.min_z) / z_extent;

    let position_z = select(raw_z, 0.0, params.flatten == 1u);
    let tex_x = f32(row) / max(f32(x_len - 1u), 1.0);
    let tex_y = f32(col) / max(f32(y_len - 1u), 1.0);
    let color = sample_color(norm_z) * vec4<f32>(1.0, 1.0, 1.0, params.alpha);

    var vertex: VertexRaw;
    vertex.data[0u] = px;
    vertex.data[1u] = py;
    vertex.data[2u] = position_z;

    vertex.data[3u] = color.x;
    vertex.data[4u] = color.y;
    vertex.data[5u] = color.z;
    vertex.data[6u] = color.w;

    vertex.data[7u] = 0.0;
    vertex.data[8u] = 0.0;
    vertex.data[9u] = 1.0;

    vertex.data[10u] = tex_x;
    vertex.data[11u] = tex_y;

    out_vertices[idx] = vertex;
}
"#;

pub const F64: &str = r#"override WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw {
    data: array<f32, 12u>;
};

struct SurfaceParams {
    min_z: f32,
    max_z: f32,
    alpha: f32,
    flatten: u32,
    x_len: u32,
    y_len: u32,
    color_table_len: u32,
    _pad: u32,
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
var<uniform> params: SurfaceParams;

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

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x_len = params.x_len;
    let y_len = params.y_len;
    let total = x_len * y_len;

    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let row = idx % x_len;
    let col = idx / x_len;

    let px = buf_x[row];
    let py = buf_y[col];
    let raw_z64 = buf_z[idx];
    let raw_z = f32(raw_z64);
    let z_extent = max(params.max_z - params.min_z, 1e-6);
    let norm_z = (raw_z - params.min_z) / z_extent;

    let position_z = select(raw_z, 0.0, params.flatten == 1u);
    let tex_x = f32(row) / max(f32(x_len - 1u), 1.0);
    let tex_y = f32(col) / max(f32(y_len - 1u), 1.0);
    let color = sample_color(norm_z) * vec4<f32>(1.0, 1.0, 1.0, params.alpha);

    var vertex: VertexRaw;
    vertex.data[0u] = px;
    vertex.data[1u] = py;
    vertex.data[2u] = position_z;

    vertex.data[3u] = color.x;
    vertex.data[4u] = color.y;
    vertex.data[5u] = color.z;
    vertex.data[6u] = color.w;

    vertex.data[7u] = 0.0;
    vertex.data[8u] = 0.0;
    vertex.data[9u] = 1.0;

    vertex.data[10u] = tex_x;
    vertex.data[11u] = tex_y;

    out_vertices[idx] = vertex;
}
"#;
