pub const F32: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw {
    data: array<f32, 12u>,
};

struct ScalarPoint {
    pos: vec2<f32>,
    value: f32,
};

struct Poly {
    points: array<ScalarPoint, 5u>,
    count: u32,
};

struct ContourFillParams {
    base_z: f32,
    alpha: f32,
    x_len: u32,
    y_len: u32,
    color_table_len: u32,
    band_count: u32,
    cell_count: u32,
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
var<storage, read> buf_z: array<f32>;

@group(0) @binding(3)
var<storage, read> color_table: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read> level_values: array<f32>;

@group(0) @binding(5)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(6)
var<uniform> params: ContourFillParams;

@group(0) @binding(7)
var<storage, read_write> indirect: IndirectArgs;

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

fn make_point(pos: vec2<f32>, value: f32) -> ScalarPoint {
    var point: ScalarPoint;
    point.pos = pos;
    point.value = value;
    return point;
}

fn push_point(poly: ptr<function, Poly>, point: ScalarPoint) {
    if ((*poly).count < 5u) {
        (*poly).points[(*poly).count] = point;
        (*poly).count = (*poly).count + 1u;
    }
}

fn interpolate_point(a: ScalarPoint, b: ScalarPoint, threshold: f32) -> ScalarPoint {
    let delta = b.value - a.value;
    let t = if (abs(delta) <= 1e-6) { 0.5 } else { clamp((threshold - a.value) / delta, 0.0, 1.0) };
    return make_point(mix(a.pos, b.pos, t), threshold);
}

fn inside_lower(value: f32, threshold: f32) -> bool {
    return value >= threshold;
}

fn inside_upper(value: f32, threshold: f32, inclusive: bool) -> bool {
    if (inclusive) {
        return value <= threshold;
    }
    return value < threshold;
}

fn clip_lower(input: Poly, threshold: f32) -> Poly {
    var out: Poly;
    out.count = 0u;
    if (input.count == 0u) {
        return out;
    }
    var prev = input.points[input.count - 1u];
    var prev_inside = inside_lower(prev.value, threshold);
    for (var i: u32 = 0u; i < input.count; i = i + 1u) {
        let curr = input.points[i];
        let curr_inside = inside_lower(curr.value, threshold);
        if (curr_inside != prev_inside) {
            push_point(&out, interpolate_point(prev, curr, threshold));
        }
        if (curr_inside) {
            push_point(&out, curr);
        }
        prev = curr;
        prev_inside = curr_inside;
    }
    return out;
}

fn clip_upper(input: Poly, threshold: f32, inclusive: bool) -> Poly {
    var out: Poly;
    out.count = 0u;
    if (input.count == 0u) {
        return out;
    }
    var prev = input.points[input.count - 1u];
    var prev_inside = inside_upper(prev.value, threshold, inclusive);
    for (var i: u32 = 0u; i < input.count; i = i + 1u) {
        let curr = input.points[i];
        let curr_inside = inside_upper(curr.value, threshold, inclusive);
        if (curr_inside != prev_inside) {
            push_point(&out, interpolate_point(prev, curr, threshold));
        }
        if (curr_inside) {
            push_point(&out, curr);
        }
        prev = curr;
        prev_inside = curr_inside;
    }
    return out;
}

fn emit_triangle(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, color: vec4<f32>) {
    let base = atomicAdd(&(indirect.vertex_count), 3u);
    out_vertices[base] = encode_vertex(vec3<f32>(a, params.base_z), color);
    out_vertices[base + 1u] = encode_vertex(vec3<f32>(b, params.base_z), color);
    out_vertices[base + 2u] = encode_vertex(vec3<f32>(c, params.base_z), color);
}

fn emit_band_triangle(a: ScalarPoint, b: ScalarPoint, c: ScalarPoint, lo: f32, hi: f32, include_hi: bool, color: vec4<f32>) {
    var poly: Poly;
    poly.count = 3u;
    poly.points[0u] = a;
    poly.points[1u] = b;
    poly.points[2u] = c;
    let clipped_lower = clip_lower(poly, lo);
    let clipped = clip_upper(clipped_lower, hi, include_hi);
    if (clipped.count < 3u) {
        return;
    }
    let origin = clipped.points[0u].pos;
    for (var i: u32 = 1u; i + 1u < clipped.count; i = i + 1u) {
        emit_triangle(origin, clipped.points[i].pos, clipped.points[i + 1u].pos, color);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.cell_count * params.band_count;
    let invocation = gid.x;
    if (invocation >= total) {
        return;
    }

    let band_idx = invocation % params.band_count;
    let cell_idx = invocation / params.band_count;
    let cells_x = params.x_len - 1u;
    let row = cell_idx % cells_x;
    let col = cell_idx / cells_x;
    let base_index = row + col * params.x_len;
    let idx00 = base_index;
    let idx10 = idx00 + 1u;
    let idx01 = idx00 + params.x_len;
    let idx11 = idx01 + 1u;

    let p0 = make_point(vec2<f32>(buf_x[row], buf_y[col]), buf_z[idx00]);
    let p1 = make_point(vec2<f32>(buf_x[row + 1u], buf_y[col]), buf_z[idx10]);
    let p2 = make_point(vec2<f32>(buf_x[row + 1u], buf_y[col + 1u]), buf_z[idx11]);
    let p3 = make_point(vec2<f32>(buf_x[row], buf_y[col + 1u]), buf_z[idx01]);
    let lo = level_values[band_idx];
    let hi = level_values[band_idx + 1u];
    let include_hi = band_idx + 1u == params.band_count;
    let base_color = color_table[min(band_idx, params.color_table_len - 1u)];
    let color = vec4<f32>(base_color.xyz, base_color.w * params.alpha);
    emit_band_triangle(p0, p1, p2, lo, hi, include_hi, color);
    emit_band_triangle(p0, p2, p3, lo, hi, include_hi, color);
}
"#;

pub const F64: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct VertexRaw {
    data: array<f32, 12u>,
};

struct ScalarPoint {
    pos: vec2<f32>,
    value: f32,
};

struct Poly {
    points: array<ScalarPoint, 5u>,
    count: u32,
};

struct ContourFillParams {
    base_z: f32,
    alpha: f32,
    x_len: u32,
    y_len: u32,
    color_table_len: u32,
    band_count: u32,
    cell_count: u32,
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
var<storage, read> buf_z: array<f64>;

@group(0) @binding(3)
var<storage, read> color_table: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read> level_values: array<f32>;

@group(0) @binding(5)
var<storage, read_write> out_vertices: array<VertexRaw>;

@group(0) @binding(6)
var<uniform> params: ContourFillParams;

@group(0) @binding(7)
var<storage, read_write> indirect: IndirectArgs;

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

fn make_point(pos: vec2<f32>, value: f32) -> ScalarPoint {
    var point: ScalarPoint;
    point.pos = pos;
    point.value = value;
    return point;
}

fn push_point(poly: ptr<function, Poly>, point: ScalarPoint) {
    if ((*poly).count < 5u) {
        (*poly).points[(*poly).count] = point;
        (*poly).count = (*poly).count + 1u;
    }
}

fn interpolate_point(a: ScalarPoint, b: ScalarPoint, threshold: f32) -> ScalarPoint {
    let delta = b.value - a.value;
    let t = if (abs(delta) <= 1e-6) { 0.5 } else { clamp((threshold - a.value) / delta, 0.0, 1.0) };
    return make_point(mix(a.pos, b.pos, t), threshold);
}

fn inside_lower(value: f32, threshold: f32) -> bool {
    return value >= threshold;
}

fn inside_upper(value: f32, threshold: f32, inclusive: bool) -> bool {
    if (inclusive) {
        return value <= threshold;
    }
    return value < threshold;
}

fn clip_lower(input: Poly, threshold: f32) -> Poly {
    var out: Poly;
    out.count = 0u;
    if (input.count == 0u) {
        return out;
    }
    var prev = input.points[input.count - 1u];
    var prev_inside = inside_lower(prev.value, threshold);
    for (var i: u32 = 0u; i < input.count; i = i + 1u) {
        let curr = input.points[i];
        let curr_inside = inside_lower(curr.value, threshold);
        if (curr_inside != prev_inside) {
            push_point(&out, interpolate_point(prev, curr, threshold));
        }
        if (curr_inside) {
            push_point(&out, curr);
        }
        prev = curr;
        prev_inside = curr_inside;
    }
    return out;
}

fn clip_upper(input: Poly, threshold: f32, inclusive: bool) -> Poly {
    var out: Poly;
    out.count = 0u;
    if (input.count == 0u) {
        return out;
    }
    var prev = input.points[input.count - 1u];
    var prev_inside = inside_upper(prev.value, threshold, inclusive);
    for (var i: u32 = 0u; i < input.count; i = i + 1u) {
        let curr = input.points[i];
        let curr_inside = inside_upper(curr.value, threshold, inclusive);
        if (curr_inside != prev_inside) {
            push_point(&out, interpolate_point(prev, curr, threshold));
        }
        if (curr_inside) {
            push_point(&out, curr);
        }
        prev = curr;
        prev_inside = curr_inside;
    }
    return out;
}

fn emit_triangle(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, color: vec4<f32>) {
    let base = atomicAdd(&(indirect.vertex_count), 3u);
    out_vertices[base] = encode_vertex(vec3<f32>(a, params.base_z), color);
    out_vertices[base + 1u] = encode_vertex(vec3<f32>(b, params.base_z), color);
    out_vertices[base + 2u] = encode_vertex(vec3<f32>(c, params.base_z), color);
}

fn emit_band_triangle(a: ScalarPoint, b: ScalarPoint, c: ScalarPoint, lo: f32, hi: f32, include_hi: bool, color: vec4<f32>) {
    var poly: Poly;
    poly.count = 3u;
    poly.points[0u] = a;
    poly.points[1u] = b;
    poly.points[2u] = c;
    let clipped_lower = clip_lower(poly, lo);
    let clipped = clip_upper(clipped_lower, hi, include_hi);
    if (clipped.count < 3u) {
        return;
    }
    let origin = clipped.points[0u].pos;
    for (var i: u32 = 1u; i + 1u < clipped.count; i = i + 1u) {
        emit_triangle(origin, clipped.points[i].pos, clipped.points[i + 1u].pos, color);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.cell_count * params.band_count;
    let invocation = gid.x;
    if (invocation >= total) {
        return;
    }

    let band_idx = invocation % params.band_count;
    let cell_idx = invocation / params.band_count;
    let cells_x = params.x_len - 1u;
    let row = cell_idx % cells_x;
    let col = cell_idx / cells_x;
    let base_index = row + col * params.x_len;
    let idx00 = base_index;
    let idx10 = idx00 + 1u;
    let idx01 = idx00 + params.x_len;
    let idx11 = idx01 + 1u;

    let p0 = make_point(vec2<f32>(f32(buf_x[row]), f32(buf_y[col])), f32(buf_z[idx00]));
    let p1 = make_point(vec2<f32>(f32(buf_x[row + 1u]), f32(buf_y[col])), f32(buf_z[idx10]));
    let p2 = make_point(vec2<f32>(f32(buf_x[row + 1u]), f32(buf_y[col + 1u])), f32(buf_z[idx11]));
    let p3 = make_point(vec2<f32>(f32(buf_x[row]), f32(buf_y[col + 1u])), f32(buf_z[idx01]));
    let lo = level_values[band_idx];
    let hi = level_values[band_idx + 1u];
    let include_hi = band_idx + 1u == params.band_count;
    let base_color = color_table[min(band_idx, params.color_table_len - 1u)];
    let color = vec4<f32>(base_color.xyz, base_color.w * params.alpha);
    emit_band_triangle(p0, p1, p2, lo, hi, include_hi, color);
    emit_band_triangle(p0, p2, p3, lo, hi, include_hi, color);
}
"#;
