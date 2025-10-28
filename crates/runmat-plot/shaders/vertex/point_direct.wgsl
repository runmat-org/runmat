// Direct point shader: expands each input point into a small screen-space quad
// using uniform viewport mapping and a fixed marker size in pixels.

struct Uniforms {
    data_min: vec2<f32>,
    data_max: vec2<f32>,
    viewport_min: vec2<f32>,
    viewport_max: vec2<f32>,
    viewport_px: vec2<f32>, // width, height in pixels
}

struct PointStyleUniforms {
    face_color: vec4<f32>,
    edge_color: vec4<f32>,
    edge_thickness_px: f32,
    marker_shape: u32,
    _pad: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;
@group(1) @binding(0)
var<uniform> styleU: PointStyleUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tex_coords: vec2<f32>,
}

struct VSOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) center_ndc: vec2<f32>,
    @location(3) half_ndc: vec2<f32>,
    @location(4) size_px: f32,
}

// We render each vertex as two triangles (a quad) by issuing 6 vertices per input point on the CPU.
// To keep things simple here, we interpret tex_coords.x as a small index 0..5 indicating which
// corner of the quad this vertex corresponds to. The CPU packs that when building the vertex buffer.

@vertex
fn vs_main(input: VertexInput) -> VSOut {
    var out: VSOut;

    // Map data to viewport NDC
    let data_range = uniforms.data_max - uniforms.data_min;
    let viewport_range = uniforms.viewport_max - uniforms.viewport_min;
    let normalized = (input.position.xy - uniforms.data_min) / data_range;
    let center = uniforms.viewport_min + normalized * viewport_range;

    // Marker size in pixels (use normal.z to pass size if desired; default 4px)
    let size_px = max(2.0, input.normal.z);
    // Convert half-size from pixels into NDC inside the viewport
    let half_w_ndc = (0.5 * size_px) / uniforms.viewport_px.x * viewport_range.x;
    let half_h_ndc = (0.5 * size_px) / uniforms.viewport_px.y * viewport_range.y;

    let corner = input.tex_coords; // expects values in {-1,1}
    let offset_ndc = vec2<f32>(corner.x * half_w_ndc, corner.y * half_h_ndc);

    let ndc = center + offset_ndc;
    out.clip_position = vec4<f32>(ndc, 0.0, 1.0);
    out.color = input.color;
    out.uv = (corner + vec2<f32>(1.0,1.0)) * 0.5; // map to 0..1
    out.center_ndc = center;
    out.half_ndc = vec2<f32>(half_w_ndc, half_h_ndc);
    out.size_px = size_px;
    return out;
}

fn inside_triangle(pt: vec2<f32>) -> bool {
    let a = vec2<f32>(-1.0, -1.0);
    let b = vec2<f32>( 1.0, -1.0);
    let c = vec2<f32>( 0.0,  1.0);
    let v0 = c - a;
    let v1 = b - a;
    let v2 = pt - a;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    if (denom == 0.0) { return false; }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    return (u >= 0.0) && (v >= 0.0) && (w >= 0.0);
}

fn inside_hexagon(pt: vec2<f32>) -> bool {
    let a = vec2<f32>(1.0, 0.0);
    let b = vec2<f32>(0.5, 0.8660254);
    let c = vec2<f32>(-0.5, 0.8660254);
    let da = abs(dot(pt, a));
    let db = abs(dot(pt, b));
    let dc = abs(dot(pt, c));
    return max(da, max(db, dc)) <= 1.0;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
    // Compute distance from center in normalized quad space (-1..1)
    let p = input.uv * 2.0 - vec2<f32>(1.0, 1.0);
    let r = length(p);

    // Shape selection
    var inside: bool;
    switch (styleU.marker_shape) {
        case 0u: { // circle
            inside = r <= 1.0;
        }
        case 1u: { // square
            inside = true;
        }
        case 2u: { // triangle (up)
            inside = inside_triangle(p);
        }
        case 3u: { // diamond (L1 ball)
            inside = (abs(p.x) + abs(p.y)) <= 1.0;
        }
        case 4u: { // plus
            let w = 0.35;
            inside = (abs(p.x) <= w) || (abs(p.y) <= w);
        }
        case 5u: { // cross (X)
            let w = 0.30;
            inside = min(abs(p.x - p.y), abs(p.x + p.y)) <= w;
        }
        case 6u: { // star (plus + cross)
            let w1 = 0.30;
            let w2 = 0.25;
            let plus = (abs(p.x) <= w1) || (abs(p.y) <= w1);
            let cross = min(abs(p.x - p.y), abs(p.x + p.y)) <= w2;
            inside = plus || cross;
        }
        case 7u: { // hexagon
            inside = inside_hexagon(p);
        }
        default: {
            inside = true;
        }
    }
    if (!inside) { discard; }

    // Edge thickness normalized to marker radius: 2*edge_px / size_px
    let edge_norm = clamp(2.0 * styleU.edge_thickness_px / max(input.size_px, 1.0), 0.0, 1.0);

    if (styleU.marker_shape == 0u) {
        if (edge_norm > 0.0 && r > (1.0 - edge_norm)) {
            return styleU.edge_color;
        }
    }

    // face (prefer style face_color; fallback to vertex color alpha)
    return vec4<f32>(mix(input.color.rgb, styleU.face_color.rgb, styleU.face_color.a), input.color.a);
}


