pub const SHADER: &str = r#"// Camera-space marker shader: renders each marker as a billboard quad.

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat3x4<f32>,
}

struct PointStyleUniforms {
    face_color: vec4<f32>,
    edge_color: vec4<f32>,
    edge_thickness_px: f32,
    marker_shape: u32,
    _pad: vec2<f32>,
}

struct MarkerScreenUniforms {
    viewport_px: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;
@group(1) @binding(0)
var<uniform> styleU: PointStyleUniforms;
@group(2) @binding(0)
var<uniform> screenU: MarkerScreenUniforms;

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
    @location(2) size_px: f32,
}

@vertex
fn vs_main(input: VertexInput) -> VSOut {
    var out: VSOut;
    let world_position = uniforms.model * vec4<f32>(input.position, 1.0);
    var clip = uniforms.view_proj * world_position;

    let size_px = max(2.0, input.normal.z);
    let vp = max(screenU.viewport_px, vec2<f32>(1.0, 1.0));
    let half_px = 0.5 * size_px;
    let ndc_per_px = vec2<f32>(2.0 / vp.x, 2.0 / vp.y);
    let ndc_offset = input.tex_coords * half_px * ndc_per_px;
    let clip_xy = clip.xy + ndc_offset * clip.w;
    clip = vec4<f32>(clip_xy, clip.z, clip.w);

    out.clip_position = clip;
    out.color = input.color;
    out.uv = (input.tex_coords + vec2<f32>(1.0, 1.0)) * 0.5;
    out.size_px = size_px;
    return out;
}

fn inside_triangle(pt: vec2<f32>) -> bool {
    let a = vec2<f32>(-1.0, -1.0);
    let b = vec2<f32>(1.0, -1.0);
    let c = vec2<f32>(0.0, 1.0);
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
    let p = input.uv * 2.0 - vec2<f32>(1.0, 1.0);
    let r = length(p);

    var inside: bool;
    switch (styleU.marker_shape) {
        case 0u: { inside = r <= 1.0; }
        case 1u: { inside = true; }
        case 2u: { inside = inside_triangle(p); }
        case 3u: { inside = (abs(p.x) + abs(p.y)) <= 1.0; }
        case 4u: {
            let w = 0.35;
            inside = (abs(p.x) <= w) || (abs(p.y) <= w);
        }
        case 5u: {
            let w = 0.30;
            inside = min(abs(p.x - p.y), abs(p.x + p.y)) <= w;
        }
        case 6u: {
            let w1 = 0.30;
            let w2 = 0.25;
            let plus = (abs(p.x) <= w1) || (abs(p.y) <= w1);
            let cross = min(abs(p.x - p.y), abs(p.x + p.y)) <= w2;
            inside = plus || cross;
        }
        case 7u: {
            inside = inside_hexagon(p);
        }
        default: {
            inside = true;
        }
    }
    if (!inside) {
        discard;
    }

    let edge_norm = clamp(2.0 * styleU.edge_thickness_px / max(input.size_px, 1.0), 0.0, 1.0);
    let face_rgb = mix(input.color.rgb, styleU.face_color.rgb, styleU.face_color.a);
    let edge_mix = clamp(styleU.edge_color.a, 0.0, 1.0);
    let edge_rgb = mix(input.color.rgb, styleU.edge_color.rgb, edge_mix);

    if (styleU.marker_shape == 0u) {
        if (edge_norm > 0.0 && r > (1.0 - edge_norm)) {
            return vec4<f32>(edge_rgb, input.color.a);
        }
    }

    return vec4<f32>(face_rgb, input.color.a);
}
"#;
