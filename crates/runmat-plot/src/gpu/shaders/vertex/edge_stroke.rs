//! Screen-space wide lines for geometry edges.
//!
//! Each segment arrives as two triangles (six vertices). A vertex carries its own endpoint in
//! `position`, the segment's other endpoint in `normal`, and `(side, width_css_px)` in
//! `tex_coords`. The vertex shader offsets the endpoint perpendicular to the projected segment by
//! half the width and extends it past the endpoint by half the width (a square cap), so chained
//! segments of a polyline meet without gaps. Width is in CSS pixels and is scaled by the screen
//! uniform's pixels-per-point, so a 1.25 px edge has the same weight at 1× and 2×.
//!
//! The depth-tested variant then moves the stroke toward the camera by `EDGE_DEPTH_SHARE` of
//! its view depth (see [`source`]): 1.5e-5 is 0.9 mm at 60 m and 4.5 µm at 0.3 m, enough to win
//! against the edge's own faces at 32-bit float depth and small enough that a frame's edges
//! 3 mm under an airliner's skin stay hidden.

/// The share of view depth a depth-tested edge stroke moves toward the camera.
pub const EDGE_DEPTH_SHARE: f32 = 1.5e-5;

/// The shader for a pipeline: `depth_tested` strokes take the view-depth offset, written for
/// reversed-Z (`reversed_z`, near = 1) or standard depth (near = 0). In clip space, moving a
/// point toward the eye by a share `e` of its distance is `z + e·z` with reversed-Z and
/// `z + e·(z − w)` with standard depth, for perspective and parallel views alike.
pub fn source(depth_tested: bool, reversed_z: bool) -> String {
    let share = if depth_tested { EDGE_DEPTH_SHARE } else { 0.0 };
    let toward = if reversed_z { "clip.z" } else { "(clip.z - clip.w)" };
    SHADER
        .replace("EDGE_DEPTH_SHARE_VALUE", &format!("{share:e}"))
        .replace("EDGE_DEPTH_TOWARD", toward)
}

pub const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat3x4<f32>,
}

struct ScreenUniforms {
    viewport_px: vec2<f32>,
    pixels_per_point: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<uniform> screen: ScreenUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) other_endpoint: vec3<f32>,
    @location(3) stroke: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let mvp = uniforms.view_proj * uniforms.model;
    let near_clip = mvp * vec4<f32>(input.position, 1.0);
    let far_clip = mvp * vec4<f32>(input.other_endpoint, 1.0);
    let viewport = max(screen.viewport_px, vec2<f32>(1.0, 1.0));
    let near_px = near_clip.xy / max(abs(near_clip.w), 0.0001) * viewport;
    let far_px = far_clip.xy / max(abs(far_clip.w), 0.0001) * viewport;
    let delta = far_px - near_px;
    let length_px = length(delta);
    var tangent = vec2<f32>(1.0, 0.0);
    if length_px > 0.0001 {
        tangent = delta / length_px;
    }
    let perpendicular = vec2<f32>(-tangent.y, tangent.x);
    let width_px = max(input.stroke.y, 0.0) * max(screen.pixels_per_point, 0.5);
    // NDC spans two units across the viewport, so a half-width offset of h px is 2h/viewport.
    let offset_px = perpendicular * input.stroke.x * width_px - tangent * width_px;
    let offset_ndc = offset_px / viewport;
    let clip = vec4<f32>(near_clip.xy + offset_ndc * near_clip.w, near_clip.zw);
    out.clip_position = vec4<f32>(clip.xy, clip.z + f32(EDGE_DEPTH_SHARE_VALUE) * EDGE_DEPTH_TOWARD, clip.w);
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if input.color.a <= 0.0 {
        discard;
    }
    return input.color;
}
"#;
