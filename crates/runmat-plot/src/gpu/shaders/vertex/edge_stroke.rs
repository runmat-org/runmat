//! Screen-space wide lines for geometry edges.
//!
//! Each segment arrives as two triangles (six vertices). A vertex carries its own endpoint in
//! `position`, the segment's other endpoint in `normal`, and `(side, width_css_px)` in
//! `tex_coords`. The vertex shader offsets the endpoint perpendicular to the projected segment by
//! half the width and extends it past the endpoint by half the width (a square cap), so chained
//! segments of a polyline meet without gaps. Width is in CSS pixels and is scaled by the screen
//! uniform's pixels-per-point, so a 1.25 px edge has the same weight at 1× and 2×.
//!
//! The depth-tested variant moves the stroke toward the camera by `EDGE_DEPTH_SHARE` of its
//! view depth (see [`source`]): 1.5e-5 is 0.9 mm at 60 m and 4.5 µm at 0.3 m, enough to win
//! against the edge's own faces at 32-bit float depth.
//!
//! A stroke is wider than the line it draws, so the depth test alone compares the stroke's
//! side pixels with whatever surface lies beside the line: an edge 3 mm under an airliner's skin
//! wins wherever the skin falls away from the eye faster than 3 mm per stroke half-width, which
//! at an overview zoom is almost everywhere (a grid of frames and stringers shows through). The
//! depth-probed variant ([`EdgeStrokeShader::depth_probe`]) decides visibility once per point of
//! the line instead: the fragment samples the opaque scene's depth across the line's own center,
//! interpolates the surface there, and drops the whole stroke width when the line lies behind
//! it. It is hidden when the depth across the line is one smooth surface in front of it, or when
//! the mean of the two samples beside the line and each side's depth rate carried on to the
//! center all lie in front of it (which holds where the smoothness test is too strict for the
//! depth's precision, such as ribs under a wing skin seen at a grazing angle). Creases and
//! silhouettes keep their own edges: there the samples aren't smooth, and an extrapolated side
//! meets the edge itself or the background.

/// The share of view depth a depth-tested edge stroke moves toward the camera.
pub const EDGE_DEPTH_SHARE: f32 = 1.5e-5;

/// Which edge stroke shader a pipeline uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeStrokeShader {
    pub depth_tested: bool,
    pub reversed_z: bool,
    /// Placements come from an instance buffer (`core::InstanceRaw`, locations 4 to 8).
    pub instanced: bool,
    /// Samples the scene depth at group 2: `Some(multisampled)` for the depth texture's kind.
    pub depth_probe: Option<bool>,
}

/// The shader for a pipeline: `depth_tested` strokes take the view-depth offset, written for
/// reversed-Z (`reversed_z`, near = 1) or standard depth (near = 0). In clip space, moving a
/// point toward the eye by a share `e` of its distance is `z + e·z` with reversed-Z and
/// `z + e·(z − w)` with standard depth, for perspective and parallel views alike.
pub fn source(depth_tested: bool, reversed_z: bool) -> String {
    source_for(EdgeStrokeShader {
        depth_tested,
        reversed_z,
        instanced: false,
        depth_probe: None,
    })
}

/// The shader for any [`EdgeStrokeShader`] variant.
pub fn source_for(shader: EdgeStrokeShader) -> String {
    let share = if shader.depth_tested {
        EDGE_DEPTH_SHARE
    } else {
        0.0
    };
    let toward = if shader.reversed_z {
        "clip.z"
    } else {
        "(clip.z - clip.w)"
    };
    let (instance_input, place_a, place_b, tint, weight) = if shader.instanced {
        (
            INSTANCE_INPUT,
            "place(inst, input.position)",
            "place(inst, input.other_endpoint)",
            "mix(input.color, vec4<f32>(inst.tint.rgb, input.color.a), inst.tint.a)",
            "max(inst.extra.x, 1.0)",
        )
    } else {
        (
            "",
            "input.position",
            "input.other_endpoint",
            "input.color",
            "1.0",
        )
    };
    let params = if shader.instanced {
        "input: VertexInput, inst: InstanceInput"
    } else {
        "input: VertexInput"
    };
    let (probe_binding, probe_fragment) = match shader.depth_probe {
        None => (String::new(), String::new()),
        Some(multisampled) => {
            let (kind, sample) = if multisampled {
                ("texture_depth_multisampled_2d", ", 0")
            } else {
                ("texture_depth_2d", ", 0")
            };
            let nearer = if shader.reversed_z { ">" } else { "<" };
            (
                PROBE_BINDING
                    .replace("DEPTH_KIND", kind)
                    .replace("SAMPLE_ARG", sample),
                PROBE_FRAGMENT.replace("NEARER", nearer),
            )
        }
    };
    SHADER
        .replace("INSTANCE_INPUT", instance_input)
        .replace("VS_PARAMS", params)
        .replace("PLACE_A", place_a)
        .replace("PLACE_B", place_b)
        .replace("TINT", tint)
        .replace("WEIGHT", weight)
        .replace("PROBE_BINDING", &probe_binding)
        .replace("PROBE_FRAGMENT", &probe_fragment)
        .replace("EDGE_DEPTH_SHARE_VALUE", &format!("{share:e}"))
        .replace("EDGE_DEPTH_TOWARD", toward)
}

const INSTANCE_INPUT: &str = r#"
struct InstanceInput {
    @location(4) row0: vec4<f32>,
    @location(5) row1: vec4<f32>,
    @location(6) row2: vec4<f32>,
    @location(7) tint: vec4<f32>,
    @location(8) extra: vec4<f32>,
}

fn place(inst: InstanceInput, p: vec3<f32>) -> vec3<f32> {
    let h = vec4<f32>(p, 1.0);
    return vec3<f32>(dot(inst.row0, h), dot(inst.row1, h), dot(inst.row2, h));
}
"#;

const PROBE_BINDING: &str = r#"
@group(2) @binding(0) var scene_depth: DEPTH_KIND;

fn load_depth(p: vec2<i32>) -> f32 {
    let size = vec2<i32>(textureDimensions(scene_depth)) - vec2<i32>(1, 1);
    return textureLoad(scene_depth, clamp(p, vec2<i32>(0, 0), size)SAMPLE_ARG);
}

// The depth at a framebuffer position, bilinear between pixel centers.
fn depth_at(p: vec2<f32>) -> f32 {
    let f = p - vec2<f32>(0.5, 0.5);
    let i = vec2<i32>(floor(f));
    let t = f - floor(f);
    let d00 = load_depth(i);
    let d10 = load_depth(i + vec2<i32>(1, 0));
    let d01 = load_depth(i + vec2<i32>(0, 1));
    let d11 = load_depth(i + vec2<i32>(1, 1));
    return mix(mix(d00, d10, t.x), mix(d01, d11, t.x), t.y);
}
"#;

const PROBE_FRAGMENT: &str = r#"
    // The line's own center at this point along it, in framebuffer pixels.
    let viewport = max(screen.viewport_px, vec2<f32>(1.0, 1.0));
    let frag_ndc = input.stroke_clip.xy / input.stroke_clip.w;
    let center_ndc = input.center_clip.xy / input.center_clip.w;
    let center = input.clip_position.xy + (center_ndc - frag_ndc) * vec2<f32>(0.5, -0.5) * viewport;
    let across = normalize(vec2<f32>(input.across.x, -input.across.y) + vec2<f32>(1.0e-12, 0.0));
    let line_depth = input.center_clip.z / input.center_clip.w;
    let m15 = depth_at(center - 1.5 * across);
    let m05 = depth_at(center - 0.5 * across);
    let p05 = depth_at(center + 0.5 * across);
    let p15 = depth_at(center + 1.5 * across);
    // The surface at the line's center, three ways: the mean of the two near samples, and each
    // side's rate carried on to the center. Over one smooth surface all three agree. Beside a
    // valley crease the mean comes nearer than the crease, but each side's extrapolation meets
    // the crease itself; on a convex surface the mean falls behind the surface; across a
    // silhouette the background side extrapolates far. The line is hidden only when all three
    // put the surface in front of it, so an edge's own faces never hide it.
    let mean = 0.5 * (m05 + p05);
    let from_left = m05 + 0.5 * (m05 - m15);
    let from_right = p05 + 0.5 * (p05 - p15);
    if mean NEARER line_depth && from_left NEARER line_depth && from_right NEARER line_depth {
        discard;
    }
    // One smooth surface across the line (its depth changes at a steady rate) in front of it.
    let left = m05 - m15;
    let right = p15 - p05;
    let smooth_surface = abs(left - right) <= 0.5 * max(abs(left), abs(right)) + 1.0e-7;
    if smooth_surface && mean NEARER line_depth {
        discard;
    }
"#;

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
PROBE_BINDING
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) other_endpoint: vec3<f32>,
    @location(3) stroke: vec2<f32>,
}
INSTANCE_INPUT
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    // The unwidened endpoint (with the depth offset) and this vertex's own clip position, so
    // the fragment finds the line's center; the stroke's perpendicular in pixels.
    @location(1) center_clip: vec4<f32>,
    @location(2) stroke_clip: vec4<f32>,
    @location(3) across: vec2<f32>,
}

@vertex
fn vs_main(VS_PARAMS) -> VertexOutput {
    var out: VertexOutput;
    let mvp = uniforms.view_proj * uniforms.model;
    let near_clip = mvp * vec4<f32>(PLACE_A, 1.0);
    let far_clip = mvp * vec4<f32>(PLACE_B, 1.0);
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
    let width_px = max(input.stroke.y, 0.0) * WEIGHT * max(screen.pixels_per_point, 0.5);
    // NDC spans two units across the viewport, so a half-width offset of h px is 2h/viewport.
    let offset_px = perpendicular * input.stroke.x * width_px - tangent * width_px;
    let offset_ndc = offset_px / viewport;
    let clip = vec4<f32>(near_clip.xy + offset_ndc * near_clip.w, near_clip.zw);
    let depth = clip.z + f32(EDGE_DEPTH_SHARE_VALUE) * EDGE_DEPTH_TOWARD;
    out.clip_position = vec4<f32>(clip.xy, depth, clip.w);
    out.center_clip = vec4<f32>(near_clip.xy, depth, near_clip.w);
    out.stroke_clip = out.clip_position;
    // One sign at both ends (each end's tangent points at the other end).
    out.across = select(-perpendicular, perpendicular, perpendicular.x > 0.0 || (perpendicular.x == 0.0 && perpendicular.y >= 0.0));
    out.color = TINT;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if input.color.a <= 0.0 {
        discard;
    }
PROBE_FRAGMENT
    return input.color;
}
"#;
