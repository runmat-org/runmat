pub const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat3x4<f32>,
}

struct ScreenUniforms {
    viewport_px: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<uniform> screen: ScreenUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    // The far endpoint of this stroke, used to find its screen-space normal.
    @location(2) other_endpoint: vec3<f32>,
    // x is the ribbon side; y selects annotation, selected edge, or selected point.
    @location(3) stroke: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) marker_uv: vec2<f32>,
    @location(2) style: f32,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let near_clip = uniforms.view_proj * uniforms.model * vec4<f32>(input.position, 1.0);
    let viewport = max(screen.viewport_px, vec2<f32>(1.0, 1.0));
    out.marker_uv = vec2<f32>(0.0, 0.0);
    out.style = input.stroke.y;
    if input.stroke.y > 1.5 && input.stroke.y < 2.5 {
        let corner = input.other_endpoint.xy;
        let radius_px = 10.0;
        let offset_ndc = corner * radius_px * 2.0 / viewport;
        out.clip_position = vec4<f32>(near_clip.xy + offset_ndc * near_clip.w, near_clip.zw);
        out.marker_uv = corner;
        out.color = vec4<f32>(input.color.rgb, 1.0);
        return out;
    }
    let far_clip = uniforms.view_proj * uniforms.model * vec4<f32>(input.other_endpoint, 1.0);
    let near_ndc = near_clip.xy / max(abs(near_clip.w), 0.0001);
    let far_ndc = far_clip.xy / max(abs(far_clip.w), 0.0001);
    let direction_px = (far_ndc - near_ndc) * viewport;
    let tangent = direction_px / max(length(direction_px), 0.0001);
    let perpendicular = vec2<f32>(-tangent.y, tangent.x);
    var stroke_width_px = 2.8;
    if input.stroke.y > 1.1 {
        stroke_width_px = 6.0;
    } else if input.stroke.y > 0.5 {
        stroke_width_px = 4.0;
    }
    let offset_ndc = perpendicular * input.stroke.x * stroke_width_px / viewport;
    out.clip_position = vec4<f32>(near_clip.xy + offset_ndc * near_clip.w, near_clip.zw);

    // A positive alpha encodes the face span for optional distributed-load arrows.
    // At overview scale only the center arrow remains; the other two appear as
    // the loaded face becomes large enough on screen.
    var opacity = 1.0;
    if input.color.a > 0.0 {
        let span_px = input.color.a * abs(uniforms.view_proj[1][1]) * viewport.y
            / (2.0 * max(abs(near_clip.w), 0.0001));
        if span_px < 105.0 {
            opacity = 0.0;
        }
    }
    out.color = vec4<f32>(input.color.rgb, opacity);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if input.style > 1.5 && input.style < 2.5 {
        let radius = length(input.marker_uv);
        if radius > 1.0 {
            discard;
        }
        // A dark rim separates the orange vertex marker from bright mesh faces.
        if radius > 0.78 {
            return vec4<f32>(0.10, 0.11, 0.15, 1.0);
        }
    }
    return input.color;
}
"#;
