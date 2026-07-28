pub const SHADER: &str = r#"// Direct coordinate transformation vertex shader for precise plot rendering
// Performs efficient data-to-viewport coordinate mapping without camera transforms

struct Uniforms {
    // Data bounds in world space
    data_min: vec2<f32>,    // (x_min, y_min)
    data_max: vec2<f32>,    // (x_max, y_max)
    // Viewport bounds in NDC space (where plot should appear)
    viewport_min: vec2<f32>, // NDC coordinates of viewport bottom-left
    viewport_max: vec2<f32>, // NDC coordinates of viewport top-right
    viewport_px: vec2<f32>,
    log_flags: vec2<u32>,
    _pad: vec2<u32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_position: vec3<f32>,
}

fn transform_axis(value: f32, is_log: u32) -> f32 {
    if (is_log != 0u) {
        if (value <= 0.0) {
            return 1e30;
        }
        return log(value) / log(10.0);
    }
    return value;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Transform data coordinates to normalized device coordinates within viewport bounds
    let data_min = vec2<f32>(
        transform_axis(uniforms.data_min.x, uniforms.log_flags.x),
        transform_axis(uniforms.data_min.y, uniforms.log_flags.y)
    );
    let data_max = vec2<f32>(
        transform_axis(uniforms.data_max.x, uniforms.log_flags.x),
        transform_axis(uniforms.data_max.y, uniforms.log_flags.y)
    );
    let position = vec2<f32>(
        transform_axis(input.position.x, uniforms.log_flags.x),
        transform_axis(input.position.y, uniforms.log_flags.y)
    );
    let data_range = data_max - data_min;
    let viewport_range = uniforms.viewport_max - uniforms.viewport_min;

    // Normalize data position to [0, 1] within data bounds
    let normalized_pos = (position - data_min) / data_range;

    // Map to viewport NDC range
    let ndc_pos = uniforms.viewport_min + normalized_pos * viewport_range;

    // Create final clip position
    out.clip_position = vec4<f32>(ndc_pos.x, ndc_pos.y, 0.0, 1.0);
    out.world_position = input.position;
    out.color = input.color;

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Simple line rendering
    return input.color;
}
"#;
