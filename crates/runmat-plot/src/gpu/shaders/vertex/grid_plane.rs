pub const SHADER: &str = r#"// Procedural XY grid plane shader (CAD-like helper)

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat3x4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct GridParams {
    major_step: f32,
    minor_step: f32,
    fade_start: f32,
    fade_end: f32,
    camera_pos: vec3<f32>,
    _pad0: f32,
    target_pos: vec3<f32>,
    _pad1: f32,
    major_color: vec4<f32>,
    minor_color: vec4<f32>,
}

@group(1) @binding(0)
var<uniform> grid: GridParams;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_position = uniforms.model * vec4<f32>(input.position, 1.0);
    out.clip_position = uniforms.view_proj * world_position;
    out.world_position = world_position.xyz;
    return out;
}

fn grid_line(coord: f32, step: f32) -> f32 {
    if !(step > 0.0) {
        return 0.0;
    }
    let u = coord / step;
    let f = fract(u);
    let d = min(f, 1.0 - f); // distance to nearest integer grid line in [0, 0.5]
    let w = fwidth(u);
    // 1-pixel-ish antialiased line in param space
    return 1.0 - smoothstep(0.0, w * 1.5, d);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let x = input.world_position.x;
    let y = input.world_position.y;

    let minor = max(grid_line(x, grid.minor_step), grid_line(y, grid.minor_step));
    let major = max(grid_line(x, grid.major_step), grid_line(y, grid.major_step));
    let minor_only = minor * (1.0 - major);

    // Fade out towards edges of the grid patch (and keep it subtle when very large).
    let dxy = length(vec2<f32>(x - grid.target_pos.x, y - grid.target_pos.y));
    let fade = 1.0 - smoothstep(grid.fade_start, grid.fade_end, dxy);

    let c = grid.major_color * major + grid.minor_color * minor_only;
    return vec4<f32>(c.rgb, c.a * fade);
}
"#;

