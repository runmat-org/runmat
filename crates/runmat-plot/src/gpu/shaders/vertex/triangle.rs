pub const SHADER: &str = r#"// Triangle vertex shader for surfaces and filled plots

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat3x4<f32>,
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
    @location(2) normal: vec3<f32>,
    @location(3) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let world_position = uniforms.model * vec4<f32>(input.position, 1.0);
    out.clip_position = uniforms.view_proj * world_position;
    out.world_position = world_position.xyz;
    out.color = input.color;
    // normal_matrix is mat3x4 for alignment (see Rust side). In WGSL, mat3x4 * vec3 -> vec4.
    out.normal = normalize((uniforms.normal_matrix * input.normal).xyz);
    out.tex_coords = input.tex_coords;

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let normal_len = length(input.normal);
    if normal_len < 0.0001 {
        return input.color;
    }

    let normal = normalize(input.normal);
    let light_dir = normalize(vec3<f32>(0.36, 0.48, 0.80));
    let fill_dir = normalize(vec3<f32>(-0.58, -0.26, 0.78));
    let diffuse = max(dot(normal, light_dir), 0.0);
    let fill = max(dot(normal, fill_dir), 0.0);
    let rim = pow(1.0 - abs(normal.z), 2.0);
    let lighting = clamp(0.56 + diffuse * 0.36 + fill * 0.10 + rim * 0.10, 0.46, 1.12);
    let highlight = vec3<f32>(0.08, 0.09, 0.10) * max(diffuse - 0.72, 0.0);
    return vec4<f32>(clamp(input.color.rgb * lighting + highlight, vec3<f32>(0.0), vec3<f32>(1.0)), input.color.a);
}
"#;
