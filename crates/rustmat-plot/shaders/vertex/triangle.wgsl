// Triangle vertex shader for surfaces and filled plots

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
    out.normal = normalize(input.normal); // Skip normal transformation for 2D plotting
    out.tex_coords = input.tex_coords;
    
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Basic lighting calculation
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let ambient = 0.3;
    let diffuse = max(0.0, dot(input.normal, light_dir));
    let lighting = ambient + diffuse * 0.7;
    
    return vec4<f32>(input.color.rgb * lighting, input.color.a);
}