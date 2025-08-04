// Point vertex shader for scatter plots and point clouds

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
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
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    let world_position = uniforms.model * vec4<f32>(input.position, 1.0);
    out.clip_position = uniforms.view_proj * world_position;
    out.world_position = world_position.xyz;
    out.color = input.color;
    out.normal = uniforms.normal_matrix * input.normal;
    
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Simple point rendering with circular shape
    let center = vec2<f32>(0.5, 0.5);
    let coord = input.clip_position.xy;
    
    // For now, just return the vertex color
    // TODO: Add proper point sprite rendering with distance-based alpha
    return input.color;
}