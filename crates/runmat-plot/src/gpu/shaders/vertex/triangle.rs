pub const SHADER: &str = r#"// Triangle vertex shader for surfaces and filled plots

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat3x4<f32>,
    cad_view_rotation: mat3x4<f32>,
    cad_lighting: vec4<u32>,
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
    let transformed_normal = (uniforms.normal_matrix * input.normal).xyz;
    out.normal = vec3<f32>(0.0);
    let normal_scale = max(max(abs(transformed_normal.x), abs(transformed_normal.y)), abs(transformed_normal.z));
    if normal_scale > 0.0 {
        out.normal = normalize(transformed_normal / normal_scale);
    }
    out.tex_coords = input.tex_coords;

    return out;
}

@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    let normal_len = length(input.normal);
    if normal_len < 0.0001 {
        return input.color;
    }

    let normal = normalize(input.normal);
    if uniforms.cad_lighting.x != 0u {
        // Camera-relative studio key/fill keeps recesses legible while orbiting.
        // The low ambient floor preserves orientation contrast; fill prevents
        // away-facing surfaces from disappearing into the background. The key sits
        // up and to the left of the eye, so a face seen straight on (a sketch plane
        // in Normal to, a standard view) reads light, as in SolidWorks and CATIA.
        let world_normal = select(-normal, normal, front_facing);
        let n = normalize((uniforms.cad_view_rotation * world_normal).xyz);
        let key = normalize(vec3<f32>(-0.42, 0.52, 0.74));
        let fill = normalize(vec3<f32>(0.55, -0.15, 0.55));
        let diffuse = max(dot(n, key), 0.0);
        let fill_diffuse = max(dot(n, fill), 0.0);
        let hemisphere = 0.5 + 0.5 * n.y;
        let half_direction = normalize(key + vec3<f32>(0.0, 0.0, 1.0));
        let highlight = 0.025 * pow(max(dot(n, half_direction), 0.0), 16.0);
        let lighting = clamp(0.18 + 0.65 * diffuse + 0.10 * fill_diffuse + 0.05 * hemisphere, 0.18, 0.98);
        return vec4<f32>(clamp(input.color.rgb * lighting + vec3<f32>(highlight), vec3<f32>(0.0), vec3<f32>(1.0)), input.color.a);
    }

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
