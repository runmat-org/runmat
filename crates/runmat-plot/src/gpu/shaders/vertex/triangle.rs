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
        // Camera-relative studio lighting, as SolidWorks and CATIA light a part: a headlight
        // along the view makes a face seen straight on (a sketch plane in Normal to, a face in
        // a standard view) read at nearly its full colour; a key up and left of the eye
        // separates the faces of a box seen in an iso view; a dim fill and a floor keep
        // away-facing faces and recesses legible.
        // High key, as CATIA and SolidWorks show a light part: face on reads at the full colour; an
        // iso box reads ~ 0.88 / 0.77 / 0.71 on its three faces, and a side face at 70 degrees
        // ~ 0.54 to 0.74, so a white fuselage's flank still reads white rather than grey.
        let world_normal = select(-normal, normal, front_facing);
        let n = normalize((uniforms.cad_view_rotation * world_normal).xyz);
        let key = normalize(vec3<f32>(-0.42, 0.52, 0.74));
        let fill = normalize(vec3<f32>(0.55, -0.15, 0.55));
        let diffuse = max(dot(n, key), 0.0);
        let head = max(n.z, 0.0);
        let fill_diffuse = max(dot(n, fill), 0.0);
        let hemisphere = 0.5 + 0.5 * n.y;
        let half_direction = normalize(key + vec3<f32>(0.0, 0.0, 1.0));
        let highlight = 0.03 * pow(max(dot(n, half_direction), 0.0), 24.0);
        // A bounce from opposite the key lifts faces turned away from it (a housing's inner
        // walls, the far side of a part) so they still read; faces the key lights don't change.
        let bounce = 0.10 * max(dot(n, normalize(vec3<f32>(0.42, -0.52, 0.30))), 0.0);
        let lighting = clamp(0.32 + 0.37 * diffuse + 0.30 * head + 0.08 * fill_diffuse + 0.04 * hemisphere + bounce, 0.32, 1.0);
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

/// The triangle shader for an instanced node: the mesh is in its prototype's coordinates and
/// each placement (`core::InstanceRaw`, locations 4 to 8) moves its points and normals, then
/// paints its color toward the placement's tint (selection, hover, role highlights).
pub fn instanced() -> String {
    SHADER
        .replace(
            "struct VertexOutput {",
            "struct InstanceInput {\n    @location(4) row0: vec4<f32>,\n    @location(5) row1: vec4<f32>,\n    @location(6) row2: vec4<f32>,\n    @location(7) tint: vec4<f32>,\n    @location(8) extra: vec4<f32>,\n}\n\nstruct VertexOutput {",
        )
        .replace("fn vs_main(input: VertexInput) -> VertexOutput {", "fn vs_main(input: VertexInput, inst: InstanceInput) -> VertexOutput {")
        .replace(
            "let world_position = uniforms.model * vec4<f32>(input.position, 1.0);",
            "let local = vec4<f32>(input.position, 1.0);\n    let placed = vec3<f32>(dot(inst.row0, local), dot(inst.row1, local), dot(inst.row2, local));\n    let placed_normal = vec3<f32>(dot(inst.row0.xyz, input.normal), dot(inst.row1.xyz, input.normal), dot(inst.row2.xyz, input.normal));\n    let world_position = uniforms.model * vec4<f32>(placed, 1.0);",
        )
        .replace("out.color = input.color;", "out.color = vec4<f32>(mix(input.color.rgb, inst.tint.rgb, inst.tint.a), input.color.a);")
        .replace("(uniforms.normal_matrix * input.normal).xyz", "(uniforms.normal_matrix * placed_normal).xyz")
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_instanced_variant_places_points_normals_and_paint() {
        let s = super::instanced();
        for needle in ["inst: InstanceInput", "dot(inst.row0, local)", "placed_normal).xyz", "inst.tint.a"] {
            assert!(s.contains(needle), "missing {needle}");
        }
    }
}
