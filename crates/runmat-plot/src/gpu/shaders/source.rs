//! Materialize the source consumed by native and browser plotting pipelines.

const FINITE_F32: &str = r#"
fn isFinite(value: f32) -> bool {
    let bits = bitcast<u32>(value);
    return (bits & 0x7f800000u) != 0x7f800000u;
}
"#;

pub(crate) fn compute_source(template: &str, workgroup_size: u32) -> String {
    template
        .replace("{{WORKGROUP_SIZE}}", &workgroup_size.to_string())
        .replace("{{FINITE_F32_HELPER}}", FINITE_F32)
}
