use super::super::pipeline_recipe::PipelineRecipe;
use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn std_canonical_primitive_rejects_shader_layout_and_workgroup_substitution() {
    for kind in [
        StdPrimitivePipelineKind::DimensionalSum,
        StdPrimitivePipelineKind::Binary,
        StdPrimitivePipelineKind::Fill,
    ] {
        let recipe = || PipelineRecipe {
            shader: "@compute @workgroup_size(64) fn main() {}".into(),
            layout: kind.layout_tag().into(),
            workgroup: 64,
        };
        let canonical = recipe();
        assert!(kind.matches_recipe(&canonical, &canonical));
        let mut changed = recipe();
        changed.shader.push_str(" // changed");
        assert!(!kind.matches_recipe(&changed, &canonical));
        let mut changed = recipe();
        changed.layout.push_str("-changed");
        assert!(!kind.matches_recipe(&changed, &canonical));
        let mut changed = recipe();
        changed.workgroup = 128;
        assert!(!kind.matches_recipe(&changed, &canonical));
    }
}
