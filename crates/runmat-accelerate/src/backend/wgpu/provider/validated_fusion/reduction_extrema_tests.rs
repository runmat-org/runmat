use super::super::pipeline_recipe::PipelineRecipe;
use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn extrema_canonical_recipe_rejects_shader_layout_and_workgroup_substitution() {
    for kind in [
        ReductionExtremaPipelineKind::Global,
        ReductionExtremaPipelineKind::Dimensional,
        ReductionExtremaPipelineKind::IntegerDimensional,
    ] {
        let recipe = || PipelineRecipe {
            shader: "@compute @workgroup_size(64) fn main() {}".into(),
            layout: kind.layout_tag().into(),
            workgroup: 64,
        };
        let canonical = recipe();
        assert!(kind.matches_recipe(&canonical, &canonical));
        let mut forged = recipe();
        forged.shader.push_str(" // changed");
        assert!(!kind.matches_recipe(&forged, &canonical));
        let mut forged = recipe();
        forged.layout.push_str("-changed");
        assert!(!kind.matches_recipe(&forged, &canonical));
        let mut forged = recipe();
        forged.workgroup = 128;
        assert!(!kind.matches_recipe(&forged, &canonical));
    }
}
