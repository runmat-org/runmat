use super::*;

#[test]
fn selected_source_layout_and_workgroup_preserve_existing_pipeline_identity() {
    let source = "@compute @workgroup_size(@WG@) fn main() {}";
    for workgroup in [64, 256, 512] {
        let selected = source.replace("@WG@", &workgroup.to_string());
        for inputs in [1, 2, 4] {
            let single =
                PipelineRecipe::fusion(source, inputs, FusionPipelineLayout::Single, workgroup);
            assert_eq!(
                single.key(),
                key::compute_pipeline_hash_bytes(
                    selected.as_bytes(),
                    &format!("runmat-fusion-layout-{inputs}"),
                    Some(workgroup)
                )
            );
            for outputs in [1, 2, 4] {
                let multi = PipelineRecipe::fusion(
                    source,
                    inputs,
                    FusionPipelineLayout::Multiple(outputs),
                    workgroup,
                );
                assert_eq!(
                    multi.key(),
                    key::compute_pipeline_hash_bytes(
                        selected.as_bytes(),
                        &format!("runmat-fusion-multi-layout-{inputs}-{outputs}"),
                        Some(workgroup)
                    )
                );
                assert_ne!(single.key(), multi.key());
            }
        }
    }
}

#[test]
fn cast_recipe_distinguishes_both_physical_precisions_and_workgroup() {
    let mut keys = std::collections::HashSet::new();
    for source in [NumericPrecision::F32, NumericPrecision::F64] {
        for target in [NumericPrecision::F32, NumericPrecision::F64] {
            for workgroup in [64, 256, 512] {
                let recipe = PipelineRecipe::floating_cast(source, target, workgroup);
                assert!(keys.insert(recipe.key()));
                assert!(recipe
                    .shader
                    .contains(&format!("@workgroup_size({workgroup})")));
                assert_eq!(recipe.layout, "runmat-floating-cast-bgl");
            }
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn integer_sum_conversion_recipe_matches_the_execution_shader_identity() {
    for workgroup in [64, 256, 512] {
        let recipe = PipelineRecipe::integer_sum_cast(workgroup);
        assert_eq!(recipe.layout, "runmat-integer-sum-floating-cast-bgl");
        assert_eq!(
            recipe.key(),
            key::compute_pipeline_hash_bytes(
                crate::backend::wgpu::shaders::integer::floating_sum_cast_shader(workgroup)
                    .as_bytes(),
                "runmat-integer-sum-floating-cast-bgl",
                Some(workgroup)
            )
        );
        assert_ne!(
            recipe.key(),
            PipelineRecipe::floating_cast(NumericPrecision::F32, NumericPrecision::F64, workgroup)
                .key()
        );
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_dimension_n_sum_recipe_validates_and_keys_exact_source() {
    use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};
    for workgroup in [64, 128, 256] {
        let recipe = PipelineRecipe::dimension_n_sum_f64(workgroup);
        assert_eq!(recipe.layout, "runmat-nd-sum-f64-bgl");
        assert_eq!(recipe.workgroup, workgroup);
        assert_eq!(
            recipe.key(),
            key::compute_pipeline_hash_bytes(
                recipe.shader.as_bytes(),
                "runmat-nd-sum-f64-bgl",
                Some(workgroup),
            ),
        );
        validate_generated(
            &format!("selected-nd-sum-f64-wg{workgroup}"),
            &recipe.shader,
            workgroup,
            8,
            RequiredCaps::Float64,
        );
    }
}
