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
