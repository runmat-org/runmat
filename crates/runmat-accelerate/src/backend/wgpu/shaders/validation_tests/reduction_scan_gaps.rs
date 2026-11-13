//! Production reduction/scan templates not covered by the NaN-predicate cohort.

use super::{validate_generated, RequiredCaps};
use crate::backend::wgpu::shaders::{ieee_words, reduction, scan};
use std::collections::BTreeSet;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn remaining_reduction_and_trapezoid_sources_validate_selected_workgroups() {
    let mut labels = BTreeSet::new();
    for workgroup in [64, 128, 256] {
        let f32_sources = [
            ("dim-minmax", reduction::REDUCE_DIM_MINMAX_SHADER_F32),
            ("nd-mean", reduction::REDUCE_ND_MEAN_SHADER_F32),
            ("nd-moments", reduction::REDUCE_ND_MOMENTS_SHADER_F32),
            ("trapezoid", scan::TRAPEZOID_SHADER_F32),
        ];
        for (name, source) in f32_sources {
            let label = format!("{name}-f32-wg{workgroup}");
            assert!(labels.insert(label.clone()));
            validate_generated(&label, source, workgroup, 8, RequiredCaps::None);
        }

        // The min/max template uses scalar F64 IEEE word helpers; the other
        // three Double templates use native Float64 but no integer64 words.
        let f64_sources = [
            (
                "dim-minmax",
                ieee_words::inject(reduction::REDUCE_DIM_MINMAX_SHADER_F64),
                RequiredCaps::Float64Int64,
            ),
            (
                "nd-mean",
                reduction::REDUCE_ND_MEAN_SHADER_F64.to_owned(),
                RequiredCaps::Float64,
            ),
            (
                "nd-moments",
                reduction::REDUCE_ND_MOMENTS_SHADER_F64.to_owned(),
                RequiredCaps::Float64,
            ),
            (
                "trapezoid",
                scan::TRAPEZOID_SHADER_F64.to_owned(),
                RequiredCaps::Float64,
            ),
        ];
        for (name, source, caps) in f64_sources {
            let label = format!("{name}-f64-wg{workgroup}");
            assert!(labels.insert(label.clone()));
            validate_generated(&label, &source, workgroup, 8, caps);
        }
    }
    assert_eq!(labels.len(), 24, "reduction/scan source inventory changed");
}
