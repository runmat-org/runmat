//! A bounded polynomial/bandwidth static template cohort.

use super::static_cases::{validate_pairs, PrecisionPair};
use crate::backend::wgpu::shaders::{bandwidth, polyder, polyint, polyval};

const SELECTED_WORKGROUPS: &[(u32, u32)] = &[(64, 8), (128, 8), (256, 8)];

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn polynomial_and_bandwidth_templates_validate_selected_workgroups_and_precisions() {
    validate_pairs(
        &[
            PrecisionPair {
                name: "polyder",
                single: polyder::POLYDER_SHADER_F32,
                double: polyder::POLYDER_SHADER_F64,
            },
            PrecisionPair {
                name: "polyint",
                single: polyint::POLYINT_SHADER_F32,
                double: polyint::POLYINT_SHADER_F64,
            },
            PrecisionPair {
                name: "polyval",
                single: polyval::POLYVAL_SHADER_F32,
                double: polyval::POLYVAL_SHADER_F64,
            },
            PrecisionPair {
                name: "bandwidth",
                single: bandwidth::BANDWIDTH_SHADER_F32,
                double: bandwidth::BANDWIDTH_SHADER_F64,
            },
        ],
        4,
        SELECTED_WORKGROUPS,
    );
}
