//! A bounded static spatial-gradient/filter/convolution/window template cohort.

use super::static_cases::{validate_pairs, PrecisionPair};
use crate::backend::wgpu::shaders::{conv, filter, gradient, window};

const SELECTED_WORKGROUPS: &[(u32, u32)] = &[(64, 8), (128, 8), (256, 8)];

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn spatial_templates_validate_selected_workgroups_and_precisions() {
    validate_pairs(
        &[
            PrecisionPair {
                name: "gradient",
                single: gradient::GRADIENT_SHADER_F32,
                double: gradient::GRADIENT_SHADER_F64,
            },
            PrecisionPair {
                name: "gradient-coordinates",
                single: gradient::GRADIENT_COORDINATES_SHADER_F32,
                double: gradient::GRADIENT_COORDINATES_SHADER_F64,
            },
            PrecisionPair {
                name: "filter",
                single: filter::FILTER_SHADER_F32,
                double: filter::FILTER_SHADER_F64,
            },
            PrecisionPair {
                name: "conv1d",
                single: conv::CONV1D_SHADER_F32,
                double: conv::CONV1D_SHADER_F64,
            },
            PrecisionPair {
                name: "window",
                single: window::WINDOW_SHADER_F32,
                double: window::WINDOW_SHADER_F64,
            },
        ],
        5,
        SELECTED_WORKGROUPS,
    );
}
