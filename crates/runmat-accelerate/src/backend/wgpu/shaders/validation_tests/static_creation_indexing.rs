use super::static_cases::{validate_pairs, validate_singles, PrecisionPair};
use crate::backend::wgpu::shaders::{creation, diag, find, index_select, scatter};

const SCALAR_WORKGROUPS: &[(u32, u32)] = &[(64, 8), (128, 8), (256, 8)];

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn creation_templates_validate_at_selected_scalar_workgroups() {
    validate_pairs(
        &[
            PrecisionPair {
                name: "eye",
                single: creation::EYE_SHADER_F32,
                double: creation::EYE_SHADER_F64,
            },
            PrecisionPair {
                name: "fill",
                single: creation::FILL_SHADER_F32,
                double: creation::FILL_SHADER_F64,
            },
            PrecisionPair {
                name: "linspace",
                single: creation::LINSPACE_SHADER_F32,
                double: creation::LINSPACE_SHADER_F64,
            },
            PrecisionPair {
                name: "fspecial",
                single: creation::FSPECIAL_SHADER_F32,
                double: creation::FSPECIAL_SHADER_F64,
            },
            PrecisionPair {
                name: "random-uniform",
                single: creation::RANDOM_UNIFORM_SHADER_F32,
                double: creation::RANDOM_UNIFORM_SHADER_F64,
            },
            PrecisionPair {
                name: "random-normal",
                single: creation::RANDOM_NORMAL_SHADER_F32,
                double: creation::RANDOM_NORMAL_SHADER_F64,
            },
            PrecisionPair {
                name: "randperm",
                single: creation::RANDPERM_SHADER_F32,
                double: creation::RANDPERM_SHADER_F64,
            },
            PrecisionPair {
                name: "peaks",
                single: creation::PEAKS_SHADER_F32,
                double: creation::PEAKS_SHADER_F64,
            },
            PrecisionPair {
                name: "peaks-xy",
                single: creation::PEAKS_XY_SHADER_F32,
                double: creation::PEAKS_XY_SHADER_F64,
            },
            PrecisionPair {
                name: "random-exprnd",
                single: creation::RANDOM_EXPRND_SHADER_F32,
                double: creation::RANDOM_EXPRND_SHADER_F64,
            },
            PrecisionPair {
                name: "random-normrnd",
                single: creation::RANDOM_NORMRND_SHADER_F32,
                double: creation::RANDOM_NORMRND_SHADER_F64,
            },
            PrecisionPair {
                name: "random-unifrnd",
                single: creation::RANDOM_UNIFRND_SHADER_F32,
                double: creation::RANDOM_UNIFRND_SHADER_F64,
            },
        ],
        12,
        SCALAR_WORKGROUPS,
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn indexing_templates_validate_at_selected_scalar_workgroups() {
    validate_pairs(
        &[
            PrecisionPair {
                name: "diag-from-vector",
                single: diag::DIAG_FROM_VECTOR_SHADER_F32,
                double: diag::DIAG_FROM_VECTOR_SHADER_F64,
            },
            PrecisionPair {
                name: "diag-extract",
                single: diag::DIAG_EXTRACT_SHADER_F32,
                double: diag::DIAG_EXTRACT_SHADER_F64,
            },
            PrecisionPair {
                name: "find",
                single: find::FIND_SHADER_F32,
                double: find::FIND_SHADER_F64,
            },
        ],
        3,
        SCALAR_WORKGROUPS,
    );
    validate_singles(
        &[
            ("scatter-col", scatter::SCATTER_COL_SHADER_F32),
            ("scatter-row", scatter::SCATTER_ROW_SHADER_F32),
            (
                "linear-gather-words",
                index_select::LINEAR_GATHER_WORDS_SHADER,
            ),
            (
                "linear-scatter-words",
                index_select::LINEAR_SCATTER_WORDS_SHADER,
            ),
        ],
        4,
        SCALAR_WORKGROUPS,
    );
}
