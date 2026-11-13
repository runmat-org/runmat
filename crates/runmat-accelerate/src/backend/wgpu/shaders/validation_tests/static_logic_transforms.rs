use super::static_cases::{validate_pairs, PrecisionPair};
use crate::backend::wgpu::shaders::{
    circshift, diff, flip, kron, logical, permute, repmat, transpose, tril, triu,
};

const SCALAR_WORKGROUPS: &[(u32, u32)] = &[(64, 8), (128, 8), (256, 8)];

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn logical_templates_validate_each_selected_operation_and_precision() {
    validate_pairs(
        &[
            PrecisionPair {
                name: "logical-and",
                single: logical::LOGICAL_AND_SHADER_F32,
                double: logical::LOGICAL_AND_SHADER_F64,
            },
            PrecisionPair {
                name: "logical-or",
                single: logical::LOGICAL_OR_SHADER_F32,
                double: logical::LOGICAL_OR_SHADER_F64,
            },
            PrecisionPair {
                name: "logical-xor",
                single: logical::LOGICAL_XOR_SHADER_F32,
                double: logical::LOGICAL_XOR_SHADER_F64,
            },
            PrecisionPair {
                name: "logical-not",
                single: logical::LOGICAL_NOT_SHADER_F32,
                double: logical::LOGICAL_NOT_SHADER_F64,
            },
            PrecisionPair {
                name: "elem-eq",
                single: logical::ELEM_EQ_SHADER_F32,
                double: logical::ELEM_EQ_SHADER_F64,
            },
            PrecisionPair {
                name: "elem-ne",
                single: logical::ELEM_NE_SHADER_F32,
                double: logical::ELEM_NE_SHADER_F64,
            },
            PrecisionPair {
                name: "elem-lt",
                single: logical::ELEM_LT_SHADER_F32,
                double: logical::ELEM_LT_SHADER_F64,
            },
            PrecisionPair {
                name: "elem-le",
                single: logical::ELEM_LE_SHADER_F32,
                double: logical::ELEM_LE_SHADER_F64,
            },
            PrecisionPair {
                name: "elem-gt",
                single: logical::ELEM_GT_SHADER_F32,
                double: logical::ELEM_GT_SHADER_F64,
            },
            PrecisionPair {
                name: "elem-ge",
                single: logical::ELEM_GE_SHADER_F32,
                double: logical::ELEM_GE_SHADER_F64,
            },
        ],
        10,
        SCALAR_WORKGROUPS,
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn transform_templates_validate_each_geometry_and_precision() {
    validate_pairs(
        &[
            PrecisionPair {
                name: "transpose",
                single: transpose::TRANSPOSE_SHADER_F32,
                double: transpose::TRANSPOSE_SHADER_F64,
            },
            PrecisionPair {
                name: "permute",
                single: permute::PERMUTE_SHADER_F32,
                double: permute::PERMUTE_SHADER_F64,
            },
            PrecisionPair {
                name: "flip",
                single: flip::FLIP_SHADER_F32,
                double: flip::FLIP_SHADER_F64,
            },
            PrecisionPair {
                name: "circshift",
                single: circshift::CIRCSHIFT_SHADER_F32,
                double: circshift::CIRCSHIFT_SHADER_F64,
            },
            PrecisionPair {
                name: "repmat",
                single: repmat::REPMAT_SHADER_F32,
                double: repmat::REPMAT_SHADER_F64,
            },
            PrecisionPair {
                name: "tril",
                single: tril::TRIL_SHADER_F32,
                double: tril::TRIL_SHADER_F64,
            },
            PrecisionPair {
                name: "triu",
                single: triu::TRIU_SHADER_F32,
                double: triu::TRIU_SHADER_F64,
            },
            PrecisionPair {
                name: "kron",
                single: kron::KRON_SHADER_F32,
                double: kron::KRON_SHADER_F64,
            },
            PrecisionPair {
                name: "diff",
                single: diff::DIFF_SHADER_F32,
                double: diff::DIFF_SHADER_F64,
            },
        ],
        9,
        SCALAR_WORKGROUPS,
    );
}
