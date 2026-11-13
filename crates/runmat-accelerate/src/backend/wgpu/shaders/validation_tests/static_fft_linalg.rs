use super::static_cases::{validate_pairs, validate_singles, PrecisionPair};
use super::{validate_generated, RequiredCaps};
use crate::backend::wgpu::shaders::{
    centered_gram, fft, matmul, matmul_smallk, matmul_tall_skinny, syrk,
};
use runmat_accelerate_api::ProviderPrecision;

const SCALAR_WORKGROUPS: &[(u32, u32)] = &[(64, 8), (128, 8), (256, 8)];
const MATRIX_TILES: &[(u32, u32)] = &[(64, 8), (64, 16)];

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn fft_templates_validate_at_selected_scalar_workgroups() {
    validate_pairs(
        &[
            PrecisionPair {
                name: "fft-init",
                single: fft::FFT_INIT_SHADER_F32,
                double: fft::FFT_INIT_SHADER_F64,
            },
            PrecisionPair {
                name: "fft-stage",
                single: fft::FFT_STAGE_SHADER_F32,
                double: fft::FFT_STAGE_SHADER_F64,
            },
            PrecisionPair {
                name: "fft-reorder",
                single: fft::FFT_REORDER_SHADER_F32,
                double: fft::FFT_REORDER_SHADER_F64,
            },
            PrecisionPair {
                name: "fft-direct",
                single: fft::FFT_DIRECT_SHADER_F32,
                double: fft::FFT_DIRECT_SHADER_F64,
            },
            PrecisionPair {
                name: "fft-extract-real",
                single: fft::FFT_EXTRACT_REAL_SHADER_F32,
                double: fft::FFT_EXTRACT_REAL_SHADER_F64,
            },
            PrecisionPair {
                name: "fft-stage3",
                single: fft::FFT_STAGE3_SHADER_F32,
                double: fft::FFT_STAGE3_SHADER_F64,
            },
            PrecisionPair {
                name: "fft-reorder3",
                single: fft::FFT_REORDER3_SHADER_F32,
                double: fft::FFT_REORDER3_SHADER_F64,
            },
            PrecisionPair {
                name: "fft-stage5",
                single: fft::FFT_STAGE5_SHADER_F32,
                double: fft::FFT_STAGE5_SHADER_F64,
            },
            PrecisionPair {
                name: "fft-reorder5",
                single: fft::FFT_REORDER5_SHADER_F32,
                double: fft::FFT_REORDER5_SHADER_F64,
            },
            PrecisionPair {
                name: "fft-reorder-mixed",
                single: fft::FFT_REORDER_MIXED_SHADER_F32,
                double: fft::FFT_REORDER_MIXED_SHADER_F64,
            },
            PrecisionPair {
                name: "fft-bluestein-prep",
                single: fft::FFT_BLUESTEIN_PREP_SHADER_F32,
                double: fft::FFT_BLUESTEIN_PREP_SHADER_F64,
            },
            PrecisionPair {
                name: "fft-bluestein-kernel",
                single: fft::FFT_BLUESTEIN_KERNEL_SHADER_F32,
                double: fft::FFT_BLUESTEIN_KERNEL_SHADER_F64,
            },
            PrecisionPair {
                name: "fft-pointwise-broadcast",
                single: fft::FFT_POINTWISE_BROADCAST_SHADER_F32,
                double: fft::FFT_POINTWISE_BROADCAST_SHADER_F64,
            },
            PrecisionPair {
                name: "fft-bluestein-finalize",
                single: fft::FFT_BLUESTEIN_FINALIZE_SHADER_F32,
                double: fft::FFT_BLUESTEIN_FINALIZE_SHADER_F64,
            },
        ],
        14,
        SCALAR_WORKGROUPS,
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn matrix_templates_validate_at_selected_tile_sizes() {
    validate_pairs(
        &[
            PrecisionPair {
                name: "matmul",
                single: matmul::MATMUL_SHADER_F32,
                double: matmul::MATMUL_SHADER_F64,
            },
            PrecisionPair {
                name: "matmul-smallk",
                single: matmul_smallk::MATMUL_SMALLK_SHADER_F32,
                double: matmul_smallk::MATMUL_SMALLK_SHADER_F64,
            },
            PrecisionPair {
                name: "centered-gram",
                single: centered_gram::CENTERED_GRAM_SHADER_F32,
                double: centered_gram::CENTERED_GRAM_SHADER_F64,
            },
            PrecisionPair {
                name: "syrk",
                single: syrk::SYRK_SHADER_F32,
                double: syrk::SYRK_SHADER_F64,
            },
        ],
        4,
        MATRIX_TILES,
    );
    validate_pairs(
        &[PrecisionPair {
            name: "matmul-tall-skinny",
            single: matmul_tall_skinny::MATMUL_TALL_SKINNY_F32,
            double: matmul_tall_skinny::MATMUL_TALL_SKINNY_F64,
        }],
        1,
        &[(64, 8)],
    );
    validate_singles(
        &[("matmul-vec4", matmul::MATMUL_SHADER_VEC4_F32)],
        1,
        MATRIX_TILES,
    );
    // The F64 vec4 aliases are the same source constants as their ordinary
    // matmul/SYRK routes, already checked by the pairs above.
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn matmul_epilogue_validates_the_owning_power_helper_not_a_raw_template() {
    for tile in [8, 16] {
        for (precision, caps) in [
            (ProviderPrecision::F32, RequiredCaps::None),
            (ProviderPrecision::F64, RequiredCaps::Float64Int64),
        ] {
            let source = matmul::epilogue_shader(precision, tile);
            validate_generated(
                &format!("matmul-epilogue-{precision:?}-tile{tile}"),
                &source,
                64,
                tile,
                caps,
            );
        }
    }
}
