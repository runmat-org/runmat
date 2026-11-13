//! Validate every shader emitted by the shared NaN predicate owner with the
//! Naga version used by WGPU. This checks the final shader, not a fragment.

use super::*;
use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};
use crate::backend::wgpu::shaders::{nan, reduction, scan};

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn all_nan_predicate_shaders_parse_and_validate() {
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        // Both storage classes are handled as u32 IEEE words, with no shader
        // floating-point or integer64 capability required.
        validate_generated(
            &format!("nan-map-{precision:?}"),
            &nan::map_source(precision),
            64,
            8,
            RequiredCaps::None,
        );
        validate_generated(
            &format!("nan-mask-{precision:?}"),
            &nan::mask_source(precision),
            64,
            8,
            RequiredCaps::None,
        );

        let arithmetic_caps = match precision {
            NumericPrecision::F32 => RequiredCaps::None,
            NumericPrecision::F64 => RequiredCaps::Float64Int64,
        };

        let (dim, cumsum, cumprod, cummin, cummax) = match precision {
            NumericPrecision::F32 => (
                reduction::REDUCE_DIM_SHADER_F32,
                scan::CUMSUM_SHADER_F32,
                scan::CUMPROD_SHADER_F32,
                scan::CUMMIN_SHADER_F32,
                scan::CUMMAX_SHADER_F32,
            ),
            NumericPrecision::F64 => (
                reduction::REDUCE_DIM_SHADER_F64,
                scan::CUMSUM_SHADER_F64,
                scan::CUMPROD_SHADER_F64,
                scan::CUMMIN_SHADER_F64,
                scan::CUMMAX_SHADER_F64,
            ),
        };
        validate_generated(
            &format!("reduce-global-{precision:?}"),
            &reduction::global_source(precision),
            64,
            8,
            arithmetic_caps,
        );
        for (name, template) in [
            ("reduce-dim", dim),
            ("cumsum", cumsum),
            ("cumprod", cumprod),
            ("cummin", cummin),
            ("cummax", cummax),
        ] {
            validate_generated(
                &format!("{name}-{precision:?}"),
                &inject(template, precision),
                64,
                8,
                arithmetic_caps,
            );
        }
        let (first, second) = reduction::build_two_pass_shaders(precision, 64);
        validate_generated(
            &format!("reduce-two-pass-first-{precision:?}"),
            &first,
            64,
            8,
            arithmetic_caps,
        );
        validate_generated(
            &format!("reduce-two-pass-second-{precision:?}"),
            &second,
            64,
            8,
            arithmetic_caps,
        );
    }

    // Ordered mean accumulates in Double for either input class, while its
    // publication class can independently be Single or Double.
    for input in [NumericPrecision::F32, NumericPrecision::F64] {
        for output in [NumericPrecision::F32, NumericPrecision::F64] {
            validate_generated(
                &format!("ordered-mean-{input:?}-{output:?}"),
                &reduction::ordered_mean::source(input, output),
                64,
                8,
                RequiredCaps::Float64Int64,
            );
        }
    }
}
