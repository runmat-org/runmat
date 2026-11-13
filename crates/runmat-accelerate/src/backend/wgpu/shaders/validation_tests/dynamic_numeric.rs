//! Validate the production numeric builders, including mixed-precision probes.

use super::{validate_generated, RequiredCaps};
use crate::backend::wgpu::shaders::{classification, power, remainder};
use crate::backend::wgpu::types::NumericPrecision;
use runmat_accelerate_api::PowerResultDomain;
use runmat_builtins::NumericClassificationPredicate;
use runmat_types::FloatingRemainderKind;

fn validate(name: &str, source: &str, caps: RequiredCaps) {
    for workgroup in [64, 128, 256] {
        validate_generated(name, source, workgroup, 8, caps);
    }
}

fn arithmetic_caps(precision: NumericPrecision) -> RequiredCaps {
    match precision {
        NumericPrecision::F32 => RequiredCaps::None,
        NumericPrecision::F64 => RequiredCaps::Float64Int64,
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn power_variants_and_mixed_precision_probes_validate() {
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        for (name, mode) in [
            ("classify", power::PowerShader::Classify),
            (
                "real",
                power::PowerShader::Evaluate(PowerResultDomain::Real),
            ),
            (
                "complex",
                power::PowerShader::Evaluate(PowerResultDomain::Complex),
            ),
        ] {
            validate(
                &format!("power-{precision:?}-{name}"),
                &power::shader(precision, mode),
                arithmetic_caps(precision),
            );
        }
        for input_precision in [NumericPrecision::F32, NumericPrecision::F64] {
            // The probe reads its original input storage, but evaluates the
            // numeric-domain decision in the selected output precision.
            let caps = match (input_precision, precision) {
                (_, NumericPrecision::F64) => RequiredCaps::Float64Int64,
                (NumericPrecision::F64, NumericPrecision::F32) => RequiredCaps::Float64,
                (NumericPrecision::F32, NumericPrecision::F32) => RequiredCaps::None,
            };
            validate(
                &format!("power-probe-{input_precision:?}-to-{precision:?}"),
                &power::classification_shader(input_precision, precision),
                caps,
            );
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn remainder_variants_validate_with_ieee_requirements() {
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        for kind in [
            FloatingRemainderKind::Modulus,
            FloatingRemainderKind::Remainder,
        ] {
            validate(
                &format!("remainder-{precision:?}-{kind:?}"),
                &remainder::shader(precision, kind),
                arithmetic_caps(precision),
            );
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn classification_variants_need_only_the_output_precision() {
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        let caps = match precision {
            NumericPrecision::F32 => RequiredCaps::None,
            NumericPrecision::F64 => RequiredCaps::Float64,
        };
        for predicate in [
            NumericClassificationPredicate::Nan,
            NumericClassificationPredicate::Infinite,
            NumericClassificationPredicate::Finite,
        ] {
            // Classification reads u32 words, so even wide inputs do not
            // require native integer64 arithmetic.
            validate(
                &format!("classification-{precision:?}-{predicate:?}"),
                &classification::classification_shader(precision, predicate),
                caps,
            );
        }
    }
}
