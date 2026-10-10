//! GPU-free validation of complete loss, softmax, and optimizer sources.

use super::*;
use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};
use std::collections::BTreeSet;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn crossentropy_source_options_validate_with_selected_precision() {
    let mut labels = BTreeSet::new();
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        for weights_enabled in [0, 1] {
            for mask_enabled in [0, 1] {
                let params = CrossentropyParams {
                    total: 16,
                    offset: 0,
                    chunk: 16,
                    mode: 0,
                    weights_enabled,
                    mask_enabled,
                    _pad0: 0,
                    _pad1: 0,
                };
                let ordinary_caps = match precision {
                    NumericPrecision::F32 => RequiredCaps::None,
                    NumericPrecision::F64 => RequiredCaps::Float64,
                };
                for (phase, source) in [
                    ("positive", crossentropy_positive_shader(precision, &params)),
                    (
                        "complementary",
                        crossentropy_complementary_shader(precision, &params),
                    ),
                ] {
                    let name = format!(
                        "crossentropy-{phase}-{precision:?}-weights{weights_enabled}-mask{mask_enabled}"
                    );
                    assert!(labels.insert(name.clone()), "duplicate {name}");
                    validate_generated(
                        &name,
                        &source,
                        crate::backend::wgpu::config::WORKGROUP_SIZE,
                        8,
                        ordinary_caps,
                    );
                }
                for mode in [0, 1] {
                    let params = CrossentropyParams { mode, ..params };
                    let caps = match precision {
                        NumericPrecision::F32 => RequiredCaps::None,
                        NumericPrecision::F64 => RequiredCaps::Float64Int64,
                    };
                    let name = format!(
                        "crossentropy-validation-{precision:?}-mode{mode}-weights{weights_enabled}-mask{mask_enabled}"
                    );
                    assert!(labels.insert(name.clone()), "duplicate {name}");
                    validate_generated(
                        &name,
                        &crossentropy_validation_shader(precision, &params),
                        crate::backend::wgpu::config::WORKGROUP_SIZE,
                        8,
                        caps,
                    );
                }
            }
        }
    }
    assert_eq!(labels.len(), 32, "crossentropy source inventory changed");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn softmax_and_adam_generators_validate_with_distinct_f64_caps() {
    let mut labels = BTreeSet::new();
    for workgroup in [128, 256, 512] {
        for (precision, caps) in [
            (NumericPrecision::F32, RequiredCaps::None),
            (NumericPrecision::F64, RequiredCaps::Float64),
        ] {
            let name = format!("softmax-{precision:?}-wg{workgroup}");
            assert!(labels.insert(name.clone()), "duplicate {name}");
            validate_generated(
                &name,
                &softmax_rows_shader(precision, workgroup),
                workgroup,
                8,
                caps,
            );
        }
    }
    for (name, source, caps) in [
        (
            "adam-f32",
            adam_update_shader(NumericPrecision::F32),
            RequiredCaps::None,
        ),
        (
            "adam-f64",
            adam_update_shader(NumericPrecision::F64),
            RequiredCaps::Float64Int64,
        ),
    ] {
        assert!(
            labels.insert(name.to_owned()),
            "duplicate deep-learning shader: {name}"
        );
        validate_generated(
            name,
            &source,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
            8,
            caps,
        );
    }
    assert_eq!(labels.len(), 8, "softmax/Adam source inventory changed");
}
