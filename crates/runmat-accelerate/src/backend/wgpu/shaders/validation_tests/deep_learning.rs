//! GPU-free validation of the production resident deep-learning shaders.

use super::{validate_generated, RequiredCaps};
use crate::backend::wgpu::{
    config,
    shaders::{conv, deep_learning, deep_learning_normalization},
    types::NumericPrecision,
};
use std::collections::BTreeSet;

fn validate_rows(rows: [(&str, String, RequiredCaps, u32); 4]) {
    let mut labels = BTreeSet::new();
    for (name, source, caps, workgroup) in rows {
        assert!(
            labels.insert(name),
            "duplicate deep-learning shader: {name}"
        );
        validate_generated(name, &source, workgroup, 8, caps);
    }
    assert_eq!(labels.len(), 4, "deep-learning shader inventory changed");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn normalization_statistics_and_affine_validate_with_exact_ieee_caps() {
    let wg = deep_learning_normalization::NORMALIZATION_WORKGROUP_SIZE;
    validate_rows([
        (
            "normalization-statistics-f32",
            deep_learning_normalization::statistics(NumericPrecision::F32),
            RequiredCaps::None,
            wg,
        ),
        (
            "normalization-statistics-f64",
            deep_learning_normalization::statistics(NumericPrecision::F64),
            RequiredCaps::Float64Int64,
            wg,
        ),
        (
            "normalization-affine-f32",
            deep_learning_normalization::affine(NumericPrecision::F32),
            RequiredCaps::None,
            wg,
        ),
        (
            "normalization-affine-f64",
            deep_learning_normalization::affine(NumericPrecision::F64),
            RequiredCaps::Float64Int64,
            wg,
        ),
    ]);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn stable_pool_and_convolution_validate_with_distinct_f64_caps() {
    validate_rows([
        (
            "stable-pool-f32",
            deep_learning::stable_pool_1d(NumericPrecision::F32),
            RequiredCaps::None,
            deep_learning::POOL_WORKGROUP_SIZE,
        ),
        (
            "stable-pool-f64",
            deep_learning::stable_pool_1d(NumericPrecision::F64),
            RequiredCaps::Float64Int64,
            deep_learning::POOL_WORKGROUP_SIZE,
        ),
        (
            "conv1d-f32",
            conv::CONV1D_SHADER_F32.to_owned(),
            RequiredCaps::None,
            config::WORKGROUP_SIZE,
        ),
        (
            "conv1d-f64",
            conv::CONV1D_SHADER_F64.to_owned(),
            RequiredCaps::Float64,
            config::WORKGROUP_SIZE,
        ),
    ]);
}
