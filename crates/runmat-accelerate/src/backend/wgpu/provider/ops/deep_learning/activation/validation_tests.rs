//! Validate the exact forward and adjoint activation sources used by the provider.

use super::*;
use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};
use std::collections::BTreeSet;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn activation_forward_and_adjoint_sources_validate_with_exact_caps() {
    let rows = [
        ("relu-f32", RELU_SHADER_F32.to_owned(), RequiredCaps::None),
        (
            "relu-f64",
            RELU_SHADER_F64.to_owned(),
            RequiredCaps::Float64,
        ),
        (
            "elu-f32",
            elu_shader(NumericPrecision::F32).to_owned(),
            RequiredCaps::None,
        ),
        (
            "elu-f64",
            elu_shader(NumericPrecision::F64).to_owned(),
            RequiredCaps::Float64,
        ),
        (
            "relu-adjoint-f32",
            RELU_ADJOINT_F32.to_owned(),
            RequiredCaps::None,
        ),
        (
            "relu-adjoint-f64",
            crate::backend::wgpu::shaders::ieee_words::inject(RELU_ADJOINT_F64),
            RequiredCaps::Float64Int64,
        ),
        (
            "elu-adjoint-f32",
            ELU_ADJOINT_F32.to_owned(),
            RequiredCaps::None,
        ),
        (
            "elu-adjoint-f64",
            ELU_ADJOINT_F64.to_owned(),
            RequiredCaps::Float64,
        ),
    ];
    let mut names = BTreeSet::new();
    for (name, source, caps) in rows {
        assert!(names.insert(name), "duplicate activation shader: {name}");
        validate_generated(
            name,
            &source,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
            8,
            caps,
        );
    }
    assert_eq!(names.len(), 8, "activation shader inventory changed");
}
