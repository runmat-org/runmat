//! The four fixed-workgroup templates selected by the basic elementwise bundles.
//! F64 general/Atan2 binary and specialized unary sources have separate coverage.

use super::{validate_generated, RequiredCaps};
use crate::backend::wgpu::shaders::elementwise;
use std::collections::BTreeSet;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_basic_binary_and_scalar_pipeline_sources_validate() {
    let cases = [
        (
            "binary-f32",
            elementwise::BINARY_SHADER_F32,
            RequiredCaps::None,
        ),
        (
            "binary-broadcast-f32",
            elementwise::BINARY_BROADCAST_SHADER_F32,
            RequiredCaps::None,
        ),
        (
            "scalar-f32",
            elementwise::SCALAR_SHADER_F32,
            RequiredCaps::None,
        ),
        (
            "scalar-f64",
            elementwise::SCALAR_SHADER_F64,
            RequiredCaps::Float64,
        ),
    ];
    let mut labels = BTreeSet::new();
    for (name, source, caps) in cases {
        assert!(labels.insert(name));
        // These selected templates fix WG512 in their source; unlike the
        // tokenized families, varying the validator's WG argument would not
        // exercise a different production variant.
        assert!(source.contains("@compute @workgroup_size(512)"));
        assert!(!source.contains("@WG@"));
        validate_generated(name, source, 512, 8, caps);
    }
    assert_eq!(
        labels.len(),
        4,
        "basic elementwise bundle inventory changed"
    );
}
