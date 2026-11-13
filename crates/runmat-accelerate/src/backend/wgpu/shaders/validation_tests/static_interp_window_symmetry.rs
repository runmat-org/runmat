//! Validate selected static interpolation, window, and symmetry sources.

use std::collections::HashSet;

use super::{validate_generated, RequiredCaps};
use crate::backend::wgpu::shaders::{ieee_words, interp1, moving_window, nan_predicate, symmetry};
use crate::backend::wgpu::types::NumericPrecision;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn static_selected_numeric_sources_validate_with_exact_capabilities() {
    let mut labels = HashSet::new();
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        let caps = match precision {
            NumericPrecision::F32 => RequiredCaps::None,
            NumericPrecision::F64 => RequiredCaps::Float64Int64,
        };
        let interpolation = nan_predicate::inject(
            match precision {
                NumericPrecision::F32 => interp1::INTERP1_SHADER_F32,
                NumericPrecision::F64 => interp1::INTERP1_SHADER_F64,
            },
            precision,
        );
        let symmetric = nan_predicate::inject(
            match precision {
                NumericPrecision::F32 => symmetry::SYMMETRY_SHADER_F32,
                NumericPrecision::F64 => symmetry::SYMMETRY_SHADER_F64,
            },
            precision,
        );
        let window = match precision {
            NumericPrecision::F32 => moving_window::MOVING_WINDOW_SHADER_F32.to_owned(),
            NumericPrecision::F64 => ieee_words::inject(moving_window::MOVING_WINDOW_SHADER_F64),
        };
        for (family, source) in [
            ("interp1", interpolation.as_str()),
            ("symmetry", symmetric.as_str()),
            ("moving-window", window.as_str()),
        ] {
            for workgroup in [64, 128, 256] {
                let label = format!("{family}-{precision:?}-wg{workgroup}");
                assert!(
                    labels.insert(label.clone()),
                    "duplicate shader case {label}"
                );
                validate_generated(&label, source, workgroup, 8, caps);
            }
        }
    }
    assert_eq!(labels.len(), 2 * 3 * 3);
}
