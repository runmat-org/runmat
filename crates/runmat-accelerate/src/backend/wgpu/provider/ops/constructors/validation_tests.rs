//! GPU-free validation of the private ndgrid production builder.

use super::*;
use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn ndgrid_builder_validates_both_selected_numeric_classes() {
    for workgroup in [64, 128] {
        for (precision, scalar, caps) in [
            (NumericPrecision::F32, "f32", RequiredCaps::None),
            (NumericPrecision::F64, "f64", RequiredCaps::Float64),
        ] {
            let source = ndgrid_shader(precision, workgroup);
            assert!(source.contains(&format!("data: array<{scalar}>")));
            assert!(source.contains("let coord = (linear / params.stride) % params.axis_len;"));
            assert!(source.contains("out.data[linear] = axis.data[coord];"));
            validate_generated(
                &format!("ndgrid-{precision:?}-wg{workgroup}"),
                &source,
                workgroup,
                8,
                caps,
            );
        }
    }
}
