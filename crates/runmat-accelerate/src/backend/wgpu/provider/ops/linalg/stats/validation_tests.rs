//! GPU-free validation of the covariance-to-correlation production builder.

use super::*;
use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn covariance_to_correlation_builder_validates_both_precisions_and_ieee_caps() {
    for (precision, scalar, tolerance, caps) in [
        (NumericPrecision::F32, "f32", "1.0e-5", RequiredCaps::None),
        (
            NumericPrecision::F64,
            "f64",
            "1.0e-10",
            RequiredCaps::Float64Int64,
        ),
    ] {
        let source = covariance_to_correlation_shader(precision);
        assert!(source.contains(&format!("data: array<{scalar}>")));
        assert!(source.contains(&format!("{scalar}({tolerance})")));
        assert!(source.contains("atomicMax(&errors.code, code);"));
        validate_generated(
            &format!("covariance-to-correlation-{precision:?}"),
            &source,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
            8,
            caps,
        );
    }
}
