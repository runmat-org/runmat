//! Validate both round-to-digits modes emitted by the production builder.

use super::{validate_generated, RequiredCaps};
use crate::backend::wgpu::shaders::elementwise::round_digits_shader;
use crate::backend::wgpu::types::NumericPrecision;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn round_digits_sources_validate_both_modes_and_precisions() {
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        let caps = match precision {
            NumericPrecision::F32 => RequiredCaps::None,
            NumericPrecision::F64 => RequiredCaps::Float64Int64,
        };
        for significant in [false, true] {
            for workgroup in [64, 128, 256, 512] {
                let source = round_digits_shader(precision, significant, workgroup);
                validate_generated(
                    &format!("round-digits-{precision:?}-significant{significant}-wg{workgroup}"),
                    &source,
                    workgroup,
                    8,
                    caps,
                );
            }
        }
    }
}
