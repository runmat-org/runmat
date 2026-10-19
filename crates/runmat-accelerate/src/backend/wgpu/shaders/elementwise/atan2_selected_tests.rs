use super::*;
use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn ordinary_f64_binary_compiles_without_integer64_but_atan2_requires_it() {
    for broadcast in [false, true] {
        for (kind, caps) in [
            (RealBinaryShaderKind::General, RequiredCaps::Float64),
            (
                RealBinaryShaderKind::SignedZeroAtan2,
                RequiredCaps::Float64Int64,
            ),
        ] {
            let source = binary_shader_f64(kind, broadcast);
            // These production sources have a literal workgroup size of 512;
            // each requested substitution checks the same emitted source.
            for requested_workgroup in [64, 128, 256] {
                validate_generated(
                    &format!(
                        "binary-{kind:?}-broadcast-{broadcast}-requested-wg{requested_workgroup}"
                    ),
                    &source,
                    requested_workgroup,
                    8,
                    caps,
                );
            }
        }
    }
}
