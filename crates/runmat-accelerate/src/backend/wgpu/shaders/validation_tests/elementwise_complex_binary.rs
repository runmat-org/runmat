//! Validate all complex/real operand layouts through the production builders.

use super::{validate_generated, RequiredCaps};
use crate::backend::wgpu::shaders::elementwise::{
    complex_binary_broadcast_shader, complex_binary_shader, complex_from_real_imag_shader,
    complex_from_real_shader, ComplexBinaryOp,
};
use crate::backend::wgpu::types::NumericPrecision;

const OPS: [ComplexBinaryOp; 4] = [
    ComplexBinaryOp::Add,
    ComplexBinaryOp::Sub,
    ComplexBinaryOp::Mul,
    ComplexBinaryOp::Div,
];

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_complex_binary_sources_validate_all_operand_classes_and_broadcast_routes() {
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        for op in OPS {
            let caps = match (precision, op) {
                (NumericPrecision::F32, _) => RequiredCaps::None,
                (NumericPrecision::F64, ComplexBinaryOp::Div) => RequiredCaps::Float64Int64,
                (NumericPrecision::F64, _) => RequiredCaps::Float64,
            };
            for lhs_complex in [false, true] {
                for rhs_complex in [false, true] {
                    for (broadcast, source) in [
                        (
                            false,
                            complex_binary_shader(op, precision, lhs_complex, rhs_complex),
                        ),
                        (
                            true,
                            complex_binary_broadcast_shader(
                                op,
                                precision,
                                lhs_complex,
                                rhs_complex,
                            ),
                        ),
                    ] {
                        for workgroup in [64, 128] {
                            validate_generated(
                                &format!(
                                    "complex-binary-{op:?}-{precision:?}-lhs{lhs_complex}-rhs{rhs_complex}-broadcast{broadcast}-wg{workgroup}"
                                ),
                                &source,
                                workgroup,
                                8,
                                caps,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn complex_construction_sources_validate_each_scalar_layout_and_precision() {
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        let caps = match precision {
            NumericPrecision::F32 => RequiredCaps::None,
            NumericPrecision::F64 => RequiredCaps::Float64,
        };
        let source = complex_from_real_shader(precision);
        for workgroup in [64, 128] {
            validate_generated(
                &format!("complex-from-real-{precision:?}-wg{workgroup}"),
                &source,
                workgroup,
                8,
                caps,
            );
        }
        for real_scalar in [false, true] {
            for imag_scalar in [false, true] {
                let source = complex_from_real_imag_shader(precision, real_scalar, imag_scalar);
                for workgroup in [64, 128] {
                    validate_generated(
                        &format!(
                            "complex-from-real-imag-{precision:?}-real{real_scalar}-imag{imag_scalar}-wg{workgroup}"
                        ),
                        &source,
                        workgroup,
                        8,
                        caps,
                    );
                }
            }
        }
    }
}
