//! Validate every selected complex-unary shader, including real-valued outputs.

use super::{validate_generated, RequiredCaps};
use crate::backend::wgpu::shaders::elementwise::{
    complex_unary_requires_ieee_words, complex_unary_shader, ComplexUnaryOp,
};
use crate::backend::wgpu::types::NumericPrecision;

const OPS: [ComplexUnaryOp; 12] = [
    ComplexUnaryOp::Real,
    ComplexUnaryOp::Imag,
    ComplexUnaryOp::Abs,
    ComplexUnaryOp::Conj,
    ComplexUnaryOp::Angle,
    ComplexUnaryOp::Sin,
    ComplexUnaryOp::Sinc,
    ComplexUnaryOp::Cos,
    ComplexUnaryOp::Sinh,
    ComplexUnaryOp::Cosh,
    ComplexUnaryOp::Tan,
    ComplexUnaryOp::Sign,
];

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_complex_unary_sources_validate_for_each_operation_and_precision() {
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        for op in OPS {
            let caps = match precision {
                NumericPrecision::F32 => RequiredCaps::None,
                NumericPrecision::F64 if complex_unary_requires_ieee_words(op) => {
                    RequiredCaps::Float64Int64
                }
                NumericPrecision::F64 => RequiredCaps::Float64,
            };
            let source = complex_unary_shader(op, precision);
            for workgroup in [64, 128] {
                validate_generated(
                    &format!("complex-unary-{op:?}-{precision:?}-wg{workgroup}"),
                    &source,
                    workgroup,
                    8,
                    caps,
                );
            }
        }
    }
}
