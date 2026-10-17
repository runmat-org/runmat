//! Validate the production specialized real-unary source for every opcode.

use super::{validate_generated, RequiredCaps};
use crate::backend::wgpu::shaders::elementwise::{
    real_unary_requires_ieee_words, real_unary_shader,
};
use crate::backend::wgpu::types::{NumericPrecision, UnaryOpCode};

const OPS: [UnaryOpCode; 40] = [
    UnaryOpCode::Sin,
    UnaryOpCode::Cos,
    UnaryOpCode::Abs,
    UnaryOpCode::Exp,
    UnaryOpCode::Log,
    UnaryOpCode::Sqrt,
    UnaryOpCode::Sign,
    UnaryOpCode::Real,
    UnaryOpCode::Imag,
    UnaryOpCode::Conj,
    UnaryOpCode::Angle,
    UnaryOpCode::Expm1,
    UnaryOpCode::Log1p,
    UnaryOpCode::Log10,
    UnaryOpCode::Log2,
    UnaryOpCode::Pow2,
    UnaryOpCode::Floor,
    UnaryOpCode::Ceil,
    UnaryOpCode::Fix,
    UnaryOpCode::Tan,
    UnaryOpCode::Asin,
    UnaryOpCode::Acos,
    UnaryOpCode::Atan,
    UnaryOpCode::Sinh,
    UnaryOpCode::Cosh,
    UnaryOpCode::Tanh,
    UnaryOpCode::Asinh,
    UnaryOpCode::Acosh,
    UnaryOpCode::Atanh,
    UnaryOpCode::Gamma,
    UnaryOpCode::Factorial,
    UnaryOpCode::Single,
    UnaryOpCode::NextPow2,
    UnaryOpCode::Sinc,
    UnaryOpCode::Heaviside,
    UnaryOpCode::Erf,
    UnaryOpCode::Gammaln,
    UnaryOpCode::Round,
    UnaryOpCode::Erfcinv,
    UnaryOpCode::Spacing,
];

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn specialized_real_unary_sources_validate_for_all_opcodes_and_precisions() {
    assert_eq!(OPS.len(), UnaryOpCode::Spacing as usize + 1);
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        for op in OPS {
            let caps = match precision {
                NumericPrecision::F32 => RequiredCaps::None,
                NumericPrecision::F64 if real_unary_requires_ieee_words(op) => {
                    RequiredCaps::Float64Int64
                }
                NumericPrecision::F64 => RequiredCaps::Float64,
            };
            let source = real_unary_shader(op, precision);
            for workgroup in [64, 128] {
                validate_generated(
                    &format!("real-unary-op{}-{precision:?}-wg{workgroup}", op as u32),
                    &source,
                    workgroup,
                    8,
                    caps,
                );
            }
        }
    }
}
