//! Shared numerical expression owner for already admitted real unary math.
//! This is not name-based admission: canonical declarations select instructions;
//! the resolver must still prove their class, domain, purity and binding.
use super::{FusionError, ScalarType};
use runmat_types::ScalarUnaryMath;

pub fn unary_math_expression(
    operation: ScalarUnaryMath,
    scalar: ScalarType,
    input: &str,
) -> Result<String, FusionError> {
    if !matches!(scalar, ScalarType::F32 | ScalarType::F64) {
        return Err(FusionError::UnsupportedPrecision(scalar));
    }
    // Preserve the canonical templates' same-precision intrinsic. In
    // particular, Double operands are never routed through an f32 intrinsic.
    let instruction = match operation {
        ScalarUnaryMath::Exp => "exp",
        ScalarUnaryMath::Sin => "sin",
        ScalarUnaryMath::Cos => "cos",
        ScalarUnaryMath::Tan => "tan",
        ScalarUnaryMath::Sinh => "sinh",
        ScalarUnaryMath::Cosh => "cosh",
        ScalarUnaryMath::Tanh => "tanh",
        ScalarUnaryMath::Atan => "atan",
        ScalarUnaryMath::Asinh => "asinh",
        ScalarUnaryMath::Acosh => "acosh",
        ScalarUnaryMath::Atanh => "atanh",
        // These instructions are used by ordinary physical unary recipes.
        // Adding their vocabulary does not admit a new fused numerical route.
        ScalarUnaryMath::Log
        | ScalarUnaryMath::Log1p
        | ScalarUnaryMath::Log10
        | ScalarUnaryMath::Log2
        | ScalarUnaryMath::Sqrt
        | ScalarUnaryMath::Asin
        | ScalarUnaryMath::Acos => {
            return Err(FusionError::Message(
                "unary instruction has no fused implementation",
            ));
        }
    };
    Ok(format!("{instruction}({input})"))
}
