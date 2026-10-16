//! Shared numerical expression owner for already admitted real unary math.
//! This is not name-based admission: canonical declarations select instructions;
//! the resolver must still prove their class, domain, purity and binding.
use super::{FusionError, FusionExprContext, FusionExprWriter, ScalarType};
use runmat_types::ScalarUnaryMath;

pub fn unary_math_expression(
    operation: ScalarUnaryMath,
    scalar: ScalarType,
    input: &str,
) -> Result<String, FusionError> {
    let mut measured = FusionExprWriter::counting();
    write_unary_math_scalar_expression(operation, scalar, input, &mut measured)?;
    let mut output = String::new();
    output
        .try_reserve_exact(measured.bytes())
        .map_err(|_| FusionError::ExpressionTooLarge)?;
    write_unary_math_scalar_expression(operation, scalar, input, &mut output)?;
    if output.len() != measured.bytes() {
        return Err(FusionError::ExpressionTooLarge);
    }
    Ok(output)
}

/// The checked scalar path and its allocation preflight use one instruction
/// spelling; the input can be a real expression or a measured fragment.
pub fn write_unary_math_scalar_expression(
    operation: ScalarUnaryMath,
    scalar: ScalarType,
    input: impl std::fmt::Display,
    writer: &mut impl std::fmt::Write,
) -> Result<(), FusionError> {
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
    write!(writer, "{instruction}({input})").map_err(|_| FusionError::ExpressionTooLarge)
}

/// Exact counted and emitted WGSL for the already admitted unary instruction.
pub fn write_unary_math_expression(
    operation: ScalarUnaryMath,
    context: &FusionExprContext<'_>,
    writer: &mut FusionExprWriter<'_>,
) -> Result<(), FusionError> {
    write_unary_math_expression_with_arity(operation, context, writer, true)
}

/// Historical fusion declarations that only inspect the first operand retain
/// that admission behavior until their optional-argument policy is migrated.
pub fn write_unary_math_first_expression(
    operation: ScalarUnaryMath,
    context: &FusionExprContext<'_>,
    writer: &mut FusionExprWriter<'_>,
) -> Result<(), FusionError> {
    write_unary_math_expression_with_arity(operation, context, writer, false)
}

fn write_unary_math_expression_with_arity(
    operation: ScalarUnaryMath,
    context: &FusionExprContext<'_>,
    writer: &mut FusionExprWriter<'_>,
    exact_arity: bool,
) -> Result<(), FusionError> {
    if exact_arity && context.inputs.len() != 1 {
        return Err(FusionError::Message(
            "unary fusion requires exactly one data operand",
        ));
    }
    let input = context.inputs.first().ok_or(FusionError::MissingInput(0))?;
    write_unary_math_scalar_expression(operation, context.scalar_ty, input, writer)
}
