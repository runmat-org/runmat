//! Original request-local scalar words and physical equivalent selection.
use super::*;
use runmat_accelerate_api::scalar_program::ExactBinary32Product;
use runmat_types::{OperatorKind, RealFloatingRecipe, ScalarOperation};

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct ExactProductReceipt {
    pub base: ValueId,
    pub control: ValueId,
    pub exact: ExactBinary32Product,
}

pub(super) fn record(
    program: &mut ResolvedProgram,
    source: &FusionOp,
    output: ValueId,
    inputs: &[ValueId],
    values: &[Option<&Value>],
) -> Result<()> {
    let [left, right] = inputs else { return Ok(()) };
    let operator = if let Some(declaration) = program.declaration(output) {
        match declaration
            .semantics
            .and_then(|semantic| semantic.scalar_operation)
        {
            Some(ScalarOperation::Operator(operator)) => operator,
            _ => return Ok(()),
        }
    } else {
        match source {
            FusionOp::Primitive { op, .. } => op.operator_kind(),
            _ => return Ok(()),
        }
    };
    let Some(recipe) =
        RealFloatingRecipe::for_facts(operator, program.fact(*left)?, program.fact(*right)?)
    else {
        return Ok(());
    };
    if recipe.operator() != OperatorKind::ElementwiseMultiply
        || recipe.computation_class() != NumericClass::Double
        || recipe.result_class() != NumericClass::Single
    {
        return Ok(());
    }
    for (base, control, position) in [(*left, *right, 1), (*right, *left, 0)] {
        if program.floating_class(base)? != NumericClass::Single
            || program.floating_class(control)? != NumericClass::Double
            || program.fact(control)?.shape.element_count() != Some(1)
        {
            continue;
        }
        let Some(runmat_runtime::builtins::common::spec::FusionScalarControl::DoubleBits(bits)) =
            exact_scalar_control(values.get(position).copied().flatten())
        else {
            continue;
        };
        if let Some(exact) = ExactBinary32Product::for_double_bits(bits) {
            program.exact_products.insert(
                output,
                ExactProductReceipt {
                    base,
                    control,
                    exact,
                },
            )?;
            break;
        }
    }
    Ok(())
}
