use runmat_hir::{ExprId, HirError, HirExpr};

use crate::{MirOperand, MirStmt};

use super::super::{
    evaluation_order::expression_children, expr::lower_operand_with_replacements,
    MirLoweringContext,
};

pub(crate) fn prepare_expr_prefix(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    awaited: ExprId,
    output: &mut Vec<MirStmt>,
    replacements: &mut crate::lowering::AwaitReplacements,
) -> Result<bool, HirError> {
    prepare_expr_prefix_in_context(ctx, expr, awaited, output, replacements, None)
}

pub(crate) fn prepare_expr_prefix_in_context(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    awaited: ExprId,
    output: &mut Vec<MirStmt>,
    replacements: &mut crate::lowering::AwaitReplacements,
    context: Option<crate::lowering::fixed_target::SelectorRegionContext<'_>>,
) -> Result<bool, HirError> {
    if replacements.contains_key(&expr.id) {
        return Ok(false);
    }
    if expr.id == awaited {
        return Ok(true);
    }
    for child in expression_children(expr) {
        if prepare_expr_prefix_in_context(ctx, child, awaited, output, replacements, context)? {
            return Ok(true);
        }
        cache_expr_in_context(ctx, child, output, replacements, context)?;
    }
    Ok(false)
}

pub(super) fn cache_expr(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    output: &mut Vec<MirStmt>,
    replacements: &mut crate::lowering::AwaitReplacements,
) -> Result<(), HirError> {
    cache_expr_in_context(ctx, expr, output, replacements, None)
}

fn cache_expr_in_context(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    output: &mut Vec<MirStmt>,
    replacements: &mut crate::lowering::AwaitReplacements,
    context: Option<crate::lowering::fixed_target::SelectorRegionContext<'_>>,
) -> Result<(), HirError> {
    if !replacements.contains_key(&expr.id) {
        if let Some(context) = context {
            let destination = if expr.expands_call_argument_sequence() {
                crate::MirFixedSelectorDestination::Sequence(ctx.fresh_sequence_local())
            } else {
                crate::MirFixedSelectorDestination::Scalar(ctx.fresh_temp(expr.span))
            };
            let region = crate::lowering::fixed_target::expression_region(
                ctx,
                expr,
                replacements,
                destination,
            )?;
            context.publish(output, region, destination);
            match destination {
                crate::MirFixedSelectorDestination::Scalar(local) => {
                    replacements.insert(expr.id, MirOperand::Local(local));
                }
                crate::MirFixedSelectorDestination::Sequence(sequence) => {
                    replacements.insert_sequence(expr.id, sequence)
                }
            }
            return Ok(());
        }
        if expr.expands_call_argument_sequence() {
            let argument = super::super::expr::lower_call_arg(ctx, expr, output, replacements)?;
            let crate::MirCallArg::CapturedSequence(sequence) = argument else {
                return Err(HirError::new(
                    "expanded await prefix did not produce a captured sequence",
                ));
            };
            replacements.insert_sequence(expr.id, sequence);
            return Ok(());
        }
        let operand = lower_operand_with_replacements(ctx, expr, output, replacements)?;
        replacements.insert(expr.id, operand);
    }
    Ok(())
}

pub(crate) fn operand_in_context(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    output: &mut Vec<MirStmt>,
    replacements: &crate::lowering::AwaitReplacements,
    context: Option<crate::lowering::fixed_target::SelectorRegionContext<'_>>,
) -> Result<MirOperand, HirError> {
    let Some(context) = context else {
        return lower_operand_with_replacements(ctx, expr, output, replacements);
    };
    let local = ctx.fresh_temp(expr.span);
    let destination = crate::MirFixedSelectorDestination::Scalar(local);
    let region =
        crate::lowering::fixed_target::expression_region(ctx, expr, replacements, destination)?;
    context.publish(output, region, destination);
    Ok(MirOperand::Local(local))
}
