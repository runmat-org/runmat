use std::collections::HashMap;

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
    replacements: &mut HashMap<ExprId, MirOperand>,
) -> Result<bool, HirError> {
    if replacements.contains_key(&expr.id) {
        return Ok(false);
    }
    if expr.id == awaited {
        return Ok(true);
    }
    for child in expression_children(expr) {
        if prepare_expr_prefix(ctx, child, awaited, output, replacements)? {
            return Ok(true);
        }
        cache_expr(ctx, child, output, replacements)?;
    }
    Ok(false)
}

pub(super) fn cache_expr(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    output: &mut Vec<MirStmt>,
    replacements: &mut HashMap<ExprId, MirOperand>,
) -> Result<(), HirError> {
    if !replacements.contains_key(&expr.id) {
        let operand = lower_operand_with_replacements(ctx, expr, output, replacements)?;
        replacements.insert(expr.id, operand);
    }
    Ok(())
}
