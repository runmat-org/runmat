use crate::{MirPlace, MirStmt};
use runmat_hir::{HirExpr, HirExprKind, HirPlace, SemanticError};

use super::{
    expr::{lower_indexing, lower_operand},
    MirLoweringContext,
};

pub(crate) fn lower_place(
    ctx: &MirLoweringContext,
    place: &HirPlace,
    temps: &mut Vec<MirStmt>,
) -> Result<MirPlace, SemanticError> {
    Ok(match place {
        HirPlace::Binding(binding) => MirPlace::Local(ctx.local_for_binding(*binding)?),
        HirPlace::Member(base, member) => MirPlace::Member(
            Box::new(lower_expr_place(ctx, base, temps)?),
            member.clone(),
        ),
        HirPlace::MemberDynamic(base, member) => MirPlace::DynamicMember(
            Box::new(lower_expr_place(ctx, base, temps)?),
            lower_operand(ctx, member, temps)?,
        ),
        HirPlace::Index(base, indexing) | HirPlace::IndexCell(base, indexing) => MirPlace::Index(
            Box::new(lower_expr_place(ctx, base, temps)?),
            lower_indexing(ctx, indexing, temps)?,
        ),
    })
}

fn lower_expr_place(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    temps: &mut Vec<MirStmt>,
) -> Result<MirPlace, SemanticError> {
    Ok(match &expr.kind {
        HirExprKind::Binding(binding) => MirPlace::Local(ctx.local_for_binding(*binding)?),
        HirExprKind::Member(base, member) => MirPlace::Member(
            Box::new(lower_expr_place(ctx, base, temps)?),
            member.clone(),
        ),
        HirExprKind::MemberDynamic(base, member) => MirPlace::DynamicMember(
            Box::new(lower_expr_place(ctx, base, temps)?),
            lower_operand(ctx, member, temps)?,
        ),
        HirExprKind::Index(base, indexing) => MirPlace::Index(
            Box::new(lower_expr_place(ctx, base, temps)?),
            lower_indexing(ctx, indexing, temps)?,
        ),
        _ => return Err(SemanticError::new("expression is not a simple MIR place")),
    })
}
