use crate::{MirPlace, MirStmt};
use runmat_hir::{HirError, HirExpr, HirExprKind, HirPlace};

use super::{
    expr::{lower_indexing, lower_operand},
    MirLoweringContext,
};

pub(crate) fn lower_place(
    ctx: &MirLoweringContext,
    place: &HirPlace,
    temps: &mut Vec<MirStmt>,
) -> Result<MirPlace, HirError> {
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
) -> Result<MirPlace, HirError> {
    let lowered = match &expr.kind {
        HirExprKind::Binding(binding) => MirPlace::Local(ctx.local_for_binding(*binding)?),
        HirExprKind::Member { base, member, .. } => MirPlace::Member(
            Box::new(lower_expr_place(ctx, base, temps)?),
            member.clone(),
        ),
        HirExprKind::MemberDynamic { base, member, .. } => MirPlace::DynamicMember(
            Box::new(lower_expr_place(ctx, base, temps)?),
            lower_operand(ctx, member, temps)?,
        ),
        HirExprKind::Index(base, indexing) => MirPlace::Index(
            Box::new(lower_expr_place(ctx, base, temps)?),
            lower_indexing(ctx, indexing, temps)?,
        ),
        _ => {
            let operand = lower_operand(ctx, expr, temps)?;
            match operand {
                crate::MirOperand::Local(local) => MirPlace::Local(local),
                _ => return Err(HirError::new("expression is not a simple MIR place")),
            }
        }
    };
    Ok(lowered)
}

pub(crate) fn lower_expr_place_with_replacements(
    ctx: &MirLoweringContext,
    expr: &HirExpr,
    temps: &mut Vec<MirStmt>,
    await_replacements: &std::collections::HashMap<runmat_hir::ExprId, crate::MirOperand>,
) -> Result<MirPlace, HirError> {
    if let Some(operand) = await_replacements.get(&expr.id) {
        return match operand {
            crate::MirOperand::Local(local) => Ok(MirPlace::Local(*local)),
            _ => Err(HirError::new(
                "await replacement cannot serve as an assignment destination",
            )),
        };
    }
    Ok(match &expr.kind {
        HirExprKind::Binding(binding) => MirPlace::Local(ctx.local_for_binding(*binding)?),
        HirExprKind::Member { base, member, .. } => MirPlace::Member(
            Box::new(lower_expr_place_with_replacements(
                ctx,
                base,
                temps,
                await_replacements,
            )?),
            member.clone(),
        ),
        HirExprKind::MemberDynamic { base, member, .. } => MirPlace::DynamicMember(
            Box::new(lower_expr_place_with_replacements(
                ctx,
                base,
                temps,
                await_replacements,
            )?),
            super::expr::lower_operand_with_replacements(ctx, member, temps, await_replacements)?,
        ),
        HirExprKind::Index(base, indexing) => MirPlace::Index(
            Box::new(lower_expr_place_with_replacements(
                ctx,
                base,
                temps,
                await_replacements,
            )?),
            super::expr::lower_indexing_with_replacements(
                ctx,
                indexing,
                temps,
                await_replacements,
            )?,
        ),
        _ => {
            let operand =
                super::expr::lower_operand_with_replacements(ctx, expr, temps, await_replacements)?;
            match operand {
                crate::MirOperand::Local(local) => MirPlace::Local(local),
                _ => return Err(HirError::new("expression is not a simple MIR place")),
            }
        }
    })
}
