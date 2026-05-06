use crate::{MirIndexing, MirPlace};
use runmat_hir::{HirExpr, HirExprKind, HirPlace, SemanticError};

use super::MirLoweringContext;

pub(crate) fn lower_place(
    ctx: &MirLoweringContext,
    place: &HirPlace,
) -> Result<MirPlace, SemanticError> {
    Ok(match place {
        HirPlace::Binding(binding) => MirPlace::Local(ctx.local_for_binding(*binding)?),
        HirPlace::Member(base, member) => {
            MirPlace::Member(Box::new(lower_expr_place(ctx, base)?), member.clone())
        }
        HirPlace::Index(base, indexing) | HirPlace::IndexCell(base, indexing) => MirPlace::Index(
            Box::new(lower_expr_place(ctx, base)?),
            MirIndexing {
                kind: indexing.kind.clone(),
                components: indexing.components.clone(),
                result_context: indexing.result_context.clone(),
            },
        ),
        _ => {
            return Err(SemanticError::new(
                "MIR lowering for place is not implemented yet",
            ))
        }
    })
}

fn lower_expr_place(ctx: &MirLoweringContext, expr: &HirExpr) -> Result<MirPlace, SemanticError> {
    Ok(match &expr.kind {
        HirExprKind::Binding(binding) => MirPlace::Local(ctx.local_for_binding(*binding)?),
        HirExprKind::Member(base, member) => {
            MirPlace::Member(Box::new(lower_expr_place(ctx, base)?), member.clone())
        }
        HirExprKind::Index(base, indexing) => MirPlace::Index(
            Box::new(lower_expr_place(ctx, base)?),
            MirIndexing {
                kind: indexing.kind.clone(),
                components: indexing.components.clone(),
                result_context: indexing.result_context.clone(),
            },
        ),
        _ => return Err(SemanticError::new("expression is not a simple MIR place")),
    })
}
