use crate::{MirOutputTarget, MirOutputTargetList, MirStmt, MirStmtKind};
use runmat_hir::{HirStmt, HirStmtKind, OutputTarget, SemanticError};

use super::{expr::lower_expr, place::lower_place, MirLoweringContext};

pub(crate) fn lower_stmt(
    ctx: &MirLoweringContext,
    stmt: &HirStmt,
) -> Result<Vec<MirStmt>, SemanticError> {
    Ok(match &stmt.kind {
        HirStmtKind::Assign(place, expr, _) => vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: lower_place(ctx, place)?,
                value: lower_expr(ctx, expr)?,
            },
            span: stmt.span,
        }],
        HirStmtKind::MultiAssign(targets, expr, _) => vec![MirStmt {
            kind: MirStmtKind::MultiAssign {
                targets: MirOutputTargetList {
                    targets: targets
                        .targets
                        .iter()
                        .map(|target| lower_output_target(ctx, target))
                        .collect::<Result<_, _>>()?,
                },
                value: lower_expr(ctx, expr)?,
            },
            span: stmt.span,
        }],
        HirStmtKind::ExprStmt(expr, _) => vec![MirStmt {
            kind: MirStmtKind::Expr(lower_expr(ctx, expr)?),
            span: stmt.span,
        }],
        HirStmtKind::Return => Vec::new(),
        _ => {
            return Err(SemanticError::new(
                "MIR lowering for statement is not implemented yet",
            ))
        }
    })
}

fn lower_output_target(
    ctx: &MirLoweringContext,
    target: &OutputTarget,
) -> Result<MirOutputTarget, SemanticError> {
    Ok(match target {
        OutputTarget::Place(place) => MirOutputTarget::Place(lower_place(ctx, place)?),
        OutputTarget::Discard => MirOutputTarget::Discard,
        OutputTarget::VarargoutExpansion => MirOutputTarget::VarargoutExpansion,
    })
}
