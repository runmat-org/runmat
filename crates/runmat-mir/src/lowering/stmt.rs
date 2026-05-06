use crate::{MirOutputTarget, MirOutputTargetList, MirStmt, MirStmtKind};
use runmat_hir::{HirStmt, HirStmtKind, OutputTarget, SemanticError, WorkspaceEffect};

use super::{expr::lower_expr, place::lower_place, MirLoweringContext};

pub(crate) fn lower_stmt(
    ctx: &MirLoweringContext,
    stmt: &HirStmt,
) -> Result<Vec<MirStmt>, SemanticError> {
    Ok(match &stmt.kind {
        HirStmtKind::Assign(place, expr, _) => {
            let mut stmts = Vec::new();
            let value = lower_expr(ctx, expr, &mut stmts)?;
            stmts.push(MirStmt {
                kind: MirStmtKind::Assign {
                    place: lower_place(ctx, place)?,
                    value,
                },
                span: stmt.span,
            });
            stmts
        }
        HirStmtKind::MultiAssign(targets, expr, _) => {
            let mut stmts = Vec::new();
            let value = lower_expr(ctx, expr, &mut stmts)?;
            stmts.push(MirStmt {
                kind: MirStmtKind::MultiAssign {
                    targets: MirOutputTargetList {
                        targets: targets
                            .targets
                            .iter()
                            .map(|target| lower_output_target(ctx, target))
                            .collect::<Result<_, _>>()?,
                    },
                    value,
                },
                span: stmt.span,
            });
            stmts
        }
        HirStmtKind::ExprStmt(expr, _) => {
            let mut stmts = Vec::new();
            let value = lower_expr(ctx, expr, &mut stmts)?;
            stmts.push(MirStmt {
                kind: MirStmtKind::Expr(value),
                span: stmt.span,
            });
            stmts
        }
        HirStmtKind::Global(bindings) => vec![MirStmt {
            kind: MirStmtKind::WorkspaceEffect {
                effect: WorkspaceEffect::MutatesGlobal,
                bindings: bindings
                    .iter()
                    .map(|binding| ctx.local_for_binding(*binding))
                    .collect::<Result<_, _>>()?,
            },
            span: stmt.span,
        }],
        HirStmtKind::Persistent(bindings) => vec![MirStmt {
            kind: MirStmtKind::WorkspaceEffect {
                effect: WorkspaceEffect::MutatesPersistent,
                bindings: bindings
                    .iter()
                    .map(|binding| ctx.local_for_binding(*binding))
                    .collect::<Result<_, _>>()?,
            },
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
